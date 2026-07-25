#!/usr/bin/env python3
"""Matched reconstruction-MSE evaluation for the diffusion-vs-FM ablation.

Builds the seed-42 val split via data_factory, takes the first --n anchors in
loader order, feeds identical x_noise to every model, and reports mean±sem MSE
split by anchor_survey. Persists the eval set as results/recon_eval_manifest.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


def _ping(webhook: str, content: str) -> None:
    """Post a partial-progress line to Discord; never raises. Echoes to stdout."""
    print(f"[ping] {content}", flush=True)
    if not webhook:
        return
    try:
        req = urllib.request.Request(
            webhook,
            data=json.dumps({"content": content}).encode(),
            # Discord/Cloudflare 403s the default Python-urllib user agent.
            headers={
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0 (compatible; recon-mse-eval)",
            },
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception as exc:  # noqa: BLE001
        print(f"[eval] ping failed (ignored): {exc}")


def _ping_line(by_survey: Dict[str, List[float]]) -> str:
    parts = []
    for survey in ("hsc", "legacy"):
        vals = by_survey.get(survey, [])
        if vals:
            parts.append(f"{survey.upper()} {sum(vals) / len(vals):.4f}")
    return " | ".join(parts) or "no data"

REPO = Path(__file__).resolve().parents[3]  # tess-generative
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from galaxy_images.galaxy_model.config import load_experiment_config
from galaxy_images.galaxy_model.data_factory import build_neighbors_dataloaders
from galaxy_images.galaxy_model.diffusion_ablation.double_train_ddpm_neighbors import (
    ConditionalDDPMModule,
)
from galaxy_images.galaxy_model.double_train_fm_neighbors import ConditionalFlowMatchingModule

ABL_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ABL_DIR / "results" / "recon_eval_manifest.json"
PAPER_CKPT = (
    Path(__file__).resolve().parents[1] / "checkpoints" / "base" / "snapshot.ckpt"
)


def _sem(values: List[float]) -> float:
    n = len(values)
    if n < 2:
        return float("nan")
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(var / n)


def _unpack_batch(batch):
    if len(batch) == 5:
        x_1, samegal, sameins, masks, metadata = batch
    else:
        x_1, samegal, sameins, metadata = batch
        B, k, _, _, _ = sameins.shape
        masks = torch.ones((B, k), dtype=torch.bool)
    return x_1, samegal, sameins, masks, metadata


def _collect_eval_set(
    val_loader,
    n: int,
) -> Tuple[List[Dict[str, Any]], List[int], List[Any]]:
    """Iterate val loader in order; return per-anchor records + dataset indices + catalog idxs."""
    records: List[Dict[str, Any]] = []
    dataset_indices: List[int] = []
    catalog_idxs: List[Any] = []
    # Reconstruct dataset indices from the Subset if present.
    dataset = val_loader.dataset
    subset_indices = getattr(dataset, "indices", None)

    seen = 0
    batch_offset = 0
    for batch in val_loader:
        x_1, samegal, sameins, masks, metadata = _unpack_batch(batch)
        B = x_1.shape[0]
        for i in range(B):
            if seen >= n:
                break
            if subset_indices is not None:
                ds_idx = int(subset_indices[batch_offset + i])
            else:
                ds_idx = batch_offset + i
            cat_idx = metadata[i].get("idx", ds_idx)
            records.append(
                {
                    "batch_local_i": i,
                    "dataset_index": ds_idx,
                    "catalog_idx": cat_idx,
                    "anchor_survey": metadata[i]["anchor_survey"],
                }
            )
            dataset_indices.append(ds_idx)
            catalog_idxs.append(cat_idx)
            seen += 1
        batch_offset += B
        if seen >= n:
            break

    if seen < n:
        raise RuntimeError(f"Val loader only yielded {seen} anchors; need --n={n}")
    return records, dataset_indices, catalog_idxs


def _load_or_create_manifest(
    manifest_path: Path,
    val_loader,
    n: int,
    noise_seed: int,
    num_steps,
    val_ratio: float,
    split_seed: int,
    batch_size: int,
) -> Dict[str, Any]:
    records, dataset_indices, catalog_idxs = _collect_eval_set(val_loader, n)
    rebuilt = {
        "n": n,
        "noise_seed": noise_seed,
        "num_steps": num_steps,
        "val_ratio": val_ratio,
        "split_seed": split_seed,
        # batch_size is part of the protocol identity: noise is drawn per-batch,
        # so a different batch size silently reassigns noise across anchors.
        "batch_size": batch_size,
        "dataset_indices": dataset_indices,
        "catalog_idxs": catalog_idxs,
        "anchor_surveys": [r["anchor_survey"] for r in records],
    }
    compared = (
        "dataset_indices", "catalog_idxs", "n", "val_ratio", "split_seed",
        "noise_seed", "batch_size",
    )
    if manifest_path.exists():
        with manifest_path.open() as f:
            stored = json.load(f)
        for key in compared:
            if key in stored and stored[key] != rebuilt[key]:
                raise RuntimeError(
                    f"Manifest mismatch on {key!r}: stored={stored[key]!r} "
                    f"rebuilt={rebuilt[key]!r}. Refusing to eval on a different set. "
                    f"Delete {manifest_path} only if intentional."
                )
        missing = [k for k in compared if k not in stored]
        if missing:
            stored.update({k: rebuilt[k] for k in missing})
            with manifest_path.open("w") as f:
                json.dump(stored, f, indent=2)
            print(f"[eval] manifest upgraded with {missing}: {manifest_path}")
        print(f"[eval] manifest OK: {manifest_path}")
        return stored

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as f:
        json.dump(rebuilt, f, indent=2)
    print(f"[eval] wrote manifest: {manifest_path}")
    return rebuilt


def _load_model(path: Path, kind: str, device: torch.device):
    if kind == "ddpm":
        cls = ConditionalDDPMModule
    else:
        cls = ConditionalFlowMatchingModule
    model = cls.load_from_checkpoint(str(path), map_location="cpu")
    model.eval()
    torch.set_grad_enabled(False)
    return model.to(device)


@torch.no_grad()
def _eval_model(
    model,
    val_loader,
    n: int,
    num_steps: int,
    noise_seed: int,
    device: torch.device,
    eta: Optional[float] = None,
) -> Dict[str, List[float]]:
    """Return per-survey lists of per-image MSE."""
    by_survey: Dict[str, List[float]] = defaultdict(list)
    seen = 0
    batch_idx = 0
    is_ddpm = isinstance(model, ConditionalDDPMModule)

    for batch in val_loader:
        x_1, samegal, sameins, masks, metadata = _unpack_batch(batch)
        B = x_1.shape[0]
        take = min(B, n - seen)
        if take <= 0:
            break

        x_1 = x_1[:take].to(device)
        samegal = samegal[:take].to(device)
        sameins = sameins[:take].to(device)
        masks = masks[:take].to(device)
        metadata = metadata[:take]

        gen = torch.Generator(device=device)
        gen.manual_seed(noise_seed + batch_idx)
        noise = torch.randn(
            take,
            model.in_channels,
            model.image_size,
            model.image_size,
            device=device,
            generator=gen,
        )

        sample_kwargs = dict(
            cond_image_samegal=samegal,
            cond_image_sameins=sameins,
            masks=masks,
            num_steps=num_steps,
            x_noise=noise,
        )
        if is_ddpm and eta is not None:
            sample_kwargs["eta"] = eta
            if eta > 0:
                step_gen = torch.Generator(device=device)
                step_gen.manual_seed(noise_seed + 10_000 + batch_idx)
                sample_kwargs["generator"] = step_gen

        pred = model.sample(**sample_kwargs)
        per_image = ((pred - x_1) ** 2).mean(dim=(1, 2, 3))

        for i, m in enumerate(metadata):
            by_survey[m["anchor_survey"]].append(float(per_image[i].item()))

        seen += take
        batch_idx += 1
        if seen >= n:
            break

    return by_survey


def _aggregate_rows(
    label: str,
    objective: str,
    eta: Optional[float],
    by_survey: Dict[str, List[float]],
    note: str = "",
    num_steps: Optional[int] = None,
) -> List[Dict[str, Any]]:
    rows = []
    for survey in ("hsc", "legacy"):
        vals = by_survey.get(survey, [])
        rows.append(
            {
                "model_label": label,
                "objective": objective,
                "eta": "" if eta is None else eta,
                "num_steps": "" if num_steps is None else num_steps,
                "anchor_survey": survey,
                "n": len(vals),
                "mse_mean": sum(vals) / len(vals) if vals else float("nan"),
                "mse_sem": _sem(vals) if vals else float("nan"),
                "note": note,
            }
        )
    return rows


def _print_markdown(rows: List[Dict[str, Any]]) -> None:
    print()
    print("| model | objective | η | steps | survey | n | MSE mean | SEM | note |")
    print("|---|---|---|---|---|---:|---:|---:|---|")
    for r in rows:
        eta = r["eta"] if r["eta"] != "" else "—"
        steps = r.get("num_steps", "") or "—"
        note = r.get("note", "")
        print(
            f"| {r['model_label']} | {r['objective']} | {eta} | {steps} | {r['anchor_survey']} | "
            f"{r['n']} | {r['mse_mean']:.6f} | {r['mse_sem']:.6f} | {note} |"
        )
    print()


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--fm-checkpoint",
        action="append",
        default=[],
        metavar="LABEL:PATH",
        help="FM checkpoint as label:path (repeatable). Bare PATH uses label 'fm'.",
    )
    p.add_argument("--ddpm-checkpoint", type=Path, default=None)
    p.add_argument("--config", type=Path, required=True, help="DDPM (or FM-control) JSON for loaders")
    p.add_argument("--n", type=int, default=256)
    p.add_argument(
        "--num-steps",
        type=int,
        action="append",
        default=None,
        help="Sampler step counts (repeatable). Default: 250",
    )
    p.add_argument(
        "--eta",
        type=float,
        action="append",
        default=None,
        help="DDIM η for DDPM rows (repeatable). Default: 0.0",
    )
    p.add_argument("--noise-seed", type=int, default=1234)
    p.add_argument(
        "--ddpm-clip-range",
        type=float,
        default=None,
        help="If set, rebuild the DDPM inference scheduler with clip_sample=True and "
        "this clip_sample_range — a wide x0-space guard against low-SNR 1/sqrt(alpha) "
        "blowups that is harmless to z-scored pixel values (e.g. 50).",
    )
    p.add_argument("--discord-webhook", type=str, default="")
    p.add_argument("--ping-prefix", type=str, default="recon-mse partial")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument(
        "--paper-checkpoint",
        type=Path,
        default=PAPER_CKPT,
        help="Paper base FM checkpoint (context row). Pass empty string to skip.",
    )
    p.add_argument("--skip-paper", action="store_true")
    args = p.parse_args(argv)

    etas = args.eta if args.eta is not None else [0.0]
    steps_list = args.num_steps if args.num_steps else [250]
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[eval] WARNING: CUDA unavailable, falling back to CPU")
        device = torch.device("cpu")

    config = load_experiment_config(args.config)
    # Deterministic split matching training.
    import pytorch_lightning as pl

    pl.seed_everything(config.trainer.seed, workers=True)
    _, val_loader, _ = build_neighbors_dataloaders(config, config.data.batch_size)

    _load_or_create_manifest(
        manifest_path=args.manifest,
        val_loader=val_loader,
        n=args.n,
        noise_seed=args.noise_seed,
        num_steps=steps_list,
        val_ratio=config.data.val_ratio,
        split_seed=config.trainer.seed,
        batch_size=config.data.batch_size,
    )

    rows: List[Dict[str, Any]] = []
    fieldnames = [
        "model_label",
        "objective",
        "eta",
        "num_steps",
        "anchor_survey",
        "n",
        "mse_mean",
        "mse_sem",
        "note",
    ]

    def _flush() -> None:
        # Persist after every model/η block: a late failure must not lose earlier rows.
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    # FM checkpoints
    for spec in args.fm_checkpoint:
        if ":" in spec and not Path(spec).exists():
            label, path_s = spec.split(":", 1)
        else:
            label, path_s = "fm", spec
        path = Path(path_s)
        print(f"[eval] FM {label}: {path}")
        model = _load_model(path, "fm", device)
        for steps in steps_list:
            by_survey = _eval_model(
                model, val_loader, args.n, steps, args.noise_seed, device, eta=None
            )
            rows.extend(
                _aggregate_rows(label, "flow_matching", None, by_survey, num_steps=steps)
            )
            _flush()
            _ping(
                args.discord_webhook,
                f"🧩 {args.ping_prefix} · `{label}` steps={steps} → {_ping_line(by_survey)}",
            )
        del model
        torch.cuda.empty_cache()

    # DDPM checkpoint at each η
    if args.ddpm_checkpoint is None:
        print("[eval] WARNING: --ddpm-checkpoint not given — DDPM rows omitted entirely")
    if args.ddpm_checkpoint is not None:
        print(f"[eval] DDPM: {args.ddpm_checkpoint}")
        model = _load_model(args.ddpm_checkpoint, "ddpm", device)
        clip_note = ""
        if args.ddpm_clip_range is not None:
            from diffusers import DDIMScheduler

            sched_cfg = dict(model.inference_scheduler.config)
            sched_cfg.update(
                clip_sample=True, clip_sample_range=args.ddpm_clip_range
            )
            model.inference_scheduler = DDIMScheduler.from_config(sched_cfg)
            clip_note = f"x0 clip ±{args.ddpm_clip_range:g}"
            print(f"[eval] DDPM inference scheduler rebuilt with {clip_note}")
        for steps in steps_list:
            for eta in etas:
                by_survey = _eval_model(
                    model, val_loader, args.n, steps, args.noise_seed, device, eta=eta
                )
                rows.extend(
                    _aggregate_rows(
                        "ddpm-eps",
                        "ddpm_epsilon",
                        eta,
                        by_survey,
                        num_steps=steps,
                        note=clip_note,
                    )
                )
                _flush()
                _ping(
                    args.discord_webhook,
                    f"🧩 {args.ping_prefix} · `ddpm-eps` steps={steps} η={eta:g}"
                    f"{' ' + clip_note if clip_note else ''} → {_ping_line(by_survey)}",
                )
        del model
        torch.cuda.empty_cache()

    # Paper base — context row only; must never take down the main results.
    # NOTE: Path("") normalizes to PosixPath('.') which is truthy and exists,
    # so test the string form, not the Path.
    paper_ckpt = str(args.paper_checkpoint) if args.paper_checkpoint else ""
    if not args.skip_paper and paper_ckpt not in ("", ".") and Path(paper_ckpt).is_file():
        try:
            print(f"[eval] paper base: {paper_ckpt}")
            model = _load_model(Path(paper_ckpt), "fm", device)
            paper_steps = max(steps_list)
            by_survey = _eval_model(
                model, val_loader, args.n, paper_steps, args.noise_seed, device, eta=None
            )
            rows.extend(
                _aggregate_rows(
                    "fm-paper",
                    "flow_matching",
                    None,
                    by_survey,
                    note="different training setup (paper run)",
                    num_steps=paper_steps,
                )
            )
            _flush()
            del model
            torch.cuda.empty_cache()
        except Exception as exc:
            print(f"[eval] WARNING: paper-base context row failed and was skipped: {exc}")
    elif not args.skip_paper:
        print(
            "[eval] WARNING: paper row skipped — checkpoint not usable: "
            f"{args.paper_checkpoint!r}"
        )

    _flush()
    _print_markdown(rows)
    print(f"[eval] wrote {args.out}")


if __name__ == "__main__":
    main()
