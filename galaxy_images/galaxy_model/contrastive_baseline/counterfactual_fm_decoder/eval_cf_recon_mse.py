#!/usr/bin/env python3
"""Matched reconstruction-MSE: paper base + fm-control + CF frozen-embedding decoders.

Rebuilds the seed-42 val split via data_factory (default: fm-control ram48 config),
takes the first --n anchors in loader order, feeds identical x_noise to every model,
and reports mean±SEM MSE split by anchor_survey.

Image-conditioned models (paper / fm-control) sample from samegal+sameins images.
CF decoders look up precomputed embeddings by (catalog_idx, survey) from the CF H5
so comparison stays galaxy-aligned even when the image loader uses ram48 and the
H5 was built from efficient_neighs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

REPO = Path(__file__).resolve().parents[4]  # tess-generative
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from galaxy_images.galaxy_model.config import load_experiment_config
from galaxy_images.galaxy_model.data_factory import build_neighbors_dataloaders
from galaxy_images.galaxy_model.double_train_fm_neighbors import ConditionalFlowMatchingModule
from galaxy_images.galaxy_model.contrastive_baseline.counterfactual_fm_decoder.cf_embedding_dataset import (
    CFEmbeddingDataset,
    build_catalog_index,
)
from galaxy_images.galaxy_model.contrastive_baseline.counterfactual_fm_decoder.cf_fm_module import (
    CFEmbeddingFlowMatching,
)

PKG_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PKG_DIR / "results" / "recon_eval_manifest.json"
DEFAULT_CONFIG = (
    REPO / "galaxy_images/galaxy_model/diffusion_ablation/configs/neighbors_fm_control.json"
)
PAPER_CKPT = (
    REPO / "galaxy_images/galaxy_model/checkpoints/base/snapshot.ckpt"
)
DEFAULT_FM_CONTROL = Path(
    "/orcd/pool/007/pablomer/checkpoints_new/diffusion_ablation/best/"
    "fm-control-ram48-h200-75k/best-epoch=270-step=104000.ckpt"
)
# Workspace twin if ORCD path missing
_POOL_FM_CONTROL = Path(
    "/home/pablomer/orcd/pool/checkpoints_new/diffusion_ablation/best/"
    "fm-control-ram48-h200-75k/best-epoch=270-step=104000.ckpt"
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


def _collect_eval_set(val_loader, n: int):
    records: List[Dict[str, Any]] = []
    dataset_indices: List[int] = []
    catalog_idxs: List[Any] = []
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
    data_mode: str = "",
    data_dir: str = "",
) -> Dict[str, Any]:
    records, dataset_indices, catalog_idxs = _collect_eval_set(val_loader, n)
    rebuilt = {
        "n": n,
        "noise_seed": noise_seed,
        "num_steps": num_steps,
        "val_ratio": val_ratio,
        "split_seed": split_seed,
        "batch_size": batch_size,
        "data_mode": data_mode,
        "data_dir": data_dir,
        "dataset_indices": dataset_indices,
        "catalog_idxs": catalog_idxs,
        "anchor_surveys": [r["anchor_survey"] for r in records],
    }
    # num_steps is recorded but deliberately NOT compared: sweeping ODE steps on the same
    # galaxy set is legitimate, and every row carries its own num_steps in the CSV.
    # data_mode/data_dir ARE compared — two stores sharing a catalog produce identical
    # index lists, so without them a store swap would pass unnoticed.
    compared = (
        "dataset_indices", "catalog_idxs", "n", "val_ratio", "split_seed",
        "noise_seed", "batch_size", "data_mode", "data_dir",
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
        print(f"[eval] manifest OK: {manifest_path}")
        return stored

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as f:
        json.dump(rebuilt, f, indent=2)
    print(f"[eval] wrote manifest: {manifest_path}")
    return rebuilt


def _aggregate_rows(
    label: str,
    objective: str,
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
    print("| model | objective | steps | survey | n | MSE mean | SEM | note |")
    print("|---|---|---|---|---:|---:|---:|---|")
    for r in rows:
        steps = r.get("num_steps", "") or "—"
        note = r.get("note", "")
        print(
            f"| {r['model_label']} | {r['objective']} | {steps} | {r['anchor_survey']} | "
            f"{r['n']} | {r['mse_mean']:.6f} | {r['mse_sem']:.6f} | {note} |"
        )
    print()


@torch.no_grad()
def _eval_image_fm(
    model: ConditionalFlowMatchingModule,
    val_loader,
    n: int,
    num_steps: int,
    noise_seed: int,
    device: torch.device,
) -> Dict[str, List[float]]:
    by_survey: Dict[str, List[float]] = defaultdict(list)
    seen = 0
    batch_idx = 0
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
            take, model.in_channels, model.image_size, model.image_size,
            device=device, generator=gen,
        )
        pred = model.sample(
            cond_image_samegal=samegal,
            cond_image_sameins=sameins,
            masks=masks,
            num_steps=num_steps,
            x_noise=noise,
        )
        per_image = ((pred - x_1) ** 2).mean(dim=(1, 2, 3))
        for i, m in enumerate(metadata):
            by_survey[m["anchor_survey"]].append(float(per_image[i].item()))

        seen += take
        batch_idx += 1
        if seen >= n:
            break
    return by_survey


def _lookup_cf_batch(
    cf_ds: CFEmbeddingDataset,
    catalog_index: dict,
    metadata: List[dict],
    device: torch.device,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Build CF cond (+ optional tok_mask) and targets for a batch via (row_idx, survey)."""
    conds = []
    masks = []
    targets = []
    for m in metadata:
        survey_code = 0 if m["anchor_survey"] == "hsc" else 1
        key = (int(m["idx"]), survey_code)
        if key not in catalog_index:
            raise KeyError(
                f"CF H5 missing (row_idx={key[0]}, survey={m['anchor_survey']}). "
                f"Rebuild H5 from the same catalog or check join keys."
            )
        h5_i = catalog_index[key]
        item = cf_ds[h5_i]
        if len(item) == 4:
            cond, tok_mask, target, _ = item
            conds.append(cond)
            masks.append(tok_mask)
            targets.append(target)
        else:
            cond, target, _ = item
            conds.append(cond)
            targets.append(target)
    cond_t = torch.stack(conds).to(device)
    tgt_t = torch.stack(targets).to(device)
    tok_m = torch.stack(masks).to(device) if masks else None
    return cond_t, tok_m, tgt_t


@torch.no_grad()
def _eval_cf_fm(
    model: CFEmbeddingFlowMatching,
    cf_ds: CFEmbeddingDataset,
    catalog_index: dict,
    val_loader,
    n: int,
    num_steps: int,
    noise_seed: int,
    device: torch.device,
    target_tol: float = 1e-2,
) -> Dict[str, List[float]]:
    by_survey: Dict[str, List[float]] = defaultdict(list)
    seen = 0
    batch_idx = 0
    for batch in val_loader:
        x_1, samegal, sameins, masks, metadata = _unpack_batch(batch)
        B = x_1.shape[0]
        take = min(B, n - seen)
        if take <= 0:
            break
        metadata = metadata[:take]

        cond, tok_mask, target = _lookup_cf_batch(
            cf_ds, catalog_index, metadata, device,
        )
        # The CF decoder is scored against the H5 target (its training target) while the
        # image models are scored against the loader's x_1. Those must be the SAME pixels
        # or the MSE columns are not comparable — verify once on the first batch instead
        # of assuming it. Built from ram48 the two agree exactly; built from the 160x160
        # store they differ by float16 rounding (~1e-3), still well inside the tolerance.
        if batch_idx == 0 and target_tol > 0:
            delta = (x_1[:take].to(device) - target).abs().max().item()
            print(f"[eval] CF target vs loader x_1: max|Δ|={delta:.3e} (tol {target_tol:g})")
            if delta > target_tol:
                raise RuntimeError(
                    f"CF H5 targets disagree with the val loader's images by {delta:.3e} "
                    f"(> {target_tol:g}). The CF row would not be comparable to the image-model "
                    f"rows. Rebuild the H5 from the same store the eval config uses "
                    f"(cf_ds.source_data_dir={cf_ds.source_data_dir!r}), or raise --target-tol "
                    f"deliberately."
                )

        gen = torch.Generator(device=device)
        gen.manual_seed(noise_seed + batch_idx)
        noise = torch.randn(
            take, model.in_channels, model.image_size, model.image_size,
            device=device, generator=gen,
        )
        pred = model.sample(cond, num_steps=num_steps, x_noise=noise, tok_mask=tok_mask)
        per_image = ((pred - target) ** 2).mean(dim=(1, 2, 3))
        for i, m in enumerate(metadata):
            by_survey[m["anchor_survey"]].append(float(per_image[i].item()))

        seen += take
        batch_idx += 1
        if seen >= n:
            break
    return by_survey


def _resolve_fm_control(path: Optional[Path]) -> Path:
    if path is not None:
        return path
    if DEFAULT_FM_CONTROL.is_file():
        return DEFAULT_FM_CONTROL
    if _POOL_FM_CONTROL.is_file():
        return _POOL_FM_CONTROL
    return DEFAULT_FM_CONTROL


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                   help="JSON for image-model val loader (default: fm-control ram48).")
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--num-steps", type=int, default=250)
    p.add_argument("--noise-seed", type=int, default=1234)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--paper-checkpoint", type=Path, default=PAPER_CKPT)
    p.add_argument("--skip-paper", action="store_true")
    p.add_argument("--fm-control-checkpoint", type=Path, default=None)
    p.add_argument("--skip-fm-control", action="store_true")
    p.add_argument(
        "--cf-checkpoint",
        action="append",
        default=[],
        metavar="LABEL:PATH",
        help="CF decoder ckpt as label:path (repeatable). Requires --cf-h5.",
    )
    p.add_argument("--cf-h5", type=Path, default=None,
                   help="CF embeddings H5 (required if --cf-checkpoint given).")
    p.add_argument("--target-tol", type=float, default=1e-2,
                   help="Max allowed max|Δ| between the CF H5 targets and the val loader's "
                        "images, checked on the first batch. 0 disables the check.")
    p.add_argument(
        "--cf-cond-style",
        default=None,
        choices=["tokens", "meanpool"],
        help="Override cond_style for all CF ckpts (else read from hparams).",
    )
    args = p.parse_args(argv)

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[eval] WARNING: CUDA unavailable, falling back to CPU")
        device = torch.device("cpu")

    config = load_experiment_config(args.config)
    import pytorch_lightning as pl
    pl.seed_everything(config.trainer.seed, workers=True)
    _, val_loader, _ = build_neighbors_dataloaders(config, config.data.batch_size)

    _load_or_create_manifest(
        manifest_path=args.manifest,
        val_loader=val_loader,
        n=args.n,
        noise_seed=args.noise_seed,
        num_steps=args.num_steps,
        val_ratio=config.data.val_ratio,
        split_seed=config.trainer.seed,
        batch_size=config.data.batch_size,
        data_mode=str(config.data.mode),
        data_dir=str(config.data.efficient_data_dir),
    )

    rows: List[Dict[str, Any]] = []
    fieldnames = [
        "model_label", "objective", "num_steps", "anchor_survey",
        "n", "mse_mean", "mse_sem", "note",
    ]

    def _flush() -> None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    # ---- fm-control
    if not args.skip_fm_control:
        ctrl = _resolve_fm_control(args.fm_control_checkpoint)
        if ctrl.is_file():
            print(f"[eval] fm-control: {ctrl}")
            model = ConditionalFlowMatchingModule.load_from_checkpoint(
                str(ctrl), map_location="cpu",
            )
            model.eval().to(device)
            torch.set_grad_enabled(False)
            by_survey = _eval_image_fm(
                model, val_loader, args.n, args.num_steps, args.noise_seed, device,
            )
            rows.extend(_aggregate_rows(
                "fm-control", "flow_matching", by_survey,
                note="ram48 diffusion-ablation control", num_steps=args.num_steps,
            ))
            _flush()
            del model
            torch.cuda.empty_cache()
        else:
            print(f"[eval] WARNING: fm-control ckpt missing: {ctrl}")

    # ---- paper base
    paper_ckpt = str(args.paper_checkpoint) if args.paper_checkpoint else ""
    if not args.skip_paper and paper_ckpt not in ("", ".") and Path(paper_ckpt).is_file():
        try:
            print(f"[eval] paper base: {paper_ckpt}")
            model = ConditionalFlowMatchingModule.load_from_checkpoint(
                paper_ckpt, map_location="cpu",
            )
            model.eval().to(device)
            by_survey = _eval_image_fm(
                model, val_loader, args.n, args.num_steps, args.noise_seed, device,
            )
            rows.extend(_aggregate_rows(
                "fm-paper", "flow_matching", by_survey,
                note="paper base (may have seen this val split)",
                num_steps=args.num_steps,
            ))
            _flush()
            del model
            torch.cuda.empty_cache()
        except Exception as exc:
            print(f"[eval] WARNING: paper-base row failed and was skipped: {exc}")
    elif not args.skip_paper:
        print(f"[eval] WARNING: paper row skipped — ckpt not usable: {args.paper_checkpoint!r}")

    # ---- CF frozen-embedding decoders
    if args.cf_checkpoint:
        if args.cf_h5 is None or not Path(args.cf_h5).is_file():
            raise SystemExit("--cf-checkpoint requires an existing --cf-h5")

        # Load once per cond_style as needed.
        cf_ds_cache: Dict[str, Tuple[CFEmbeddingDataset, dict]] = {}

        def _get_cf_ds(style: str):
            if style not in cf_ds_cache:
                ds = CFEmbeddingDataset(str(args.cf_h5), cond_style=style)
                cf_ds_cache[style] = (ds, build_catalog_index(ds))
            return cf_ds_cache[style]

        for spec in args.cf_checkpoint:
            if ":" in spec and not Path(spec).exists():
                label, path_s = spec.split(":", 1)
            else:
                label, path_s = "cf-fm", spec
            path = Path(path_s)
            print(f"[eval] CF {label}: {path}")
            model = CFEmbeddingFlowMatching.load_from_checkpoint(
                str(path), map_location="cpu",
            )
            model.eval().to(device)
            style = args.cf_cond_style or getattr(model, "cond_style", None)
            if style is None:
                style = getattr(model.hparams, "cond_style", "tokens")
            cf_ds, cat_index = _get_cf_ds(str(style))
            by_survey = _eval_cf_fm(
                model, cf_ds, cat_index, val_loader,
                args.n, args.num_steps, args.noise_seed, device,
                target_tol=args.target_tol,
            )
            rows.extend(_aggregate_rows(
                label, "cf_frozen_embedding_fm", by_survey,
                note=f"cond_style={style}", num_steps=args.num_steps,
            ))
            _flush()
            del model
            torch.cuda.empty_cache()

    _flush()
    _print_markdown(rows)
    print(f"[eval] wrote {args.out}")


if __name__ == "__main__":
    main()
