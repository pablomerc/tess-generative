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
import os
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


def _ping(webhook: str, content: str) -> None:
    """Best-effort Discord notification. Never raises — a dead webhook must not kill an eval."""
    if not webhook:
        return
    try:
        req = urllib.request.Request(
            webhook,
            data=json.dumps({"content": content}).encode(),
            headers={"Content-Type": "application/json",
                     "User-Agent": "Mozilla/5.0 (compatible; cf-recon-mse)"},
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception as exc:  # noqa: BLE001
        print(f"[eval] discord ping failed (ignored): {exc}", flush=True)


class _Progress:
    """Per-model progress reporter: prints always, pings Discord every `every` images.

    Sampling is the whole cost here (num_steps ODE evaluations per image), so images/s
    is stable after the first batch and the ETA is meaningful almost immediately.
    """

    def __init__(self, webhook: str, label: str, total: int, every: int,
                 model_i: int, model_n: int):
        self.webhook, self.label, self.total = webhook, label, total
        self.every = max(0, int(every))
        self.model_i, self.model_n = model_i, model_n
        self.t0 = time.time()
        self._next = self.every if self.every else None

    def update(self, done: int) -> None:
        el = max(1e-9, time.time() - self.t0)
        rate = done / el
        eta = (self.total - done) / rate if rate > 0 else float("inf")
        line = (f"[eval] {self.label} {done}/{self.total} "
                f"({100.0*done/max(1,self.total):.0f}%) {rate:.2f} img/s eta {eta/60:.1f} min")
        print(line, flush=True)
        if self._next is not None and done >= self._next:
            while self._next is not None and done >= self._next:
                self._next += self.every
            _ping(self.webhook,
                  f"cf-recon-mse ⏳ [{self.model_i}/{self.model_n}] `{self.label}` "
                  f"{done}/{self.total} ({100.0*done/max(1,self.total):.0f}%) · "
                  f"{rate:.2f} img/s · eta {eta/60:.1f} min")

    def finish(self, by_survey: Dict[str, List[float]]) -> None:
        el = time.time() - self.t0
        parts = []
        for s in ("hsc", "legacy"):
            v = by_survey.get(s, [])
            if v:
                parts.append(f"{s} {sum(v)/len(v):.5f}±{_sem(v):.5f}")
        _ping(self.webhook,
              f"cf-recon-mse ✅ [{self.model_i}/{self.model_n}] `{self.label}` done in "
              f"{el/60:.1f} min · MSE " + " · ".join(parts))

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


def _lookup_cf_batch(
    cf_ds: CFEmbeddingDataset,
    catalog_index: dict,
    metadata: List[dict],
    device: torch.device,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Build CF cond (+ optional tok_mask) and targets for a chunk via (row_idx, survey)."""
    conds, masks, targets = [], [], []
    for m in metadata:
        survey_code = 0 if m["anchor_survey"] == "hsc" else 1
        key = (int(m["idx"]), survey_code)
        if key not in catalog_index:
            raise KeyError(
                f"CF H5 missing (row_idx={key[0]}, survey={m['anchor_survey']}). "
                f"Rebuild H5 from the same catalog or check join keys."
            )
        item = cf_ds[catalog_index[key]]
        if len(item) == 4:
            cond, tok_mask, target, _ = item
            conds.append(cond); masks.append(tok_mask); targets.append(target)
        else:
            cond, target, _ = item
            conds.append(cond); targets.append(target)
    cond_t = torch.stack(conds).to(device)
    tgt_t = torch.stack(targets).to(device)
    tok_m = torch.stack(masks).to(device) if masks else None
    return cond_t, tok_m, tgt_t


def _chunk_spans(take: int, first_chunk: int, is_first_batch: bool):
    """Split the first batch so an early three-way report is available cheaply.

    Only the FIRST batch is split (into first_chunk + remainder); every later batch is
    sampled whole. Sampling is per-image independent and the noise is sliced from the
    same per-batch draw, so chunking changes throughput slightly but NOT the numbers.
    Splitting every batch down to 16 would underutilise the GPU and make the job slower.
    """
    if is_first_batch and 0 < first_chunk < take:
        return [(0, first_chunk), (first_chunk, take)]
    return [(0, take)]


@torch.no_grad()
def _predict_chunk(entry: Dict[str, Any], ctx: Dict[str, Any], num_steps: int,
                   device: torch.device, check_target: bool) -> List[float]:
    """Per-image MSE for one model on one chunk, against that model's own target."""
    if entry["kind"] == "image":
        pred = entry["model"].sample(
            cond_image_samegal=ctx["samegal"], cond_image_sameins=ctx["sameins"],
            masks=ctx["masks"], num_steps=num_steps, x_noise=ctx["noise"],
        )
        target = ctx["x_1"]
    else:
        cond, tok_mask, target = _lookup_cf_batch(
            entry["cf_ds"], entry["cat_index"], ctx["metadata"], device,
        )
        # The CF decoder scores against the H5 target (its training target) while the
        # image models score against the loader's x_1. Those must be the SAME pixels or
        # the columns are not comparable — verify once rather than assume.
        tol = entry.get("target_tol", 0.0)
        if check_target and tol > 0:
            delta = (ctx["x_1"] - target).abs().max().item()
            print(f"[eval] {entry['label']}: target vs loader x_1 max|Δ|={delta:.3e} "
                  f"(tol {tol:g})", flush=True)
            if delta > tol:
                raise RuntimeError(
                    f"CF H5 targets disagree with the val loader's images by {delta:.3e} "
                    f"(> {tol:g}); the CF row would not be comparable to the image rows. "
                    f"Rebuild the H5 from the store the eval config uses "
                    f"(source_data_dir={entry['cf_ds'].source_data_dir!r}) or raise --target-tol."
                )
        pred = entry["model"].sample(
            cond, num_steps=num_steps, x_noise=ctx["noise"], tok_mask=tok_mask,
        )
    return ((pred - target) ** 2).mean(dim=(1, 2, 3)).tolist()


@torch.no_grad()
def _run_interleaved(
    entries: List[Dict[str, Any]],
    val_loader,
    n: int,
    num_steps: int,
    noise_seed: int,
    device: torch.device,
    report_at: List[int],
    first_chunk: int,
    on_report,
) -> Dict[str, Dict[str, List[float]]]:
    """Evaluate EVERY model on each chunk before advancing.

    Interleaving (rather than one model at a time) means a complete three-way table
    exists after the first few galaxies, and every model is loaded exactly once — which
    matters because the 2.2 GB checkpoints are the dominant fixed cost on this filesystem.
    All models share the same x_noise per chunk, so the comparison stays paired.
    """
    results = {e["label"]: defaultdict(list) for e in entries}
    ch = {int(e["model"].in_channels) for e in entries}
    sz = {int(e["model"].image_size) for e in entries}
    if len(ch) != 1 or len(sz) != 1:
        raise RuntimeError(f"models disagree on sample shape: channels={ch} sizes={sz}")
    in_channels, image_size = ch.pop(), sz.pop()

    pending = sorted({int(r) for r in report_at if 0 < int(r) <= n})
    seen, batch_idx, t0 = 0, 0, time.time()

    for batch in val_loader:
        x_1, samegal, sameins, masks, metadata = _unpack_batch(batch)
        take = min(x_1.shape[0], n - seen)
        if take <= 0:
            break
        gen = torch.Generator(device=device)
        gen.manual_seed(noise_seed + batch_idx)
        noise_full = torch.randn(take, in_channels, image_size, image_size,
                                 device=device, generator=gen)

        for a, b in _chunk_spans(take, first_chunk, batch_idx == 0):
            md = metadata[a:b]
            ctx = {
                "x_1": x_1[a:b].to(device), "samegal": samegal[a:b].to(device),
                "sameins": sameins[a:b].to(device), "masks": masks[a:b].to(device),
                "metadata": md, "noise": noise_full[a:b],
            }
            for e in entries:
                vals = _predict_chunk(e, ctx, num_steps, device,
                                      check_target=(batch_idx == 0 and a == 0))
                for i, m in enumerate(md):
                    results[e["label"]][m["anchor_survey"]].append(float(vals[i]))
            seen += b - a
            while pending and seen >= pending[0]:
                pending.pop(0)
                on_report(results, seen, time.time() - t0, final=False)

        batch_idx += 1
        if seen >= n:
            break

    on_report(results, seen, time.time() - t0, final=True)
    return results


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
    p.add_argument("--mmap-store", action="store_true",
                   help="ram48 only: memory-map the image store instead of slurping ~17 GB "
                        "into RAM. This eval reads at most --n*7 images (~66 MB at n=512), so "
                        "the slurp is a pure startup tax -- ~45 min on the contended pool "
                        "volume. Same galaxies, same split, same numbers.")
    p.add_argument("--discord-webhook", default="",
                   help="Discord webhook for partial-result reports (else $DISCORD_WEBHOOK).")
    p.add_argument("--report-at", type=lambda s: [int(x) for x in s.split(",") if x.strip()],
                   default=[16, 32, 64, 128, 256, 512, 1024, 2048],
                   help="Cumulative galaxy counts at which to emit a full running table "
                        "(CSV + markdown + Discord). Costs nothing: the per-image MSEs are "
                        "already accumulated.")
    p.add_argument("--first-chunk", type=int, default=16,
                   help="Split only the FIRST batch into this many galaxies + remainder so an "
                        "early three-way table appears within minutes. Later batches run "
                        "whole. 0 disables. Changes no number -- noise is sliced from the "
                        "same per-batch draw.")
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
    if args.mmap_store:
        config.data.load_into_ram = False
        print("[eval] --mmap-store: memory-mapping the image store (skipping the ~17 GB slurp)")
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

    fieldnames = [
        "model_label", "objective", "num_steps", "anchor_survey",
        "n", "mse_mean", "mse_sem", "note",
    ]
    webhook = args.discord_webhook or os.environ.get("DISCORD_WEBHOOK", "")

    def _rows_from(results: Dict[str, Dict[str, List[float]]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for e in entries:
            out.extend(_aggregate_rows(
                e["label"], e["objective"], results[e["label"]],
                note=e["note"], num_steps=args.num_steps,
            ))
        return out

    def _write_csv(rows: List[Dict[str, Any]]) -> None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    # ---------------------------------------------------------------- load models once
    entries: List[Dict[str, Any]] = []
    torch.set_grad_enabled(False)

    def _load_image(label: str, ckpt: Path, note: str) -> None:
        t = time.time()
        print(f"[eval] loading {label}: {ckpt}", flush=True)
        m = ConditionalFlowMatchingModule.load_from_checkpoint(str(ckpt), map_location="cpu")
        m.eval().to(device)
        entries.append({"label": label, "kind": "image", "model": m,
                        "objective": "flow_matching", "note": note})
        print(f"[eval] loaded {label} in {time.time()-t:.0f}s", flush=True)

    if not args.skip_fm_control:
        ctrl = _resolve_fm_control(args.fm_control_checkpoint)
        if ctrl.is_file():
            _load_image("fm-control", ctrl, "ram48 diffusion-ablation control")
        else:
            print(f"[eval] WARNING: fm-control ckpt missing: {ctrl}")

    paper_ckpt = str(args.paper_checkpoint) if args.paper_checkpoint else ""
    if not args.skip_paper and paper_ckpt not in ("", ".") and Path(paper_ckpt).is_file():
        try:
            _load_image("fm-paper", Path(paper_ckpt),
                        "paper base; same seed-42 val split (configs/neighbors_efficient.json)")
        except Exception as exc:  # noqa: BLE001
            print(f"[eval] WARNING: paper-base row skipped: {exc}")
    elif not args.skip_paper:
        print(f"[eval] WARNING: paper row skipped — ckpt not usable: {args.paper_checkpoint!r}")

    if args.cf_checkpoint:
        if args.cf_h5 is None or not Path(args.cf_h5).is_file():
            raise SystemExit("--cf-checkpoint requires an existing --cf-h5")
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
            t = time.time()
            print(f"[eval] loading CF {label}: {path_s}", flush=True)
            m = CFEmbeddingFlowMatching.load_from_checkpoint(path_s, map_location="cpu")
            m.eval().to(device)
            style = args.cf_cond_style or getattr(m, "cond_style", None) \
                or getattr(m.hparams, "cond_style", "tokens")
            cf_ds, cat_index = _get_cf_ds(str(style))
            entries.append({
                "label": label, "kind": "cf", "model": m,
                "objective": "cf_frozen_embedding_fm",
                "note": f"cond_style={style}; h5={Path(args.cf_h5).name}",
                "cf_ds": cf_ds, "cat_index": cat_index, "target_tol": args.target_tol,
            })
            print(f"[eval] loaded CF {label} in {time.time()-t:.0f}s", flush=True)

    if not entries:
        raise SystemExit("[eval] nothing to evaluate — every model was skipped or missing")

    labels = ", ".join(e["label"] for e in entries)
    print(f"[eval] interleaving {len(entries)} models over n={args.n}: {labels}", flush=True)
    _ping(webhook, f"cf-recon-mse 🚀 start · {len(entries)} models × n={args.n} "
                   f"({args.num_steps} ODE steps) · `{labels}` · first report at "
                   f"{args.first_chunk} galaxies")

    # ---------------------------------------------------------------- interleaved pass
    def _on_report(results, seen, elapsed, final):
        rows = _rows_from(results)
        _write_csv(rows)
        _print_markdown(rows)
        rate = seen / max(1e-9, elapsed)
        eta = (args.n - seen) / rate if rate > 0 else float("inf")
        head = "🏁 FINAL" if final else "⏳ partial"
        lines = [f"cf-recon-mse {head} · {seen}/{args.n} galaxies · "
                 f"{elapsed/60:.1f} min elapsed" + ("" if final else f" · eta {eta/60:.1f} min")]
        for e in entries:
            bs = results[e["label"]]
            seg = []
            for s in ("hsc", "legacy"):
                v = bs.get(s, [])
                if v:
                    seg.append(f"{s} {sum(v)/len(v):.5f}±{_sem(v):.5f}")
            allv = [x for s in ("hsc", "legacy") for x in bs.get(s, [])]
            if allv:
                seg.append(f"all {sum(allv)/len(allv):.5f}")
            lines.append(f"`{e['label']}` " + " · ".join(seg))
        _ping(webhook, "\n".join(lines))
        print(f"[eval] {head}: {seen}/{args.n} in {elapsed/60:.1f} min "
              f"({rate:.2f} img/s)", flush=True)

    _run_interleaved(
        entries=entries, val_loader=val_loader, n=args.n, num_steps=args.num_steps,
        noise_seed=args.noise_seed, device=device,
        report_at=args.report_at, first_chunk=args.first_chunk, on_report=_on_report,
    )
    print(f"[eval] wrote {args.out}")


if __name__ == "__main__":
    main()
