#!/usr/bin/env python3
"""Shuffle-conditioning ablation (reviewer SYJm Q2).

Paired inference-time corruption of z_phy / z_ins on the paper FM checkpoint.
Reuses the diffusion-ablation eval-set + noise protocol so numbers are comparable.

Run from repo root:
  python -m galaxy_images.galaxy_model.shuffle_ablation.run_shuffle_ablation ...
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]  # tess-generative
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from galaxy_images.galaxy_model.config import load_experiment_config
from galaxy_images.galaxy_model.data_factory import build_neighbors_dataloaders
from galaxy_images.galaxy_model.diffusion_ablation.eval_recon_mse import (
    _aggregate_rows,
    _load_model,
    _load_or_create_manifest,
    _ping,
    _sem,
    _unpack_batch,
)
from galaxy_images.galaxy_model.shuffle_ablation.metrics import (
    adjacent_diff_sigma,
    band_mean_sigma,
    corner_sky_rms,
    masked_mad_sigma,
    mean_high_k_power,
)

ABL_DIR = Path(__file__).resolve().parent
DIFF_DIR = ABL_DIR.parent / "diffusion_ablation"
DEFAULT_CONFIG = DIFF_DIR / "configs" / "neighbors_fm_control.json"
DEFAULT_RECON_MANIFEST = DIFF_DIR / "results" / "recon_eval_manifest.json"
PAPER_CKPT = ABL_DIR.parent / "checkpoints" / "base" / "snapshot.ckpt"
CONDITIONS = ("C0", "C1", "C2", "C3")
CONDITION_NAMES = {
    "C0": "intact",
    "C1": "shuffle-phy",
    "C2": "shuffle-ins",
    "C3": "shuffle-both",
}


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def within_survey_cyclic_pi(anchor_surveys: List[str]) -> np.ndarray:
    """π = cyclic shift by 1 within each survey group (no fixed points, no RNG)."""
    n = len(anchor_surveys)
    pi = np.arange(n, dtype=np.int64)
    by_survey: Dict[str, List[int]] = defaultdict(list)
    for i, s in enumerate(anchor_surveys):
        by_survey[s].append(i)
    for idxs in by_survey.values():
        m = len(idxs)
        if m < 2:
            raise RuntimeError(
                f"Survey group too small for fixed-point-free cycle: {idxs}"
            )
        for j, i in enumerate(idxs):
            pi[i] = idxs[(j + 1) % m]
    # Guard: no fixed points.
    if np.any(pi == np.arange(n)):
        raise RuntimeError("π has fixed points; expected none")
    return pi


def _pad_sameins(
    sameins_list: List[torch.Tensor],
    masks_list: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stack variable-k sameins/masks to a common k (pad with zeros / False)."""
    ks = [t.shape[0] for t in sameins_list]
    k_max = max(ks)
    c, h, w = sameins_list[0].shape[1:]
    n = len(sameins_list)
    out = torch.zeros((n, k_max, c, h, w), dtype=sameins_list[0].dtype)
    masks = torch.zeros((n, k_max), dtype=torch.bool)
    for i, (t, m) in enumerate(zip(sameins_list, masks_list)):
        k = t.shape[0]
        out[i, :k] = t
        masks[i, :k] = m[:k]
    return out, masks


def materialize_eval_set(
    val_loader,
    n: int,
) -> Dict[str, Any]:
    """Load first n anchors in loader order; re-pad sameins to common k."""
    x1_list: List[torch.Tensor] = []
    samegal_list: List[torch.Tensor] = []
    sameins_list: List[torch.Tensor] = []
    masks_list: List[torch.Tensor] = []
    surveys: List[str] = []
    catalog_idxs: List[Any] = []
    dataset_indices: List[int] = []

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
            x1_list.append(x_1[i].cpu())
            samegal_list.append(samegal[i].cpu())
            sameins_list.append(sameins[i].cpu())
            masks_list.append(masks[i].cpu())
            surveys.append(metadata[i]["anchor_survey"])
            catalog_idxs.append(metadata[i].get("idx", ds_idx))
            dataset_indices.append(ds_idx)
            seen += 1
        batch_offset += B
        if seen >= n:
            break

    if seen < n:
        raise RuntimeError(f"Val loader only yielded {seen} anchors; need n={n}")

    sameins, masks = _pad_sameins(sameins_list, masks_list)
    return {
        "x_1": torch.stack(x1_list),  # (N, C, H, W)
        "samegal": torch.stack(samegal_list),
        "sameins": sameins,
        "masks": masks,
        "anchor_surveys": surveys,
        "catalog_idxs": catalog_idxs,
        "dataset_indices": dataset_indices,
    }


def _mse_per_image(pred: torch.Tensor, target: torch.Tensor, crop: Optional[int] = None) -> torch.Tensor:
    """Per-image MSE over (C,H,W); optional center crop."""
    diff = pred - target
    if crop is not None:
        _, _, h, w = diff.shape
        sy = (h - crop) // 2
        sx = (w - crop) // 2
        diff = diff[:, :, sy : sy + crop, sx : sx + crop]
    return (diff ** 2).mean(dim=(1, 2, 3))


def _noise_seed_for(m: int, batch_idx: int, noise_seed: int) -> int:
    if m == 0:
        return noise_seed + batch_idx
    return noise_seed + 20_000 * m + batch_idx


@torch.no_grad()
def run_ablation(
    model,
    data: Dict[str, Any],
    pi: np.ndarray,
    *,
    batch_size: int,
    num_steps: int,
    noise_seed: int,
    m_values: Tuple[int, ...],
    device: torch.device,
    conditions: Tuple[str, ...] = CONDITIONS,
    discord_webhook: str = "",
    ping_prefix: str = "shuffle-ablation",
    flush_csv_path: Optional[Path] = None,
) -> Tuple[Dict[str, np.ndarray], List[Dict[str, Any]]]:
    """Generate under each condition; return gens[condition]=(N,M,C,H,W) and tidy rows.

    ``m_values`` are the posterior-sample indices to run (seeds depend only on
    (m, batch_idx), so any subset is exact and reproducible — this is what lets
    an M=8 run be split across jobs). ``flush_csv_path`` rewrites the per-anchor
    CSV after every (batch, m) block so a wall-clock kill never loses work.
    """
    n = data["x_1"].shape[0]
    if n % batch_size != 0:
        raise RuntimeError(
            f"n={n} not divisible by batch_size={batch_size}; "
            "batch boundaries are part of the noise protocol identity"
        )
    n_batches = n // batch_size
    n_samples = len(m_values)
    c = model.in_channels
    h = w = model.image_size

    gens: Dict[str, np.ndarray] = {
        cond: np.empty((n, n_samples, c, h, w), dtype=np.float32) for cond in conditions
    }
    rows: List[Dict[str, Any]] = []

    # Precompute own/donor target σ (independent of generation).
    x1_np = data["x_1"].numpy()
    own_sigma = {
        "corner": band_mean_sigma(corner_sky_rms(x1_np)),
        "adjdiff": band_mean_sigma(adjacent_diff_sigma(x1_np)),
        "mad": band_mean_sigma(masked_mad_sigma(x1_np)),
        "high_k": mean_high_k_power(x1_np),
    }
    donor_x1 = x1_np[pi]
    donor_sigma = {
        "corner": band_mean_sigma(corner_sky_rms(donor_x1)),
        "adjdiff": band_mean_sigma(adjacent_diff_sigma(donor_x1)),
        "mad": band_mean_sigma(masked_mad_sigma(donor_x1)),
        "high_k": mean_high_k_power(donor_x1),
    }

    for batch_idx in range(n_batches):
        sl = slice(batch_idx * batch_size, (batch_idx + 1) * batch_size)
        idx = np.arange(sl.start, sl.stop)
        pi_b = pi[idx]

        x_1_b = data["x_1"][idx].to(device)
        samegal_b = data["samegal"][idx].to(device)
        sameins_b = data["sameins"][idx].to(device)
        masks_b = data["masks"][idx].to(device)

        samegal_pi = data["samegal"][pi_b].to(device)
        sameins_pi = data["sameins"][pi_b].to(device)
        masks_pi = data["masks"][pi_b].to(device)
        x_1_pi = data["x_1"][pi_b].to(device)

        cond_inputs = {
            "C0": (samegal_b, sameins_b, masks_b),
            "C1": (samegal_pi, sameins_b, masks_b),
            "C2": (samegal_b, sameins_pi, masks_pi),
            "C3": (samegal_pi, sameins_pi, masks_pi),
        }

        for m_pos, m in enumerate(m_values):
            gen = torch.Generator(device=device)
            gen.manual_seed(_noise_seed_for(m, batch_idx, noise_seed))
            noise = torch.randn(batch_size, c, h, w, device=device, generator=gen)

            for cond in conditions:
                sg, si, mk = cond_inputs[cond]
                pred = model.sample(
                    cond_image_samegal=sg,
                    cond_image_sameins=si,
                    masks=mk,
                    num_steps=num_steps,
                    x_noise=noise,
                )
                gens[cond][idx, m_pos] = pred.detach().cpu().numpy()

                mse_own = _mse_per_image(pred, x_1_b)
                mse_donor = _mse_per_image(pred, x_1_pi)
                mse_own_32 = _mse_per_image(pred, x_1_b, crop=32)
                mse_donor_32 = _mse_per_image(pred, x_1_pi, crop=32)

                pred_np = pred.detach().cpu().numpy()
                sig_corner = band_mean_sigma(corner_sky_rms(pred_np))
                sig_adj = band_mean_sigma(adjacent_diff_sigma(pred_np))
                sig_mad = band_mean_sigma(masked_mad_sigma(pred_np))
                high_k = mean_high_k_power(pred_np)

                for i_local, i_global in enumerate(idx):
                    rows.append(
                        {
                            "anchor_id": int(i_global),
                            "catalog_idx": data["catalog_idxs"][i_global],
                            "dataset_index": data["dataset_indices"][i_global],
                            "survey": data["anchor_surveys"][i_global],
                            "donor_id": int(pi[i_global]),
                            "condition": cond,
                            "condition_name": CONDITION_NAMES[cond],
                            "m": m,
                            "mse_own": float(mse_own[i_local].item()),
                            "mse_donor": float(mse_donor[i_local].item()),
                            "mse_own_32": float(mse_own_32[i_local].item()),
                            "mse_donor_32": float(mse_donor_32[i_local].item()),
                            "sigma_corner": float(sig_corner[i_local]),
                            "sigma_adjdiff": float(sig_adj[i_local]),
                            "sigma_mad": float(sig_mad[i_local]),
                            "high_k_power": float(high_k[i_local]),
                            "sigma_corner_own": float(own_sigma["corner"][i_global]),
                            "sigma_adjdiff_own": float(own_sigma["adjdiff"][i_global]),
                            "sigma_mad_own": float(own_sigma["mad"][i_global]),
                            "high_k_power_own": float(own_sigma["high_k"][i_global]),
                            "sigma_corner_donor": float(donor_sigma["corner"][i_global]),
                            "sigma_adjdiff_donor": float(donor_sigma["adjdiff"][i_global]),
                            "sigma_mad_donor": float(donor_sigma["mad"][i_global]),
                            "high_k_power_donor": float(donor_sigma["high_k"][i_global]),
                        }
                    )

            if flush_csv_path is not None:
                _write_csv(flush_csv_path, rows)

            _ping(
                discord_webhook,
                f"🧩 {ping_prefix} · batch {batch_idx + 1}/{n_batches} "
                f"m={m} ({m_pos + 1}/{n_samples}) done",
            )

        print(
            f"[shuffle] batch {batch_idx + 1}/{n_batches} complete "
            f"({n_samples} samples × {len(conditions)} conditions)",
            flush=True,
        )

    return gens, rows


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No rows to write")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summary_markdown(rows: List[Dict[str, Any]]) -> str:
    """Quick mean±sem MSE_own / MSE_donor by condition × survey (lowest m for headline)."""
    by: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(
        lambda: {"mse_own": [], "mse_donor": []}
    )
    m_ref = min((r["m"] for r in rows), default=0)
    for r in rows:
        if r["m"] != m_ref:
            continue
        key = (r["condition"], r["survey"])
        by[key]["mse_own"].append(r["mse_own"])
        by[key]["mse_donor"].append(r["mse_donor"])

    lines = [
        "",
        "| condition | survey | n | MSE_own | MSE_donor |",
        "|---|---|---:|---:|---:|",
    ]
    for cond in CONDITIONS:
        for survey in ("hsc", "legacy"):
            d = by.get((cond, survey))
            if not d or not d["mse_own"]:
                continue
            n = len(d["mse_own"])
            mo = sum(d["mse_own"]) / n
            md = sum(d["mse_donor"]) / n
            lines.append(
                f"| {cond} ({CONDITION_NAMES[cond]}) | {survey} | {n} | "
                f"{mo:.6f}±{_sem(d['mse_own']):.6f} | "
                f"{md:.6f}±{_sem(d['mse_donor']):.6f} |"
            )
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--checkpoint", type=Path, default=PAPER_CKPT)
    p.add_argument("--recon-manifest", type=Path, default=DEFAULT_RECON_MANIFEST)
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--num-steps", type=int, default=250)
    p.add_argument("--noise-seed", type=int, default=1234)
    p.add_argument("--n-samples", type=int, default=8, help="Posterior samples M (locked 8; use 1 for smoke)")
    p.add_argument(
        "--m-list",
        type=str,
        default="",
        help=(
            "Explicit posterior-sample indices, e.g. '0,1,2,3' (overrides --n-samples). "
            "Seeds depend only on (m, batch_idx), so splitting M across jobs is exact — "
            "use this to keep each job well under the partition wall."
        ),
    )
    p.add_argument(
        "--conditions",
        type=str,
        default="C0,C1,C2,C3",
        help="Comma-separated subset of C0,C1,C2,C3",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out-dir", type=Path, default=ABL_DIR / "results")
    p.add_argument("--discord-webhook", type=str, default="")
    p.add_argument("--ping-prefix", type=str, default="shuffle-ablation")
    p.add_argument(
        "--skip-generations",
        action="store_true",
        help="Skip writing .npz generation tensors (CSV + manifest only).",
    )
    args = p.parse_args(argv)

    conditions = tuple(c.strip() for c in args.conditions.split(",") if c.strip())
    for c in conditions:
        if c not in CONDITION_NAMES:
            raise SystemExit(f"Unknown condition {c!r}; choose from {list(CONDITION_NAMES)}")

    if args.m_list.strip():
        m_values = tuple(int(v) for v in args.m_list.split(",") if v.strip())
        if len(set(m_values)) != len(m_values):
            raise SystemExit(f"--m-list has duplicates: {m_values}")
        if any(m < 0 for m in m_values):
            raise SystemExit(f"--m-list must be non-negative: {m_values}")
    else:
        m_values = tuple(range(args.n_samples))
    print(f"[shuffle] posterior samples m={list(m_values)}")

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[shuffle] WARNING: CUDA unavailable, falling back to CPU")
        device = torch.device("cpu")

    config = load_experiment_config(args.config)
    import pytorch_lightning as pl

    pl.seed_everything(config.trainer.seed, workers=True)
    _, val_loader, _ = build_neighbors_dataloaders(config, config.data.batch_size)
    batch_size = int(config.data.batch_size)

    # Guard eval-set identity against the diffusion-ablation manifest.
    recon_manifest = _load_or_create_manifest(
        manifest_path=args.recon_manifest,
        val_loader=val_loader,
        n=args.n,
        noise_seed=args.noise_seed,
        num_steps=args.num_steps,
        val_ratio=config.data.val_ratio,
        split_seed=config.trainer.seed,
        batch_size=batch_size,
    )
    print(f"[shuffle] recon manifest OK (n={recon_manifest['n']})")

    print("[shuffle] materializing eval set…")
    data = materialize_eval_set(val_loader, args.n)
    if data["dataset_indices"] != recon_manifest["dataset_indices"]:
        raise RuntimeError("Materialized dataset_indices disagree with recon manifest")
    if data["anchor_surveys"] != recon_manifest["anchor_surveys"]:
        raise RuntimeError("Materialized anchor_surveys disagree with recon manifest")

    pi = within_survey_cyclic_pi(data["anchor_surveys"])
    n_hsc = sum(1 for s in data["anchor_surveys"] if s == "hsc")
    n_leg = sum(1 for s in data["anchor_surveys"] if s == "legacy")
    print(f"[shuffle] π ready: {n_hsc} hsc / {n_leg} legacy; k_sameins={data['sameins'].shape[1]}")

    print(f"[shuffle] loading checkpoint: {args.checkpoint}")
    ckpt_sha = _sha256_file(args.checkpoint)
    model = _load_model(args.checkpoint, "fm", device)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "shuffle_per_anchor.csv"

    _ping(
        args.discord_webhook,
        f"🚀 {args.ping_prefix} started · n={args.n} m={list(m_values)} "
        f"steps={args.num_steps} conditions={','.join(conditions)}",
    )

    gens, rows = run_ablation(
        model,
        data,
        pi,
        batch_size=batch_size,
        num_steps=args.num_steps,
        noise_seed=args.noise_seed,
        m_values=m_values,
        device=device,
        conditions=conditions,
        discord_webhook=args.discord_webhook,
        ping_prefix=args.ping_prefix,
        flush_csv_path=csv_path,
    )

    _write_csv(csv_path, rows)
    print(f"[shuffle] wrote {csv_path} ({len(rows)} rows)")

    if not args.skip_generations:
        # Also persist targets + pi for figure scripts.
        np.savez_compressed(
            out_dir / "targets.npz",
            x_1=data["x_1"].numpy(),
            pi=pi,
            m_values=np.array(m_values),
            anchor_surveys=np.array(data["anchor_surveys"]),
            catalog_idxs=np.array(data["catalog_idxs"]),
            dataset_indices=np.array(data["dataset_indices"]),
        )
        for cond, arr in gens.items():
            path = out_dir / f"gens_{cond}.npz"
            np.savez_compressed(path, gens=arr)
            print(f"[shuffle] wrote {path} shape={arr.shape}")

    shuffle_manifest = {
        "n": args.n,
        "batch_size": batch_size,
        "noise_seed": args.noise_seed,
        "num_steps": args.num_steps,
        "n_samples": len(m_values),
        "m_values": list(m_values),
        "split_seed": config.trainer.seed,
        "val_ratio": config.data.val_ratio,
        "conditions": list(conditions),
        "condition_names": {c: CONDITION_NAMES[c] for c in conditions},
        "pi": pi.tolist(),
        "pi_rule": "within-survey cyclic shift by 1",
        "anchor_surveys": data["anchor_surveys"],
        "dataset_indices": data["dataset_indices"],
        "catalog_idxs": data["catalog_idxs"],
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": ckpt_sha,
        "recon_manifest": str(args.recon_manifest),
        "config": str(args.config),
        "noise_seed_rule": {
            "m0": "noise_seed + batch_idx",
            "m_ge_1": "noise_seed + 20000*m + batch_idx",
        },
        "survey_counts": {"hsc": n_hsc, "legacy": n_leg},
    }
    man_path = out_dir / "shuffle_manifest.json"
    with man_path.open("w") as f:
        json.dump(shuffle_manifest, f, indent=2)
    print(f"[shuffle] wrote {man_path}")

    md = _summary_markdown(rows)
    print(md)
    (out_dir / "summary_m0.md").write_text(md)

    # Also emit a diffusion-ablation-compatible C0 aggregate row (fm-paper).
    by_survey: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        if r["condition"] == "C0" and r["m"] == 0:
            by_survey[r["survey"]].append(r["mse_own"])
    if by_survey:
        agg = _aggregate_rows(
            "fm-paper",
            "flow_matching",
            None,
            by_survey,
            note="shuffle-ablation C0 (paper base)",
            num_steps=args.num_steps,
        )
        agg_path = out_dir / "c0_fm_paper_aggregate.csv"
        with agg_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
            writer.writeheader()
            writer.writerows(agg)
        print(f"[shuffle] wrote {agg_path}")

    _ping(
        args.discord_webhook,
        f"✅ {args.ping_prefix} finished · wrote {csv_path.name}",
    )
    print("[shuffle] done")


if __name__ == "__main__":
    main()
