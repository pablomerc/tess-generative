#!/usr/bin/env python3
"""Direct collapse check on a JEPA checkpoint's EMA encoder (see README 'Watching
for collapse'). Reports the diagnostics that actually matter for representation
quality, independent of the training loss:

  norm_std   mean per-dim std of L2-normalized embeddings.
             Healthy ~ 1/sqrt(D); ->0 means collapse.
  rel_std    norm_std / (1/sqrt(D)) — 1.0 is the isotropic reference.
  cen_std    same, but MEAN-CENTERED before normalizing. This is the honest
             spread measure: a space where every vector shares a large common
             offset has a small rel_std while being perfectly well-spread around
             that offset, and a probe (whose first linear layer absorbs offsets)
             is unaffected. Low rel_std + healthy cen_std/eff_rank = offset-
             dominated, NOT collapse.
  eff_rank   exp(entropy of the PCA eigenvalue spectrum). The primary criterion:
             a collapsed space has eff_rank ~ 1 even when norm_std looks fine.
  top1_var   fraction of variance in the leading PC (near 1.0 = one direction).

Verdict is driven by eff_rank (rank structure) with cen_std as the secondary
check; rel_std alone is deliberately NOT decisive, because it conflates a shared
mean offset with genuine collapse.

Usage:
  python check_collapse.py --arm hsc-vit --checkpoint <ckpt> [--n 1024]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from galaxy_images.galaxy_model.jepa_ablation.datasets import load_umap_pairs  # noqa: E402
from galaxy_images.galaxy_model.jepa_ablation.regenerate_umap_jepa import (  # noqa: E402
    ARM_SPECS,
    load_jepa_checkpoint,
)


def diagnostics(z: torch.Tensor) -> dict:
    z = z.float()
    d = z.shape[1]
    zn = F.normalize(z, dim=1)
    norm_std = float(zn.std(dim=0).mean())
    # Centered spread: removes a shared mean offset, which depresses norm_std
    # without indicating any loss of information.
    zc = F.normalize(z - z.mean(dim=0, keepdim=True), dim=1)
    cen_std = float(zc.std(dim=0).mean())
    x = z - z.mean(dim=0, keepdim=True)
    # Eigenvalues of the covariance via SVD of the centered matrix.
    sv = torch.linalg.svdvals(x)
    ev = (sv ** 2) / max(1, z.shape[0] - 1)
    p = (ev / ev.sum().clamp(min=1e-12)).clamp(min=1e-12)
    eff_rank = float(torch.exp(-(p * p.log()).sum()))
    return {
        "dim": d,
        "norm_std": norm_std,
        "rel_std": norm_std * (d ** 0.5),
        "cen_std": cen_std * (d ** 0.5),
        "eff_rank": eff_rank,
        "top1_var": float(p[0]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(ARM_SPECS), required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--data-dir", default="/orcd/pool/007/pablomer/efficient_neighs_48")
    args = ap.parse_args()

    model, single, _hp = load_jepa_checkpoint(args.checkpoint, args.arm)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    hsc, legacy = load_umap_pairs(args.data_dir, n=args.n, seed=7)
    print(f"[collapse] {args.arm}  ckpt={Path(args.checkpoint).name}  n={hsc.shape[0]}  "
          f"single_encoder={single}  device={device.type}")

    towers = [("physics/e1", model.encoder_galaxy)]
    if not single:
        towers.append(("instrument/e2", model.encoder_instrument))

    for name, enc in towers:
        outs = []
        for imgs in (hsc, legacy):
            chunks = []
            for i in range(0, imgs.shape[0], 128):
                x = imgs[i : i + 128].to(device)
                chunks.append(enc(x).flatten(start_dim=1).detach().float().cpu())
            outs.append(torch.cat(chunks))
        z = torch.cat(outs, dim=0)
        d = diagnostics(z)
        # eff_rank is the primary criterion; cen_std guards against a space that is
        # high-rank but vanishingly small. rel_std is reported, not used to judge —
        # it is depressed by a shared mean offset that a probe absorbs harmlessly.
        if d["eff_rank"] < 2 or d["top1_var"] > 0.9 or d["cen_std"] < 0.1:
            verdict = "COLLAPSED"
        elif d["eff_rank"] < 0.05 * d["dim"] or d["cen_std"] < 0.3:
            verdict = "weak"
        else:
            verdict = "healthy"
        offset = " [offset-dominated]" if (d["rel_std"] < 0.5 <= d["cen_std"]) else ""
        print(f"  {name:14s} dim={d['dim']:4d}  norm_std={d['norm_std']:.4f} "
              f"(rel {d['rel_std']:.2f}, centered {d['cen_std']:.2f})  "
              f"eff_rank={d['eff_rank']:.1f}  top1_var={d['top1_var']:.3f}  "
              f"-> {verdict}{offset}")


if __name__ == "__main__":
    main()
