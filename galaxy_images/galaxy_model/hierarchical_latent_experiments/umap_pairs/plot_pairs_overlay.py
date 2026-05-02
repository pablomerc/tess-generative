"""Overlay K randomly-chosen HSC/Legacy pairs on the umap_pairs scatter.

Reads the cache produced by run_umap_pairs.py:
  <out-dir>/umap_pairs_<variant>_data.npz   (hsc_phys, leg_phys, hsc_inst, leg_inst,
                                              emb_p, emb_i, n_hsc)

Produces:
  <out-dir>/umap_pairs_<variant>_overlay.png

Usage:
  python -m galaxy_images.galaxy_model.hierarchical_latent_experiments.umap_pairs.plot_pairs_overlay \
      --data-dir .../outputs/spatial_flat__hier_small --variant spatial_flat --k 8

For each highlighted pair we draw the HSC point and Legacy point with the same
distinct color and a number, plus a thin connecting line so the user can see how
far apart paired galaxies land.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_THIS = Path(__file__).resolve()
_PKG_ROOT = _THIS.parents[4]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from galaxy_images.galaxy_model.hierarchical_latent_experiments.common.discord_utils import (
    post_image, post_text,
)


PAIR_PALETTE = [
    "#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9A6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9",
]


def plot_overlay(emb_combined: np.ndarray, n_hsc: int, pair_idx: np.ndarray,
                 ax, title: str):
    ax.scatter(emb_combined[:n_hsc, 0], emb_combined[:n_hsc, 1],
               s=4, c="#8eb8e8", alpha=0.35, label="HSC", rasterized=True)
    ax.scatter(emb_combined[n_hsc:, 0], emb_combined[n_hsc:, 1],
               s=4, c="#e8c4a0", alpha=0.35, label="Legacy", rasterized=True)
    for i, p in enumerate(pair_idx):
        color = PAIR_PALETTE[i % len(PAIR_PALETTE)]
        x_h, y_h = emb_combined[p]
        x_l, y_l = emb_combined[n_hsc + p]
        ax.plot([x_h, x_l], [y_h, y_l], "-", color=color, alpha=0.8, linewidth=1.2)
        ax.scatter([x_h], [y_h], s=140, c=color, marker="o", edgecolors="black", linewidths=0.8, zorder=3)
        ax.scatter([x_l], [y_l], s=140, c=color, marker="s", edgecolors="black", linewidths=0.8, zorder=3)
        ax.annotate(str(i + 1), (x_h, y_h), color="black", fontsize=8, fontweight="bold",
                    ha="center", va="center", zorder=4)
        ax.annotate(str(i + 1), (x_l, y_l), color="black", fontsize=8, fontweight="bold",
                    ha="center", va="center", zorder=4)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True,
                    help="Directory containing umap_pairs_<variant>_data.npz")
    ap.add_argument("--variant", required=True)
    ap.add_argument("--k", type=int, default=8, help="Number of pairs to highlight")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pair-indices", type=str, default=None,
                    help="Optional explicit comma-separated 0-based pair indices (overrides --k random).")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-discord", action="store_true")
    args = ap.parse_args()

    npz = args.data_dir / f"umap_pairs_{args.variant}_data.npz"
    if not npz.is_file():
        raise FileNotFoundError(f"Cache not found: {npz}. Re-run run_umap_pairs.py first.")
    cache = np.load(npz)
    emb_p = cache["emb_p"]; emb_i = cache["emb_i"]; n_hsc = int(cache["n_hsc"])
    print(f"loaded {npz}: emb_p={emb_p.shape}, emb_i={emb_i.shape}, n_hsc={n_hsc}")

    if args.pair_indices:
        pair_idx = np.array([int(x) for x in args.pair_indices.split(",")], dtype=np.int64)
    else:
        rng = np.random.default_rng(args.seed)
        pair_idx = rng.choice(n_hsc, size=min(args.k, n_hsc), replace=False)
        pair_idx.sort()
    print(f"highlighting pairs (0-based): {pair_idx.tolist()}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
    plot_overlay(emb_p, n_hsc, pair_idx, ax1,
                 f"Physics latent — {args.variant} — {len(pair_idx)} highlighted pairs")
    plot_overlay(emb_i, n_hsc, pair_idx, ax2,
                 f"Instrument latent — {len(pair_idx)} highlighted pairs")
    fig.suptitle(
        f"UMAP HSC/Legacy pairs — {args.variant} — circles=HSC, squares=Legacy, lines connect same-galaxy pairs",
        fontsize=13,
    )
    plt.tight_layout()

    out_path = args.out or (args.data_dir / f"umap_pairs_{args.variant}_overlay.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")

    if not args.no_discord:
        post_image(out_path, message=(
            f"📊 [umap_pairs/overlay/{args.variant}] {len(pair_idx)} pairs highlighted "
            f"(indices: {','.join(map(str, pair_idx.tolist()))})"
        ))


if __name__ == "__main__":
    main()
