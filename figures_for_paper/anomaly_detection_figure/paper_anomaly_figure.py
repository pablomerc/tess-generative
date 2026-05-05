"""
Standalone, paper-ready 3-column anomaly figure.

This is a self-contained copy that reads only from the precomputed `_cache/`
directory in this folder — no access to the original anomaly_scores HDF5 or
the neighbours_v2 dataset is required.

Cached top-N per source (top_raw indices, NLL percentiles, HSC images) supports
N up to 24. To request a larger N, regenerate the cache with the upstream
script at:
    galaxy_images/galaxy_model/anomaly_detection/paper_anomaly_figure.py

Usage:
    python paper_anomaly_figure.py             # generates 8, 9 (3x3), 12
    python paper_anomaly_figure.py --top-n 9 --n-cols 3
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

_here = Path(__file__).resolve().parent
CACHE_DIR = _here / "_cache"
OUT_DIR = _here

SOURCES = [
    {
        "label": "Ours (Physics)",
        "scores_file": "anomaly_scores_ours_100k.h5",
        "score_key": "ours/hsc_mean/flow",
    },
    {
        "label": "AION-1",
        "scores_file": "anomaly_scores_aion_100k.h5",
        "score_key": "aion/hsc_mean_pca64/flow",
    },
    {
        "label": "Ours (Instrument)",
        "scores_file": "anomaly_scores_ins_100k.h5",
        "score_key": "ours/hsc_flat/flow",
    },
]

# Background colors — lightened from downstream_eval/final/makeplot_v2.py palette.
BG_COLORS = ["#E7F3FC", "#DFECE3", "#FCE8EC"]


def _to_rgb(img_chw, pct_lo=1, pct_hi=99):
    rgb = img_chw[:3].astype(np.float32)
    lo = np.percentile(rgb, pct_lo, axis=(1, 2), keepdims=True)
    hi = np.percentile(rgb, pct_hi, axis=(1, 2), keepdims=True)
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)
    return (rgb.transpose(1, 2, 0) * 255).astype(np.uint8)


def _cache_path(src, cache_n):
    stem = Path(src["scores_file"]).stem
    key = src["score_key"].replace("/", "_")
    return CACHE_DIR / f"{stem}__{key}__top{cache_n}.npz"


def _load_source_top(src, top_n, cache_n=24):
    """Load (top_raw, top_pcts, hsc_imgs[:top_n]) from the bundled cache."""
    cache = _cache_path(src, cache_n)
    if not cache.exists():
        raise FileNotFoundError(
            f"Missing cache file: {cache}\n"
            f"Available caches: {sorted(p.name for p in CACHE_DIR.glob('*.npz'))}"
        )
    if top_n > cache_n:
        raise ValueError(
            f"Requested top_n={top_n} but bundled cache only contains {cache_n}. "
            f"Regenerate the cache from the upstream script for a larger top_n."
        )
    with np.load(cache) as data:
        return data["top_raw"][:top_n], data["top_pcts"][:top_n], data["hsc_imgs"][:top_n]


def make_figure(top_n, out_path, n_cols_per=4, img_size=1.35, group_gap=0.35):
    assert top_n % n_cols_per == 0, f"top_n={top_n} must be divisible by n_cols_per={n_cols_per}"
    n_rows = top_n // n_cols_per
    n_groups = len(SOURCES)

    fig_w = n_groups * n_cols_per * img_size + (n_groups - 1) * group_gap
    fig_h = n_rows * img_size + 0.7

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")

    gs_left, gs_right = 0.005, 0.995
    wspace_frac = group_gap / (n_cols_per * img_size)
    outer = gridspec.GridSpec(
        1, n_groups, figure=fig,
        left=gs_left, right=gs_right,
        top=0.82, bottom=0.03,
        wspace=wspace_frac,
    )

    col_w_fig = (gs_right - gs_left) / (n_groups + wspace_frac * (n_groups - 1))
    gap = wspace_frac * col_w_fig
    pad_x = gap * 0.40

    bg_ax = fig.add_axes([0, 0, 1, 1], zorder=-1)
    bg_ax.set_xlim(0, 1)
    bg_ax.set_ylim(0, 1)
    bg_ax.set_axis_off()
    bg_ax.patch.set_visible(False)

    for col_idx in range(n_groups):
        pos = outer[col_idx].get_position(fig)
        x0 = pos.x0 - (pad_x if col_idx > 0 else 0)
        x1 = pos.x1 + (pad_x if col_idx < n_groups - 1 else 0)
        bg_ax.axvspan(x0, x1, facecolor=BG_COLORS[col_idx], linewidth=0)

    print(f"Building figure: top_n={top_n}, n_rows={n_rows}, n_cols={n_cols_per}")

    for col_idx, src in enumerate(SOURCES):
        top_raw, top_pcts, hsc_imgs = _load_source_top(src, top_n)
        print(f"  {src['label']}: loaded {len(top_raw)} from cache")

        inner = gridspec.GridSpecFromSubplotSpec(
            n_rows, n_cols_per,
            subplot_spec=outer[col_idx],
            hspace=0.30, wspace=0.04,
        )

        for i in range(top_n):
            r, c = divmod(i, n_cols_per)
            ax = fig.add_subplot(inner[r, c])
            ax.set_facecolor("none")
            ax.set_zorder(2)
            ax.imshow(_to_rgb(hsc_imgs[i]))
            ax.axis("off")
            ax.set_title(
                f"#{i+1}",
                fontsize=11, pad=3,
                color="#222222", fontweight="bold",
            )

        pos = outer[col_idx].get_position(fig)
        x_mid = (pos.x0 + pos.x1) / 2
        fig.text(
            x_mid, 0.88,
            src["label"],
            ha="center", va="bottom",
            fontsize=15, fontweight="bold",
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    pdf_path = out_path.with_suffix('.pdf')
    plt.savefig(pdf_path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {pdf_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=None,
                        help="If set, generate a single figure with this top_n.")
    parser.add_argument("--n-cols", type=int, default=None,
                        help="Columns per source group (default 3 if top_n=9, else 4).")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR

    if args.top_n is not None:
        n_cols = args.n_cols or (3 if args.top_n == 9 else 4)
        make_figure(args.top_n, out_dir / f"paper_anomaly_{args.top_n}.png", n_cols_per=n_cols)
    else:
        make_figure(8,  out_dir / "paper_anomaly_8.png",  n_cols_per=4)
        make_figure(9,  out_dir / "paper_anomaly_9.png",  n_cols_per=3)
        make_figure(12, out_dir / "paper_anomaly_12.png", n_cols_per=4)

    print("\nDone.")


if __name__ == "__main__":
    main()
