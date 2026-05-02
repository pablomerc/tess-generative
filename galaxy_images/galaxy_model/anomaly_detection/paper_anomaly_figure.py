"""
Paper-ready 3-column anomaly figure.

Columns (left→right):
  1. Ours (Physics)   — ours/hsc_mean/flow   from anomaly_scores_ours_100k.h5
  2. AION             — aion/hsc_mean_pca64/flow from anomaly_scores_aion_100k.h5
  3. Ours (Instrument)— ours/hsc_flat/flow    from anomaly_scores_ins_100k.h5

Produces two figures:
  paper_anomaly_8.png   — 2 rows × 4 cols per column group (8 examples each)
  paper_anomaly_12.png  — 3 rows × 4 cols per column group (12 examples each)

Each image is labelled with its rank and NLL percentile among all scores in that model.

Run from galaxy_model/:
  python anomaly_detection/paper_anomaly_figure.py [--out-dir anomaly_detection/outputs/figures_compare]
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

_here = Path(__file__).resolve().parent
OUTPUTS_DIR = _here / "outputs"
NEIGHBORS_HDF5 = "/work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5"

SOURCES = [
    {
        "label": "Ours (Physics)",
        "scores_file": "anomaly_scores_ours_100k.h5",
        "score_key": "ours/hsc_mean/flow",
    },
    {
        "label": "AION",
        "scores_file": "anomaly_scores_aion_100k.h5",
        "score_key": "aion/hsc_mean_pca64/flow",
    },
    {
        "label": "Ours (Instrument)",
        "scores_file": "anomaly_scores_ins_100k.h5",
        "score_key": "ours/hsc_flat/flow",
    },
]

N_COLS_PER = 4

# Background colors — lightened from downstream_eval/final/makeplot_v2.py palette:
#   Physics:    #8AC3EE (light blue bar)  → ~15% tint
#   AION:       #5DA271 (green)           → ~15% tint
#   Instrument: #E5254E (red bar)         → ~10% tint (softer)
BG_COLORS = ["#E7F3FC", "#DFECE3", "#FCE8EC"]


def _to_rgb(img_chw, pct_lo=1, pct_hi=99):
    rgb = img_chw[:3].astype(np.float32)
    lo = np.percentile(rgb, pct_lo, axis=(1, 2), keepdims=True)
    hi = np.percentile(rgb, pct_hi, axis=(1, 2), keepdims=True)
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)
    return (rgb.transpose(1, 2, 0) * 255).astype(np.uint8)


def _load_hsc_images(h5_data, raw_indices):
    sort_order = np.argsort(raw_indices)
    sorted_idx = raw_indices[sort_order]
    imgs = h5_data["images_hsc"][sorted_idx]
    return imgs[np.argsort(sort_order)]


def _top_n_with_percentiles(scores_path, score_key, n):
    """Return top-n raw indices and their NLL percentile among all finite scores."""
    with h5py.File(scores_path, "r") as f:
        raw_index = f["raw_index"][:]
        parts = score_key.split("/")
        node = f
        for p in parts:
            node = node[p]
        scores = node[:]
    finite = np.isfinite(scores)
    sorted_all = np.sort(scores[finite])
    order = np.argsort(scores * finite.astype(float))[::-1]
    top_idx = order[:n]
    top_raw = raw_index[top_idx]
    top_scores = scores[top_idx]
    # percentile: fraction of all finite scores strictly below this score × 100
    top_pcts = np.array([
        np.searchsorted(sorted_all, s, side="left") / len(sorted_all) * 100
        for s in top_scores
    ])
    return top_raw, top_pcts


def make_figure(top_n, out_path, img_size=1.35, group_gap=0.35):
    n_rows = top_n // N_COLS_PER
    n_groups = len(SOURCES)

    fig_w = n_groups * N_COLS_PER * img_size + (n_groups - 1) * group_gap
    fig_h = n_rows * img_size + 0.7  # extra room for larger header

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")

    gs_left, gs_right = 0.005, 0.995
    wspace_frac = group_gap / (N_COLS_PER * img_size)
    outer = gridspec.GridSpec(
        1, n_groups, figure=fig,
        left=gs_left, right=gs_right,
        top=0.82, bottom=0.03,
        wspace=wspace_frac,
    )

    # Compute how much to expand each rectangle into the gap on each side.
    # Use 40% of the half-gap so rectangles don't touch.
    col_w_fig = (gs_right - gs_left) / (n_groups + wspace_frac * (n_groups - 1))
    gap = wspace_frac * col_w_fig
    pad_x = gap * 0.40

    # Background axes — rendered before all image axes (zorder=-1).
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

    print(f"Building figure: top_n={top_n}, n_rows={n_rows}")

    with h5py.File(NEIGHBORS_HDF5, "r") as h5_data:
        for col_idx, src in enumerate(SOURCES):
            scores_path = OUTPUTS_DIR / src["scores_file"]
            print(f"  {src['label']}: loading top {top_n} ...")
            top_raw, top_pcts = _top_n_with_percentiles(scores_path, src["score_key"], top_n)
            hsc_imgs = _load_hsc_images(h5_data, top_raw)

            inner = gridspec.GridSpecFromSubplotSpec(
                n_rows, N_COLS_PER,
                subplot_spec=outer[col_idx],
                hspace=0.30, wspace=0.04,
            )

            for i in range(top_n):
                r, c = divmod(i, N_COLS_PER)
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

            # column group header
            pos = outer[col_idx].get_position(fig)
            x_mid = (pos.x0 + pos.x1) / 2
            fig.text(
                x_mid, 0.88,
                src["label"],
                ha="center", va="bottom",
                fontsize=15, fontweight="bold",
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as PNG
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {out_path}")

    # Save as PDF
    pdf_path = out_path.with_suffix('.pdf')
    plt.savefig(pdf_path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {pdf_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else OUTPUTS_DIR / "figures_compare"

    make_figure(8,  out_dir / "paper_anomaly_8.png")
    make_figure(12, out_dir / "paper_anomaly_12.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
