"""
Show top anomalies as they appear after 48x48 training preprocessing.

Same layout as paper_anomaly_8.png (Ours Physics + AION, 8 examples each),
but images are run through the full training preprocessing pipeline:
  center-crop to 48x48, clamp, zeropoint rescale, arcsinh range compression.

Run from galaxy_model/:
  python anomaly_detection/counterfactual_anomaly_experiment/plot_preprocessed_anomalies.py \
    [--out-dir anomaly_detection/counterfactual_anomaly_experiment/outputs]
"""
import argparse
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
_model_dir = _here.parent.parent
_repo_root = _model_dir.parent.parent
for p in [str(_model_dir), str(_repo_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from galaxy_images.image_preprocessing import preprocess_image_v2

NEIGHBORS_HDF5 = "/work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5"
OUTPUTS_DIR = _here.parent / "outputs"

SOURCES = [
    {
        "label": "Ours (Physics)",
        "scores_file": "anomaly_scores_ours_100k.h5",
        "score_key": "ours/hsc_mean/flow",
        "bg_color": "#E7F3FC",
    },
    {
        "label": "AION",
        "scores_file": "anomaly_scores_aion_100k.h5",
        "score_key": "aion/hsc_mean_pca64/flow",
        "bg_color": "#DFECE3",
    },
]

N_COLS_PER = 4


def _to_rgb(img_chw, pct_lo=1, pct_hi=99):
    """Percentile stretch first 3 channels → uint8 HxWx3."""
    rgb = img_chw[:3].astype(np.float32)
    lo = np.percentile(rgb, pct_lo, axis=(1, 2), keepdims=True)
    hi = np.percentile(rgb, pct_hi, axis=(1, 2), keepdims=True)
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)
    return (rgb.transpose(1, 2, 0) * 255).astype(np.uint8)


def _preprocess_48(raw_chw: np.ndarray) -> np.ndarray:
    """Apply 48x48 training preprocessing; return float32 (C,48,48)."""
    t = torch.from_numpy(raw_chw).float()
    out = preprocess_image_v2(t, crop_size=48, survey="hsc")
    return out.numpy()


def _top_n_raw(scores_path, score_key, n):
    with h5py.File(scores_path, "r") as f:
        raw_index = f["raw_index"][:]
        node = f
        for p in score_key.split("/"):
            node = node[p]
        scores = node[:]
    finite = np.isfinite(scores)
    order = np.argsort(scores * finite.astype(float))[::-1]
    return raw_index[order[:n]]


def make_figure(top_n, out_path, img_size=1.35, group_gap=0.35):
    import math
    n_rows = math.ceil(top_n / N_COLS_PER)
    n_groups = len(SOURCES)

    fig_w = n_groups * N_COLS_PER * img_size + (n_groups - 1) * group_gap
    fig_h = n_rows * img_size + 0.7

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

    col_w_fig = (gs_right - gs_left) / (n_groups + wspace_frac * (n_groups - 1))
    gap = wspace_frac * col_w_fig
    pad_x = gap * 0.40

    bg_ax = fig.add_axes([0, 0, 1, 1], zorder=-1)
    bg_ax.set_xlim(0, 1)
    bg_ax.set_ylim(0, 1)
    bg_ax.set_axis_off()
    bg_ax.patch.set_visible(False)

    for col_idx, src in enumerate(SOURCES):
        pos = outer[col_idx].get_position(fig)
        x0 = pos.x0 - (pad_x if col_idx > 0 else 0)
        x1 = pos.x1 + (pad_x if col_idx < n_groups - 1 else 0)
        bg_ax.axvspan(x0, x1, facecolor=src["bg_color"], linewidth=0)

    with h5py.File(NEIGHBORS_HDF5, "r") as h5_data:
        for col_idx, src in enumerate(SOURCES):
            scores_path = OUTPUTS_DIR / src["scores_file"]
            print(f"  {src['label']}: loading top {top_n} ...")
            top_raw = _top_n_raw(scores_path, src["score_key"], top_n)

            # Load raw 160x160 and preprocess to 48x48
            sort_order = np.argsort(top_raw)
            sorted_idx = top_raw[sort_order]
            raw_imgs = h5_data["images_hsc"][sorted_idx]           # (N,5,160,160)
            raw_imgs = raw_imgs[np.argsort(sort_order)]            # restore original order

            processed = np.stack([_preprocess_48(raw_imgs[i]) for i in range(top_n)])

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
                ax.imshow(_to_rgb(processed[i]), interpolation="nearest")
                ax.axis("off")
                ax.set_title(f"#{i+1}", fontsize=11, pad=3,
                             color="#222222", fontweight="bold")

            pos = outer[col_idx].get_position(fig)
            x_mid = (pos.x0 + pos.x1) / 2
            fig.text(x_mid, 0.88, src["label"],
                     ha="center", va="bottom", fontsize=15, fontweight="bold")

    fig.text(0.5, 0.96, "Preprocessed to 48×48 (training pipeline)",
             ha="center", va="bottom", fontsize=12, color="#555555")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else _here / "outputs"
    make_figure(8, out_dir / "preprocessed_anomalies_48x48.png")
    print("\nDone.")


if __name__ == "__main__":
    main()
