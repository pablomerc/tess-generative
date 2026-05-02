"""
Generate browse plots for the top 512 anomalies (8 batches of 64) from
Ours (Physics) and AION, in two versions:
  - original:     raw 160x160 images with percentile stretch
  - preprocessed: 48x48 training pipeline then percentile stretch

Outputs go to:
  counterfactual_anomaly_experiment/outputs/top500_browse/original/batch_NNN_MMM.png
  counterfactual_anomaly_experiment/outputs/top500_browse/preprocessed/batch_NNN_MMM.png

Also saves an index file:
  counterfactual_anomaly_experiment/outputs/top500_browse/rank_to_raw_index.npz

  Contains arrays 'physics_raw' and 'aion_raw', each length 512,
  where physics_raw[i] is the HDF5 raw_index for rank i+1.

Run from galaxy_model/:
  python anomaly_detection/counterfactual_anomaly_experiment/generate_top500_browse.py
"""
import sys
from pathlib import Path
import math

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
BROWSE_DIR = _here / "outputs" / "top500_browse"

SOURCES = [
    {
        "label": "Ours (Physics)",
        "scores_file": "anomaly_scores_ours_100k.h5",
        "score_key": "ours/hsc_mean/flow",
        "bg_color": "#E7F3FC",
        "key": "physics",
    },
    {
        "label": "AION",
        "scores_file": "anomaly_scores_aion_100k.h5",
        "score_key": "aion/hsc_mean_pca64/flow",
        "bg_color": "#DFECE3",
        "key": "aion",
    },
]

TOTAL = 512
BATCH_SIZE = 64
N_COLS = 4


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


def _load_raw(h5_data, raw_indices):
    sort_order = np.argsort(raw_indices)
    sorted_idx = raw_indices[sort_order]
    imgs = h5_data["images_hsc"][sorted_idx]
    return imgs[np.argsort(sort_order)]


def _to_rgb(img_chw, pct_lo=1, pct_hi=99):
    rgb = img_chw[:3].astype(np.float32)
    lo = np.percentile(rgb, pct_lo, axis=(1, 2), keepdims=True)
    hi = np.percentile(rgb, pct_hi, axis=(1, 2), keepdims=True)
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)
    return (rgb.transpose(1, 2, 0) * 255).astype(np.uint8)


def _preprocess_48(raw_chw: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(raw_chw).float()
    return preprocess_image_v2(t, crop_size=48, survey="hsc").numpy()


def _make_batch_figure(imgs_per_source, rank_offset, mode, img_size=1.35, group_gap=0.35):
    """
    imgs_per_source: list of (N,5,H,W) arrays, one per source, already in rank order.
    rank_offset: global rank of imgs_per_source[*][0] (1-indexed).
    mode: 'original' or 'preprocessed'
    """
    n = len(imgs_per_source[0])
    n_rows = math.ceil(n / N_COLS)
    n_groups = len(SOURCES)

    fig_w = n_groups * N_COLS * img_size + (n_groups - 1) * group_gap
    fig_h = n_rows * img_size + 0.65

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")

    gs_left, gs_right = 0.005, 0.995
    wspace_frac = group_gap / (N_COLS * img_size)
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

    for col_idx, src in enumerate(SOURCES):
        raw_imgs = imgs_per_source[col_idx]  # (N,5,H,W)

        if mode == "preprocessed":
            display = [_to_rgb(_preprocess_48(raw_imgs[i])) for i in range(n)]
        else:
            display = [_to_rgb(raw_imgs[i]) for i in range(n)]

        inner = gridspec.GridSpecFromSubplotSpec(
            n_rows, N_COLS,
            subplot_spec=outer[col_idx],
            hspace=0.30, wspace=0.04,
        )

        for i in range(n):
            r, c = divmod(i, N_COLS)
            ax = fig.add_subplot(inner[r, c])
            ax.set_facecolor("none")
            ax.set_zorder(2)
            interp = "nearest" if mode == "preprocessed" else "bilinear"
            ax.imshow(display[i], interpolation=interp)
            ax.axis("off")
            ax.set_title(f"#{rank_offset + i}", fontsize=8, pad=2,
                         color="#222222", fontweight="bold")

        pos = outer[col_idx].get_position(fig)
        x_mid = (pos.x0 + pos.x1) / 2
        fig.text(x_mid, 0.88, src["label"],
                 ha="center", va="bottom", fontsize=13, fontweight="bold")

    suffix = "48×48 preprocessed" if mode == "preprocessed" else "original 160×160"
    rank_end = rank_offset + n - 1
    fig.text(0.5, 0.96, f"Ranks #{rank_offset}–#{rank_end}  —  {suffix}",
             ha="center", va="bottom", fontsize=10, color="#555555")

    return fig


def main():
    print("Loading top-512 raw indices for each source...")
    all_raw = {}
    for src in SOURCES:
        scores_path = OUTPUTS_DIR / src["scores_file"]
        raw = _top_n_raw(scores_path, src["score_key"], TOTAL)
        all_raw[src["key"]] = raw
        print(f"  {src['label']}: {len(raw)} indices")

    # Save index file for later retrieval
    idx_path = BROWSE_DIR / "rank_to_raw_index.npz"
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(idx_path,
             physics_raw=all_raw["physics"],
             aion_raw=all_raw["aion"])
    print(f"Saved index: {idx_path}")

    orig_dir = BROWSE_DIR / "original"
    prep_dir = BROWSE_DIR / "preprocessed"
    orig_dir.mkdir(parents=True, exist_ok=True)
    prep_dir.mkdir(parents=True, exist_ok=True)

    n_batches = math.ceil(TOTAL / BATCH_SIZE)

    with h5py.File(NEIGHBORS_HDF5, "r") as h5_data:
        for b in range(n_batches):
            start = b * BATCH_SIZE
            end = min(start + BATCH_SIZE, TOTAL)
            rank_offset = start + 1  # 1-indexed

            print(f"Batch {b+1}/{n_batches}: ranks #{rank_offset}–#{end} ...")

            # Load raw images for this batch for each source
            imgs_per_source = []
            for src in SOURCES:
                batch_raw = all_raw[src["key"]][start:end]
                imgs = _load_raw(h5_data, batch_raw)
                imgs_per_source.append(imgs)

            fname = f"batch_{rank_offset:03d}_{end:03d}.png"

            for mode, out_dir in [("original", orig_dir), ("preprocessed", prep_dir)]:
                fig = _make_batch_figure(imgs_per_source, rank_offset, mode)
                out_path = out_dir / fname
                fig.savefig(out_path, dpi=180, bbox_inches="tight")
                plt.close(fig)
                print(f"  [{mode}] Saved: {out_path.name}")

    print("\nDone.")
    print(f"Index file: {idx_path}")
    print(f"Browse dir: {BROWSE_DIR}")


if __name__ == "__main__":
    main()
