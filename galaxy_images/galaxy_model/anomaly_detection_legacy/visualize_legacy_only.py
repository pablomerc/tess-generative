"""
Single-score, Legacy-thumbnail anomaly visualization.

Given a scores file and one score key, plot a top-N grid (5 cols) showing
ONLY the Legacy image for each top-anomaly example.

Usage:
  python anomaly_detection_legacy/visualize_legacy_only.py \
    --scores anomaly_detection_legacy/outputs/anomaly_scores_ours_legacy_100k.h5 \
    --score-key ours/hsc_flat/flow \
    --data /work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5 \
    --out anomaly_detection_legacy/outputs/figures_focus/ours_legacy_flat_flow.png \
    --top-n 25 --label "Ours (physics) — legacy_flat / flow"
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _to_rgb(img_chw, pct_lo=1, pct_hi=99):
    rgb = img_chw[:3].astype(np.float32)
    lo = np.percentile(rgb, pct_lo, axis=(1, 2), keepdims=True)
    hi = np.percentile(rgb, pct_hi, axis=(1, 2), keepdims=True)
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)
    return (rgb.transpose(1, 2, 0) * 255).astype(np.uint8)


def _load_legacy_images(h5_data, raw_indices):
    sort_order = np.argsort(raw_indices)
    sorted_idx = raw_indices[sort_order]
    imgs = h5_data["images_legacy"][sorted_idx]
    return imgs[np.argsort(sort_order)]


def _read_score(scores_path, score_key):
    with h5py.File(scores_path, "r") as f:
        raw_index = f["raw_index"][:]
        node = f
        for p in score_key.split("/"):
            node = node[p]
        scores = node[:]
    return raw_index, scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--score-key", required=True, help="e.g. ours/hsc_flat/flow")
    parser.add_argument("--data", default="/work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5")
    parser.add_argument("--out", required=True, help="Output PNG path")
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--label", default=None, help="Figure title")
    args = parser.parse_args()

    raw_index, scores = _read_score(args.scores, args.score_key)
    finite = np.isfinite(scores)
    if not finite.any():
        raise SystemExit(f"All scores NaN for {args.score_key} in {args.scores}")
    order = np.argsort(scores * finite.astype(float))[::-1]
    top_raw = raw_index[order[:args.top_n]]

    n_cols = 5
    n_rows = (args.top_n + n_cols - 1) // n_cols

    with h5py.File(args.data, "r") as h5_data:
        leg_imgs = _load_legacy_images(h5_data, top_raw)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    if n_rows == 1:
        axes = axes.reshape(1, n_cols)

    for i in range(args.top_n):
        r, c = divmod(i, n_cols)
        ax = axes[r, c]
        ax.imshow(_to_rgb(leg_imgs[i]))
        ax.set_title(f"#{i+1}\nidx={top_raw[i]}", fontsize=7)
        ax.axis("off")

    # Hide unused axes if top_n < n_rows*n_cols
    for j in range(args.top_n, n_rows * n_cols):
        r, c = divmod(j, n_cols)
        axes[r, c].axis("off")

    title = args.label or f"{args.score_key} — top {args.top_n}"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
