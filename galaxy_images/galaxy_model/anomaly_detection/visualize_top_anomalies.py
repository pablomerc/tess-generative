"""
Load anomaly scores, retrieve galaxy images by raw_index, and plot grids of top anomalies.

Produces per (method × latent) combination:
  - A 5×5 grid figure showing the top 25 most anomalous galaxies (HSC top, Legacy bottom).
  - A Spearman rank correlation matrix across all score combinations.
  - UMAP of hsc_flat colored by each method's scores (requires ours latents file).

Run from galaxy_model/ or anomaly_detection/:
  python anomaly_detection/visualize_top_anomalies.py \
    --scores outputs/anomaly_scores_best87k.h5 \
    --ours   outputs/anomaly_latents_ours_best87k.h5 \
    --data   /work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5 \
    --suffix best87k \
    [--top-n 25] [--out-dir outputs/figures_best87k]
"""
import argparse
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from tqdm import tqdm


NEIGHBORS_HDF5_DEFAULT = "/work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5"


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _to_rgb(img_chw, pct_lo=1, pct_hi=99):
    """Convert a (C, H, W) float32 array to displayable (H, W, 3) uint8."""
    rgb = img_chw[:3].astype(np.float32)
    lo = np.percentile(rgb, pct_lo, axis=(1, 2), keepdims=True)
    hi = np.percentile(rgb, pct_hi, axis=(1, 2), keepdims=True)
    rgb = (rgb - lo) / (hi - lo + 1e-8)
    rgb = np.clip(rgb, 0, 1)
    return (rgb.transpose(1, 2, 0) * 255).astype(np.uint8)


def _load_images_batch(h5_file, raw_indices):
    """Load HSC and Legacy images for given raw HDF5 indices."""
    # h5py requires fancy indices in strictly increasing order
    sort_order = np.argsort(raw_indices)
    sorted_idx = raw_indices[sort_order]
    hsc_imgs = h5_file["images_hsc"][sorted_idx]
    leg_imgs = h5_file["images_legacy"][sorted_idx]
    # restore original (score-ranked) order
    restore = np.argsort(sort_order)
    return hsc_imgs[restore], leg_imgs[restore]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_top_anomalies(raw_indices_top, h5_data, score_label, top_n, out_path):
    """5×5 grid: top row half = HSC, bottom row half = Legacy, 5 columns."""
    n_cols = 5
    n_rows_per = max(1, top_n // n_cols)
    n_show = n_cols * n_rows_per

    hsc_imgs, leg_imgs = _load_images_batch(h5_data, raw_indices_top[:n_show])

    fig, axes = plt.subplots(n_rows_per * 2, n_cols, figsize=(n_cols * 2, n_rows_per * 4))
    if axes.ndim == 1:
        axes = axes.reshape(2, n_cols)

    for i in range(n_show):
        row_hsc = (i // n_cols) * 2
        row_leg = row_hsc + 1
        col = i % n_cols

        axes[row_hsc, col].imshow(_to_rgb(hsc_imgs[i]))
        axes[row_hsc, col].set_title(f"#{i+1} HSC\nidx={raw_indices_top[i]}", fontsize=7)
        axes[row_hsc, col].axis("off")

        axes[row_leg, col].imshow(_to_rgb(leg_imgs[i]))
        axes[row_leg, col].set_title(f"#{i+1} Legacy", fontsize=7)
        axes[row_leg, col].axis("off")

    fig.suptitle(f"Top {n_show} anomalies — {score_label}", fontsize=10, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_correlation_matrix(score_keys, all_scores, out_path):
    n = len(score_keys)
    corr = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i == j:
                corr[i, j] = 1.0
            elif j > i:
                si, sj = all_scores[i], all_scores[j]
                mask = np.isfinite(si) & np.isfinite(sj)
                if mask.sum() > 10:
                    corr[i, j] = spearmanr(si[mask], sj[mask]).statistic
                    corr[j, i] = corr[i, j]

    short_keys = [k.replace("ours/", "").replace("aion/", "aion:") for k in score_keys]

    fig, ax = plt.subplots(figsize=(max(8, n * 0.6), max(7, n * 0.55)))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    plt.colorbar(im, ax=ax, fraction=0.03)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_keys, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short_keys, fontsize=8)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="black" if abs(corr[i, j]) < 0.7 else "white")
    ax.set_title("Spearman rank correlation across anomaly scores")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved correlation matrix: {out_path}")


def plot_umap_colored(hsc_flat, score_keys, all_scores, out_dir, suffix):
    try:
        import umap as umap_lib
    except ImportError:
        print("  umap-learn not installed — skipping UMAP. Install with: pip install umap-learn")
        return

    print("  Computing UMAP of hsc_flat...")
    reducer = umap_lib.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric="euclidean", random_state=42)
    embedding = reducer.fit_transform(hsc_flat)

    n_panels = min(len(score_keys), 16)
    n_cols = 4
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    axes = np.array(axes).flatten()

    for i, key in enumerate(score_keys[:n_panels]):
        scores = all_scores[i]
        if not np.isfinite(scores).any():
            axes[i].set_visible(False)
            continue
        finite_mask = np.isfinite(scores)
        vmin, vmax = np.percentile(scores[finite_mask], [2, 98])
        sc = axes[i].scatter(embedding[:, 0], embedding[:, 1], c=scores, s=1, alpha=0.5,
                             cmap="viridis", vmin=vmin, vmax=vmax, rasterized=True)
        plt.colorbar(sc, ax=axes[i], fraction=0.04)
        axes[i].set_title(key.replace("ours/", "").replace("aion/", "aion:"), fontsize=8)
        axes[i].axis("off")

    for i in range(n_panels, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("UMAP of hsc_flat colored by anomaly score", fontsize=12)
    plt.tight_layout()
    out_path = out_dir / f"umap_anomaly_scores_{suffix}.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved UMAP: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True, help="Path to anomaly_scores_*.h5")
    parser.add_argument("--ours", default=None, help="Path to anomaly_latents_ours_*.h5 (for UMAP)")
    parser.add_argument("--data", default=NEIGHBORS_HDF5_DEFAULT, help="Path to neighbours_v2.h5")
    parser.add_argument("--suffix", default="best87k")
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    scores_path = Path(args.scores)
    if not scores_path.is_absolute() and not scores_path.exists():
        scores_path = _here / "outputs" / args.scores

    out_dir = Path(args.out_dir) if args.out_dir else _here / "outputs" / f"figures_{args.suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading scores from {scores_path}")
    with h5py.File(scores_path, "r") as f:
        raw_index = f["raw_index"][:]
        score_keys = list(f.attrs.get("score_keys", []))
        if not score_keys:
            score_keys = [k for k in f.keys() if k != "raw_index"]
        all_scores = [f[k][:] for k in score_keys]

    print(f"  N={len(raw_index)}, score keys: {score_keys}")

    hsc_flat = None
    if args.ours is not None:
        ours_path = Path(args.ours)
        if not ours_path.is_absolute() and not ours_path.exists():
            ours_path = _here / "outputs" / args.ours
        with h5py.File(ours_path, "r") as f:
            hsc_flat = f["hsc_flat"][:]

    print(f"\nOpening galaxy data from {args.data}")
    with h5py.File(args.data, "r") as h5_data:
        for key, scores in zip(score_keys, all_scores):
            if not np.isfinite(scores).any():
                print(f"  Skipping {key} (all NaN)")
                continue
            finite_mask = np.isfinite(scores)
            sorted_idx = np.argsort(scores * finite_mask.astype(float))[::-1]
            top_raw = raw_index[sorted_idx[:args.top_n]]

            label = key.replace("/", "_")
            out_path = out_dir / f"top_anomalies_{label}_{args.suffix}.png"
            print(f"  Plotting top {args.top_n} for {key}...")
            plot_top_anomalies(top_raw, h5_data, key, args.top_n, out_path)
            print(f"    Saved: {out_path}")

    print("\nPlotting Spearman correlation matrix...")
    corr_path = out_dir / f"score_correlation_{args.suffix}.png"
    plot_correlation_matrix(score_keys, all_scores, corr_path)

    if hsc_flat is not None:
        print("\nPlotting UMAP colored by scores...")
        plot_umap_colored(hsc_flat, score_keys, all_scores, out_dir, args.suffix)

    print(f"\nDone. All figures saved to {out_dir}")


if __name__ == "__main__":
    main()
