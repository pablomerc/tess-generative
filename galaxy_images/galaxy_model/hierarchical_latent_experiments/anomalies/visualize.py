"""Plot top anomalies, score correlation, and UMAP-colored anomaly scores.

Forks anomaly_detection/visualize_top_anomalies.py.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

_THIS = Path(__file__).resolve()
_PKG_ROOT = _THIS.parents[4]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from galaxy_images.galaxy_model.hierarchical_latent_experiments.common.discord_utils import (
    post_image, post_text,
)

NEIGHBORS_HDF5_DEFAULT = "/work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5"


def _to_rgb(img_chw, pct_lo=1, pct_hi=99):
    rgb = img_chw[:3].astype(np.float32)
    lo = np.percentile(rgb, pct_lo, axis=(1, 2), keepdims=True)
    hi = np.percentile(rgb, pct_hi, axis=(1, 2), keepdims=True)
    rgb = (rgb - lo) / (hi - lo + 1e-8)
    rgb = np.clip(rgb, 0, 1)
    return (rgb.transpose(1, 2, 0) * 255).astype(np.uint8)


def _load_images_batch(h5_file, raw_indices):
    sort_order = np.argsort(raw_indices)
    sorted_idx = raw_indices[sort_order]
    hsc_imgs = h5_file["images_hsc"][sorted_idx]
    leg_imgs = h5_file["images_legacy"][sorted_idx]
    restore = np.argsort(sort_order)
    return hsc_imgs[restore], leg_imgs[restore]


def plot_top_anomalies(raw_indices_top, h5_data, score_label, top_n, out_path):
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
    fig, ax = plt.subplots(figsize=(max(8, n * 0.6), max(7, n * 0.55)))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    plt.colorbar(im, ax=ax, fraction=0.03)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(score_keys, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(score_keys, fontsize=8)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="black" if abs(corr[i, j]) < 0.7 else "white")
    ax.set_title("Spearman rank correlation across anomaly scores")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_umap_colored(latent_for_umap, score_keys, all_scores, out_path):
    try:
        import umap as umap_lib
    except ImportError:
        print("  umap not installed — skipping UMAP panel")
        return False
    reducer = umap_lib.UMAP(n_neighbors=15, min_dist=0.1, n_components=2,
                            metric="euclidean", random_state=42)
    embedding = reducer.fit_transform(latent_for_umap)
    n_panels = min(len(score_keys), 16)
    n_cols = 4
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    axes = np.array(axes).flatten()
    for i, key in enumerate(score_keys[:n_panels]):
        scores = all_scores[i]
        if not np.isfinite(scores).any():
            axes[i].set_visible(False); continue
        finite = np.isfinite(scores)
        vmin, vmax = np.percentile(scores[finite], [2, 98])
        sc = axes[i].scatter(embedding[:, 0], embedding[:, 1], c=scores, s=1, alpha=0.5,
                             cmap="viridis", vmin=vmin, vmax=vmax, rasterized=True)
        plt.colorbar(sc, ax=axes[i], fraction=0.04)
        axes[i].set_title(key, fontsize=8); axes[i].axis("off")
    for i in range(n_panels, len(axes)):
        axes[i].set_visible(False)
    fig.suptitle("UMAP of physics latent colored by anomaly score", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--latents", required=True)
    ap.add_argument("--data", default=NEIGHBORS_HDF5_DEFAULT)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--top-n", type=int, default=25)
    ap.add_argument("--no-discord", action="store_true")
    args = ap.parse_args()

    if not args.no_discord:
        post_text(f"🟢 [anomalies/viz/{args.variant}] starting")

    t0 = time.time()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.scores, "r") as f:
        raw_index = f["raw_index"][:]
        score_keys = [k.decode() if isinstance(k, bytes) else k for k in f.attrs.get("score_keys", [])]
        if not score_keys:
            score_keys = [k for k in f.keys() if k != "raw_index"]
        all_scores = [f[k][:] for k in score_keys]
    print(f"loaded {len(score_keys)} score columns over N={len(raw_index)}")

    with h5py.File(args.latents, "r") as f:
        hsc_flat = f["hsc_flat"][:]

    posted = []
    print("Plotting top anomalies …")
    with h5py.File(args.data, "r") as h5_data:
        for key, scores in zip(score_keys, all_scores):
            if not np.isfinite(scores).any():
                print(f"  skip {key} (all NaN)"); continue
            mask = np.isfinite(scores)
            order = np.argsort(scores * mask.astype(float))[::-1]
            top = raw_index[order[:args.top_n]]
            label = key.replace("/", "_")
            out = out_dir / f"top_anomalies_{label}.png"
            plot_top_anomalies(top, h5_data, key, args.top_n, out)
            posted.append(out)

    print("Plotting correlation matrix …")
    corr_path = out_dir / "score_correlation.png"
    plot_correlation_matrix(score_keys, all_scores, corr_path)

    print("Plotting UMAP coloring …")
    umap_path = out_dir / "umap_anomaly_scores.png"
    has_umap = plot_umap_colored(hsc_flat, score_keys, all_scores, umap_path)

    elapsed = time.time() - t0
    print(f"done in {elapsed:.1f}s; figures in {out_dir}")

    if not args.no_discord:
        # Headline figures only — top-N grids would flood Discord with one-per-method.
        post_image(corr_path, message=f"📈 [anomalies/{args.variant}] score correlation matrix")
        if has_umap:
            post_image(umap_path, message=f"🧭 [anomalies/{args.variant}] UMAP of physics latent colored by anomaly score")
        # Send the top-anomaly grids for the most informative methods only.
        priority = [k for k in score_keys if k.endswith("/iforest") or k.endswith("/flow")]
        for key in priority[:4]:
            label = key.replace("/", "_")
            f = out_dir / f"top_anomalies_{label}.png"
            if f.exists():
                post_image(f, message=f"🚨 [anomalies/{args.variant}] top-{args.top_n} {key}")
        post_text(f"✅ [anomalies/viz/{args.variant}] done in {elapsed:.0f}s — {len(posted)} top-N grids saved in {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        try:
            from galaxy_images.galaxy_model.hierarchical_latent_experiments.common.discord_utils import post_text as _pt
            _pt(f"❌ [anomalies/viz] FAILED: {type(exc).__name__}: {exc}")
        except Exception:
            pass
        raise
