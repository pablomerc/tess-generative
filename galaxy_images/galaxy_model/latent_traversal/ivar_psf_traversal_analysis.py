"""
Traversal analysis: HSC ivar × PSF FWHM → sum score + PCA score.

Steps:
  1. Load hsc_mean_ivar, hsc_psf_fwhm_avg, hsc_images, hdf5_row_idx from stats HDF5
  2. Visualise zero-ivar examples (≤ 8), drop them
  3. StandardScaler normalise each axis independently
  4. Compute two metrics: sum score (z_ivar - z_psf) and PCA score (PC1)
  5. Hexbin 2D density + marginal 1D histograms + PC arrows → hsc_ivar_psf_pca_hexbin.png
  6. 6 inspection grids (2 metrics × 3 shifts, 2×5 images each)
  7. Save intermediate .npz cache for fast replay
  8. Post all to Discord

Usage:
  python ivar_psf_traversal_analysis.py --stats-path /work1/.../hsc_ivar_psf_stats.h5
  python ivar_psf_traversal_analysis.py --stats-path ... --cache-path .../ivar_psf_cache.npz
"""

import argparse
import io
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ============= CONFIGURATION =============

STATS_DEFAULT = Path("/work1/jeroenaudenaert/pablomer/data/hsc_ivar_psf_stats.h5")
OUTPUT_DIR    = Path(__file__).resolve().parent
DISCORD_WEBHOOK = (
    "https://discord.com/api/webhooks/1497979386144493680/"
    "VA-xWhfTWzc-oeC5EvPzyqEk_MW52wZsK2RyLS0egfhHHHhBxrmb9NGawy0rIpfvn3Zo"
)
CROP_SIZE = 48
INSPECTION_N = 10       # images per inspection figure
INSPECTION_COLS = 5     # columns per row in inspection figure


# ============= UTILITIES =============

def tensor_to_rgb(img: np.ndarray, percentile_clip: float = 99.5) -> np.ndarray:
    """img: (C, H, W) float → (H, W, 3) RGB in [0, 1]."""
    rgb = img[:3].copy()
    rgb = np.transpose(rgb, (1, 2, 0))
    for i in range(3):
        p_lo = np.percentile(rgb[:, :, i], 100 - percentile_clip)
        p_hi = np.percentile(rgb[:, :, i], percentile_clip)
        rgb[:, :, i] = np.clip(rgb[:, :, i], p_lo, p_hi)
        ch = rgb[:, :, i]
        lo, hi = ch.min(), ch.max()
        rgb[:, :, i] = (ch - lo) / (hi - lo) if hi > lo else 0.0
    return rgb


def send_to_discord(webhook_url: str, file_path: Path, message: str = ""):
    with open(file_path, "rb") as fh:
        data = fh.read()
    resp = requests.post(
        webhook_url,
        data={"content": message} if message else {},
        files={"file": (file_path.name, io.BytesIO(data), "image/png")},
    )
    if resp.status_code in (200, 204):
        print(f"  Sent {file_path.name} to Discord.")
    else:
        print(f"  WARNING: Discord {resp.status_code}: {resp.text[:200]}")


# ============= DATA LOADING =============

def load_stats(stats_path: Path):
    """Return (ivar, psf, images, row_idx) from stats HDF5."""
    with h5py.File(stats_path, "r") as f:
        ivar    = np.array(f["hsc_mean_ivar"],    dtype=np.float32)
        psf     = np.array(f["hsc_psf_fwhm_avg"], dtype=np.float32)
        images  = np.array(f["hsc_images"],        dtype=np.float32)   # (N, 4, 48, 48)
        row_idx = np.array(f["hdf5_row_idx"],      dtype=np.int32)
    print(
        f"Loaded {len(ivar):,} rows from {stats_path.name}\n"
        f"  ivar: [{ivar.min():.3g}, {ivar.max():.3g}]  zeros={(ivar == 0).sum()}\n"
        f"  psf:  [{psf.min():.3f}, {psf.max():.3f}] arcsec"
    )
    return ivar, psf, images, row_idx


# ============= STEP 1: ZERO-IVAR VISUALISATION =============

def plot_zero_ivar_examples(ivar: np.ndarray, images: np.ndarray,
                             output_path: Path, n_show: int = 8):
    zero_mask = ivar < 1e-6
    n_zeros = zero_mask.sum()
    print(f"Zero-ivar examples: {n_zeros:,}")
    if n_zeros == 0:
        print("  No zero-ivar examples — skipping plot.")
        return

    n_show = min(n_show, n_zeros)
    zero_imgs = images[zero_mask][:n_show]   # (n_show, 4, 48, 48)
    zero_ivar = ivar[zero_mask][:n_show]

    fig, axes = plt.subplots(1, n_show, figsize=(n_show * 2.2, 2.5))
    if n_show == 1:
        axes = [axes]
    for i, (img, iv) in enumerate(zip(zero_imgs, zero_ivar)):
        rgb = tensor_to_rgb(img)
        axes[i].imshow(rgb)
        axes[i].set_title(f"ivar={iv:.2g}", fontsize=7.5)
        axes[i].axis("off")

    fig.suptitle(
        f"HSC examples with ivar ≈ 0  (showing {n_show} / {n_zeros:,} total)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


# ============= STEP 4: HEXBIN + MARGINALS + PC ARROWS =============

def _percentile_drop(arr: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0):
    """Return mask of values inside [p_lo, p_hi] percentiles."""
    lo = np.nanpercentile(arr, p_lo)
    hi = np.nanpercentile(arr, p_hi)
    return (arr >= lo) & (arr <= hi), lo, hi


def _draw_hexbin_panel(fig, inner, ix, px, pca_components, pca_explained_variance,
                       xlabel, ylabel, title, n_total):
    pct = pca_explained_variance / pca_explained_variance.sum() * 100

    ax_top   = fig.add_subplot(inner[0, 0])
    ax_main  = fig.add_subplot(inner[1, 0])
    ax_right = fig.add_subplot(inner[1, 1])
    ax_cb    = fig.add_subplot(inner[0, 1])
    ax_cb.set_visible(False)

    hb = ax_main.hexbin(ix, px, gridsize=60, cmap="viridis", mincnt=1, linewidths=0.2)
    fig.colorbar(hb, ax=ax_cb, fraction=0.8, pad=0.05).set_label("Count", fontsize=8)
    ax_main.set_xlabel(xlabel, fontsize=9)
    ax_main.set_ylabel(ylabel, fontsize=9)
    ax_main.axhline(0, color="white", lw=0.4, alpha=0.4)
    ax_main.axvline(0, color="white", lw=0.4, alpha=0.4)

    # Draw PC arrows as fixed fraction of axis span, centered at plot midpoint
    xlim = ax_main.get_xlim()
    ylim = ax_main.get_ylim()
    cx   = (xlim[0] + xlim[1]) / 2
    cy   = (ylim[0] + ylim[1]) / 2
    span = min(xlim[1] - xlim[0], ylim[1] - ylim[0]) * 0.20   # 20% of shorter axis
    colors = ["tomato", "orange"]
    for k in range(min(2, len(pca_components))):
        v  = pca_components[k]
        dx = v[0] * span
        dy = v[1] * span
        ax_main.annotate(
            "", xy=(cx + dx, cy + dy), xytext=(cx, cy),
            arrowprops=dict(arrowstyle="->", color=colors[k], lw=2),
        )
        ax_main.text(cx + dx * 1.25, cy + dy * 1.25,
                     f"PC{k+1} ({pct[k]:.0f}%)", color=colors[k], fontsize=8,
                     ha="center", va="center")

    ax_top.hist(ix, bins=80, color="steelblue", edgecolor="none")
    ax_top.set_xlim(ax_main.get_xlim())
    ax_top.set_xticks([])
    ax_top.set_ylabel("Count", fontsize=8)
    ax_top.tick_params(axis="y", labelsize=7)
    ax_top.set_title(title, fontsize=10, fontweight="bold", pad=4)

    ax_right.hist(px, bins=80, color="tomato", edgecolor="none", orientation="horizontal")
    ax_right.set_ylim(ax_main.get_ylim())
    ax_right.set_yticks([])
    ax_right.set_xlabel("Count", fontsize=8)
    ax_right.tick_params(axis="x", labelsize=7)


def plot_hexbin(z_ivar: np.ndarray, z_psf: np.ndarray,
                pca_components: np.ndarray, pca_explained_variance: np.ndarray,
                n_total: int, n_dropped: int, output_path: Path):
    # Panel 2: p1-p99 drop-and-normalize
    mask_iv, iv_lo, iv_hi = _percentile_drop(z_ivar)
    mask_ps, ps_lo, ps_hi = _percentile_drop(z_psf)
    keep = mask_iv & mask_ps
    n_clipped = (~keep).sum()
    iv_f  = (z_ivar[keep] - iv_lo) / (iv_hi - iv_lo + 1e-10)
    psf_f = (z_psf[keep]  - ps_lo) / (ps_hi - ps_lo + 1e-10)

    fig = plt.figure(figsize=(18, 8))
    outer = fig.add_gridspec(1, 2, wspace=0.35)

    # Panel 1: raw z-scores
    inner1 = outer[0].subgridspec(
        2, 2, height_ratios=[1, 4], width_ratios=[4, 1], hspace=0.05, wspace=0.05
    )
    _draw_hexbin_panel(
        fig, inner1, z_ivar, z_psf, pca_components, pca_explained_variance,
        xlabel="z_ivar  (higher = more signal)",
        ylabel="z_psf  (lower = sharper)",
        title=f"Raw z-scores  (n={n_total:,}, {n_dropped:,} zeros dropped)",
        n_total=n_total,
    )

    # Panel 2: p1-p99 drop-and-normalize
    inner2 = outer[1].subgridspec(
        2, 2, height_ratios=[1, 4], width_ratios=[4, 1], hspace=0.05, wspace=0.05
    )
    _draw_hexbin_panel(
        fig, inner2, iv_f, psf_f, pca_components, pca_explained_variance,
        xlabel=f"z_ivar  norm [p1={iv_lo:.2f}, p99={iv_hi:.2f}]",
        ylabel=f"z_psf  norm [p1={ps_lo:.2f}, p99={ps_hi:.2f}]",
        title=f"p1–p99 drop-and-normalize  ({n_clipped:,} dropped)",
        n_total=len(iv_f),
    )

    fig.suptitle(
        "HSC ivar × PSF FWHM  —  z-score space + marginals + PC axes",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


# ============= STEP 5: INSPECTION FIGURES =============

def plot_inspection(images: np.ndarray, ivar: np.ndarray, psf: np.ndarray,
                    score: np.ndarray, metric_name: str, shift: int,
                    output_path: Path):
    """
    2 rows × INSPECTION_COLS columns of images sampled at linspace(5, 95, INSPECTION_N)
    percentile ranks of score, with an optional integer shift applied.
    """
    n = len(score)
    sorted_idx = np.argsort(score)
    ranks = np.linspace(5, 95, INSPECTION_N)
    positions = np.clip(
        (ranks / 100 * n + shift).astype(int), 0, n - 1
    )
    chosen = sorted_idx[positions]

    rows = 2
    cols = INSPECTION_COLS
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.5))

    for plot_i, idx in enumerate(chosen):
        r, c = divmod(plot_i, cols)
        img = images[idx]
        rgb = tensor_to_rgb(img)
        axes[r, c].imshow(rgb)
        axes[r, c].set_title(
            f"#{plot_i+1}  ivar={ivar[idx]:.0f}  PSF={psf[idx]:.2f}\"",
            fontsize=7,
        )
        axes[r, c].axis("off")

    fig.suptitle(
        f"Inspection: {metric_name}  shift={shift:+d}  "
        f"(rows={rows}, cols={cols}, n={n:,})",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


# ============= MAIN =============

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-path", type=Path, default=STATS_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--cache-path", type=Path, default=None,
                        help="Path to existing .npz cache; skip extraction if present")
    parser.add_argument("--n-zero-show", type=int, default=8)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    auto_cache = args.output_dir / "ivar_psf_cache.npz"

    # ---- Load or compute ----
    cache_path = args.cache_path or (auto_cache if auto_cache.exists() else None)

    if cache_path is not None and cache_path.exists():
        print(f"Loading cache from {cache_path}")
        c = np.load(cache_path)
        z_ivar           = c["z_ivar"]
        z_psf            = c["z_psf"]
        score_sum        = c["score_sum"]
        score_pca        = c["score_pca"]
        ivar             = c["ivar"]
        psf              = c["psf"]
        row_idx          = c["hdf5_row_idx"]
        pca_components   = c["pca_components"]
        pca_exp_var      = c["pca_explained_variance"]
        n_dropped        = int(c["n_dropped"])
        # images loaded on-demand below
        print(f"  Loaded {len(ivar):,} examples from cache.")
        images = None  # deferred
    else:
        ivar_raw, psf_raw, images_raw, row_idx_raw = load_stats(args.stats_path)

        # Step 1: zero-ivar visualisation
        zero_path = args.output_dir / "hsc_zero_ivar_examples.png"
        plot_zero_ivar_examples(ivar_raw, images_raw, zero_path, args.n_zero_show)

        # Drop zeros
        keep_mask = ivar_raw >= 1e-6
        n_dropped = int((~keep_mask).sum())
        ivar    = ivar_raw[keep_mask]
        psf     = psf_raw[keep_mask]
        images  = images_raw[keep_mask]
        row_idx = row_idx_raw[keep_mask]
        print(f"After dropping zeros: {len(ivar):,} examples remain")

        # Step 2: StandardScaler normalisation
        features = np.column_stack([ivar, psf]).astype(np.float64)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)
        z_ivar = scaled[:, 0].astype(np.float32)
        z_psf  = scaled[:, 1].astype(np.float32)

        # Step 3: metrics
        score_sum = (z_ivar - z_psf).astype(np.float32)

        pca = PCA(n_components=2)
        pca_coords = pca.fit_transform(scaled)
        score_pca = pca_coords[:, 0].astype(np.float32)
        # Orient so positive PC1 = better quality (higher ivar, lower psf)
        pc1 = pca.components_[0]
        if pc1[0] < 0:   # ivar loading negative → flip
            score_pca = -score_pca
            pca.components_[0] = -pca.components_[0]
        pca_components = pca.components_.astype(np.float32)
        pca_exp_var    = pca.explained_variance_.astype(np.float32)

        np.savez(
            auto_cache,
            z_ivar=z_ivar, z_psf=z_psf,
            score_sum=score_sum, score_pca=score_pca,
            ivar=ivar, psf=psf,
            hdf5_row_idx=row_idx,
            pca_components=pca_components,
            pca_explained_variance=pca_exp_var,
            n_dropped=np.array(n_dropped),
        )
        print(f"Saved cache → {auto_cache}")

    # If images weren't loaded (cache hit), load them now
    if images is None:
        with h5py.File(args.stats_path, "r") as f:
            all_ivar_raw = np.array(f["hsc_mean_ivar"], dtype=np.float32)
            keep_mask = all_ivar_raw >= 1e-6
            images = np.array(f["hsc_images"], dtype=np.float32)[keep_mask]
        print(f"Loaded images from {args.stats_path.name}")

    n_total = len(ivar)

    # ---- Hexbin plot ----
    hexbin_path = args.output_dir / "hsc_ivar_psf_pca_hexbin.png"
    plot_hexbin(z_ivar, z_psf, pca_components, pca_exp_var,
                n_total, n_dropped, hexbin_path)

    # ---- Inspection figures ----
    metrics = [
        ("sum",  score_sum,  "sum score (z_ivar − z_psf)"),
        ("pca",  score_pca,  "PCA score (PC1)"),
    ]
    inspection_paths = []
    for tag, score, label in metrics:
        for shift in (0, 1, 2):
            out = args.output_dir / f"hsc_inspection_{tag}_shift{shift}.png"
            plot_inspection(images, ivar, psf, score, label, shift, out)
            inspection_paths.append((out, tag, shift))

    # ---- Discord ----
    print("\n=== Sending to Discord ===")

    # zero-ivar plot only if it was produced (not from cache, or file exists)
    zero_path = args.output_dir / "hsc_zero_ivar_examples.png"
    if zero_path.exists():
        send_to_discord(DISCORD_WEBHOOK, zero_path,
                        f"**HSC zero-ivar examples** (n_zeros dropped={n_dropped:,})")

    send_to_discord(
        DISCORD_WEBHOOK, hexbin_path,
        f"**HSC ivar × PSF hexbin** (n={n_total:,}, {n_dropped:,} zeros dropped)",
    )

    # Sum inspection batch
    sum_paths = [(p, t, s) for p, t, s in inspection_paths if t == "sum"]
    for p, _, shift in sum_paths:
        send_to_discord(DISCORD_WEBHOOK, p,
                        f"**Inspection: sum score  shift={shift:+d}**")

    # PCA inspection batch
    pca_paths = [(p, t, s) for p, t, s in inspection_paths if t == "pca"]
    for p, _, shift in pca_paths:
        send_to_discord(DISCORD_WEBHOOK, p,
                        f"**Inspection: PCA score  shift={shift:+d}**")

    print("\nDone.")


if __name__ == "__main__":
    main()
