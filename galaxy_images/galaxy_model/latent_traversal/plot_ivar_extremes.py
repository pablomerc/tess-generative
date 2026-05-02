"""
Explore legacy_image_ivar as a candidate noise axis for latent traversal.

Produces three figures and posts them to Discord:
  1. ivar_histogram.png      — distribution of mean ivar across MMU examples
  2. ivar_zero_examples.png  — 8 examples with the lowest mean ivar (near-zero)
  3. ivar_psf_density.png    — 2D density: normalized mean ivar vs normalized PSF size

Run:
  python plot_ivar_extremes.py [--n-examples 200000] [--n-show 8] [--output-dir .]
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

# ============= CONFIGURATION =============

NEIGHBORS_HDF5 = "/work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5"
OUTPUT_DIR = Path(__file__).resolve().parent
CROP_SIZE = 48
DISCORD_WEBHOOK = (
    "https://discord.com/api/webhooks/1497979386144493680/"
    "VA-xWhfTWzc-oeC5EvPzyqEk_MW52wZsK2RyLS0egfhHHHhBxrmb9NGawy0rIpfvn3Zo"
)
LEGACY_PSF_BANDS = ["legacy_PSFSIZE_G", "legacy_PSFSIZE_I", "legacy_PSFSIZE_R", "legacy_PSFSIZE_Z"]


# ============= HELPERS =============

def center_crop(img: np.ndarray, size: int) -> np.ndarray:
    """img: (C, H, W). Returns (C, size, size) center crop."""
    _, h, w = img.shape
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return img[:, y0:y0 + size, x0:x0 + size]


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
    with open(file_path, "rb") as f:
        data = f.read()
    resp = requests.post(
        webhook_url,
        data={"content": message} if message else {},
        files={"file": (file_path.name, io.BytesIO(data), "image/png")},
    )
    if resp.status_code in (200, 204):
        print(f"  Sent {file_path.name} to Discord.")
    else:
        print(f"  WARNING: Discord returned {resp.status_code}: {resp.text[:200]}")


# ============= DATA LOADING =============

def load_data(hdf5_path: str, n_examples: int):
    """Return (indexes_mmu, mean_ivar, ivar_all, psf_avg) for the first n_examples MMU rows."""
    with h5py.File(hdf5_path, "r") as f:
        sources = f["source_type"][:]
        indexes_mmu = np.where(sources == 0)[0]
        n_use = min(n_examples, len(indexes_mmu))
        indexes_mmu = indexes_mmu[:n_use]

        ivar = f["legacy_image_ivar"][indexes_mmu]  # (n_use, 4, H, W)

        psf_bands = []
        for key in LEGACY_PSF_BANDS:
            psf_bands.append(np.asarray(f[key][indexes_mmu], dtype=np.float32))

    mean_ivar = ivar.mean(axis=(1, 2, 3))  # (n_use,)
    psf_avg = np.nanmean(np.stack(psf_bands, axis=0), axis=0)  # (n_use,)

    print(
        f"Loaded {n_use} MMU rows.\n"
        f"  mean_ivar: [{mean_ivar.min():.2f}, {mean_ivar.max():.2f}]  "
        f"(zeros: {(mean_ivar == 0).sum()}  <1: {(mean_ivar < 1).sum()})\n"
        f"  psf_avg:   [{np.nanmin(psf_avg):.3f}, {np.nanmax(psf_avg):.3f}] arcsec"
    )
    return indexes_mmu, mean_ivar, ivar, psf_avg


def percentile_filter_normalize(arr: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0):
    """Return mask of values inside [p_lo, p_hi] percentiles and their [0,1] normalization.
    Values outside are dropped (not clipped), so there are no spikes at 0 or 1."""
    lo = np.nanpercentile(arr, p_lo)
    hi = np.nanpercentile(arr, p_hi)
    mask = (arr >= lo) & (arr <= hi)
    normed = (arr[mask] - lo) / (hi - lo + 1e-10)
    return normed, mask, lo, hi


# ============= PLOTS =============

def plot_histogram(mean_ivar: np.ndarray, output_path: Path):
    p1  = np.percentile(mean_ivar, 1)
    p99 = np.percentile(mean_ivar, 99)
    norm_filtered, mask, _, _ = percentile_filter_normalize(mean_ivar)
    n_dropped = (~mask).sum()

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    ax = axes[0]
    ax.hist(mean_ivar, bins=200, color="steelblue", edgecolor="none")
    ax.axvline(p1,  color="tomato", linestyle="--", label=f"p1  = {p1:.1f}")
    ax.axvline(p99, color="orange", linestyle="--", label=f"p99 = {p99:.1f}")
    ax.set_xlabel("Mean ivar (raw)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Raw distribution", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

    ax2 = axes[1]
    p10 = np.percentile(mean_ivar, 10)
    ax2.hist(mean_ivar[mean_ivar < p10], bins=100, color="tomato", edgecolor="none")
    ax2.axvline(p1, color="black", linestyle="--", label=f"p1 = {p1:.1f}")
    ax2.set_xlabel("Mean ivar (raw)", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title(f"Zoom: bottom 10% (< {p10:.1f})", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)

    ax3 = axes[2]
    ax3.hist(norm_filtered, bins=200, color="mediumseagreen", edgecolor="none")
    ax3.set_xlabel("Normalized ivar [0, 1]", fontsize=11)
    ax3.set_ylabel("Count", fontsize=11)
    ax3.set_title(
        f"p1–p99 drop-and-normalize  ({n_dropped:,} examples dropped)",
        fontsize=11, fontweight="bold",
    )

    fig.suptitle("legacy_image_ivar distribution (MMU examples)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_zero_examples(
    indexes_mmu: np.ndarray,
    mean_ivar: np.ndarray,
    ivar_all: np.ndarray,
    hdf5_path: str,
    n_show: int,
    output_path: Path,
):
    """Show n_show examples with the lowest mean ivar: Legacy image + mean ivar map."""
    worst_pos = np.argsort(mean_ivar)[:n_show]  # positions in the n_use slice

    print(f"\nLoading {n_show} lowest-ivar Legacy images ...")
    with h5py.File(hdf5_path, "r") as f:
        images = [
            center_crop(np.asarray(f["images_legacy"][int(indexes_mmu[pos])], dtype=np.float32), CROP_SIZE)
            for pos in worst_pos
        ]

    # 3 rows: Legacy image | mean ivar map (cropped) | per-channel ivar maps
    fig, axes = plt.subplots(2, n_show, figsize=(n_show * 2.2, 5.5))

    for col, pos in enumerate(worst_pos):
        raw_idx = int(indexes_mmu[pos])
        mv = mean_ivar[pos]
        img = images[col]           # (4, 48, 48)
        ivar_map = ivar_all[pos]    # (4, H, W) — full res

        # Row 0: Legacy RGB
        rgb = tensor_to_rgb(img)
        axes[0, col].imshow(rgb)
        axes[0, col].set_title(f"row={raw_idx}\nivar={mv:.1f}", fontsize=7.5)
        axes[0, col].axis("off")

        # Row 1: mean ivar map (across channels, cropped to 48x48)
        ivar_mean_map = center_crop(ivar_map, CROP_SIZE).mean(axis=0)  # (48, 48)
        im = axes[1, col].imshow(ivar_mean_map, cmap="inferno", vmin=0)
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Legacy image", fontsize=9)
    axes[1, 0].set_ylabel("Mean ivar map", fontsize=9)

    fig.suptitle(
        f"{n_show} examples with lowest raw (unclipped) mean legacy_image_ivar — expect ivar ≈ 0",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_2d_density(
    mean_ivar: np.ndarray,
    psf_avg: np.ndarray,
    output_path: Path,
    bins: int = 120,
):
    """2D density with marginal 1D distributions, three normalization variants."""
    valid = np.isfinite(mean_ivar) & np.isfinite(psf_avg)
    ivar_v = mean_ivar[valid]
    psf_v  = psf_avg[valid]

    # Drop outside p1-p99 for each axis, then normalize — no clipping, no spikes
    _, ivar_mask, ivar_lo, ivar_hi = percentile_filter_normalize(ivar_v)
    _, psf_mask,  psf_lo,  psf_hi  = percentile_filter_normalize(psf_v)
    keep = ivar_mask & psf_mask
    n_dropped = (~keep).sum()
    ivar_f = (ivar_v[keep] - ivar_lo) / (ivar_hi - ivar_lo + 1e-10)
    psf_f  = (psf_v[keep]  - psf_lo)  / (psf_hi  - psf_lo  + 1e-10)

    ivar_label = f"ivar  [0={ivar_lo:.0f}, 1={ivar_hi:.0f}]"
    psf_label  = f"PSF  [0={psf_lo:.2f}\", 1={psf_hi:.2f}\"]"
    configs = [
        (ivar_f, psf_f, f"p1–p99 drop-and-normalize (linear)\n{n_dropped:,} dropped", False),
        (ivar_f, psf_f, f"p1–p99 drop-and-normalize (log count)\n{n_dropped:,} dropped", True),
    ]

    # Each panel: 2×2 inner grid — [top hist | colorbar] / [2D hist | right hist]
    n_panels = len(configs)
    fig = plt.figure(figsize=(n_panels * 9, 8))
    outer = fig.add_gridspec(1, n_panels, wspace=0.38)

    for col_idx, (ix, px, title, log_scale) in enumerate(configs):
        inner = outer[col_idx].subgridspec(
            2, 2, height_ratios=[1, 4], width_ratios=[4, 1], hspace=0.05, wspace=0.05
        )
        ax_top   = fig.add_subplot(inner[0, 0])
        ax_main  = fig.add_subplot(inner[1, 0])
        ax_right = fig.add_subplot(inner[1, 1])
        ax_cb    = fig.add_subplot(inner[0, 1])
        ax_cb.set_visible(False)

        norm = plt.matplotlib.colors.LogNorm() if log_scale else None
        _, _, _, im = ax_main.hist2d(ix, px, bins=bins, cmap="viridis", norm=norm)
        fig.colorbar(im, ax=ax_cb, fraction=0.8, pad=0.05).set_label("Count", fontsize=8)
        ax_main.set_xlabel(ivar_label, fontsize=9)
        ax_main.set_ylabel(psf_label, fontsize=9)

        ax_top.hist(ix, bins=bins, color="steelblue", edgecolor="none")
        ax_top.set_xlim(ax_main.get_xlim())
        ax_top.set_xticks([])
        ax_top.set_ylabel("Count", fontsize=8)
        ax_top.tick_params(axis="y", labelsize=7)
        ax_top.set_title(title, fontsize=10, fontweight="bold", pad=4)

        ax_right.hist(px, bins=bins, color="tomato", edgecolor="none",
                      orientation="horizontal")
        ax_right.set_ylim(ax_main.get_ylim())
        ax_right.set_yticks([])
        ax_right.set_xlabel("Count", fontsize=8)
        ax_right.tick_params(axis="x", labelsize=7)

    fig.suptitle(
        "2D density: mean legacy_image_ivar vs Legacy PSF size  (+marginals)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


# ============= MAIN =============

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5-path", default=NEIGHBORS_HDF5)
    parser.add_argument("--n-examples", type=int, default=200_000)
    parser.add_argument("--n-show", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Loading data ===")
    indexes_mmu, mean_ivar, ivar_all, psf_avg = load_data(args.hdf5_path, args.n_examples)

    print("\n=== Plotting histogram ===")
    hist_path = args.output_dir / "ivar_histogram.png"
    plot_histogram(mean_ivar, hist_path)

    print("\n=== Plotting lowest-ivar examples ===")
    examples_path = args.output_dir / "ivar_zero_examples.png"
    plot_zero_examples(indexes_mmu, mean_ivar, ivar_all, args.hdf5_path, args.n_show, examples_path)

    print("\n=== Plotting 2D density (ivar vs PSF) ===")
    density_path = args.output_dir / "ivar_psf_density.png"
    plot_2d_density(mean_ivar, psf_avg, density_path)

    print("\n=== Sending to Discord ===")
    send_to_discord(DISCORD_WEBHOOK, hist_path,
                    f"**ivar histogram** (n={args.n_examples:,} MMU examples)")
    send_to_discord(DISCORD_WEBHOOK, examples_path,
                    f"**{args.n_show} lowest mean-ivar examples** — checking for bad data")
    send_to_discord(DISCORD_WEBHOOK, density_path,
                    "**2D density: normalized mean ivar vs normalized PSF size**")

    print("\nDone.")


if __name__ == "__main__":
    main()
