"""
Sample the 8 galaxies with the largest and 8 with the smallest average Legacy PSF Size
from the MMU rows in neighbours_v2.h5, and plot their Legacy images in one figure.

Layout (2 rows × 8 cols):
  Row 0 — Legacy images for 8 LARGEST  avg Legacy PSF
  Row 1 — Legacy images for 8 SMALLEST avg Legacy PSF

Outputs:
  psf_size_extremes.png

Run:
  python plot_psf_extremes.py [--n-examples 8000]
"""

import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ============= CONFIGURATION =============

NEIGHBORS_HDF5 = "/work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5"
DEFAULT_NUM_EXAMPLES = 8000
CROP_SIZE = 48          # center-crop from native 160×160
OUTPUT_DIR = Path(__file__).resolve().parent

LEGACY_PSF_BANDS = ["legacy_PSFSIZE_G", "legacy_PSFSIZE_I", "legacy_PSFSIZE_R", "legacy_PSFSIZE_Z"]


# ============= HELPERS =============

def center_crop(img: np.ndarray, size: int) -> np.ndarray:
    """img: (C, H, W) numpy array. Returns (C, size, size)."""
    _, h, w = img.shape
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return img[:, y0:y0 + size, x0:x0 + size]


def tensor_to_rgb(img: np.ndarray, percentile_clip: float = 99.5) -> np.ndarray:
    """img: (C, H, W) numpy float array. Returns (H, W, 3) RGB in [0, 1]."""
    rgb = img[:3].copy()               # (3, H, W) — use first 3 channels
    rgb = np.transpose(rgb, (1, 2, 0)) # (H, W, 3)

    for i in range(3):
        p_lo = np.percentile(rgb[:, :, i], 100 - percentile_clip)
        p_hi = np.percentile(rgb[:, :, i], percentile_clip)
        rgb[:, :, i] = np.clip(rgb[:, :, i], p_lo, p_hi)
        ch = rgb[:, :, i]
        lo, hi = ch.min(), ch.max()
        if hi > lo:
            rgb[:, :, i] = (ch - lo) / (hi - lo)
        else:
            rgb[:, :, i] = 0.0
    return rgb


# ============= DATA LOADING =============

def load_psf_and_indexes(hdf5_path: str, n_examples: int):
    """Load avg Legacy PSF size and the raw HDF5 row indexes for MMU examples."""
    with h5py.File(hdf5_path, "r") as f:
        sources = f["source_type"][:]
        indexes_mmu = np.where(sources == 0)[0]
        n_use = min(n_examples, len(indexes_mmu))
        indexes_mmu = indexes_mmu[:n_use]

        psf_bands = []
        for band_key in LEGACY_PSF_BANDS:
            if band_key not in f:
                raise KeyError(f"Key '{band_key}' not found in HDF5 — check band names.")
            vals = np.asarray(f[band_key][indexes_mmu], dtype=np.float32)
            psf_bands.append(vals)

    psf_stack = np.stack(psf_bands, axis=0)          # (4, N)
    psf_avg = np.nanmean(psf_stack, axis=0)           # (N,)
    print(f"Loaded {n_use} MMU examples. Legacy PSF avg: "
          f"min={np.nanmin(psf_avg):.3f}, max={np.nanmax(psf_avg):.3f} arcsec")
    return indexes_mmu, psf_avg


def load_legacy_images_for_raw_rows(hdf5_path: str, raw_row_indexes: list) -> list:
    """Load and center-crop Legacy images for given raw HDF5 row indices."""
    leg_imgs = []
    with h5py.File(hdf5_path, "r") as f:
        for raw_idx in raw_row_indexes:
            leg = np.asarray(f["images_legacy"][raw_idx], dtype=np.float32)
            leg_imgs.append(center_crop(leg, CROP_SIZE))
    return leg_imgs


# ============= PLOTTING =============

def make_plot(largest_leg, largest_psf, smallest_leg, smallest_psf, output_dir: Path):
    n = 8
    fig, axes = plt.subplots(2, n, figsize=(n * 2.5, 2 * 2.8))

    groups = [
        (largest_leg, largest_psf, "Largest PSF"),
        (smallest_leg, smallest_psf, "Smallest PSF"),
    ]

    for row_idx, (imgs, psf_vals, row_label) in enumerate(groups):
        for col_idx in range(n):
            ax = axes[row_idx, col_idx]
            rgb = tensor_to_rgb(imgs[col_idx])
            ax.imshow(rgb)
            ax.set_axis_off()
            ax.set_title(f"PSF={psf_vals[col_idx]:.2f}\"", fontsize=9)
        # Row label on the left
        ax0 = axes[row_idx, 0]
        ax0.set_axis_on()
        ax0.set_xticks([])
        ax0.set_yticks([])
        for spine in ax0.spines.values():
            spine.set_visible(False)
        ax0.set_ylabel(row_label, fontsize=11, fontweight="bold", labelpad=6)

    fig.suptitle(
        "Legacy images at Legacy PSF size extremes (avg across g/i/r/z bands)",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    out_path = output_dir / "psf_size_extremes.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ============= MAIN =============

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-examples", type=int, default=DEFAULT_NUM_EXAMPLES)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--hdf5-path", type=str, default=NEIGHBORS_HDF5)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load PSF metadata
    indexes_mmu, psf_avg = load_psf_and_indexes(args.hdf5_path, args.n_examples)

    # 2. Identify 8 largest / 8 smallest (ignoring NaN)
    valid_mask = np.isfinite(psf_avg)
    valid_dataset_idxs = np.where(valid_mask)[0]

    sorted_valid = valid_dataset_idxs[np.argsort(psf_avg[valid_dataset_idxs])]
    smallest_dataset_idxs = sorted_valid[:8]
    largest_dataset_idxs = sorted_valid[-8:][::-1]   # descending order

    print("\n8 LARGEST avg Legacy PSF:")
    for di in largest_dataset_idxs:
        print(f"  dataset_idx={di}, raw_row={indexes_mmu[di]}, psf={psf_avg[di]:.3f} arcsec")

    print("\n8 SMALLEST avg Legacy PSF:")
    for di in smallest_dataset_idxs:
        print(f"  dataset_idx={di}, raw_row={indexes_mmu[di]}, psf={psf_avg[di]:.3f} arcsec")

    # 3. Load images
    largest_raw_rows = [int(indexes_mmu[di]) for di in largest_dataset_idxs]
    smallest_raw_rows = [int(indexes_mmu[di]) for di in smallest_dataset_idxs]

    print("\nLoading images for 8 largest PSF examples...")
    largest_leg = load_legacy_images_for_raw_rows(args.hdf5_path, largest_raw_rows)
    print("Loading images for 8 smallest PSF examples...")
    smallest_leg = load_legacy_images_for_raw_rows(args.hdf5_path, smallest_raw_rows)

    largest_psf_vals = [psf_avg[di] for di in largest_dataset_idxs]
    smallest_psf_vals = [psf_avg[di] for di in smallest_dataset_idxs]

    # 4. Plot
    print("\nGenerating figure...")
    make_plot(largest_leg, largest_psf_vals, smallest_leg, smallest_psf_vals, args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
