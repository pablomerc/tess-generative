"""
Plot distributions of metadata properties from the neighbors dataset.

Loads metadata directly from neighbours_v2.h5 (no model needed), derives
HSC PSF FWHM from sdssshape moments, averages multi-band properties across
channels, and produces histogram plots.

Outputs:
  - property_distributions.png  — grid of histograms for all averaged properties
  - psf_distributions.png       — dedicated HSC vs Legacy PSF size comparison

Run:
  python plot_distributions.py [--n-examples 8000]
"""

import argparse
import re
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ============= CONFIGURATION =============

NEIGHBORS_HDF5 = "/work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5"
DEFAULT_NUM_EXAMPLES = 8000
OUTPUT_DIR = Path(__file__).resolve().parent

NEIGHBORS_SIMPLE_EXCLUDE_KEYS = frozenset({
    "source_type",
    "images_hsc",
    "images_legacy",
    "neighbor_idx_hsc",
    "neighbor_idx_legacy",
    "neighbor_dist_hsc",
    "neighbor_dist_legacy",
})

SKIP_META_KEYS = frozenset({"idx", "index_mmu"})

AVERAGE_PATTERNS = {
    r"^legacy_GALDEPTH_": "legacy_GALDEPTH",
    r"^legacy_NOBS_": "legacy_NOBS",
    r"^legacy_PSFSIZE_": "legacy_PSFSIZE",
    r"^legacy_PSFDEPTH_": "legacy_PSFDEPTH",
    r"^hsc_.*_variance_value$": "hsc_variance_value",
    r"^hsc_.*_psf_fwhm$": "hsc_psf_fwhm",
}

DISPLAY_LABELS = {
    "hsc_variance_value": "HSC Variance",
    "hsc_psf_fwhm": "HSC PSF Size (FWHM, arcsec)",
    "legacy_GALDEPTH": "Legacy Galaxy Depth",
    "legacy_NOBS": "Legacy # Observations",
    "legacy_PSFSIZE": "Legacy PSF Size (arcsec)",
    "legacy_PSFDEPTH": "Legacy PSF Depth",
    "EBV": "E(B-V)",
}

BAND_DISPLAY = {
    "legacy_PSFSIZE_G": "Legacy g",
    "legacy_PSFSIZE_I": "Legacy i",
    "legacy_PSFSIZE_R": "Legacy r",
    "legacy_PSFSIZE_Z": "Legacy z",
    "hsc_g_psf_fwhm": "HSC g",
    "hsc_i_psf_fwhm": "HSC i",
    "hsc_r_psf_fwhm": "HSC r",
    "hsc_z_psf_fwhm": "HSC z",
}


# ============= DATA LOADING =============

def _metadata_value_from_h5(val):
    if hasattr(val, "shape") and val.shape == ():
        return val.item()
    if np.isscalar(val):
        if isinstance(val, (np.floating, np.float32, np.float64)):
            return float(val)
        if isinstance(val, (np.integer, np.int64, np.int32)):
            return int(val)
        if isinstance(val, np.bool_):
            return bool(val)
    if hasattr(val, "tolist"):
        return val.tolist()
    return val


def compute_hsc_psf_fwhm(shape11, shape22):
    pixel_scale_hsc = 0.168
    return 2.355 * np.sqrt((shape11 + shape22) / 2) * pixel_scale_hsc


def load_metadata(hdf5_path, n_examples):
    """Load numeric metadata columns from neighbours_v2.h5 for the first n_examples MMU rows."""
    with h5py.File(hdf5_path, "r") as f:
        sources = f["source_type"][:]
        indexes_mmu = np.where(sources == 0)[0]
        n_use = min(n_examples, len(indexes_mmu))
        indexes_mmu = indexes_mmu[:n_use]

        meta_keys = [k for k in f.keys() if k not in NEIGHBORS_SIMPLE_EXCLUDE_KEYS]

        param_arrays = {}
        valid_params = []

        for key in meta_keys:
            if key in SKIP_META_KEYS:
                continue
            try:
                vals = f[key][indexes_mmu]
                arr = np.asarray(vals, dtype=np.float64)
                if arr.ndim != 1:
                    continue
                finite_frac = np.isfinite(arr).mean()
                if finite_frac < 0.5:
                    continue
                valid_params.append(key)
                param_arrays[key] = arr.astype(np.float32)
            except (TypeError, ValueError):
                continue

    print(f"Loaded {n_use} examples, {len(valid_params)} numeric metadata columns")

    # Derive HSC PSF FWHM from shape moments
    for band in ("g", "i", "r", "z"):
        k11 = f"hsc_{band}_sdssshape_psf_shape11"
        k22 = f"hsc_{band}_sdssshape_psf_shape22"
        if k11 in param_arrays and k22 in param_arrays:
            fwhm = compute_hsc_psf_fwhm(param_arrays[k11], param_arrays[k22])
            name = f"hsc_{band}_psf_fwhm"
            param_arrays[name] = fwhm.astype(np.float32)
            valid_params.append(name)
            print(f"  Derived: {name}")

    return valid_params, param_arrays, n_use


def average_multiband_properties(valid_params, param_arrays):
    """Average multi-band properties across channels using AVERAGE_PATTERNS."""
    consumed = set()
    averaged_params = []
    averaged_arrays = {}
    per_band_groups = {}

    for pattern, avg_name in AVERAGE_PATTERNS.items():
        matching = [p for p in valid_params if re.search(pattern, p)]
        if not matching:
            continue
        stacked = np.stack([param_arrays[p] for p in matching], axis=0)
        averaged_arrays[avg_name] = np.nanmean(stacked, axis=0).astype(np.float32)
        averaged_params.append(avg_name)
        per_band_groups[avg_name] = {p: param_arrays[p] for p in matching}
        consumed.update(matching)

    for p in valid_params:
        if p not in consumed:
            averaged_params.append(p)
            averaged_arrays[p] = param_arrays[p]

    return averaged_params, averaged_arrays, per_band_groups


# ============= PLOTTING =============

def plot_all_distributions(averaged_params, averaged_arrays, n_examples, output_dir):
    """Grid of histograms, one per averaged property."""
    n = len(averaged_params)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
    axes = np.atleast_2d(axes)

    for i, param in enumerate(averaged_params):
        row, col = divmod(i, ncols)
        ax = axes[row, col]
        vals = averaged_arrays[param]
        finite = vals[np.isfinite(vals)]
        label = DISPLAY_LABELS.get(param, param)

        lo, hi = np.percentile(finite, [1, 99])
        clipped = finite[(finite >= lo) & (finite <= hi)]
        ax.hist(clipped, bins=60, color="#5b8abf", edgecolor="white", linewidth=0.3, alpha=0.85)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Value", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.tick_params(labelsize=8)

        median = np.nanmedian(finite)
        ax.axvline(median, color="#c0392b", ls="--", lw=1.2, label=f"median={median:.3g}")
        ax.legend(fontsize=7, loc="upper right")

    for j in range(i + 1, nrows * ncols):
        row, col = divmod(j, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle(f"Property Distributions (N={n_examples})", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = output_dir / "property_distributions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_psf_distributions(per_band_groups, averaged_arrays, n_examples, output_dir):
    """Dedicated PSF size comparison: per-band overlaid + channel-averaged."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # --- Panel 1: Legacy PSF SIZE per band ---
    ax = axes[0]
    legacy_bands = per_band_groups.get("legacy_PSFSIZE", {})
    colors_leg = ["#2980b9", "#27ae60", "#e67e22", "#8e44ad"]
    for (key, vals), c in zip(sorted(legacy_bands.items()), colors_leg):
        finite = vals[np.isfinite(vals)]
        lo, hi = np.percentile(finite, [1, 99])
        clipped = finite[(finite >= lo) & (finite <= hi)]
        label = BAND_DISPLAY.get(key, key)
        ax.hist(clipped, bins=50, alpha=0.55, color=c, edgecolor="white", linewidth=0.3, label=label)
    ax.set_title("Legacy PSF Size per Band", fontsize=12, fontweight="bold")
    ax.set_xlabel("PSF Size (arcsec)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=9)

    # --- Panel 2: HSC PSF FWHM per band ---
    ax = axes[1]
    hsc_bands = per_band_groups.get("hsc_psf_fwhm", {})
    colors_hsc = ["#c0392b", "#d4ac0d", "#1abc9c", "#2c3e50"]
    for (key, vals), c in zip(sorted(hsc_bands.items()), colors_hsc):
        finite = vals[np.isfinite(vals)]
        lo, hi = np.percentile(finite, [1, 99])
        clipped = finite[(finite >= lo) & (finite <= hi)]
        label = BAND_DISPLAY.get(key, key)
        ax.hist(clipped, bins=50, alpha=0.55, color=c, edgecolor="white", linewidth=0.3, label=label)
    ax.set_title("HSC PSF FWHM per Band", fontsize=12, fontweight="bold")
    ax.set_xlabel("PSF FWHM (arcsec)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=9)

    # --- Panel 3: Averaged Legacy vs HSC ---
    ax = axes[2]
    for key, color, label in [
        ("legacy_PSFSIZE", "#3498db", "Legacy PSF Size (avg)"),
        ("hsc_psf_fwhm", "#e74c3c", "HSC PSF FWHM (avg)"),
    ]:
        if key not in averaged_arrays:
            continue
        vals = averaged_arrays[key]
        finite = vals[np.isfinite(vals)]
        lo, hi = np.percentile(finite, [1, 99])
        clipped = finite[(finite >= lo) & (finite <= hi)]
        ax.hist(clipped, bins=50, alpha=0.55, color=color, edgecolor="white", linewidth=0.3, label=label)
    ax.set_title("Legacy vs HSC PSF Size (Channel-Averaged)", fontsize=12, fontweight="bold")
    ax.set_xlabel("PSF Size (arcsec)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=9)

    fig.suptitle(f"PSF Size Distributions (N={n_examples})", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = output_dir / "psf_distributions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ============= MAIN =============

def main():
    parser = argparse.ArgumentParser(description="Plot property distributions from neighbors dataset")
    parser.add_argument("--n-examples", type=int, default=DEFAULT_NUM_EXAMPLES)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--hdf5-path", type=str, default=NEIGHBORS_HDF5)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"HDF5: {args.hdf5_path}")
    print(f"Num examples: {args.n_examples}")
    print(f"Output dir: {output_dir}")

    valid_params, param_arrays, n_use = load_metadata(args.hdf5_path, args.n_examples)

    avg_params, avg_arrays, per_band_groups = average_multiband_properties(valid_params, param_arrays)
    print(f"\nAfter channel averaging: {len(avg_params)} properties:")
    for p in avg_params:
        label = DISPLAY_LABELS.get(p, p)
        arr = avg_arrays[p]
        print(f"  {label:40s} ({p})  range=[{np.nanmin(arr):.4g}, {np.nanmax(arr):.4g}]")

    print("\n--- Figure 1: All property distributions ---")
    plot_all_distributions(avg_params, avg_arrays, n_use, output_dir)

    print("\n--- Figure 2: PSF size distributions ---")
    plot_psf_distributions(per_band_groups, avg_arrays, n_use, output_dir)

    print(f"\nDone! Figures saved to {output_dir}")


if __name__ == "__main__":
    main()
