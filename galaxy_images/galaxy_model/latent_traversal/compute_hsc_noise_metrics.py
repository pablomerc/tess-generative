"""
Compute 6 noise-level metrics for each HSC cutout, in i-band and as a 4-band
average over g/r/i/z. Output: /work1/jeroenaudenaert/pablomer/data/hsc_noise_metrics.h5

Metrics (higher value = noisier in all cases — SNR is stored negated):
  1. corner_sky_rms        — std of sigma-clipped pixels in 8x8 corner patches
  2. outer_annulus_sigma   — median 1/sqrt(ivar) over outer annulus (r>18 px)
  3. adjacent_diff_sigma   — MAD of neighbor-pixel differences * 1.4826 / sqrt(2)
  4. masked_mad_sigma      — MAD after iterative >3·MAD source masking
  5. high_freq_power       — sum |FFT|^2 over k > 0.4·Nyquist
  6. snr_neg               — -(cmodel_flux / cmodel_fluxerr); NaN where flux/err invalid
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from discord_notify import notify as _notify

STATS_HDF5     = Path("/work1/jeroenaudenaert/pablomer/data/hsc_ivar_psf_stats.h5")
IVAR_MAPS_HDF5 = Path("/work1/jeroenaudenaert/pablomer/data/hsc_ivar_maps.h5")
NEIGHBORS_HDF5 = "/work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5"
OUTPUT_DEFAULT = Path("/work1/jeroenaudenaert/pablomer/data/hsc_noise_metrics.h5")

CROP = 48
N_BANDS = 4  # g/r/i/z
I_BAND = 2

CORNER = 8        # corner patch side
OUTER_R = 18.0    # annulus inner radius (px) on 48x48
HIGH_K_FRAC = 0.4 # high-frequency cutoff as fraction of Nyquist


# ============= PRECOMPUTED MASKS =============

def _outer_annulus_mask() -> np.ndarray:
    yy, xx = np.mgrid[:CROP, :CROP]
    cx = cy = (CROP - 1) / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return r > OUTER_R  # (48,48)


def _high_k_mask() -> np.ndarray:
    ky = np.fft.fftfreq(CROP)[:, None]
    kx = np.fft.fftfreq(CROP)[None, :]
    k = np.sqrt(ky ** 2 + kx ** 2)
    return k > HIGH_K_FRAC * 0.5  # Nyquist = 0.5 cycles/px


# ============= METRICS (vectorized over batch + band) =============

def corner_sky_rms(flux: np.ndarray) -> np.ndarray:
    """flux: (N, B, 48, 48) → (N, B). Std after 5–95 percentile clip on corner patches."""
    c = CORNER
    parts = [
        flux[..., :c, :c],   flux[..., :c, -c:],
        flux[..., -c:, :c],  flux[..., -c:, -c:],
    ]
    pix = np.concatenate([p.reshape(*flux.shape[:2], -1) for p in parts], axis=-1)
    lo = np.percentile(pix,  5, axis=-1, keepdims=True)
    hi = np.percentile(pix, 95, axis=-1, keepdims=True)
    keep = (pix >= lo) & (pix <= hi)
    masked = np.where(keep, pix, np.nan)
    return np.nanstd(masked, axis=-1)


def outer_annulus_sigma(ivar: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """ivar: (N, B, 48, 48) → (N, B). Median 1/sqrt(ivar) over annulus pixels (ivar>0)."""
    pix = ivar[..., mask]                          # (N, B, n_pix)
    pix = np.where(pix > 0, pix, np.nan)
    sigma = 1.0 / np.sqrt(pix)
    return np.nanmedian(sigma, axis=-1)


def adjacent_diff_sigma(flux: np.ndarray) -> np.ndarray:
    """flux: (N, B, 48, 48) → (N, B). MAD-based σ from neighbor-pixel differences."""
    dx = (flux[..., :, 1:] - flux[..., :, :-1]).reshape(*flux.shape[:2], -1)
    dy = (flux[..., 1:, :] - flux[..., :-1, :]).reshape(*flux.shape[:2], -1)
    diffs = np.concatenate([dx, dy], axis=-1)
    med = np.median(diffs, axis=-1, keepdims=True)
    mad = np.median(np.abs(diffs - med), axis=-1)
    return mad * 1.4826 / np.sqrt(2.0)


def masked_mad_sigma(flux: np.ndarray, n_iter: int = 3, k_sigma: float = 3.0) -> np.ndarray:
    """flux: (N, B, 48, 48) → (N, B). Iteratively mask pixels above median+k·σ_MAD; final σ_MAD."""
    pix = flux.reshape(*flux.shape[:2], -1).astype(np.float32, copy=True)  # (N, B, P)
    valid = np.ones_like(pix, dtype=bool)
    for _ in range(n_iter):
        masked = np.where(valid, pix, np.nan)
        med = np.nanmedian(masked, axis=-1, keepdims=True)
        mad = np.nanmedian(np.abs(masked - med), axis=-1, keepdims=True)
        sig = mad * 1.4826
        valid = valid & (pix <= med + k_sigma * sig)
    masked = np.where(valid, pix, np.nan)
    med = np.nanmedian(masked, axis=-1, keepdims=True)
    mad = np.nanmedian(np.abs(masked - med), axis=-1)
    return mad * 1.4826


def high_freq_power(flux: np.ndarray, mask_k: np.ndarray) -> np.ndarray:
    """flux: (N, B, 48, 48) → (N, B). Sum |FFT|^2 in outer-k annulus."""
    F = np.fft.fft2(flux, axes=(-2, -1))
    P = (F.real ** 2 + F.imag ** 2)
    return P[..., mask_k].sum(axis=-1)


# ============= SNR =============

def load_cmodel_snr(neighbors_path: str, hdf5_row_idx: np.ndarray) -> np.ndarray:
    """Returns (N, 4) SNR for bands g/r/i/z. NaN where flux<=0 or fluxerr<=0/NaN.
    h5py fancy indexing requires strictly-increasing unique indices, so we read
    by unique-sorted indices then expand back via inverse mapping."""
    bands = ["g", "r", "i", "z"]
    n = len(hdf5_row_idx)
    snr = np.full((n, N_BANDS), np.nan, dtype=np.float64)
    unique_idx, inverse = np.unique(hdf5_row_idx, return_inverse=True)
    unique_idx = unique_idx.astype(np.int64)
    with h5py.File(neighbors_path, "r") as f:
        for b, band in enumerate(bands):
            flux = f[f"hsc_{band}_cmodel_flux"][unique_idx][inverse].astype(np.float64)
            ferr = f[f"hsc_{band}_cmodel_fluxerr"][unique_idx][inverse].astype(np.float64)
            ok = np.isfinite(flux) & np.isfinite(ferr) & (ferr > 0) & (flux > 0)
            snr[ok, b] = flux[ok] / ferr[ok]
    return snr


# ============= MAIN =============

METRIC_NAMES = [
    "corner_sky_rms",
    "outer_annulus_sigma",
    "adjacent_diff_sigma",
    "masked_mad_sigma",
    "high_freq_power",
    "snr_neg",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-path",   type=Path, default=STATS_HDF5)
    parser.add_argument("--ivar-path",    type=Path, default=IVAR_MAPS_HDF5)
    parser.add_argument("--neighbors",    default=NEIGHBORS_HDF5)
    parser.add_argument("--output",       type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--chunk",        type=int, default=8192,
                        help="Chunk size for vectorized metric loop")
    args = parser.parse_args()

    print(f"=== Loading {args.stats_path} ===")
    with h5py.File(args.stats_path, "r") as f:
        hdf5_row_idx = np.array(f["hdf5_row_idx"], dtype=np.int64)
        flux_all = np.array(f["hsc_images"], dtype=np.float32)  # (N,4,48,48)
    n = len(hdf5_row_idx)
    print(f"  N={n:,}  flux shape={flux_all.shape}")

    print(f"=== Loading {args.ivar_path} ===")
    with h5py.File(args.ivar_path, "r") as f:
        ivar_rows = np.array(f["hdf5_row_idx"], dtype=np.int64)
        ivar_all = np.array(f["hsc_ivar_maps"], dtype=np.float32)  # (N,4,48,48)
    if not np.array_equal(ivar_rows, hdf5_row_idx):
        raise RuntimeError("ivar_maps rows misaligned with stats file rows")
    print(f"  ivar shape={ivar_all.shape}  alignment OK")

    print("=== Loading cmodel flux/err from neighbours_v2.h5 ===")
    snr_per_band = load_cmodel_snr(args.neighbors, hdf5_row_idx)  # (N,4) g/r/i/z
    print(f"  SNR finite count per band: {np.isfinite(snr_per_band).sum(axis=0)}")

    annulus_mask = _outer_annulus_mask()
    high_k = _high_k_mask()
    print(f"  annulus pixels: {annulus_mask.sum()}, high-k bins: {high_k.sum()}")

    print("=== Computing metrics ===")
    _notify(f"🧮 noise metrics: computing 6 metrics × 4 bands on {n:,} images")
    results = {name: np.empty((n, N_BANDS), dtype=np.float64) for name in METRIC_NAMES}
    chunk = args.chunk
    n_chunks = (n + chunk - 1) // chunk
    notify_every = max(1, n_chunks // 5)
    for ci, start in enumerate(range(0, n, chunk)):
        end = min(n, start + chunk)
        flux = flux_all[start:end]
        ivar = ivar_all[start:end]

        results["corner_sky_rms"][start:end]      = corner_sky_rms(flux)
        results["outer_annulus_sigma"][start:end] = outer_annulus_sigma(ivar, annulus_mask)
        results["adjacent_diff_sigma"][start:end] = adjacent_diff_sigma(flux)
        results["masked_mad_sigma"][start:end]    = masked_mad_sigma(flux)
        results["high_freq_power"][start:end]     = high_freq_power(flux, high_k)
        if start % (chunk * 4) == 0:
            print(f"  {end:,}/{n:,}")
        if (ci + 1) % notify_every == 0 or end == n:
            _notify(f"⏳ noise metrics: {end:,}/{n:,} images")

    # SNR (already loaded)
    results["snr_neg"] = -snr_per_band  # (N, 4)

    # Build i-band and 4-band-average columns
    print("\n=== Summary stats (i-band) ===")
    summary_iband = {}
    summary_4band = {}
    for name in METRIC_NAMES:
        arr = results[name]                      # (N, 4)
        iband = arr[:, I_BAND]
        avg4 = np.nanmean(arr, axis=1)
        summary_iband[name] = iband
        summary_4band[name] = avg4
        finite = iband[np.isfinite(iband)]
        if finite.size:
            print(
                f"  {name:22s}  iband min={finite.min():.4g}  med={np.median(finite):.4g}  "
                f"max={finite.max():.4g}  nan={np.isnan(iband).sum()}"
            )

    # Pearson correlation matrix among i-band metrics on rows where all are finite
    print("\n=== Pearson correlation (i-band, on rows finite for all 6) ===")
    cols = np.stack([summary_iband[n] for n in METRIC_NAMES], axis=1)  # (N,6)
    valid = np.all(np.isfinite(cols), axis=1)
    corr_text = ""
    if valid.sum() > 100:
        sub = cols[valid]
        c = np.corrcoef(sub.T)
        header = "              " + "  ".join(f"{n[:10]:>10}" for n in METRIC_NAMES)
        print(header)
        lines = [header]
        for i, n in enumerate(METRIC_NAMES):
            line = f"  {n[:12]:12s}  " + "  ".join(f"{c[i,j]:+.3f}     " for j in range(len(METRIC_NAMES)))
            print(line)
            lines.append(line)
        corr_text = "\n".join(lines)
        _notify(f"📊 noise metrics i-band Pearson corr (n={valid.sum():,}):\n```\n{corr_text}\n```")

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as f:
        f.create_dataset("hdf5_row_idx", data=hdf5_row_idx.astype(np.int32))
        for name in METRIC_NAMES:
            f.create_dataset(f"{name}_per_band", data=results[name].astype(np.float32))
            f.create_dataset(f"{name}_iband",    data=summary_iband[name].astype(np.float32))
            f.create_dataset(f"{name}_4band",    data=summary_4band[name].astype(np.float32))
        f.attrs["bands"] = ["g", "r", "i", "z"]
        f.attrs["i_band_index"] = I_BAND
        f.attrs["metric_names"] = METRIC_NAMES
    print(f"\nSaved {args.output}")
    _notify(f"💾 noise metrics: saved {args.output.name}")


if __name__ == "__main__":
    main()
