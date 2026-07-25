"""Noise / PSD metrics for the shuffle-conditioning ablation.

Copied (not imported) from:
  - latent_traversal/compute_hsc_noise_metrics.py  (σ estimators + CORNER)
  - generative_evaluation/power_autocorrelation_analysis_v3_heldout.py  (PSD/autocorr)

Importing those modules pulls ROCm/torch/discord side effects; these helpers are
pure numpy/scipy.
"""
from __future__ import annotations

import numpy as np
from numpy import fft
from scipy import signal

CORNER = 8  # corner patch side (px) on 48×48
HIGH_K_FRAC = 0.4  # high-frequency cutoff as fraction of Nyquist


# ---------------------------------------------------------------------------
# σ estimators  (flux: (N, B, 48, 48) → (N, B))
# ---------------------------------------------------------------------------

def corner_sky_rms(flux: np.ndarray) -> np.ndarray:
    """Std after 5–95 percentile clip on corner patches."""
    c = CORNER
    parts = [
        flux[..., :c, :c],
        flux[..., :c, -c:],
        flux[..., -c:, :c],
        flux[..., -c:, -c:],
    ]
    pix = np.concatenate([p.reshape(*flux.shape[:2], -1) for p in parts], axis=-1)
    lo = np.percentile(pix, 5, axis=-1, keepdims=True)
    hi = np.percentile(pix, 95, axis=-1, keepdims=True)
    keep = (pix >= lo) & (pix <= hi)
    masked = np.where(keep, pix, np.nan)
    return np.nanstd(masked, axis=-1)


def adjacent_diff_sigma(flux: np.ndarray) -> np.ndarray:
    """MAD-based σ from neighbor-pixel differences."""
    dx = (flux[..., :, 1:] - flux[..., :, :-1]).reshape(*flux.shape[:2], -1)
    dy = (flux[..., 1:, :] - flux[..., :-1, :]).reshape(*flux.shape[:2], -1)
    diffs = np.concatenate([dx, dy], axis=-1)
    med = np.median(diffs, axis=-1, keepdims=True)
    mad = np.median(np.abs(diffs - med), axis=-1)
    return mad * 1.4826 / np.sqrt(2.0)


def masked_mad_sigma(flux: np.ndarray, n_iter: int = 3, k_sigma: float = 3.0) -> np.ndarray:
    """Iteratively mask pixels above median+k·σ_MAD; final σ_MAD."""
    pix = flux.reshape(*flux.shape[:2], -1).astype(np.float32, copy=True)
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


def band_mean_sigma(sigma_nb: np.ndarray) -> np.ndarray:
    """(N, B) → (N,) mean over bands, ignoring NaNs."""
    return np.nanmean(sigma_nb, axis=-1)


# ---------------------------------------------------------------------------
# PSD / autocorrelation  (per-band 2D float64 input)
# ---------------------------------------------------------------------------

def radial_average(image_2d: np.ndarray):
    center = np.array(image_2d.shape) // 2
    y, x = np.ogrid[: image_2d.shape[0], : image_2d.shape[1]]
    r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    r_int = r.astype(int)

    max_r = int(min(center) * np.sqrt(2)) - 1
    mask = r_int <= max_r

    counts = np.bincount(r_int[mask].ravel(), minlength=max_r + 1)
    sums = np.bincount(r_int[mask].ravel(), image_2d[mask].ravel(), minlength=max_r + 1)

    valid = counts > 0
    radial_mean = np.zeros_like(sums)
    radial_mean[valid] = sums[valid] / counts[valid]

    radii = np.arange(len(radial_mean))
    return radii, radial_mean


def apply_window(image_2d: np.ndarray) -> np.ndarray:
    window = np.hanning(image_2d.shape[0])[:, None] * np.hanning(image_2d.shape[1])[None, :]
    return image_2d * window


def compute_radial_power_spectrum(image_2d: np.ndarray):
    image_windowed = apply_window(image_2d)
    fft_img = fft.fft2(image_windowed)
    fft_shifted = fft.fftshift(fft_img)
    power_2d = np.abs(fft_shifted) ** 2
    freqs, radial_power = radial_average(power_2d)
    return freqs, radial_power


def compute_radial_correlation_direct(image_2d: np.ndarray):
    autocorr_2d = signal.correlate2d(image_2d, image_2d, mode="same", boundary="fill")
    autocorr_2d /= autocorr_2d.max()
    lags, radial_corr = radial_average(autocorr_2d)
    return lags, radial_corr


def high_k_power_from_psd(freqs: np.ndarray, radial_power: np.ndarray, nyquist_bin: float | None = None) -> float:
    """Sum radial power for k > HIGH_K_FRAC · Nyquist.

    ``freqs`` from ``radial_average`` are integer radial bin indices (pixels from
    center after fftshift). Nyquist sits near ``image_size / 2`` bins.
    """
    if nyquist_bin is None:
        # Conservative default for 48×48: Nyquist ≈ 24 bins from center.
        nyquist_bin = 24.0
    cutoff = HIGH_K_FRAC * nyquist_bin
    keep = freqs > cutoff
    if not np.any(keep):
        return float("nan")
    return float(radial_power[keep].sum())


def mean_high_k_power(images: np.ndarray) -> np.ndarray:
    """images: (N, B, H, W) → (N,) mean high-k power over bands."""
    n, b, h, w = images.shape
    out = np.empty(n, dtype=np.float64)
    nyquist_bin = min(h, w) / 2.0
    for i in range(n):
        band_vals = []
        for j in range(b):
            freqs, power = compute_radial_power_spectrum(images[i, j].astype(np.float64))
            band_vals.append(high_k_power_from_psd(freqs, power, nyquist_bin=nyquist_bin))
        out[i] = float(np.nanmean(band_vals))
    return out


def mean_radial_psd(images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average radial PSD over N·B images. Returns (freqs, mean_power)."""
    n, b, _, _ = images.shape
    acc = None
    freqs = None
    count = 0
    for i in range(n):
        for j in range(b):
            f, p = compute_radial_power_spectrum(images[i, j].astype(np.float64))
            if acc is None:
                freqs = f
                acc = np.zeros_like(p, dtype=np.float64)
            acc += p
            count += 1
    if acc is None or count == 0:
        return np.array([]), np.array([])
    return freqs, acc / count
