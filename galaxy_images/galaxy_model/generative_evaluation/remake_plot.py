'''
Reload saved intermediate .npz files from figures_v2/ and regenerate plots.
Useful for tuning visualization without re-running the model.

Usage:
    python remake_plot.py                     # remake all samples
    python remake_plot.py --samples 0 2 4     # remake specific sample indices
'''

import sys
import argparse
from pathlib import Path

_script_dir = Path(__file__).resolve().parent

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt

CHANNEL_NAMES = ['g', 'r', 'i', 'z']

DATA_DIR = _script_dir / "figures_v2"
OUT_DIR  = _script_dir / "figures_v2"


# ============= SHARED UTILITIES (copied from v2) =============

from scipy import fft
from scipy import signal


def radial_average(image_2d):
    center = np.array(image_2d.shape) // 2
    y, x = np.ogrid[:image_2d.shape[0], :image_2d.shape[1]]
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r_int = r.astype(int)

    max_r = int(min(center) * np.sqrt(2)) - 1
    mask = r_int <= max_r

    counts = np.bincount(r_int[mask].ravel(), minlength=max_r + 1)
    sums = np.bincount(r_int[mask].ravel(), image_2d[mask].ravel(), minlength=max_r + 1)

    valid = counts > 0
    radial_mean = np.zeros_like(sums)
    radial_mean[valid] = sums[valid] / counts[valid]

    return np.arange(len(radial_mean)), radial_mean


def apply_window(image_2d):
    window = np.hanning(image_2d.shape[0])[:, None] * np.hanning(image_2d.shape[1])[None, :]
    return image_2d * window


def compute_radial_power_spectrum(image_2d):
    image_windowed = apply_window(image_2d)
    fft_img = fft.fft2(image_windowed)
    fft_shifted = fft.fftshift(fft_img)
    power_2d = np.abs(fft_shifted)**2
    return radial_average(power_2d)


def compute_radial_correlation_direct(image_2d):
    autocorr_2d = signal.correlate2d(image_2d, image_2d, mode='same', boundary='fill')
    autocorr_2d /= autocorr_2d.max()
    return radial_average(autocorr_2d)


# ============= PLOT (edit this to tune visualization) =============

def plot_individual_sample_v2(target_np, recons_np, survey, sample_idx, output_dir):
    """
    4-column (channels) x 2-row (power spectrum, autocorrelation) plot.
    Shows target, mean reconstruction, and +/- 1 std band.
    """
    n_channels = min(target_np.shape[0], len(CHANNEL_NAMES))
    fig, axes = plt.subplots(2, n_channels, figsize=(4 * n_channels, 7))

    for ch in range(n_channels):
        tch = target_np[ch].astype(np.float64)

        # --- power spectrum ---
        freq_t, ps_t = compute_radial_power_spectrum(tch)

        all_ps_r = []
        for r in range(recons_np.shape[0]):
            rch = recons_np[r, ch].astype(np.float64)
            _, ps_r = compute_radial_power_spectrum(rch)
            all_ps_r.append(ps_r)

        min_len = min(len(ps_t), min(len(a) for a in all_ps_r))
        ps_t = ps_t[:min_len]
        freq_t = freq_t[:min_len]
        all_ps_r = np.array([a[:min_len] for a in all_ps_r])
        ps_r_mean = all_ps_r.mean(0)
        ps_r_std = all_ps_r.std(0)

        ax = axes[0, ch]
        ax.semilogy(freq_t, ps_t, label='Target', color='steelblue', lw=1.5)
        ax.semilogy(freq_t, ps_r_mean, label='Mean recon', color='tomato', ls='--', lw=1.5)
        ax.fill_between(
            freq_t,
            np.maximum(ps_r_mean - ps_r_std, 1e-10),
            ps_r_mean + ps_r_std,
            color='tomato', alpha=0.25, label='±1 std'
        )
        ax.set_title(f'{CHANNEL_NAMES[ch]} band')
        ax.set_xlabel('Spatial frequency')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which='both')
        if ch == 0:
            ax.set_ylabel('Power Spectrum')

        # --- autocorrelation ---
        lag_t, ac_t = compute_radial_correlation_direct(tch)

        all_ac_r = []
        for r in range(recons_np.shape[0]):
            rch = recons_np[r, ch].astype(np.float64)
            _, ac_r = compute_radial_correlation_direct(rch)
            all_ac_r.append(ac_r)

        min_len = min(len(ac_t), min(len(a) for a in all_ac_r))
        ac_t = ac_t[:min_len]
        lag_t = lag_t[:min_len]
        all_ac_r = np.array([a[:min_len] for a in all_ac_r])
        ac_r_mean = all_ac_r.mean(0)
        ac_r_std = all_ac_r.std(0)

        ax = axes[1, ch]
        ax.plot(lag_t, ac_t, label='Target', color='steelblue', lw=1.5)
        ax.plot(lag_t, ac_r_mean, label='Mean recon', color='tomato', ls='--', lw=1.5)
        ax.fill_between(
            lag_t,
            ac_r_mean - ac_r_std,
            ac_r_mean + ac_r_std,
            color='tomato', alpha=0.25, label='±1 std'
        )
        ax.set_xlabel('Lag (pixels)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if ch == 0:
            ax.set_ylabel('Autocorrelation')

    fig.suptitle(
        f'Sample {sample_idx} — Target survey: {survey.upper()} '
        f'(mean ± 1 std of {recons_np.shape[0]} reconstructions)',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    path = output_dir / f'individual_sample_{sample_idx}_{survey}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path.name}")


# ============= MAIN =============

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', nargs='*', type=int, default=None,
                        help='Sample indices to remake. Default: all found in figures_v2/')
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--out-dir',  type=Path, default=OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(exist_ok=True)

    npz_files = sorted(args.data_dir.glob('sample_*.npz'))
    if not npz_files:
        print(f"No .npz files found in {args.data_dir}")
        sys.exit(1)

    if args.samples is not None:
        npz_files = [f for f in npz_files
                     if any(f.stem.startswith(f'sample_{idx}_') for idx in args.samples)]

    print(f"Remaking {len(npz_files)} plot(s) from {args.data_dir} …")
    for npz_path in npz_files:
        data = np.load(npz_path, allow_pickle=True)
        target_np  = data['target']
        recons_np  = data['recons']
        survey     = str(data['survey'])
        sample_idx = int(data['sample_idx'])
        print(f"  Loaded {npz_path.name} — survey={survey}, sample_idx={sample_idx}")
        plot_individual_sample_v2(target_np, recons_np, survey, sample_idx, args.out_dir)

    print("Done.")


if __name__ == '__main__':
    main()
