'''
Reload saved intermediate .npz files from figures_v2/ and regenerate plots.
Useful for tuning visualization without re-running the model.

Usage:
    python remake_plot.py                     # remake all samples
    python remake_plot.py --samples 0 2 4     # remake specific sample indices
    python remake_plot.py --combined 15:hsc 30:legacy   # one figure, multiple samples
'''

import sys
import argparse
from pathlib import Path

_script_dir = Path(__file__).resolve().parent

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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


def _target_legend_label(survey: str) -> str:
    s = survey.lower()
    if s == 'hsc':
        return 'Target (HSC)'
    if s == 'legacy':
        return 'Target (Legacy)'
    return f'Target ({survey.upper()})'


def _plot_power_spectrum_ax(ax, target_np, recons_np, ch, *, target_legend_label='Target'):
    tch = target_np[ch].astype(np.float64)
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
    ax.semilogy(freq_t, ps_t, label=target_legend_label, color='steelblue', lw=1.5)
    ax.semilogy(freq_t, ps_r_mean, label='Mean recon', color='tomato', ls='--', lw=1.5)
    ax.fill_between(
        freq_t,
        np.maximum(ps_r_mean - ps_r_std, 1e-10),
        ps_r_mean + ps_r_std,
        color='tomato', alpha=0.25, label='±1 std'
    )
    ax.set_xlabel('Spatial frequency')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')


def _plot_autocorr_ax(ax, target_np, recons_np, ch, *, target_legend_label='Target'):
    tch = target_np[ch].astype(np.float64)
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
    ax.plot(lag_t, ac_t, label=target_legend_label, color='steelblue', lw=1.5)
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


# ============= PLOT (edit this to tune visualization) =============

def plot_individual_sample_v2(target_np, recons_np, survey, sample_idx, output_dir):
    """
    4-column (channels) x 2-row (power spectrum, autocorrelation) plot.
    Shows target, mean reconstruction, and +/- 1 std band.
    """
    n_channels = min(target_np.shape[0], len(CHANNEL_NAMES))
    fig, axes = plt.subplots(2, n_channels, figsize=(4 * n_channels, 7))

    for ch in range(n_channels):
        _plot_power_spectrum_ax(axes[0, ch], target_np, recons_np, ch)
        axes[0, ch].set_title(f'{CHANNEL_NAMES[ch]} band')
        if ch == 0:
            axes[0, ch].set_ylabel('Power Spectrum')

        _plot_autocorr_ax(axes[1, ch], target_np, recons_np, ch)
        if ch == 0:
            axes[1, ch].set_ylabel('Autocorrelation')

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


def plot_combined_samples(entries, output_path: Path):
    """
    Single figure stacking multiple samples vertically. Each sample is two rows
    (power spectrum, autocorrelation) × bands. Survey is indicated in each legend
    as Target (HSC) or Target (Legacy).

    entries: list of dicts with keys: target, recons, survey (str), sample_idx (int, optional)
    """
    if not entries:
        raise ValueError('entries must be non-empty')
    n_samples = len(entries)
    n_channels = min(entries[0]['target'].shape[0], len(CHANNEL_NAMES))
    n_recon = entries[0]['recons'].shape[0]
    # Slightly taller when stacked so hspace does not over-compress panels.
    fig_h = 7 * n_samples + 0.6 * max(0, n_samples - 1)
    # Extra vertical gap between rows so block separators sit in whitespace, not on tick labels.
    fig, axes = plt.subplots(
        2 * n_samples,
        n_channels,
        figsize=(4 * n_channels, fig_h),
        gridspec_kw={'hspace': 0.40, 'wspace': 0.28},
    )

    for i, ent in enumerate(entries):
        target_np = ent['target']
        recons_np = ent['recons']
        survey = str(ent['survey'])
        tgt_lbl = _target_legend_label(survey)
        r0, r1 = 2 * i, 2 * i + 1
        for ch in range(n_channels):
            ax_ps = axes[r0, ch]
            ax_ac = axes[r1, ch]
            _plot_power_spectrum_ax(ax_ps, target_np, recons_np, ch, target_legend_label=tgt_lbl)
            _plot_autocorr_ax(ax_ac, target_np, recons_np, ch, target_legend_label=tgt_lbl)

            if i == 0:
                ax_ps.set_title(f'{CHANNEL_NAMES[ch]} band')
            if ch == 0:
                ax_ps.set_ylabel('Power Spectrum', fontsize=10)
                ax_ac.set_ylabel('Autocorrelation', fontsize=10)

    fig.suptitle(
        f'Target survey panels (mean ± 1 std of {n_recon} reconstructions per target)',
        fontsize=13,
        fontweight='bold',
    )
    # tight_layout() squeezes row gaps; use explicit margins so hspace isn't collapsed.
    fig.subplots_adjust(left=0.07, right=0.99, top=0.91, bottom=0.06, hspace=0.40, wspace=0.28)
    # Light horizontal rules between stacked sample blocks (between autocorr row and next power row)
    if n_samples > 1:
        fig.canvas.draw()
        for k in range(1, n_samples):
            ax_upper = axes[2 * k - 1, 0]  # autocorr row of block above
            ax_lower = axes[2 * k, 0]  # power row of block below
            p_up = ax_upper.get_position()
            p_lo = ax_lower.get_position()
            # Figure coords: y increases upward. Upper panel's bottom edge y0; lower panel's top y1.
            y_top = p_up.y0
            y_bot = p_lo.y1
            span = y_top - y_bot
            # Place the rule in the lower part of the gap so it clears the upper row's x-axis.
            if span > 1e-4:
                frac = 0.28
                y_mid = y_bot + frac * span
            else:
                y_mid = (y_top + y_bot) * 0.5
            x0 = min(axes[2 * k, j].get_position().x0 for j in range(n_channels))
            x1 = max(axes[2 * k, j].get_position().x1 for j in range(n_channels))
            fig.add_artist(
                Line2D(
                    [x0, x1],
                    [y_mid, y_mid],
                    transform=fig.transFigure,
                    color='0.78',
                    lw=1.6,
                    zorder=5,
                )
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {output_path.name}")


def _parse_combined_arg(spec: str):
    """Parse '15:hsc' -> (15, 'hsc')."""
    if ':' not in spec:
        raise argparse.ArgumentTypeError(f"Expected IDX:SURVEY, got {spec!r}")
    a, b = spec.split(':', 1)
    return int(a.strip()), b.strip().lower()


def _load_npz_entry(data_dir: Path, sample_idx: int, survey: str):
    npz_path = data_dir / f'sample_{sample_idx}_{survey}.npz'
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    data = np.load(npz_path, allow_pickle=True)
    return {
        'target': data['target'],
        'recons': data['recons'],
        'survey': str(data['survey']),
        'sample_idx': int(data['sample_idx']),
    }


# ============= MAIN =============

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', nargs='*', type=int, default=None,
                        help='Sample indices to remake. Default: all found in figures_v2/')
    parser.add_argument(
        '--combined',
        nargs='+',
        type=_parse_combined_arg,
        default=None,
        metavar='IDX:SURVEY',
        help='Build one stacked figure, e.g. --combined 15:hsc 30:legacy',
    )
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--out-dir',  type=Path, default=OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(exist_ok=True)

    if args.combined is not None:
        entries = []
        for sample_idx, survey in args.combined:
            print(f"  Loading sample_{sample_idx}_{survey}.npz …")
            entries.append(_load_npz_entry(args.data_dir, sample_idx, survey))
        parts = '_'.join(f"{e['sample_idx']}_{e['survey']}" for e in entries)
        out_path = args.out_dir / f'combined_sample_{parts}.png'
        print(f"Writing combined figure to {out_path.name} …")
        plot_combined_samples(entries, out_path)
        print("Done.")
        return

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
