''' This script will:
-Load a pretrained model
-Load N image pairs
-Generate model's prediction/reconstruction for a target image given its pairs
-Calculate the power spectrum of both the target image and the reconstructed image
-Calculate the autocorrelation of both the target image and the reconstructed image
-For the N_SAMPLES_IND, for each one, make a plot that has 4 columns, one per channel, and 2
rows with the value of the power spectrum (row 1) and the autocorrelation (row 2) for both the target and the reconstruction, where each channel is one column
-Clearly state if the target was HSC or legacy

-For N_SAMPLES_AVG, separate by those that used HSC as target and those that used Legacy as target
-For each of those two make a 4x2 plot that shows the avg power spectrum and autocorrelation per channel for both the target and the reconstruction, with uncertainties (+- 1 std)


'''

import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import os
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")

import matplotlib
matplotlib.use("Agg")

from scipy import fft
from scipy import signal
import torch
torch.backends.cuda.preferred_blas_library("hipblas")
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

from galaxy_images.galaxy_model.double_train_fm_neighbors import ConditionalFlowMatchingModule
from galaxy_images.galaxy_model.neighbors import NeighborsDataset, collate_neighbors

MODEL_CHECKPOINT = '/work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model/outputs/neighbors_all_attn/checkpoints/best-epoch=228-step=87000.ckpt'
HDF5_PATH = '/work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5'

N_SAMPLES_IND = 16
N_SAMPLES_AVG = 256
N_ODE_PLOTS = 16
MAX_DATASET_SIZE = 1024
BATCH_SIZE = 8
NUM_INTEGRATION_STEPS = 250
CHANNEL_NAMES = ['g', 'r', 'i', 'z']


# ============= SHARED UTILITIES =============

def radial_average(image_2d):
    """
    Compute radial average of a 2D array centered at the middle.

    Returns:
        radii: 1D array of unique integer radii
        radial_mean: 1D array of mean values at each radius
    """
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

    radii = np.arange(len(radial_mean))
    return radii, radial_mean


def apply_window(image_2d):
    """Apply 2D Hann window to reduce edge artifacts."""
    window = np.hanning(image_2d.shape[0])[:, None] * np.hanning(image_2d.shape[1])[None, :]
    return image_2d * window


# ============= POWER SPECTRUM =============

def compute_radial_power_spectrum(image_2d):
    """
    Compute radial power spectrum with windowing.

    Args:
        image_2d: (H, W) numpy array

    Returns:
        freqs: 1D array of spatial frequencies (cycles/image)
        radial_power: 1D array of power at each frequency
    """
    image_windowed = apply_window(image_2d)
    fft_img = fft.fft2(image_windowed)
    fft_shifted = fft.fftshift(fft_img)
    power_2d = np.abs(fft_shifted)**2
    freqs, radial_power = radial_average(power_2d)
    return freqs, radial_power


# ============== AUTO CORRELATION ==============

def compute_radial_correlation_direct(image_2d):
    """
    Compute radial autocorrelation via direct spatial correlation.
    """
    autocorr_2d = signal.correlate2d(
        image_2d, image_2d,
        mode='same', boundary='fill'
    )
    autocorr_2d /= autocorr_2d.max()
    lags, radial_corr = radial_average(autocorr_2d)
    return lags, radial_corr


# ============= PLOTTING =============

def plot_individual_sample(target_np, recon_np, survey, sample_idx, output_dir):
    """
    4-column (channels) x 2-row (power spectrum, autocorrelation) plot for one sample.
    Each subplot overlays target vs reconstruction.
    """
    n_channels = min(target_np.shape[0], len(CHANNEL_NAMES))
    fig, axes = plt.subplots(2, n_channels, figsize=(4 * n_channels, 7))

    for ch in range(n_channels):
        tch = target_np[ch].astype(np.float64)
        rch = recon_np[ch].astype(np.float64)

        freq_t, ps_t = compute_radial_power_spectrum(tch)
        freq_r, ps_r = compute_radial_power_spectrum(rch)
        axes[0, ch].semilogy(freq_t, ps_t, label='Target', color='steelblue', lw=1.5)
        axes[0, ch].semilogy(freq_r, ps_r, label='Reconstruction', color='tomato', ls='--', lw=1.5)
        axes[0, ch].set_title(f'{CHANNEL_NAMES[ch]} band')
        axes[0, ch].set_xlabel('Spatial frequency')
        axes[0, ch].legend(fontsize=8)
        axes[0, ch].grid(True, alpha=0.3, which='both')
        if ch == 0:
            axes[0, ch].set_ylabel('Power Spectrum')

        lag_t, ac_t = compute_radial_correlation_direct(tch)
        lag_r, ac_r = compute_radial_correlation_direct(rch)
        axes[1, ch].plot(lag_t, ac_t, label='Target', color='steelblue', lw=1.5)
        axes[1, ch].plot(lag_r, ac_r, label='Reconstruction', color='tomato', ls='--', lw=1.5)
        axes[1, ch].set_xlabel('Lag (pixels)')
        axes[1, ch].legend(fontsize=8)
        axes[1, ch].grid(True, alpha=0.3)
        if ch == 0:
            axes[1, ch].set_ylabel('Autocorrelation')

    fig.suptitle(f'Sample {sample_idx} — Target survey: {survey.upper()}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = output_dir / f'individual_sample_{sample_idx}_{survey}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path.name}")


def plot_averaged(targets_list, recons_list, survey_name, output_dir):
    """
    4-column (channels) x 2-row (power spectrum, autocorrelation) averaged plot
    with mean +/- 1 std shading.
    """
    n_channels = min(targets_list[0].shape[0], len(CHANNEL_NAMES))
    n_samples = len(targets_list)
    fig, axes = plt.subplots(2, n_channels, figsize=(4 * n_channels, 7))

    for ch in range(n_channels):
        all_ps_t, all_ps_r = [], []
        all_ac_t, all_ac_r = [], []
        freq_ref, lag_ref = None, None

        for i in range(n_samples):
            tch = targets_list[i][ch].astype(np.float64)
            rch = recons_list[i][ch].astype(np.float64)

            freq, ps = compute_radial_power_spectrum(tch)
            all_ps_t.append(ps)
            freq_ref = freq

            freq, ps = compute_radial_power_spectrum(rch)
            all_ps_r.append(ps)

            lag, ac = compute_radial_correlation_direct(tch)
            all_ac_t.append(ac)
            lag_ref = lag

            lag, ac = compute_radial_correlation_direct(rch)
            all_ac_r.append(ac)

        min_len_ps = min(len(a) for a in all_ps_t + all_ps_r)
        all_ps_t = np.array([a[:min_len_ps] for a in all_ps_t])
        all_ps_r = np.array([a[:min_len_ps] for a in all_ps_r])
        freq_ref = freq_ref[:min_len_ps]

        min_len_ac = min(len(a) for a in all_ac_t + all_ac_r)
        all_ac_t = np.array([a[:min_len_ac] for a in all_ac_t])
        all_ac_r = np.array([a[:min_len_ac] for a in all_ac_r])
        lag_ref = lag_ref[:min_len_ac]

        ps_t_mean, ps_t_std = all_ps_t.mean(0), all_ps_t.std(0)
        ps_r_mean, ps_r_std = all_ps_r.mean(0), all_ps_r.std(0)

        ax = axes[0, ch]
        ax.semilogy(freq_ref, ps_t_mean, label='Target', color='steelblue', lw=1.5)
        ax.fill_between(freq_ref,
                         np.maximum(ps_t_mean - ps_t_std, 1e-10),
                         ps_t_mean + ps_t_std, color='steelblue', alpha=0.2)
        ax.semilogy(freq_ref, ps_r_mean, label='Reconstruction', color='tomato', ls='--', lw=1.5)
        ax.fill_between(freq_ref,
                         np.maximum(ps_r_mean - ps_r_std, 1e-10),
                         ps_r_mean + ps_r_std, color='tomato', alpha=0.2)
        ax.set_title(f'{CHANNEL_NAMES[ch]} band')
        ax.set_xlabel('Spatial frequency')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which='both')
        if ch == 0:
            ax.set_ylabel('Power Spectrum')

        ac_t_mean, ac_t_std = all_ac_t.mean(0), all_ac_t.std(0)
        ac_r_mean, ac_r_std = all_ac_r.mean(0), all_ac_r.std(0)

        ax = axes[1, ch]
        ax.plot(lag_ref, ac_t_mean, label='Target', color='steelblue', lw=1.5)
        ax.fill_between(lag_ref, ac_t_mean - ac_t_std, ac_t_mean + ac_t_std,
                         color='steelblue', alpha=0.2)
        ax.plot(lag_ref, ac_r_mean, label='Reconstruction', color='tomato', ls='--', lw=1.5)
        ax.fill_between(lag_ref, ac_r_mean - ac_r_std, ac_r_mean + ac_r_std,
                         color='tomato', alpha=0.2)
        ax.set_xlabel('Lag (pixels)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if ch == 0:
            ax.set_ylabel('Autocorrelation')

    fig.suptitle(f'Averaged — Target survey: {survey_name.upper()} (N={n_samples})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = output_dir / f'averaged_{survey_name}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path.name}")


# ============= ODE PROGRESSION =============

@torch.no_grad()
def sample_with_snapshots(model, cond_image_samegal, cond_image_sameins,
                          masks, num_steps, n_snapshots=6):
    """
    Euler integration that captures snapshots at evenly spaced steps.
    Returns a list of (step_number, image_tensor) tuples for sample index 0.
    """
    device = cond_image_samegal.device
    num_samples = cond_image_samegal.shape[0]

    if masks is None:
        B, k = cond_image_sameins.shape[:2]
        masks = torch.ones((B, k), device=device, dtype=torch.bool)

    x = torch.randn(
        num_samples, model.in_channels, model.image_size, model.image_size,
        device=device,
    )
    dt = 1.0 / num_steps

    step_interval = max(1, num_steps // (n_snapshots - 1))
    snapshot_steps = set(range(0, num_steps + 1, step_interval))
    snapshot_steps.add(num_steps)

    snapshots = [(0, x[0].cpu().numpy().copy())]

    for i in range(num_steps):
        t = torch.full((num_samples,), i * dt, device=device)
        velocity = model(x, t, cond_image_samegal, cond_image_sameins, masks)
        x = x + velocity * dt

        step = i + 1
        if step in snapshot_steps:
            snapshots.append((step, x[0].cpu().numpy().copy()))

    return snapshots


def plot_ode_progression(snapshots, target_np, survey, output_dir, sample_idx=0):
    """
    Plot the ODE integration progression: one row per channel, one column
    per snapshot step.  First column shows the target for reference.
    """
    n_channels = min(target_np.shape[0], len(CHANNEL_NAMES))
    n_snaps = len(snapshots)
    n_cols = 1 + n_snaps  # target column + snapshot columns

    fig, axes = plt.subplots(n_channels, n_cols, figsize=(2.2 * n_cols, 2.2 * n_channels))

    def _to_display(img_2d):
        lo, hi = np.percentile(img_2d, (2, 98))
        if hi > lo:
            return np.clip((img_2d - lo) / (hi - lo), 0, 1)
        return np.clip(img_2d, 0, 1)

    for ch in range(n_channels):
        axes[ch, 0].imshow(_to_display(target_np[ch]), cmap='gray', origin='lower')
        axes[ch, 0].set_title('Target' if ch == 0 else '', fontsize=9)
        axes[ch, 0].set_ylabel(CHANNEL_NAMES[ch], fontsize=10, fontweight='bold')
        axes[ch, 0].set_xticks([])
        axes[ch, 0].set_yticks([])

        for j, (step, snap_np) in enumerate(snapshots):
            axes[ch, j + 1].imshow(_to_display(snap_np[ch]), cmap='gray', origin='lower')
            if ch == 0:
                axes[ch, j + 1].set_title(f'step {step}', fontsize=9)
            axes[ch, j + 1].set_xticks([])
            axes[ch, j + 1].set_yticks([])

    fig.suptitle(f'ODE progression — Sample {sample_idx} — Target survey: {survey.upper()}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = output_dir / f'ode_progression_{sample_idx}_{survey}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path.name}")


# ============= MAIN =============

def main():
    output_dir = _script_dir / "figures"
    output_dir.mkdir(exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print(f"Loading model from checkpoint …")
    model = ConditionalFlowMatchingModule.load_from_checkpoint(
        MODEL_CHECKPOINT, map_location='cpu'
    )
    model.eval()
    model.to(device)
    print("Model loaded.")

    print(f"Opening dataset (lazy h5) from {HDF5_PATH} …")
    dataset = NeighborsDataset(
        hdf5_path=HDF5_PATH,
        crop_size=48,
        max_neighbors=15,
    )
    n_use = min(MAX_DATASET_SIZE, len(dataset))
    subset = Subset(dataset, list(range(n_use)))
    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_neighbors,
    )
    print(f"Will iterate over {n_use} samples (dataset total: {len(dataset)})")

    total_needed = max(N_SAMPLES_IND, N_SAMPLES_AVG)
    results_hsc = []
    results_legacy = []

    print(f"\nGenerating reconstructions ({NUM_INTEGRATION_STEPS} integration steps) …")
    for batch_idx, batch in enumerate(loader):
        targets, samegals, padded_neighbors, neighbor_masks, metadata = batch
        targets = targets.to(device)
        samegals = samegals.to(device)
        padded_neighbors = padded_neighbors.to(device)
        neighbor_masks = neighbor_masks.to(device)

        with torch.no_grad():
            reconstructions = model.sample(
                samegals, padded_neighbors,
                masks=neighbor_masks,
                num_steps=NUM_INTEGRATION_STEPS,
            )

        for i in range(targets.shape[0]):
            survey = metadata[i]['anchor_survey']
            t_np = targets[i].cpu().numpy()
            r_np = reconstructions[i].cpu().numpy()
            if survey == 'hsc':
                results_hsc.append((t_np, r_np))
            else:
                results_legacy.append((t_np, r_np))

        n_hsc, n_leg = len(results_hsc), len(results_legacy)
        print(f"  Batch {batch_idx + 1}: {n_hsc} HSC, {n_leg} Legacy collected")

        if n_hsc >= total_needed and n_leg >= total_needed:
            break

    # ---- Individual plots ----
    print(f"\nMaking {N_SAMPLES_IND} individual plots …")
    all_individual = (
        [(t, r, 'hsc') for t, r in results_hsc] +
        [(t, r, 'legacy') for t, r in results_legacy]
    )
    for idx in range(min(N_SAMPLES_IND, len(all_individual))):
        t_np, r_np, survey = all_individual[idx]
        plot_individual_sample(t_np, r_np, survey, idx, output_dir)

    # ---- Averaged plots (one per survey) ----
    print(f"\nMaking averaged plots (N_SAMPLES_AVG={N_SAMPLES_AVG}) …")
    for survey_name, results in [('hsc', results_hsc), ('legacy', results_legacy)]:
        n = min(N_SAMPLES_AVG, len(results))
        if n == 0:
            print(f"  No {survey_name.upper()} target samples, skipping.")
            continue
        targets_arr = [r[0] for r in results[:n]]
        recons_arr = [r[1] for r in results[:n]]
        plot_averaged(targets_arr, recons_arr, survey_name, output_dir)

    # ---- ODE progression plots (N_ODE_PLOTS examples) ----
    print(f"\nMaking ODE progression plots (N_ODE_PLOTS={N_ODE_PLOTS}) …")
    ode_collected = 0
    for batch in loader:
        if ode_collected >= N_ODE_PLOTS:
            break
        targets_b, samegals_b, neighbors_b, masks_b, meta_b = batch
        targets_b = targets_b.to(device)
        samegals_b = samegals_b.to(device)
        neighbors_b = neighbors_b.to(device)
        masks_b = masks_b.to(device)

        n_in_batch = min(targets_b.shape[0], N_ODE_PLOTS - ode_collected)
        for i in range(n_in_batch):
            snapshots = sample_with_snapshots(
                model,
                samegals_b[i:i+1],
                neighbors_b[i:i+1],
                masks=masks_b[i:i+1],
                num_steps=NUM_INTEGRATION_STEPS,
                n_snapshots=6,
            )
            plot_ode_progression(
                snapshots,
                target_np=targets_b[i].cpu().numpy(),
                survey=meta_b[i]['anchor_survey'],
                output_dir=output_dir,
                sample_idx=ode_collected,
            )
            ode_collected += 1

    print(f"\nDone! Figures saved to {output_dir}")


if __name__ == '__main__':
    main()
