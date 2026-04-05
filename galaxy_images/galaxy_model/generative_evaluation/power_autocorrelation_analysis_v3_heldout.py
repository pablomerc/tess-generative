''' This script will:
-Load a pretrained model
-Load held-out validation samples saved in precomputed neighbors format
-For N_SAMPLES_IND HSC and N_SAMPLES_IND Legacy samples, generate N_RECONSTRUCTIONS
 reconstructions per sample and compute their mean and std
-For each individual sample make a 4-column (channels) x 2-row (power spectrum,
 autocorrelation) plot showing:
   1) the target
   2) the mean reconstruction
   3) +/- 1 std interval of the reconstruction
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
from galaxy_images.galaxy_model.neighbors import NeighborsPrecomputedDataset, simple_collate

MODEL_CHECKPOINT = '/work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model/outputs/neighbors_all_attn/checkpoints/best-epoch=228-step=87000.ckpt'
HELDOUT_H5_PATH = '/work1/jeroenaudenaert/pablomer/data/heldout_validation/neighbors_heldout_val_seed42.h5'

N_SAMPLES_IND = 16
N_RECONSTRUCTIONS = 32
MAX_DATASET_SIZE = 100_000
BATCH_SIZE = 8
NUM_INTEGRATION_STEPS = 250
CHANNEL_NAMES = ['g', 'r', 'i', 'z']


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

    radii = np.arange(len(radial_mean))
    return radii, radial_mean


def apply_window(image_2d):
    window = np.hanning(image_2d.shape[0])[:, None] * np.hanning(image_2d.shape[1])[None, :]
    return image_2d * window


def compute_radial_power_spectrum(image_2d):
    image_windowed = apply_window(image_2d)
    fft_img = fft.fft2(image_windowed)
    fft_shifted = fft.fftshift(fft_img)
    power_2d = np.abs(fft_shifted)**2
    freqs, radial_power = radial_average(power_2d)
    return freqs, radial_power


def compute_radial_correlation_direct(image_2d):
    autocorr_2d = signal.correlate2d(
        image_2d, image_2d,
        mode='same', boundary='fill'
    )
    autocorr_2d /= autocorr_2d.max()
    lags, radial_corr = radial_average(autocorr_2d)
    return lags, radial_corr


@torch.no_grad()
def generate_reconstructions(model, samegal, neighbors, mask, device):
    cond_samegal = samegal.unsqueeze(0).expand(N_RECONSTRUCTIONS, -1, -1, -1).to(device)
    cond_neighbors = neighbors.unsqueeze(0).expand(N_RECONSTRUCTIONS, -1, -1, -1, -1).to(device)
    cond_mask = mask.unsqueeze(0).expand(N_RECONSTRUCTIONS, -1).to(device)

    recons = model.sample(
        cond_samegal, cond_neighbors,
        masks=cond_mask,
        num_steps=NUM_INTEGRATION_STEPS,
    )
    return recons.cpu().numpy()


def plot_individual_sample_v2(target_np, recons_np, survey, sample_idx, output_dir):
    n_channels = min(target_np.shape[0], len(CHANNEL_NAMES))
    fig, axes = plt.subplots(2, n_channels, figsize=(4 * n_channels, 7))

    for ch in range(n_channels):
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


def main():
    output_dir = _script_dir / "figures_v3_heldout"
    output_dir.mkdir(exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Loading model from checkpoint ...")
    model = ConditionalFlowMatchingModule.load_from_checkpoint(
        MODEL_CHECKPOINT, map_location='cpu'
    )
    model.eval()
    model.to(device)
    print("Model loaded.")

    print(f"Opening held-out dataset from {HELDOUT_H5_PATH} ...")
    dataset = NeighborsPrecomputedDataset(HELDOUT_H5_PATH)
    n_use = min(MAX_DATASET_SIZE, len(dataset))
    subset = Subset(dataset, list(range(n_use)))
    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=simple_collate,
    )
    print(f"Using {n_use} held-out samples")

    collected_hsc = []
    collected_legacy = []

    print(f"\nCollecting {N_SAMPLES_IND} HSC and {N_SAMPLES_IND} Legacy samples ...")
    for batch in loader:
        if len(collected_hsc) >= N_SAMPLES_IND and len(collected_legacy) >= N_SAMPLES_IND:
            break

        targets, samegals, padded_neighbors, neighbor_masks, metadata = batch

        for i in range(targets.shape[0]):
            survey = metadata[i]['anchor_survey']
            if survey == 'hsc' and len(collected_hsc) < N_SAMPLES_IND:
                collected_hsc.append((
                    targets[i].numpy(),
                    samegals[i],
                    padded_neighbors[i],
                    neighbor_masks[i],
                ))
            elif survey != 'hsc' and len(collected_legacy) < N_SAMPLES_IND:
                collected_legacy.append((
                    targets[i].numpy(),
                    samegals[i],
                    padded_neighbors[i],
                    neighbor_masks[i],
                ))

        print(f"  Collected: {len(collected_hsc)} HSC, {len(collected_legacy)} Legacy")

    print(f"\nGenerating {N_RECONSTRUCTIONS} reconstructions per sample "
          f"({NUM_INTEGRATION_STEPS} integration steps) ...")

    sample_idx = 0
    for survey_tag, collected in [('hsc', collected_hsc), ('legacy', collected_legacy)]:
        for t_np, samegal, neighbors, mask in collected:
            print(f"  Sample {sample_idx} ({survey_tag.upper()}) ...")
            recons_np = generate_reconstructions(model, samegal, neighbors, mask, device)

            npz_path = output_dir / f'sample_{sample_idx}_{survey_tag}.npz'
            np.savez(npz_path, target=t_np, recons=recons_np, survey=survey_tag, sample_idx=sample_idx)
            print(f"  Saved intermediate data to {npz_path.name}")

            plot_individual_sample_v2(t_np, recons_np, survey_tag, sample_idx, output_dir)
            sample_idx += 1

    print(f"\nDone! Figures saved to {output_dir}")


if __name__ == '__main__':
    main()
