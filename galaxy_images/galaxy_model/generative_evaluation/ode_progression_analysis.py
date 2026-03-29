'''ODE progression visualisation.

For N_HSC and N_LEGACY examples, run the flow-matching ODE from noise
to final sample, capturing snapshots at evenly-spaced steps.

Each plot has 5 rows:
  - Rows 0-3: individual channels (g, r, i, z)
  - Row 4:    row-scaled RGB composite (first 3 channels as RGB,
              scaled per-channel to the target's min/max)

Columns: Target | step 0 (noise) | step k | … | step N (final)
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

import torch
torch.backends.cuda.preferred_blas_library("hipblas")
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

from galaxy_images.galaxy_model.double_train_fm_neighbors import ConditionalFlowMatchingModule
from galaxy_images.galaxy_model.neighbors import NeighborsDataset, collate_neighbors

MODEL_CHECKPOINT = '/work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model/outputs/neighbors_all_attn/checkpoints/best-epoch=228-step=87000.ckpt'
HDF5_PATH = '/work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5'

N_HSC = 16
N_LEGACY = 16
MAX_DATASET_SIZE = 1024
BATCH_SIZE = 8
NUM_INTEGRATION_STEPS = 500
N_SNAPSHOTS = 11
CHANNEL_NAMES = ['g', 'r', 'i', 'z']


# ============= ODE SAMPLING =============

@torch.no_grad()
def sample_with_snapshots(model, cond_samegal, cond_sameins, masks,
                          num_steps, n_snapshots=6):
    """
    Euler integration capturing snapshots at evenly spaced steps.
    Returns list of (step, image_np) for sample index 0.
    """
    device = cond_samegal.device
    B = cond_samegal.shape[0]

    if masks is None:
        k = cond_sameins.shape[1]
        masks = torch.ones((B, k), device=device, dtype=torch.bool)

    x = torch.randn(B, model.in_channels, model.image_size, model.image_size,
                     device=device)
    dt = 1.0 / num_steps

    step_interval = max(1, num_steps // (n_snapshots - 1))
    snap_steps = set(range(0, num_steps + 1, step_interval))
    snap_steps.add(num_steps)

    snapshots = [(0, x[0].cpu().numpy().copy())]

    for i in range(num_steps):
        t = torch.full((B,), i * dt, device=device)
        v = model(x, t, cond_samegal, cond_sameins, masks)
        x = x + v * dt

        step = i + 1
        if step in snap_steps:
            snapshots.append((step, x[0].cpu().numpy().copy()))

    return snapshots


# ============= VISUALISATION =============

def _row_scale_rgb(chw, vmin, vmax):
    """Scale (C,H,W) numpy array channels 0-2 to (H,W,3) in [0,1] using given per-channel bounds."""
    rgb = chw[:3].copy()
    for c in range(3):
        denom = vmax[c] - vmin[c]
        if denom > 0:
            rgb[c] = (rgb[c] - vmin[c]) / denom
        else:
            rgb[c] = 0.0
    return np.clip(rgb.transpose(1, 2, 0), 0, 1)


def _percentile_scale(img_2d):
    """Scale a single-channel 2D array to [0,1] using 2nd/98th percentiles."""
    lo, hi = np.percentile(img_2d, (2, 98))
    if hi > lo:
        return np.clip((img_2d - lo) / (hi - lo), 0, 1)
    return np.clip(img_2d, 0, 1)


def plot_ode_progression(snapshots, target_np, survey, output_dir, sample_idx=0):
    """
    5-row plot:
      rows 0-3 = channels g/r/i/z  (grayscale, percentile-scaled per image)
      row 4    = row-scaled RGB composite (scaled to target min/max)
    Columns: Target | snapshot_0 | … | snapshot_N
    """
    n_channels = min(target_np.shape[0], len(CHANNEL_NAMES))
    n_snaps = len(snapshots)
    n_cols = 1 + n_snaps
    n_rows = n_channels + 1  # 4 channels + 1 RGB row

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.2 * n_cols, 2.2 * n_rows))

    # per-channel min/max from target for the RGB row
    vmin = np.array([target_np[c].min() for c in range(3)])
    vmax = np.array([target_np[c].max() for c in range(3)])

    # --- Target column ---
    for ch in range(n_channels):
        axes[ch, 0].imshow(_percentile_scale(target_np[ch]),
                           cmap='gray', origin='lower')
        axes[ch, 0].set_ylabel(CHANNEL_NAMES[ch], fontsize=10, fontweight='bold')
        axes[ch, 0].set_xticks([])
        axes[ch, 0].set_yticks([])
    axes[0, 0].set_title('Target', fontsize=9)

    # RGB row – target
    axes[n_channels, 0].imshow(_row_scale_rgb(target_np, vmin, vmax),
                                origin='lower')
    axes[n_channels, 0].set_ylabel('RGB', fontsize=10, fontweight='bold')
    axes[n_channels, 0].set_xticks([])
    axes[n_channels, 0].set_yticks([])

    # --- Snapshot columns ---
    for j, (step, snap_np) in enumerate(snapshots):
        col = j + 1
        for ch in range(n_channels):
            axes[ch, col].imshow(_percentile_scale(snap_np[ch]),
                                 cmap='gray', origin='lower')
            axes[ch, col].set_xticks([])
            axes[ch, col].set_yticks([])
        if True:
            axes[0, col].set_title(f'step {step}', fontsize=9)

        # RGB row
        axes[n_channels, col].imshow(
            _row_scale_rgb(snap_np, vmin, vmax), origin='lower')
        axes[n_channels, col].set_xticks([])
        axes[n_channels, col].set_yticks([])

    fig.suptitle(
        f'ODE progression — Sample {sample_idx} — Target: {survey.upper()}',
        fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = output_dir / f'ode_progression_{sample_idx}_{survey}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path.name}")


# ============= MAIN =============

def main():
    output_dir = _script_dir / "figures_ode"
    output_dir.mkdir(exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Loading model from checkpoint …")
    model = ConditionalFlowMatchingModule.load_from_checkpoint(
        MODEL_CHECKPOINT, map_location='cpu')
    model.eval()
    model.to(device)
    print("Model loaded.")

    print(f"Opening dataset (lazy h5) from {HDF5_PATH} …")
    dataset = NeighborsDataset(hdf5_path=HDF5_PATH, crop_size=48, max_neighbors=15)
    n_use = min(MAX_DATASET_SIZE, len(dataset))
    subset = Subset(dataset, list(range(n_use)))
    loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, collate_fn=collate_neighbors)
    print(f"Will iterate over {n_use} samples (dataset total: {len(dataset)})")

    n_hsc_done = 0
    n_leg_done = 0
    sample_idx = 0

    print(f"\nGenerating ODE progressions ({NUM_INTEGRATION_STEPS} steps, "
          f"{N_SNAPSHOTS} snapshots) …")
    print(f"  Target: {N_HSC} HSC + {N_LEGACY} Legacy examples")

    for batch in loader:
        if n_hsc_done >= N_HSC and n_leg_done >= N_LEGACY:
            break

        targets_b, samegals_b, neighbors_b, masks_b, meta_b = batch
        targets_b = targets_b.to(device)
        samegals_b = samegals_b.to(device)
        neighbors_b = neighbors_b.to(device)
        masks_b = masks_b.to(device)

        for i in range(targets_b.shape[0]):
            survey = meta_b[i]['anchor_survey']
            if survey == 'hsc' and n_hsc_done >= N_HSC:
                continue
            if survey == 'legacy' and n_leg_done >= N_LEGACY:
                continue

            snapshots = sample_with_snapshots(
                model,
                samegals_b[i:i+1], neighbors_b[i:i+1],
                masks=masks_b[i:i+1],
                num_steps=NUM_INTEGRATION_STEPS,
                n_snapshots=N_SNAPSHOTS,
            )
            plot_ode_progression(
                snapshots,
                target_np=targets_b[i].cpu().numpy(),
                survey=survey,
                output_dir=output_dir,
                sample_idx=sample_idx,
            )
            sample_idx += 1
            if survey == 'hsc':
                n_hsc_done += 1
            else:
                n_leg_done += 1

        print(f"  Progress: {n_hsc_done}/{N_HSC} HSC, "
              f"{n_leg_done}/{N_LEGACY} Legacy")

    print(f"\nDone! {sample_idx} figures saved to {output_dir}")


if __name__ == '__main__':
    main()
