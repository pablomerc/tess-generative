"""Load a list of checkpoints,
for each one calculate the MSE of reconstruction
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from data import HSCLegacyTripletDataset, custom_collate_fn
from torch.utils.data import DataLoader, TensorDataset
from double_train_fm import ConditionalFlowMatchingModule


n_samples = 32
n_batches = 10  # Number of batches to evaluate over
checkpoints = [
    '/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy-flow-matching/w6hjbkct/checkpoints/latest-step=step=68000.ckpt', # z_dim = 8
    '/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy-flow-matching/27z357da/checkpoints/latest-step=step=71000.ckpt', # z_dim = 32
    '/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy_images/galaxy_model/galaxy-flow-matching/r2cvid3f/checkpoints/epoch=201-step=75000.ckpt', # z_dim = 64
    '/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy_images/galaxy_model/galaxy-flow-matching/s39qr0v8/checkpoints/epoch=201-step=75000.ckpt', # z_dim = 128
    '/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy_images/galaxy_model/galaxy-flow-matching/33mo9r3n/checkpoints/epoch=201-step=75000.ckpt', # z_dim = 512
]

figure_path = '/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy_images/galaxy_model/figures/mse_study.png'
z_dim_list = [8, 32, 64, 128, 512]
epochs = [190, 190, 201, 201, 201]


def set_device():
    # Determine device: try to find a working GPU, fallback to CPU
    device = torch.device('cpu')  # Default to CPU for safe loading
    if torch.cuda.is_available():
        # Try to find a working GPU by testing each one
        for gpu_id in range(torch.cuda.device_count()):
            try:
                # Test if GPU is accessible
                test_tensor = torch.tensor([1.0], device=f'cuda:{gpu_id}')
                del test_tensor
                torch.cuda.empty_cache()
                device = torch.device(f'cuda:{gpu_id}')
                print(f"Using GPU {gpu_id}")
                break
            except RuntimeError:
                print(f"GPU {gpu_id} is not available, trying next...")
                continue
        if device.type == 'cpu':
            print("No working GPU found, using CPU")
    return device


device = set_device()

# Start timing for the whole study
study_start_time = time.time()

start_idx = 95_000
num_ims_load = 1024

val_dataset = HSCLegacyTripletDataset(
    hdf5_path='/data/vision/billf/scratch/pablomer/legacysurvey_hsc/preprocessed_hsc_legacy_48x48_all.h5',
    idx_list=list(range(start_idx, start_idx + num_ims_load)),
    deterministic_anchor_survey=True,  # Make validation batches consistent
)


val_loader = DataLoader(
    val_dataset,
    batch_size=n_samples,
    # batch_sampler=val_batch_sampler,  # important: use batch_sampler, not batch_size/shuffle
    shuffle=False,  # Set to False for consistent batches across checkpoints
    num_workers=4,
    pin_memory=True,
    collate_fn=custom_collate_fn,  # Use same collate function
)


mse_results = []  # Will store (mean_total, std_total, mean_hsc, std_hsc, mean_legacy, std_legacy) tuples for each checkpoint
for i, cp in enumerate(checkpoints):
    print(f'\n{"="*60}')
    print(f'Loading model {i+1} of {len(checkpoints)} from checkpoint {cp}')
    print(f'{"="*60}')

    # Load model once per checkpoint
    model = ConditionalFlowMatchingModule.load_from_checkpoint(cp, map_location='cpu')
    model.eval()
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Loaded model, with total params {total_params:,}')

    # Evaluate over n_batches
    batch_mse_total_list = []
    batch_mse_hsc_list = []
    batch_mse_legacy_list = []
    data_iter = iter(val_loader)
    for batch_idx in range(n_batches):
        try:
            batch = next(data_iter)
        except StopIteration:
            # Reset iterator if we run out of batches
            data_iter = iter(val_loader)
            batch = next(data_iter)

        x_1, cond_image_samegal, cond_image_sameins, metadata = batch
        x_1 = x_1.to(device)
        cond_image_samegal = cond_image_samegal.to(device)
        cond_image_sameins = cond_image_sameins.to(device)

        print(f'  Batch {batch_idx+1}/{n_batches}: Evaluating {x_1.shape[0]} images...', end=' ')
        start = time.perf_counter()
        with torch.no_grad():
            samples = model.sample(cond_image_samegal, cond_image_sameins)
            diff = x_1 - samples
            mse_total = torch.mean(diff**2).item()

            # Extract anchor_survey from metadata and compute separate MSEs
            anchor_surveys = [m['anchor_survey'] for m in metadata]
            hsc_mask = torch.tensor([s == 'hsc' for s in anchor_surveys], device=device)
            legacy_mask = torch.tensor([s == 'legacy' for s in anchor_surveys], device=device)

            # Compute MSE for HSC samples
            if hsc_mask.any():
                diff_hsc = diff[hsc_mask]
                mse_hsc = torch.mean(diff_hsc**2).item()
            else:
                mse_hsc = 0.0

            # Compute MSE for Legacy samples
            if legacy_mask.any():
                diff_legacy = diff[legacy_mask]
                mse_legacy = torch.mean(diff_legacy**2).item()
            else:
                mse_legacy = 0.0

        end = time.perf_counter()
        batch_mse_total_list.append(mse_total)
        batch_mse_hsc_list.append(mse_hsc)
        batch_mse_legacy_list.append(mse_legacy)
        print(f'MSE_total={mse_total:.6f}, MSE_hsc={mse_hsc:.6f}, MSE_legacy={mse_legacy:.6f} ({end-start:.2f}s)')

    # Calculate mean and std for each metric
    mse_total_mean = np.mean(batch_mse_total_list)
    mse_total_std = np.std(batch_mse_total_list, ddof=1)
    mse_hsc_mean = np.mean(batch_mse_hsc_list)
    mse_hsc_std = np.std(batch_mse_hsc_list, ddof=1)
    mse_legacy_mean = np.mean(batch_mse_legacy_list)
    mse_legacy_std = np.std(batch_mse_legacy_list, ddof=1)

    mse_results.append((mse_total_mean, mse_total_std, mse_hsc_mean, mse_hsc_std, mse_legacy_mean, mse_legacy_std))

    print(f'\n  Summary (over {n_batches} batches):')
    print(f'    MSE_total = {mse_total_mean:.6f} ± {mse_total_std:.6f}')
    print(f'    MSE_hsc   = {mse_hsc_mean:.6f} ± {mse_hsc_std:.6f}')
    print(f'    MSE_legacy = {mse_legacy_mean:.6f} ± {mse_legacy_std:.6f}')

    # Clean up model from GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


print(f'\n{"="*60}')
print(f'Final Results (n_samples={n_samples}, n_batches={n_batches})')
print(f'{"="*60}')
print(f'z_dim: {z_dim_list}')
print(f'MSE (mean ± std):')

mean_mse_total_list = []
std_mse_total_list = []
mean_mse_hsc_list = []
std_mse_hsc_list = []
mean_mse_legacy_list = []
std_mse_legacy_list = []

for i, (mean_total, std_total, mean_hsc, std_hsc, mean_legacy, std_legacy) in enumerate(mse_results):
    print(f'  Checkpoint {i+1} (z_dim={z_dim_list[i]}):')
    print(f'    Total:  {mean_total:.6f} ± {std_total:.6f}')
    print(f'    HSC:    {mean_hsc:.6f} ± {std_hsc:.6f}')
    print(f'    Legacy: {mean_legacy:.6f} ± {std_legacy:.6f}')
    mean_mse_total_list.append(mean_total)
    std_mse_total_list.append(std_total)
    mean_mse_hsc_list.append(mean_hsc)
    std_mse_hsc_list.append(std_hsc)
    mean_mse_legacy_list.append(mean_legacy)
    std_mse_legacy_list.append(std_legacy)


# Plot results
x = np.array(z_dim_list)
y_total = np.array(mean_mse_total_list)
e_total = np.array(std_mse_total_list)
y_hsc = np.array(mean_mse_hsc_list)
e_hsc = np.array(std_mse_hsc_list)
y_legacy = np.array(mean_mse_legacy_list)
e_legacy = np.array(std_mse_legacy_list)

plt.figure(figsize=(10, 6))
plt.errorbar(x, y_total, yerr=e_total, fmt='-o', capsize=4, label='Total', linewidth=2, markersize=8)
plt.errorbar(x, y_hsc, yerr=e_hsc, fmt='-s', capsize=4, label='HSC', linewidth=2, markersize=8)
plt.errorbar(x, y_legacy, yerr=e_legacy, fmt='-^', capsize=4, label='Legacy', linewidth=2, markersize=8)
plt.xlabel("z_dim")
plt.ylabel("MSE")
plt.xticks(z_dim_list)  # Set x-axis ticks to correspond to z_dim values
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Create directory if it doesn't exist
os.makedirs(os.path.dirname(figure_path), exist_ok=True)
plt.savefig(figure_path, dpi=150, bbox_inches='tight')
print(f'\nFigure saved to: {figure_path}')
plt.close()  # Close the figure to free memory

# Calculate and print total study time
study_end_time = time.time()
study_total_time = study_end_time - study_start_time
hours = int(study_total_time // 3600)
minutes = int((study_total_time % 3600) // 60)
seconds = int(study_total_time % 60)
print(f'\n{"="*60}')
print(f'Total study time: {hours:02d}:{minutes:02d}:{seconds:02d} ({study_total_time:.2f} seconds)')
print(f'{"="*60}')
