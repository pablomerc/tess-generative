""" 
Script to load a trained single encoder model and test it 

# The models are saved in paths such as 
/Users/pablomercaderperez/Desktop/tess-generative/plots-galaxy/single_enc_flow_20260104_181816/model_epoch_100.pth
which contain a state dict

So first we need to recreate the model architecture
"""

import os
import sys
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import wandb

current_path=os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_path))
sys.path.insert(0,parent_dir)

from .single_encoder_model import build_model
from . import single_encoder_config as cfg
from .hsc_data_loader import HSC_Legacy_DataLoader_OneHot



def main():

    chkpoint_path = '/Users/pablomercaderperez/Desktop/tess-generative/plots-galaxy/single_enc_flow_20260104_181816/model_epoch_100.pth'
    
    # Auto-detect best available device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"[INFO] Using MPS (Apple Silicon GPU) - should be much faster for sampling!")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"[INFO] Using CUDA GPU")
    else:
        device = torch.device('cpu')
        print(f"[WARNING] Using CPU - sampling will be very slow! Consider using MPS if available.")
    
    model = build_model(device=device)
    hdf5_path = cfg.PREPROCESSED_HDF5_PATH
    save_path = cfg.PLOTS_DIR + 'single_enc_flow_20260104_181816/loading/samples_epoch100.png'
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    checkpoint = torch.load(chkpoint_path, map_location=device)
    
    # Extract model state dict from checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        # If it's already the state dict directly
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)

    model.eval()

    print("Loaded model succesfully")

    print(f"[DEBUG] Initializing data loader (load_to_memory=False, this may take a moment)...")
    data_loader_start = time.time()
    data_loader = HSC_Legacy_DataLoader_OneHot(
    hdf5_path=hdf5_path,
    seed=42,
    load_to_memory=False,
    )
    print(f"[DEBUG] Data loader initialized in {time.time() - data_loader_start:.2f}s")

    print("Visualizing samples")
    samples_hsc = visualize_samples(model, device, data_loader, num_examples=4, num_samples=4, save_path=save_path)
    print("Samples visualized")
    print(samples_hsc.shape)

    print('Now visualizing samples with unconditional "random" mode')
    path_random = save_path[:-4] + '_uncond_rdm.png'
    os.makedirs(os.path.dirname(path_random), exist_ok=True)
    samples_hsc_random = visualize_samples_unconditional(model, device, data_loader, mode='random', num_examples=4, num_samples=4, save_path=path_random)
    print("Samples visualized")

    print('Now visualizing samples with unconditional "zeros" mode')
    path_zeros = save_path[:-4] + '_uncond_zeros.png'
    os.makedirs(os.path.dirname(path_zeros), exist_ok=True)
    samples_hsc_zeros = visualize_samples_unconditional(model, device, data_loader, mode='zeros', num_examples=4, num_samples=4, save_path=path_zeros)
    print("Samples visualized")

    


def print_z_statistics(z, name="z"):
    """
    Print summary statistics for latent variable z (similar to pandas describe).
    
    Args:
        z: torch.Tensor of shape (batch_size, latent_dim)
        name: Name to display for the variable
    """
    z_np = z.cpu().numpy()  # Convert to numpy for easier statistics
    z_flat = z_np.flatten()  # Flatten for overall statistics
    
    print(f"\n{'='*60}")
    print(f"Summary Statistics for {name}")
    print(f"Shape: {z.shape}")
    print(f"{'='*60}")
    
    # Overall statistics (across all dimensions and samples)
    print(f"\nOverall Statistics (all dimensions, all samples):")
    print(f"  Count:     {len(z_flat):,}")
    print(f"  Mean:      {z_flat.mean():.6f}")
    print(f"  Std:       {z_flat.std():.6f}")
    print(f"  Min:       {z_flat.min():.6f}")
    print(f"  25%:       {np.percentile(z_flat, 25):.6f}")
    print(f"  Median:    {np.median(z_flat):.6f}")
    print(f"  75%:       {np.percentile(z_flat, 75):.6f}")
    print(f"  Max:       {z_flat.max():.6f}")
    
    # Per-dimension statistics
    print(f"\nPer-Dimension Statistics (across all samples):")
    print(f"{'Dim':<6} {'Mean':<12} {'Std':<12} {'Min':<12} {'25%':<12} {'Median':<12} {'75%':<12} {'Max':<12}")
    print(f"{'-'*96}")
    
    for dim in range(z_np.shape[1]):
        dim_values = z_np[:, dim]
        print(f"{dim:<6} {dim_values.mean():<12.6f} {dim_values.std():<12.6f} "
              f"{dim_values.min():<12.6f} {np.percentile(dim_values, 25):<12.6f} "
              f"{np.median(dim_values):<12.6f} {np.percentile(dim_values, 75):<12.6f} "
              f"{dim_values.max():<12.6f}")
    
    print(f"{'='*60}\n")


def visualize_samples(model, device, data_loader, num_examples=8, num_samples=8, save_path=None, epoch=None, use_wandb=False, normalize=False):
    '''
    Visualize generated samples for HSC based on Legacy Survey examples.
    '''
    # Convert device string to torch.device if needed
    if isinstance(device, str):
        device = torch.device(device)

    model.eval()  # Set model to evaluation mode
    batch_size = num_examples

    print(f"[DEBUG] Starting visualization at {datetime.now().strftime('%H:%M:%S')}")
    
    with torch.no_grad():  # Disable gradient computation to save memory
        print(f"[DEBUG] Step 1/6: Loading batch from data loader...")
        start_time = time.time()
        hsc_batch, legacy_batch, _, _ = data_loader.get_batch(batch_size=batch_size)
        print(f"[DEBUG] Step 1/6: Loaded batch in {time.time() - start_time:.2f}s")
        
        print(f"[DEBUG] Step 2/6: Moving tensors to device ({device})...")
        start_time = time.time()
        hsc_batch=hsc_batch.to(device)
        legacy_batch=legacy_batch.to(device)
        print(f"[DEBUG] Step 2/6: Moved tensors in {time.time() - start_time:.2f}s")

        if normalize:
            print(f"[DEBUG] Normalizing batches...")
            min_val_hsc = hsc_batch.min()
            max_val_hsc = hsc_batch.max()
            
            hsc_norm = (hsc_batch - min_val_hsc) / ((max_val_hsc - min_val_hsc)+1e-3)
            hsc_flat = hsc_norm.flatten(1)

            min_val_legacy = legacy_batch.min()
            max_val_legacy = legacy_batch.max()
            legacy_batch = (legacy_batch - min_val_legacy) / ((max_val_legacy - min_val_legacy)+1e-3)
        else:
            hsc_norm = hsc_batch
            hsc_flat = hsc_batch.flatten(1)
            
        print(f"[DEBUG] Step 3/6: Encoding legacy batch (shape: {legacy_batch.shape})...")
        start_time = time.time()
        z = model.encode(legacy_batch)
        print(f"[DEBUG] Step 3/6: Encoded in {time.time() - start_time:.2f}s (z shape: {z.shape})")
        
        # Print summary statistics for z
        print_z_statistics(z, name="z (encoded latent)")

        # Clear MPS cache before sampling to free up memory
        if device.type == 'mps':
            torch.mps.empty_cache()

        print(f"[DEBUG] Step 4/6: Sampling {num_samples} samples per example (this may take a while - ODE integration with {cfg.N_INTEGRATION_STEPS} steps)...")
        start_time = time.time()
        samples_hsc = model.sample(device=device, n_samples=num_samples, z=z) # (n_samples, n_examples, C*H*W)
        print(f"[DEBUG] Step 4/6: Sampling completed in {time.time() - start_time:.2f}s")

        # Clear cache again after sampling
        if device.type == 'mps':
            torch.mps.empty_cache()

        print(f"[DEBUG] Step 5/6: Reshaping samples...")
        start_time = time.time()
        samples_hsc = samples_hsc.view(num_samples, num_examples, cfg.NUM_CHANNELS, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)
        print(f"[DEBUG] Step 5/6: Reshaped in {time.time() - start_time:.2f}s")

        # Convert to numpy for saving and visualization
        print(f"[DEBUG] Step 6/6: Converting to numpy...")
        start_time = time.time()
        samples_hsc_np = samples_hsc.cpu().numpy()
        hsc_batch_np = hsc_norm.cpu().numpy()
        legacy_batch_np = legacy_batch.cpu().numpy()
        print(f"[DEBUG] Step 6/6: Converted to numpy in {time.time() - start_time:.2f}s")

    print(f"[DEBUG] Creating visualization plot...")
    start_time = time.time()
    num_rows = num_examples
    num_cols = num_samples + 2  # +2 for target and condition columns

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 2 * num_rows))
    
    # Ensure axes is 2D array
    if num_rows == 1:
        axes = axes.reshape(1, num_cols)


    #TODO: Check if this is what we want to do or not
    def to_rgb(channels_np):
        """Convert 4-channel (g, r, i, z) to RGB using first 3 channels with per-channel normalization."""
        rgb = np.stack([
            channels_np[0],  # g -> R
            channels_np[1],  # r -> G
            channels_np[2],  # i -> B
        ], axis=-1)  # (H, W, 3)

        for c in range(3):
            ch = rgb[:, :, c]
            ch_min, ch_max = ch.min(), ch.max()
            if ch_max > ch_min:
                rgb[:, :, c] = (ch - ch_min) / (ch_max - ch_min)
            else:
                rgb[:, :, c] = 0.0
        return rgb

    
    for j in range(num_examples):
        rgb_gt = to_rgb(hsc_batch_np[j])
        axes[j,0].imshow(rgb_gt, vmin=0, vmax=1)
        axes[j,0].set_title(f'Target (HSC)')

        rgb_cond = to_rgb(legacy_batch_np[j])
        axes[j,1].imshow(rgb_cond, vmin=0, vmax=1)
        axes[j,1].set_title(f'Cond. (Legacy)')

        for i in range(num_samples):
            rgb_hsc = to_rgb(samples_hsc_np[i,j])
            axes[j, i+2].imshow(rgb_hsc, vmin=0, vmax=1)
            axes[j, i+2].set_title(f'Sample {i+1}')
            axes[j, i+2].axis('off')
    plt.suptitle(
        f'Generated Samples (HSC) - Epoch {epoch}'
        if epoch is not None
        else 'Generated Samples (HSC)',
        fontsize=14,
    )
    plt.tight_layout()
    print(f"[DEBUG] Plot created in {time.time() - start_time:.2f}s")
    
    if save_path:
        print(f"[DEBUG] Saving plot to {save_path}...")
        save_start = time.time()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[DEBUG] Saved visualization in {time.time() - save_start:.2f}s")
        print(f"Saved visualization to {save_path}")
    
    if use_wandb:
        wandb.log({"generated_samples": wandb.Image(fig)}, step=epoch if epoch is not None else None)
    
    plt.close(fig)
    # plt.show()
    
    print(f"[DEBUG] Visualization complete at {datetime.now().strftime('%H:%M:%S')}")

    return samples_hsc



def visualize_samples_unconditional(model, device, data_loader, mode='random', num_examples=8, num_samples=8, save_path=None, epoch=None, use_wandb=False, normalize=False):
    '''
    Visualize unconditionally generated samples for HSC (not conditioned on Legacy Survey examples).
    '''
    # Convert device string to torch.device if needed
    if isinstance(device, str):
        device = torch.device(device)

    model.eval()  # Set model to evaluation mode
    batch_size = num_examples

    with torch.no_grad():  # Disable gradient computation to save memory
        hsc_batch, legacy_batch, _, _ = data_loader.get_batch(batch_size=batch_size)
        hsc_batch=hsc_batch.to(device)
        legacy_batch=legacy_batch.to(device)

        if normalize:
            min_val_hsc = hsc_batch.min()
            max_val_hsc = hsc_batch.max()
            
            hsc_norm = (hsc_batch - min_val_hsc) / ((max_val_hsc - min_val_hsc)+1e-3)

            min_val_legacy = legacy_batch.min()
            max_val_legacy = legacy_batch.max()
            legacy_batch = (legacy_batch - min_val_legacy) / ((max_val_legacy - min_val_legacy)+1e-3)
        else:
            hsc_norm = hsc_batch
            
        z_encoded = model.encode(legacy_batch)
        z_encoded_shape = z_encoded.shape

        if mode == 'random':
            z = torch.randn(z_encoded_shape, device=device)
        elif mode == 'zeros':
            z = torch.zeros(z_encoded_shape, device=device)
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'random' or 'zeros'")

        # Clear MPS cache before sampling to free up memory
        if device.type == 'mps':
            torch.mps.empty_cache()

        samples_hsc = model.sample(device=device, n_samples=num_samples, z=z) # (n_samples, n_examples, C*H*W)

        # Clear cache again after sampling
        if device.type == 'mps':
            torch.mps.empty_cache()

        samples_hsc = samples_hsc.view(num_samples, num_examples, cfg.NUM_CHANNELS, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)

        # Convert to numpy for saving and visualization
        samples_hsc_np = samples_hsc.cpu().numpy()
        hsc_batch_np = hsc_norm.cpu().numpy()
        legacy_batch_np = legacy_batch.cpu().numpy()


    num_rows = num_examples
    num_cols = num_samples + 2  # +2 for target and condition columns

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 2 * num_rows))
    
    # Ensure axes is 2D array
    if num_rows == 1:
        axes = axes.reshape(1, num_cols)


    #TODO: Check if this is what we want to do or not
    def to_rgb(channels_np):
        """Convert 4-channel (g, r, i, z) to RGB using first 3 channels with per-channel normalization."""
        rgb = np.stack([
            channels_np[0],  # g -> R
            channels_np[1],  # r -> G
            channels_np[2],  # i -> B
        ], axis=-1)  # (H, W, 3)

        for c in range(3):
            ch = rgb[:, :, c]
            ch_min, ch_max = ch.min(), ch.max()
            if ch_max > ch_min:
                rgb[:, :, c] = (ch - ch_min) / (ch_max - ch_min)
            else:
                rgb[:, :, c] = 0.0
        return rgb

    
    for j in range(num_examples):
        rgb_gt = to_rgb(hsc_batch_np[j])
        axes[j,0].imshow(rgb_gt, vmin=0, vmax=1)
        axes[j,0].set_title(f'Target (HSC)')

        rgb_cond = to_rgb(legacy_batch_np[j])
        axes[j,1].imshow(rgb_cond, vmin=0, vmax=1)
        axes[j,1].set_title(f'Cond. (Legacy)')

        for i in range(num_samples):
            rgb_hsc = to_rgb(samples_hsc_np[i,j])
            axes[j, i+2].imshow(rgb_hsc, vmin=0, vmax=1)
            axes[j, i+2].set_title(f'Sample {i+1}')
            axes[j, i+2].axis('off')
    plt.suptitle(
        f'Generated Samples (HSC) Unconditional - Epoch {epoch}'
        if epoch is not None
        else 'Generated Samples (HSC) Unconditional',
        fontsize=14,
    )
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if use_wandb:
        wandb.log({"generated_samples": wandb.Image(fig)}, step=epoch if epoch is not None else None)
    
    plt.close(fig)
    # plt.show()

    return samples_hsc


# def visualize_samples_unconditioned()

if __name__ == "__main__":
    main()
