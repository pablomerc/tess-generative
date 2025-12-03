'''
Training script for an unconditional flow matching model for galaxy images.
'''

import os
import sys
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

# Try to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")


current_path=os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_path))
sys.path.insert(0,parent_dir)

from galaxy_triplets import TripletCreator
from .unconditional_model import build_model
from . import unconditional_config as cfg
from galaxy_images.image_preprocessing import preprocess_image
from .hsc_data_loader import HSCDataLoader


def visualize_samples(model, device, num_samples=8, save_path=None, epoch=None, use_wandb=False):
    """Visualize generated samples and save both plot and raw values."""
    model.eval()

    with torch.no_grad():
        # Generate samples
        samples_flat = model.sample(batch_size=num_samples, device=device)  # (B, output_dim)

        # Reshape from flattened to (B, C, H, W)
        batch_size = samples_flat.shape[0]
        samples = samples_flat.view(batch_size, cfg.NUM_CHANNELS, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)

        # Convert to numpy for saving and visualization
        samples_np = samples.cpu().numpy()  # (B, C, H, W)

    # Save raw values as numpy array
    if save_path:
        raw_save_path = save_path.replace('.png', '_raw.npy')
        np.save(raw_save_path, samples_np)
        print(f"Saved raw samples to {raw_save_path} (shape: {samples_np.shape})")

    # Create figure for visualization
    num_cols = min(8, num_samples)
    num_rows = (num_samples + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(2*num_cols, 2*num_rows))
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for i in range(num_samples):
        # Convert 4-channel (g, r, i, z) to RGB using first 3 channels (g, r, i)
        rgb = np.stack([
            samples_np[i, 0],  # g -> R
            samples_np[i, 1],  # r -> G
            samples_np[i, 2]   # i -> B
        ], axis=-1)  # (H, W, 3)

        # Normalize each channel to [0, 1] for visualization
        for c in range(3):
            ch = rgb[:, :, c]
            ch_min, ch_max = ch.min(), ch.max()
            if ch_max > ch_min:
                rgb[:, :, c] = (ch - ch_min) / (ch_max - ch_min)
            else:
                rgb[:, :, c] = 0

        axes[i].imshow(rgb, vmin=0, vmax=1)
        axes[i].set_title(f'Sample {i+1}')
        axes[i].axis('off')

    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')

    plt.suptitle(f'Generated Samples - Epoch {epoch}' if epoch else 'Generated Samples', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    plt.close()

    return samples_np


def train_epoch(model, triplet_creator, optimizer, device, batch_size=cfg.BATCH_SIZE, use_object_mask=False, show_progress=False, profile=False):
    """Train the model for one epoch.

    Args:
        profile: If True, returns timing breakdown and FLOPs info
    """
    model.train()
    total_loss = 0
    num_batches = 0

    num_samples = cfg.NUM_SAMPLES_PER_EPOCH
    num_batches_epoch = max(1, num_samples // batch_size)

    # Define bands for legacysurvey (DES bands: g, r, i, z)
    # This matches what TripletCreator uses (num_bands=4 by default)
    # Band names must match BAND_CENTER_MAX keys in image_preprocessing.py
    legacysurvey_bands = ['DES-G', 'DES-R', 'DES-I', 'DES-Z']

    # Timing measurements
    timing_stats = defaultdict(float)

    batch_iter = tqdm(range(num_batches_epoch), desc="Batches", leave=False) if show_progress else range(num_batches_epoch)
    for batch_idx in batch_iter:
        try:
            # Time data loading
            t0 = time.time()
            anchor_survey='hsc'
            # Use appropriate anchor_survey based on data loader type
            # HSCDataLoader always uses HSC, TripletCreator can use either


            batch = triplet_creator.create_batch_triplets(
                batch_size=batch_size,
                anchor_survey=anchor_survey,
                use_object_mask=use_object_mask
            )
            t1 = time.time()
            timing_stats['data_loading'] += (t1 - t0)

            # Extract image tensor
            t0 = time.time()
            im = batch["ground_truth"]["tensor"].to(device)  # (B, C, H, W)

            # Check if data is already preprocessed (shape should be (B, 4, 96, 96))
            # If shape is (B, 4, 96, 96), data is preprocessed; otherwise, it needs preprocessing
            is_preprocessed = (im.shape[1] == 4 and im.shape[2] == cfg.IMAGE_SIZE and im.shape[3] == cfg.IMAGE_SIZE)

            if is_preprocessed:
                # Data is already preprocessed (crop, clamp, rescale, range compress)
                # Normalize to [-1, 1] range for flow matching
                min_val = im.min()
                max_val = im.max()
                if not torch.isclose(max_val, min_val):
                    im_norm = (im - min_val) / (max_val - min_val)  # [0, 1]
                    im_norm = 2.0 * im_norm - 1.0  # [-1, 1]
                else:
                    im_norm = torch.zeros_like(im)
                # Flatten it
                im_flat = im_norm.flatten(1)  # (B, num_channels * 96 * 96)
            else:
                # Extract flux channels (first num_bands channels)
                # Structure: [flux_bands, ivar_bands, mask_bands, (object_mask)]
                num_bands = 4  # Default from TripletCreator
                flux_channels = im[:, :num_bands, :, :]  # (B, num_bands, H, W)

                # Preprocess: crop to 96x96, clamp, rescale, range compress
                # preprocess_image expects (B, C, H, W) and returns (B, C, 96, 96)
                processed = preprocess_image(
                    flux_channels,
                    bands=legacysurvey_bands,
                    crop_size=cfg.IMAGE_SIZE,
                    apply_range_compression=True
                )  # (B, num_bands, 96, 96)

                # Normalize to [-1, 1] range for flow matching
                # Compute min/max for normalization
                min_val = processed.min()
                max_val = processed.max()
                if not torch.isclose(max_val, min_val):
                    processed_norm = (processed - min_val) / (max_val - min_val)  # [0, 1]
                    processed_norm = 2.0 * processed_norm - 1.0  # [-1, 1]
                else:
                    processed_norm = torch.zeros_like(processed)

                # Flatten all channels to (B, output_dim) where output_dim = num_channels * 96 * 96 = 36864
                # Shape: (B, 4, 96, 96) -> (B, 4*96*96)
                im_flat = processed_norm.flatten(1)  # (B, num_channels * 96 * 96)

            t1 = time.time()
            timing_stats['data_preprocessing'] += (t1 - t0)

            # Forward pass
            t0 = time.time()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            loss = model(im_flat)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.time()
            timing_stats['forward'] += (t1 - t0)

            # Backward pass
            t0 = time.time()
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for stability
            if hasattr(cfg, 'MAX_GRAD_NORM') and cfg.MAX_GRAD_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.MAX_GRAD_NORM)
            optimizer.step()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.time()
            timing_stats['backward'] += (t1 - t0)

            total_loss += loss.item()
            num_batches += 1

        except Exception as e:
            print(f"Error in training batch {batch_idx}: {e}")
            continue

    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')

    # Normalize timing by number of batches
    if num_batches > 0:
        for key in timing_stats:
            timing_stats[key] /= num_batches

    if profile:
        return avg_loss, timing_stats
    return avg_loss


def count_flops(model, input_shape, device):
    """Estimate FLOPs for the model using a simple forward pass analysis.

    This is a rough estimate. For more accurate FLOP counting, use thop or fvcore.
    """
    try:
        # Try to use thop if available
        import thop
        dummy_input = torch.randn(1, *input_shape).to(device)
        flops, params = thop.profile(model, inputs=(dummy_input,), verbose=False)
        return flops, params
    except ImportError:
        # Fallback: estimate based on model parameters and input size
        # Rough estimate: 2 * num_params * input_size (for forward pass)
        total_params = sum(p.numel() for p in model.parameters())
        input_size = np.prod(input_shape)
        # Very rough estimate: assume each parameter is used once per forward pass
        estimated_flops = total_params * input_size * 2  # multiply by 2 for add/multiply ops
        return estimated_flops, total_params


def print_profiling_info(timing_stats, num_batches, batch_size, flops_info=None):
    """Print detailed profiling information."""
    print("\n" + "="*60)
    print("PROFILING INFORMATION")
    print("="*60)

    total_time = sum(timing_stats.values())
    print(f"\nPer-batch timing (averaged over {num_batches} batches):")
    for key, value in timing_stats.items():
        percentage = (value / total_time * 100) if total_time > 0 else 0
        print(f"  {key:20s}: {value*1000:6.2f} ms ({percentage:5.1f}%)")

    print(f"\nTotal per-batch time: {total_time*1000:.2f} ms")
    print(f"Estimated epoch time: {total_time * num_batches:.2f} s")

    if flops_info:
        flops, params = flops_info
        print(f"\nModel complexity:")
        print(f"  Parameters: {params:,}")
        if flops > 1e12:
            print(f"  FLOPs (estimated): {flops/1e12:.2f} TFLOPs")
        elif flops > 1e9:
            print(f"  FLOPs (estimated): {flops/1e9:.2f} GFLOPs")
        elif flops > 1e6:
            print(f"  FLOPs (estimated): {flops/1e6:.2f} MFLOPs")
        else:
            print(f"  FLOPs (estimated): {flops:,.0f}")
        print(f"  FLOPs per batch: {flops/1e9:.4f} GFLOPs")
        print(f"  Throughput: {batch_size / total_time:.2f} samples/sec")

    print("="*60 + "\n")


def train(model, triplet_creator, num_epochs=cfg.NUM_EPOCHS, lr=cfg.LEARNING_RATE, plots_dir=cfg.PLOTS_DIR, device=None, weight_decay=cfg.WEIGHT_DECAY, use_wandb=True, profile_first_epoch=None):
    """Main training loop."""

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # Initialize wandb if available and requested
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project="galaxy-unconditional-flow",
            name=f"unconditional_flow_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "num_epochs": num_epochs,
                "learning_rate": lr,
                "batch_size": cfg.BATCH_SIZE,
                "weight_decay": weight_decay,
                "image_size": cfg.IMAGE_SIZE,
                "num_channels": cfg.NUM_CHANNELS,
                "output_dim": cfg.OUTPUT_DIM,
                "velocity_field_type": cfg.VELOCITY_FIELD_TYPE,
                "use_film": cfg.USE_FILM,
                "unet_channels": cfg.UNET_CHANNELS,
                "num_residual_layers": cfg.NUM_RESIDUAL_LAYERS,
                "n_integration_steps": cfg.N_INTEGRATION_STEPS,
                "num_samples_per_epoch": cfg.NUM_SAMPLES_PER_EPOCH,
                "save_interval": cfg.SAVE_INTERVAL,
                "visualization_interval": cfg.VISUALIZATION_INTERVAL,
                "device": str(device),
            }
        )
        wandb.watch(model, log="all")
        print("Wandb initialized")
    elif use_wandb and not WANDB_AVAILABLE:
        print("Warning: wandb requested but not available. Continuing without wandb.")
        use_wandb = False

    # Create output directory
    os.makedirs(plots_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(plots_dir, f'unconditional_flow_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)

    train_losses = []

    print(f"Starting training on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Output directory: {run_dir}")

    # Profile first epoch if requested (default from config)
    if profile_first_epoch is None:
        profile_first_epoch = getattr(cfg, 'PROFILE_FIRST_EPOCH', True)

    start_epoch = 0
    if profile_first_epoch:
        print("\nProfiling first epoch...")
        flops_info = count_flops(model, (cfg.OUTPUT_DIM,), device)
        train_loss, timing_stats = train_epoch(
            model, triplet_creator, optimizer, device, profile=True
        )
        num_batches_epoch = max(1, cfg.NUM_SAMPLES_PER_EPOCH // cfg.BATCH_SIZE)
        print_profiling_info(timing_stats, num_batches_epoch, cfg.BATCH_SIZE, flops_info)
        train_losses.append(train_loss)
        start_epoch = 1
        scheduler.step(train_loss)
        print(f"\nEpoch 1/{num_epochs} - Train Loss: {train_loss:.6f}")

    pbar = tqdm(range(start_epoch, num_epochs), desc="Training", initial=start_epoch, total=num_epochs)

    for epoch in pbar:
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(model, triplet_creator, optimizer, device)
        train_losses.append(train_loss)

        scheduler.step(train_loss)

        epoch_time = time.time() - epoch_start

        # Print loss at every epoch
        print(f"\nEpoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f} - Time: {epoch_time:.1f}s")

        # Log to wandb
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch_time": epoch_time,
            }, step=epoch + 1)

        # Update progress bar with loss info
        pbar.set_postfix({'loss': f'{train_loss:.4f}', 'time': f'{epoch_time:.1f}s'})

        # Visualization
        if (epoch + 1) % cfg.VISUALIZATION_INTERVAL == 0:
            try:
                samples_np = visualize_samples(
                    model, device,
                    num_samples=8,
                    save_path=os.path.join(run_dir, f'samples_epoch_{epoch+1}.png'),
                    epoch=epoch+1,
                    use_wandb=use_wandb and WANDB_AVAILABLE
                )
                # Log images to wandb
                if use_wandb and WANDB_AVAILABLE:
                    # Create a figure for wandb logging
                    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
                    axes = axes.flatten()
                    for i in range(min(8, len(samples_np))):
                        # Convert 4-channel to RGB
                        rgb = np.stack([
                            samples_np[i, 0],  # g -> R
                            samples_np[i, 1],  # r -> G
                            samples_np[i, 2]   # i -> B
                        ], axis=-1)
                        # Normalize for visualization
                        for c in range(3):
                            ch = rgb[:, :, c]
                            ch_min, ch_max = ch.min(), ch.max()
                            if ch_max > ch_min:
                                rgb[:, :, c] = (ch - ch_min) / (ch_max - ch_min)
                            else:
                                rgb[:, :, c] = 0
                        axes[i].imshow(rgb, vmin=0, vmax=1)
                        axes[i].set_title(f'Sample {i+1}')
                        axes[i].axis('off')
                    # Hide unused subplots
                    for i in range(len(samples_np), len(axes)):
                        axes[i].axis('off')
                    plt.suptitle(f'Generated Samples - Epoch {epoch+1}', fontsize=14)
                    plt.tight_layout()
                    wandb.log({"generated_samples": wandb.Image(fig)}, step=epoch + 1)
                    plt.close(fig)
            except Exception as e:
                print(f"Error creating visualization: {e}")

        # Save checkpoint
        if (epoch + 1) % cfg.SAVE_INTERVAL == 0:
            checkpoint_path = os.path.join(run_dir, f'model_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    # Finish wandb run
    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()
        print("Wandb run finished")



if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create data loader (either preprocessed HDF5 or TripletCreator)
    if getattr(cfg, 'USE_PREPROCESSED_DATA', False):
        hdf5_path = getattr(cfg, 'PREPROCESSED_HDF5_PATH', '/mnt/scratch/legacysurvey_hsc_crossmatched/preprocessed_hsc.h5')
        if not os.path.exists(hdf5_path):
            print(f"Warning: Preprocessed HDF5 file not found at {hdf5_path}")
            print("Falling back to TripletCreator. Run preprocess_hsc_images.py first to create preprocessed data.")
            triplet_creator = TripletCreator(
                dataset_path=cfg.DATA_DIR,
                split='train'
            )
        else:
            print(f"Using preprocessed HDF5 data from {hdf5_path}")
            triplet_creator = HSCDataLoader(
                hdf5_path=hdf5_path,
                seed=42
            )
    else:
        print("Using TripletCreator (raw parquet data)")
        triplet_creator = TripletCreator(
            dataset_path=cfg.DATA_DIR,
            split='train'
        )

    # Initialize model using build_model (uses unconditional_config defaults)
    model = build_model(device=device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Train
    train(model, triplet_creator, num_epochs=cfg.NUM_EPOCHS, lr=cfg.LEARNING_RATE, plots_dir=cfg.PLOTS_DIR, device=device)

    # Cleanup: close HDF5 file if using HSCDataLoader
    if isinstance(triplet_creator, HSCDataLoader):
        triplet_creator.close()
