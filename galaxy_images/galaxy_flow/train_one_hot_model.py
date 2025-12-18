'''
Training script for a one-hot flow matching model for galaxy images.

It expects data already preprocessed and loaded by.hsc_data_loader HSC_Legacy_DataLoader_Onehot

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


from .one_hot_model import build_model
from . import onehot_config as cfg
from galaxy_images.image_preprocessing import preprocess_image
from .hsc_data_loader import HSCDataLoader
from .hsc_data_loader import HSC_Legacy_DataLoader_OneHot


#TODO: Visualize samples
# def visualize_samples(model, device, num_samples=8, save_path=None, epoch=None, use_wandb=False):
#     """Visualize generated samples and save both plot and raw values."""
#     model.eval()

#     with torch.no_grad():
#         # Generate samples
#         samples_flat = model.sample(batch_size=num_samples, device=device)  # (B, output_dim)

#         # Reshape from flattened to (B, C, H, W)
#         batch_size = samples_flat.shape[0]
#         samples = samples_flat.view(batch_size, cfg.NUM_CHANNELS, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)

#         # Convert to numpy for saving and visualization
#         samples_np = samples.cpu().numpy()  # (B, C, H, W)

#     # Save raw values as numpy array
#     if save_path:
#         raw_save_path = save_path.replace('.png', '_raw.npy')
#         np.save(raw_save_path, samples_np)
#         print(f"Saved raw samples to {raw_save_path} (shape: {samples_np.shape})")

#     # Create figure for visualization
#     num_cols = min(8, num_samples)
#     num_rows = (num_samples + num_cols - 1) // num_cols
#     fig, axes = plt.subplots(num_rows, num_cols, figsize=(2*num_cols, 2*num_rows))
#     if num_rows == 1:
#         axes = axes.reshape(1, -1)
#     axes = axes.flatten()

#     for i in range(num_samples):
#         # Convert 4-channel (g, r, i, z) to RGB using first 3 channels (g, r, i)
#         rgb = np.stack([
#             samples_np[i, 0],  # g -> R
#             samples_np[i, 1],  # r -> G
#             samples_np[i, 2]   # i -> B
#         ], axis=-1)  # (H, W, 3)

#         # Normalize each channel to [0, 1] for visualization
#         for c in range(3):
#             ch = rgb[:, :, c]
#             ch_min, ch_max = ch.min(), ch.max()
#             if ch_max > ch_min:
#                 rgb[:, :, c] = (ch - ch_min) / (ch_max - ch_min)
#             else:
#                 rgb[:, :, c] = 0

#         axes[i].imshow(rgb, vmin=0, vmax=1)
#         axes[i].set_title(f'Sample {i+1}')
#         axes[i].axis('off')

#     # Hide unused subplots
#     for i in range(num_samples, len(axes)):
#         axes[i].axis('off')

#     plt.suptitle(f'Generated Samples - Epoch {epoch}' if epoch else 'Generated Samples', fontsize=14)
#     plt.tight_layout()

#     if save_path:
#         plt.savefig(save_path, dpi=150, bbox_inches='tight')
#         print(f"Saved visualization to {save_path}")

#     plt.close()

#     return samples_np


def visualize_samples(model, device, num_samples=8, save_path=None, epoch=None, use_wandb=False):
    """Visualize generated samples for HSC and Legacy Survey in two columns.

    Generates num_samples HSC-conditioned samples and num_samples Legacy-conditioned
    samples, and plots them in an (num_samples x 2) grid: left column HSC, right column Legacy.
    """
    model.eval()

    with torch.no_grad():
        # Build one-hot conditioning vectors
        hsc_one_hot = torch.tensor([1.0, 0.0], dtype=torch.float32, device=device)
        legacy_one_hot = torch.tensor([0.0, 1.0], dtype=torch.float32, device=device)

        z_hsc = hsc_one_hot.unsqueeze(0).repeat(num_samples, 1)        # (num_samples, 2)
        z_legacy = legacy_one_hot.unsqueeze(0).repeat(num_samples, 1)  # (num_samples, 2)

        # Generate samples conditioned on HSC and Legacy
        samples_hsc_flat = model.sample(batch_size=num_samples, device=device, z=z_hsc)        # (B, output_dim)
        samples_legacy_flat = model.sample(batch_size=num_samples, device=device, z=z_legacy)  # (B, output_dim)

        # Reshape from flattened to (B, C, H, W)
        samples_hsc = samples_hsc_flat.view(num_samples, cfg.NUM_CHANNELS, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)
        samples_legacy = samples_legacy_flat.view(num_samples, cfg.NUM_CHANNELS, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)

        # Convert to numpy for saving and visualization
        samples_hsc_np = samples_hsc.cpu().numpy()
        samples_legacy_np = samples_legacy.cpu().numpy()

    # Optionally save raw values as numpy arrays
    if save_path:
        base, ext = os.path.splitext(save_path)
        raw_hsc_path = f"{base}_hsc_raw.npy"
        raw_legacy_path = f"{base}_legacy_raw.npy"
        np.save(raw_hsc_path, samples_hsc_np)
        np.save(raw_legacy_path, samples_legacy_np)
        print(f"Saved raw HSC samples to {raw_hsc_path} (shape: {samples_hsc_np.shape})")
        print(f"Saved raw Legacy samples to {raw_legacy_path} (shape: {samples_legacy_np.shape})")

    # Create figure: num_samples rows, 2 columns (HSC | Legacy)
    num_rows = num_samples
    num_cols = 2
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 2 * num_rows))

    # Ensure axes is 2D array
    if num_rows == 1:
        axes = axes.reshape(1, num_cols)

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

    for i in range(num_samples):
        # HSC sample (left column)
        rgb_hsc = to_rgb(samples_hsc_np[i])
        axes[i, 0].imshow(rgb_hsc, vmin=0, vmax=1)
        axes[i, 0].set_title(f'HSC {i+1}')
        axes[i, 0].axis('off')

        # Legacy sample (right column)
        rgb_legacy = to_rgb(samples_legacy_np[i])
        axes[i, 1].imshow(rgb_legacy, vmin=0, vmax=1)
        axes[i, 1].set_title(f'Legacy {i+1}')
        axes[i, 1].axis('off')

    plt.suptitle(
        f'Generated Samples (HSC vs Legacy) - Epoch {epoch}'
        if epoch is not None
        else 'Generated Samples (HSC vs Legacy)',
        fontsize=14,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    if use_wandb and WANDB_AVAILABLE:
        wandb.log({"generated_samples": wandb.Image(fig)}, step=epoch if epoch is not None else None)

    plt.close(fig)

    return samples_hsc_np, samples_legacy_np


def visualize_shared_noise_trajectories(model, device, save_path=None, epoch=None, use_wandb=False, num_examples: int = 4):
    """Visualize how multiple shared-noise samples evolve under HSC vs Legacy conditioning.

    For each of num_examples seeds, uses the same initial noise x0 for both conditions
    and integrates the ODE once for [HSC, Legacy] to show the difference purely from conditioning.
    """
    model.eval()

    with torch.no_grad():
        # Shared initial noises: one per example
        x0 = torch.randn(num_examples, cfg.OUTPUT_DIM, device=device)  # (N, D)
        # Repeat each noise twice (HSC, Legacy) -> (2N, D)
        x0_pair = torch.cat([x0, x0], dim=0)

        # Conditioning: first N are HSC, next N are Legacy
        z_hsc = torch.tensor([[1.0, 0.0]], dtype=torch.float32, device=device).repeat(num_examples, 1)
        z_legacy = torch.tensor([[0.0, 1.0]], dtype=torch.float32, device=device).repeat(num_examples, 1)
        z = torch.cat([z_hsc, z_legacy], dim=0)  # (2N, 2)

        # Evolve from fixed noises using the decoder's ODE integrator
        x_T = model.decoder.sample_from_fixed_noise(x0_pair, z=z)  # (2N, output_dim)

        # Reshape to images
        samples = x_T.view(2 * num_examples, cfg.NUM_CHANNELS, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)
        samples_np = samples.cpu().numpy()

    # Optional raw save
    if save_path:
        base, ext = os.path.splitext(save_path)
        raw_path = f"{base}_shared_noise_raw.npy"
        np.save(raw_path, samples_np)
        print(f"Saved shared-noise samples to {raw_path} (shape: {samples_np.shape})")

    # Make a num_examples x 2 figure: left = HSC, right = Legacy for each seed
    fig, axes = plt.subplots(num_examples, 2, figsize=(8, 2 * num_examples))

    if num_examples == 1:
        axes = axes.reshape(1, 2)

    def to_rgb(channels_np):
        rgb = np.stack(
            [
                channels_np[0],  # g -> R
                channels_np[1],  # r -> G
                channels_np[2],  # i -> B
            ],
            axis=-1,
        )
        for c in range(3):
            ch = rgb[:, :, c]
            ch_min, ch_max = ch.min(), ch.max()
            if ch_max > ch_min:
                rgb[:, :, c] = (ch - ch_min) / (ch_max - ch_min)
            else:
                rgb[:, :, c] = 0.0
        return rgb

    for i in range(num_examples):
        # Indexing: i -> HSC, i + num_examples -> Legacy
        rgb_hsc = to_rgb(samples_np[i])
        rgb_legacy = to_rgb(samples_np[i + num_examples])

        axes[i, 0].imshow(rgb_hsc, vmin=0, vmax=1)
        axes[i, 0].set_title(f'HSC (seed {i+1})')
        axes[i, 0].axis("off")

        axes[i, 1].imshow(rgb_legacy, vmin=0, vmax=1)
        axes[i, 1].set_title(f'Legacy (seed {i+1})')
        axes[i, 1].axis("off")

    plt.suptitle(
        f'Shared-noise trajectories (HSC vs Legacy) - Epoch {epoch}'
        if epoch is not None
        else 'Shared-noise trajectories (HSC vs Legacy)',
        fontsize=14,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved shared-noise visualization to {save_path}")

    if use_wandb and WANDB_AVAILABLE:
        wandb.log({"shared_noise_samples": wandb.Image(fig)}, step=epoch if epoch is not None else None)

    plt.close(fig)


def train_epoch(model, data_loader, optimizer, device, batch_size=cfg.BATCH_SIZE, show_progress=False, profile=False, normalize=False):
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
    # legacysurvey_bands = ['DES-G', 'DES-R', 'DES-I', 'DES-Z']

    # Timing measurements
    timing_stats = defaultdict(float)

    batch_iter = tqdm(range(num_batches_epoch), desc="Batches", leave=False) if show_progress else range(num_batches_epoch)
    for batch_idx in batch_iter:
        try:
            # Time data loading
            t0 = time.time()
            # anchor_survey='hsc'
            # Use appropriate anchor_survey based on data loader type
            # HSCDataLoader always uses HSC, TripletCreator can use either


            images_batch, conditioning_batch = data_loader.get_training_batch(batch_size = batch_size)
            t1 = time.time()
            timing_stats['data_loading'] += (t1 - t0)

            # Extract image tensor
            t0 = time.time()
            im = images_batch.to(device)  # (B, C, H, W)
            z = conditioning_batch.to(device)


            if normalize:
                # normalize to [-1,1]?
                min_val = im.min()
                max_val = im.max()
                if not torch.isclose(max_val, min_val):
                    im_norm = (im - min_val) / (max_val - min_val)  # [0, 1]
                    im_norm = 2.0 * im_norm - 1.0  # [-1, 1]
                else:
                    im_norm = torch.zeros_like(im)

                im_flat = im_norm.flatten(1)
            else:
                im_flat = im.flatten(1)


            t1 = time.time()
            timing_stats['data_preprocessing'] += (t1 - t0)

            # Forward pass
            t0 = time.time()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            loss = model(im_flat,z)
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


def train(model, data_loader, num_epochs=cfg.NUM_EPOCHS, lr=cfg.LEARNING_RATE, plots_dir=cfg.PLOTS_DIR, device=None, weight_decay=cfg.WEIGHT_DECAY, use_wandb=True, profile_first_epoch=None):
    """Main training loop."""

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # Initialize wandb if available and requested
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project="galaxy-onehot-flow",
            name=f"onehot_flow_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
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
    run_dir = os.path.join(plots_dir, f'onehot_flow_{timestamp}')
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
            model, data_loader, optimizer, device, profile=True
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
        train_loss = train_epoch(model, data_loader, optimizer, device)
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
                # HSC vs Legacy grids
                visualize_samples(
                    model,
                    device,
                    num_samples=8,
                    save_path=os.path.join(run_dir, f'samples_epoch_{epoch+1}.png'),
                    epoch=epoch + 1,
                    use_wandb=use_wandb and WANDB_AVAILABLE,
                )

                # Shared-noise trajectories under different conditionings
                visualize_shared_noise_trajectories(
                    model,
                    device,
                    save_path=os.path.join(run_dir, f'shared_noise_epoch_{epoch+1}.png'),
                    epoch=epoch + 1,
                    use_wandb=use_wandb and WANDB_AVAILABLE,
                )
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
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Using device: {device}")

    # Create data loader (HSC + Legacy with one-hot conditioning)
    hdf5_path = getattr(cfg, 'PREPROCESSED_HDF5_PATH', '/mnt/scratch/legacysurvey_hsc_crossmatched/preprocessed_hsc_legacy.h5')

    data_loader = HSC_Legacy_DataLoader_OneHot(
        hdf5_path=hdf5_path,
        seed=42,
        load_to_memory=getattr(cfg, 'LOAD_TO_MEMORY', True),
    )

    # Initialize model using build_model (uses onehot_config defaults)
    model = build_model(device=device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Train
    train(model, data_loader, num_epochs=cfg.NUM_EPOCHS, lr=cfg.LEARNING_RATE, plots_dir=cfg.PLOTS_DIR, device=device)

    # Cleanup: close HDF5 file if using HSCDataLoader
    if isinstance(data_loader, HSCDataLoader):
        data_loader.close()
