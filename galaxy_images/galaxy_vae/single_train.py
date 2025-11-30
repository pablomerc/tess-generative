"""
Training script for Single Encoder Galaxy VAE model.
"""

import os
import sys
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Try to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import triplet creator from parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from galaxy_triplets import TripletCreator

from .model import SingleEncoderGalaxyVAE
from .config import *

# Import viz module
try:
    from . import viz
except ImportError:
    # If relative import fails, create a simple viz function
    import matplotlib.pyplot as plt
    viz = None


def slice_tensor_to_config_channels(tensor):
    """
    Slice input tensor to match NUM_CHANNELS from config based on USE_IVAR, USE_FLUX_MASK, USE_OBJECT_MASK.

    The triplet loader always returns: flux (0-3), ivar (4-7), mask (8-11), object_mask (12)
    This function selects channels based on config flags:
    - Flux (0-3): always included
    - IVAR (4-7): included if USE_IVAR=True
    - Mask (8-11): included if USE_FLUX_MASK=True
    - Object mask (12): included if USE_OBJECT_MASK=True

    Args:
        tensor: (B, C, H, W) tensor from triplet loader (should have 13 channels when use_object_mask=True)

    Returns:
        Sliced tensor with NUM_CHANNELS channels matching config
    """
    channels = []

    # Always include flux (channels 0-3)
    channels.append(tensor[:, 0:NUM_BANDS, :, :])

    # Include ivar (channels 4-7) if USE_IVAR=True
    if USE_IVAR:
        channels.append(tensor[:, NUM_BANDS:2*NUM_BANDS, :, :])

    # Include mask (channels 8-11) if USE_FLUX_MASK=True
    if USE_FLUX_MASK:
        channels.append(tensor[:, 2*NUM_BANDS:3*NUM_BANDS, :, :])

    # Include object_mask (channel 12) if USE_OBJECT_MASK=True
    if USE_OBJECT_MASK:
        channels.append(tensor[:, -1:, :, :])  # Last channel

    # Concatenate selected channels
    if len(channels) > 1:
        return torch.cat(channels, dim=1)
    else:
        return channels[0]


def vae_loss(reconstruction, target, mu, logvar, beta_kl=BETA_KL, recon_weight=RECONSTRUCTION_WEIGHT):
    """
    Compute VAE loss: reconstruction loss + KL divergence.

    Args:
        reconstruction: Reconstructed image (B, C, H, W)
        target: Target image (B, C, H, W) - same as input for standard VAE
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta_kl: Weight for KL divergence
        recon_weight: Weight for reconstruction loss

    Returns:
        tuple: (total_loss, recon_loss, kl_loss)
    """
    # Reconstruction loss (MSE)
    recon_loss = nn.functional.mse_loss(reconstruction, target, reduction='mean')

    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    # Total loss
    total_loss = recon_weight * recon_loss + beta_kl * kl_loss

    return total_loss, recon_loss, kl_loss


def train_epoch(model, triplet_creator, optimizer, device, batch_size=BATCH_SIZE, use_object_mask=USE_OBJECT_MASK):
    """Train for one epoch. use_object_mask should match USE_OBJECT_MASK in config."""
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    num_batches = 0

    # Get number of samples (approximate - will iterate until we run out)
    num_samples = NUM_SAMPLES_PER_EPOCH
    num_batches_epoch = max(1, num_samples // batch_size)

    for batch_idx in range(num_batches_epoch):
        try:
            # Create batch of triplets (we only use the anchor/ground_truth image)
            # Always load with object_mask=True to get all channels, then slice based on config
            batch = triplet_creator.create_batch_triplets(
                batch_size=batch_size,
                anchor_survey="legacysurvey",  # Use legacysurvey as anchor
                use_object_mask=True  # Always load all channels, slice based on config
            )

            # Extract anchor image (ground_truth) - this is what we'll encode and reconstruct
            anchor_image = batch["ground_truth"]["tensor"].to(device)  # (B, C, H, W)

            # Slice tensor to match NUM_CHANNELS from config
            anchor_image = slice_tensor_to_config_channels(anchor_image)

            # Forward pass: encode and reconstruct the same image
            reconstruction, z, mu, logvar = model(anchor_image)

            # Compute loss (target is the same as input for standard VAE)
            loss, recon_loss, kl_loss = vae_loss(
                reconstruction,
                anchor_image,  # Target is the same as input
                mu,
                logvar
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if GRADIENT_CLIPPING:
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

            optimizer.step()

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            num_batches += 1

            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{num_batches_epoch}: Loss={loss.item():.4f}, "
                      f"Recon={recon_loss.item():.4f}, KL={kl_loss.item():.4f}")

        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue

    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    avg_recon_loss = total_recon_loss / num_batches if num_batches > 0 else float('inf')
    avg_kl_loss = total_kl_loss / num_batches if num_batches > 0 else float('inf')

    return avg_loss, avg_recon_loss, avg_kl_loss


def train(model, triplet_creator, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE,
          plots_dir=PLOTS_DIR, device=None, use_object_mask=USE_OBJECT_MASK, use_wandb=True):
    """Main training loop."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # Initialize wandb if available and requested
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project="galaxy-vae",
            name=f"single_encoder_vae_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "num_epochs": num_epochs,
                "learning_rate": lr,
                "batch_size": BATCH_SIZE,
                "latent_dim": LATENT_DIM,
                "num_channels": NUM_CHANNELS,
                "use_object_mask": use_object_mask,
                "beta_kl": BETA_KL,
                "reconstruction_weight": RECONSTRUCTION_WEIGHT,
            }
        )
        print("Wandb initialized")
    elif use_wandb and not WANDB_AVAILABLE:
        print("Warning: wandb requested but not available. Continuing without wandb.")
        use_wandb = False

    # Create output directory (only if not using wandb for plots)
    if not use_wandb:
        os.makedirs(plots_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = os.path.join(plots_dir, f'single_encoder_vae_{timestamp}')
        os.makedirs(run_dir, exist_ok=True)
    else:
        # Still create run_dir for checkpoints
        os.makedirs(plots_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = os.path.join(plots_dir, f'single_encoder_vae_{timestamp}')
        os.makedirs(run_dir, exist_ok=True)

    train_losses = []

    print(f"Starting training on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    if not use_wandb:
        print(f"Output directory: {run_dir}")

    for epoch in range(num_epochs):
        epoch_start = time.time()
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Train
        train_loss, train_recon, train_kl = train_epoch(
            model, triplet_creator, optimizer, device, use_object_mask=use_object_mask
        )

        train_losses.append(train_loss)

        scheduler.step(train_loss)
        current_lr = optimizer.param_groups[0]['lr']

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, KL: {train_kl:.4f})")
        print(f"            - Time: {epoch_time:.1f}s")

        # Log to wandb
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_recon_loss": train_recon,
                "train_kl_loss": train_kl,
                "learning_rate": current_lr,
                "epoch_time": epoch_time,
            })

        # Save checkpoint
        if (epoch + 1) % SAVE_INTERVAL == 0:
            checkpoint_path = os.path.join(run_dir, f'model_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

        # Visualization every 2 epochs
        if (epoch + 1) % 2 == 0:
            try:
                # Create visualization showing first 3 channels (flux) for target and reconstruction
                save_path = os.path.join(run_dir, f'reconstruction_epoch_{epoch+1}.png') if not use_wandb else None
                create_single_encoder_visualization(
                    model, triplet_creator, device,
                    save_path=save_path,
                    num_examples=8,
                    use_object_mask=use_object_mask,
                    use_wandb=use_wandb,
                    epoch=epoch + 1
                )
            except Exception as e:
                print(f"Error creating visualization: {e}")

    print(f"\nTraining completed.")
    if not use_wandb:
        print(f"Results saved to: {run_dir}")
    if use_wandb:
        wandb.finish()

    return train_losses


def create_single_encoder_visualization(model, triplet_creator, device, save_path=None, num_examples=8,
                                       use_object_mask=USE_OBJECT_MASK, use_wandb=False, epoch=None):
    """Create visualization of reconstructions for single encoder VAE.

    Shows the first 3 channels (flux) for both target and reconstruction.
    If use_wandb is True, logs to wandb. Otherwise saves to save_path if provided.
    """
    model.eval()

    with torch.no_grad():
        # Get a batch of examples
        # Always load with object_mask=True to get all channels, then slice based on config
        batch = triplet_creator.create_batch_triplets(
            batch_size=num_examples,
            anchor_survey="legacysurvey",
            use_object_mask=True  # Always load all channels, slice based on config
        )

        anchor_images = batch["ground_truth"]["tensor"].to(device)

        # Slice tensor to match NUM_CHANNELS from config
        anchor_images = slice_tensor_to_config_channels(anchor_images)

        reconstructions, _, _, _ = model(anchor_images)

        # Convert to numpy for plotting
        anchor_images_np = anchor_images.cpu().numpy()
        reconstructions_np = reconstructions.cpu().numpy()

        # Create figure: 2 columns (target, reconstruction), num_examples rows
        fig, axes = plt.subplots(num_examples, 2, figsize=(10, 5 * num_examples))
        if num_examples == 1:
            axes = axes.reshape(1, -1)

        for i in range(num_examples):
            # Show first 3 channels (flux) of target as RGB
            target_rgb = np.stack([
                anchor_images_np[i, 0],  # Channel 0
                anchor_images_np[i, 1],  # Channel 1
                anchor_images_np[i, 2]   # Channel 2
            ], axis=-1)
            # Normalize to [0, 1] for each channel
            for c in range(3):
                ch = target_rgb[:, :, c]
                ch_min, ch_max = ch.min(), ch.max()
                if ch_max > ch_min:
                    target_rgb[:, :, c] = (ch - ch_min) / (ch_max - ch_min)
                else:
                    target_rgb[:, :, c] = 0

            axes[i, 0].imshow(target_rgb)
            axes[i, 0].set_title(f'Target {i+1} (flux channels 0-2)')
            axes[i, 0].axis('off')

            # Show first 3 channels (flux) of reconstruction as RGB
            recon_rgb = np.stack([
                reconstructions_np[i, 0],  # Channel 0
                reconstructions_np[i, 1],  # Channel 1
                reconstructions_np[i, 2]    # Channel 2
            ], axis=-1)
            # Normalize to [0, 1] for each channel
            for c in range(3):
                ch = recon_rgb[:, :, c]
                ch_min, ch_max = ch.min(), ch.max()
                if ch_max > ch_min:
                    recon_rgb[:, :, c] = (ch - ch_min) / (ch_max - ch_min)
                else:
                    recon_rgb[:, :, c] = 0

            axes[i, 1].imshow(recon_rgb)
            axes[i, 1].set_title(f'Reconstruction {i+1} (flux channels 0-2)')
            axes[i, 1].axis('off')

        plt.tight_layout()

        # Log to wandb or save locally
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({f"reconstructions_epoch_{epoch}": wandb.Image(fig)})
            print(f"Logged visualization to wandb for epoch {epoch}")
        elif save_path is not None:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")

        plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Single Encoder Galaxy VAE")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--data_path", type=str, default=DATA_DIR, help="Path to data directory")
    parser.add_argument("--plots_dir", type=str, default=PLOTS_DIR, help="Directory for plots/checkpoints")
    parser.add_argument("--use_object_mask", action="store_true", default=USE_OBJECT_MASK, help="Use object mask")
    parser.add_argument("--no_object_mask", dest="use_object_mask", action="store_false", help="Don't use object mask")
    parser.add_argument("--use_wandb", action="store_true", default=True, help="Use wandb for logging (default: True)")
    parser.add_argument("--no_wandb", dest="use_wandb", action="store_false", help="Don't use wandb")

    args = parser.parse_args()

    # Create triplet creator
    triplet_creator = TripletCreator(
        dataset_path=args.data_path,
        split="train"
    )

    # Create model using NUM_CHANNELS from config
    # The model will expect exactly NUM_CHANNELS channels
    model = SingleEncoderGalaxyVAE(num_channels=NUM_CHANNELS)

    print(f"Model created with NUM_CHANNELS={NUM_CHANNELS} from config")
    print(f"USE_OBJECT_MASK={USE_OBJECT_MASK} in config")
    print(f"Input tensors will be sliced to match NUM_CHANNELS")

    # Train - always load with object_mask=True to get all channels, then slice based on config
    # The slice_tensor_to_config_channels function will handle the slicing
    train(model, triplet_creator, num_epochs=args.epochs, lr=args.lr, plots_dir=args.plots_dir,
          use_object_mask=True, use_wandb=args.use_wandb)
