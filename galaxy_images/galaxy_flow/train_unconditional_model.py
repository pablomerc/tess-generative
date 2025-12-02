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


current_path=os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_path))
sys.path.insert(0,parent_dir)

from galaxy_triplets import TripletCreator
from .unconditional_model import build_model
from . import unconditional_config as cfg
from galaxy_images.image_preprocessing import preprocess_image


def visualize_samples(model, device, num_samples=8, save_path=None, epoch=None):
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


def train_epoch(model, triplet_creator, optimizer, device, batch_size=cfg.BATCH_SIZE, use_object_mask=False, show_progress=True):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    num_samples = cfg.NUM_SAMPLES_PER_EPOCH
    num_batches_epoch = max(1, num_samples // batch_size)

    # Define bands for legacysurvey (DES bands: g, r, i, z)
    # This matches what TripletCreator uses (num_bands=4 by default)
    # Band names must match BAND_CENTER_MAX keys in image_preprocessing.py
    legacysurvey_bands = ['DES-G', 'DES-R', 'DES-I', 'DES-Z']

    batch_iter = tqdm(range(num_batches_epoch), desc="Batches", leave=False) if show_progress else range(num_batches_epoch)
    for batch_idx in batch_iter:
        try:
            batch = triplet_creator.create_batch_triplets(
                batch_size=batch_size,
                anchor_survey="legacysurvey",
                use_object_mask=use_object_mask
            )

            # Extract image tensor: (B, C, H, W) where C = num_bands*3 (+ 1 if object_mask)
            # Structure: [flux_bands, ivar_bands, mask_bands, (object_mask)]
            im = batch["ground_truth"]["tensor"].to(device)  # (B, C, H, W)

            # Extract flux channels (first num_bands channels)
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

            # Flatten all channels to (B, output_dim) where output_dim = num_channels * 96 * 96 = 36864
            # Shape: (B, 4, 96, 96) -> (B, 4*96*96)
            im_flat = processed.flatten(1)  # (B, num_channels * 96 * 96)

            # Forward pass
            loss = model(im_flat)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        except Exception as e:
            print(f"Error in training batch {batch_idx}: {e}")
            continue

    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')

    return avg_loss


def train(model, triplet_creator, num_epochs=cfg.NUM_EPOCHS, lr=cfg.LEARNING_RATE, plots_dir=cfg.PLOTS_DIR, device=None, weight_decay=cfg.WEIGHT_DECAY):
    """Main training loop."""

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # Create output directory
    os.makedirs(plots_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(plots_dir, f'unconditional_flow_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)

    train_losses = []

    print(f"Starting training on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Output directory: {run_dir}")

    pbar = tqdm(range(num_epochs), desc="Training")
    for epoch in pbar:
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(model, triplet_creator, optimizer, device)
        train_losses.append(train_loss)

        scheduler.step(train_loss)

        epoch_time = time.time() - epoch_start

        # Print loss at every epoch
        print(f"\nEpoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f} - Time: {epoch_time:.1f}s")

        # Update progress bar with loss info
        pbar.set_postfix({'loss': f'{train_loss:.4f}', 'time': f'{epoch_time:.1f}s'})

        # Visualization
        if (epoch + 1) % cfg.VISUALIZATION_INTERVAL == 0:
            try:
                visualize_samples(
                    model, device,
                    num_samples=8,
                    save_path=os.path.join(run_dir, f'samples_epoch_{epoch+1}.png'),
                    epoch=epoch+1
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



if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create triplet creator
    triplet_creator = TripletCreator(
        dataset_path=cfg.DATA_DIR,
        split='train'
    )

    # Initialize model using build_model (uses unconditional_config defaults)
    model = build_model(device=device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Train
    # train(model, triplet_creator, num_epochs=cfg.NUM_EPOCHS, lr=cfg.LEARNING_RATE, plots_dir=cfg.PLOTS_DIR, device=device)
    train(model, triplet_creator, num_epochs=3, lr=cfg.LEARNING_RATE, plots_dir=cfg.PLOTS_DIR, device=device)
