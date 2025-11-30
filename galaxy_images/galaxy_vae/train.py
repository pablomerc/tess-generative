"""
Training script for Galaxy VAE models.
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

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import triplet creator from parent directory
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from galaxy_triplets import TripletCreator

from .model import DoubleEncoderGalaxyVAE
from .config import *

# Import viz module
try:
    from . import viz
except ImportError:
    # If relative import fails, create a simple viz function
    import matplotlib.pyplot as plt
    viz = None


def vae_loss(reconstruction, target, mu, logvar, beta_kl=BETA_KL, recon_weight=RECONSTRUCTION_WEIGHT):
    """
    Compute VAE loss: reconstruction loss + KL divergence.

    Args:
        reconstruction: Reconstructed image (B, C, H, W)
        target: Target image (B, C, H, W)
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
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    num_batches = 0

    # Get number of samples (approximate - will iterate until we run out)
    # num_samples = triplet_creator.num_rows
    num_samples = NUM_SAMPLES_PER_EPOCH
    num_batches_epoch = max(1, num_samples // batch_size)

    for batch_idx in range(num_batches_epoch):
        try:
            # Create batch of triplets
            batch = triplet_creator.create_batch_triplets(
                batch_size=batch_size,
                anchor_survey="legacysurvey",  # Use legacysurvey as anchor
                use_object_mask=use_object_mask
            )

            # Extract tensors
            ground_truth = batch["ground_truth"]["tensor"].to(device)  # (B, C, H, W)
            cross_instrument = batch["cross_instrument"]["tensor"].to(device)  # Same galaxy, different instrument
            different_galaxy = batch["different_galaxy"]["tensor"].to(device)  # Different galaxy, same instrument

            # Forward pass
            # Use: same_galaxy=cross_instrument (same galaxy, different instrument) -> number encoder
            #      different_galaxy=different_galaxy (different galaxy, same instrument) -> filter encoder
            reconstruction, number_z, filter_z, number_mu, number_logvar, filter_mu, filter_logvar = \
                model(cross_instrument, different_galaxy)

            # Compute loss (combine KL losses from both encoders)
            # Target is ground_truth (the anchor image)
            loss, recon_loss, kl_loss = vae_loss(
                reconstruction,
                ground_truth,
                torch.cat([number_mu, filter_mu], dim=1),
                torch.cat([number_logvar, filter_logvar], dim=1)
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


def validate(model, triplet_creator, device, batch_size=BATCH_SIZE, use_object_mask=USE_OBJECT_MASK):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    num_batches = 0

    num_samples = triplet_creator.num_rows
    num_batches_epoch = max(1, min(100, num_samples // batch_size))  # Limit validation batches

    with torch.no_grad():
        for batch_idx in range(num_batches_epoch):
            try:
                # Create batch of triplets (use test split if available)
                batch = triplet_creator.create_batch_triplets(
                    batch_size=batch_size,
                    anchor_survey="legacysurvey",
                    use_object_mask=use_object_mask
                )

                ground_truth = batch["ground_truth"]["tensor"].to(device)
                cross_instrument = batch["cross_instrument"]["tensor"].to(device)
                different_galaxy = batch["different_galaxy"]["tensor"].to(device)

                reconstruction, number_z, filter_z, number_mu, number_logvar, filter_mu, filter_logvar = \
                    model(cross_instrument, different_galaxy)

                loss, recon_loss, kl_loss = vae_loss(
                    reconstruction,
                    ground_truth,
                    torch.cat([number_mu, filter_mu], dim=1),
                    torch.cat([number_logvar, filter_logvar], dim=1)
                )

                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
                num_batches += 1

            except Exception as e:
                print(f"Error in validation batch {batch_idx}: {e}")
                continue

    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    avg_recon_loss = total_recon_loss / num_batches if num_batches > 0 else float('inf')
    avg_kl_loss = total_kl_loss / num_batches if num_batches > 0 else float('inf')

    return avg_loss, avg_recon_loss, avg_kl_loss


def train(model, triplet_creator, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE,
          plots_dir=PLOTS_DIR, device=None):
    """Main training loop."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # Create output directory
    os.makedirs(plots_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(plots_dir, f'galaxy_vae_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    print(f"Starting training on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Output directory: {run_dir}")

    for epoch in range(num_epochs):
        epoch_start = time.time()
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Train
        train_loss, train_recon, train_kl = train_epoch(
            model, triplet_creator, optimizer, device
        )

        # Validate
        val_loss, val_recon, val_kl = validate(model, triplet_creator, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, KL: {train_kl:.4f})")
        print(f"            - Val Loss: {val_loss:.4f} (Recon: {val_recon:.4f}, KL: {val_kl:.4f})")
        print(f"            - Time: {epoch_time:.1f}s")

        # Save checkpoint
        if (epoch + 1) % SAVE_INTERVAL == 0:
            checkpoint_path = os.path.join(run_dir, f'model_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_path = os.path.join(run_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_loss': best_val_loss,
            }, best_model_path)
        else:
            patience_counter += 1

        # Early stopping
        if EARLY_STOPPING_PATIENCE and patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

        # Visualization
        if (epoch + 1) % VISUALIZATION_INTERVAL == 0:
            try:
                if viz is not None:
                    viz.create_reconstruction_plot(
                        model, triplet_creator, device,
                        save_path=os.path.join(run_dir, f'reconstruction_epoch_{epoch+1}.png'),
                        num_examples=8
                    )
            except Exception as e:
                print(f"Error creating visualization: {e}")

    print(f"\nTraining completed. Best validation loss: {best_val_loss:.4f}")
    print(f"Results saved to: {run_dir}")

    return train_losses, val_losses


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Galaxy VAE")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--data_path", type=str, default=DATA_DIR, help="Path to data directory")
    parser.add_argument("--plots_dir", type=str, default=PLOTS_DIR, help="Directory for plots/checkpoints")
    parser.add_argument("--use_object_mask", action="store_true", default=USE_OBJECT_MASK, help="Use object mask")
    parser.add_argument("--no_object_mask", dest="use_object_mask", action="store_false", help="Don't use object mask")

    args = parser.parse_args()

    # Create triplet creator
    triplet_creator = TripletCreator(
        dataset_path=args.data_path,
        split="train"
    )

    # Create model using NUM_CHANNELS from config
    # The model will use the number of channels defined in config
    model = DoubleEncoderGalaxyVAE(num_channels=NUM_CHANNELS)

    # Train
    train(model, triplet_creator, num_epochs=args.epochs, lr=args.lr, plots_dir=args.plots_dir)
