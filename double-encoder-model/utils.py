"""
Utility functions for Double Encoder Model

This module contains:
1. Loss functions (reconstruction loss, KL divergence)
2. Metrics and evaluation functions
3. Visualization functions
4. Model saving/loading utilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from config import *


def reconstruction_loss(reconstruction, target):
    """
    Compute reconstruction loss (MSE)

    Args:
        reconstruction: Reconstructed image
        target: Target image

    Returns:
        torch.Tensor: Reconstruction loss
    """
    #Note: alternatives
    # Option 1: Just BCE
    # recon_loss = F.binary_cross_entropy(reconstruction, target, reduction='sum')

    # # Option 2: Combined
    # mse_loss = F.mse_loss(reconstruction, target, reduction='sum')
    # bce_loss = F.binary_cross_entropy(reconstruction, target, reduction='sum')
    # recon_loss = 0.5 * mse_loss + 0.5 * bce_loss
    return F.mse_loss(reconstruction, target, reduction='sum')


def kl_divergence_loss(mu, logvar):
    """
    Compute KL divergence loss for VAE

    Args:
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution

    Returns:
        torch.Tensor: KL divergence loss
    """
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return kl


def compute_total_loss(reconstruction, target, number_mu, number_logvar,
                      filter_mu, filter_logvar, beta=BETA_KL,
                      reconstruction_weight=RECONSTRUCTION_WEIGHT):
    """
    Compute total loss combining reconstruction and KL divergence

    Args:
        reconstruction: Reconstructed image
        target: Target image
        number_mu, number_logvar: Number encoder parameters
        filter_mu, filter_logvar: Filter encoder parameters
        beta: Weight for KL divergence
        reconstruction_weight: Weight for reconstruction loss

    Returns:
        tuple: (total_loss, recon_loss, weighted_kl_loss)
    """
    # Reconstruction loss
    recon_loss = reconstruction_loss(reconstruction, target)

    # KL divergence losses
    number_kl = kl_divergence_loss(number_mu, number_logvar)
    filter_kl = kl_divergence_loss(filter_mu, filter_logvar)
    total_kl = number_kl + filter_kl

    # Weighted KL loss for tracking
    weighted_kl = beta * total_kl

    # Total loss
    total_loss = reconstruction_weight * recon_loss + weighted_kl

    return total_loss, recon_loss, weighted_kl


def save_model(model, optimizer, epoch, loss, model_name="double_encoder", save_dir="../models", timestamp=None):
    """
    Save model checkpoint with comprehensive metadata

    Args:
        model: The model to save
        optimizer: The optimizer state
        epoch: Current epoch
        loss: Current loss value
        model_name: Name for the model file
        save_dir: Directory to save models in
        timestamp: Timestamp to use (if None, generate new one)
    """
    # Use provided timestamp or create new one
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create timestamped subfolder in models directory
    model_subfolder = f"double_encoder_model_{DATASET_TYPE}_{timestamp}"
    model_path = os.path.join(save_dir, DATASET_TYPE, model_subfolder)
    os.makedirs(model_path, exist_ok=True)

    # Save model state
    model_filename = f"{model_name}_epoch_{epoch}.pth"
    model_filepath = os.path.join(model_path, model_filename)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': {
            'DATASET_TYPE': DATASET_TYPE,
            'BETA_KL': BETA_KL,
            'RECONSTRUCTION_WEIGHT': RECONSTRUCTION_WEIGHT,
            'LEARNING_RATE': LEARNING_RATE,
            'BATCH_SIZE': BATCH_SIZE,
            'NUM_EPOCHS': NUM_EPOCHS,
            'LATENT_DIM': LATENT_DIM,
            'MIN_ROTATION_DIFF': MIN_ROTATION_DIFF,
            'SAVE_INTERVAL': SAVE_INTERVAL,
            'VISUALIZATION_INTERVAL': VISUALIZATION_INTERVAL
        }
    }

    torch.save(checkpoint, model_filepath)
    print(f"Model saved to: {model_filepath}")

    return model_path  # Return the path for potential use by other functions


def load_model(model, optimizer, filepath):
    """
    Load model checkpoint

    Args:
        model: Model to load state into
        optimizer: Optimizer to load state into
        filepath: Path to checkpoint file

    Returns:
        tuple: (epoch, loss)
    """
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    print(f"Model loaded from: {filepath}")
    print(f"Epoch: {epoch}, Loss: {loss:.4f}")

    return epoch, loss


def visualize_triplet_reconstruction(ground_truth, different_digit, same_digit,
                                   reconstruction, epoch, save_dir="../figures"):
    """
    Visualize triplet and reconstruction

    Args:
        ground_truth: Ground truth image
        different_digit: Different digit with same augmentation
        same_digit: Same digit with different augmentation
        reconstruction: Reconstructed image
        epoch: Current epoch (can be int or string like "test")
        save_dir: Directory to save figure
    """
    os.makedirs(save_dir, exist_ok=True)

    # Convert to numpy for plotting
    ground_truth_np = ground_truth.detach().cpu().numpy()
    different_digit_np = different_digit.detach().cpu().numpy()
    same_digit_np = same_digit.detach().cpu().numpy()
    reconstruction_np = reconstruction.detach().cpu().numpy()

    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Plot first batch item
    for i in range(4):
        # Row 1: Inputs
        axes[0, i].imshow(ground_truth_np[i, 0], cmap='gray')
        axes[0, i].set_title(f'Ground Truth {i+1}')
        axes[0, i].axis('off')

        # Row 2: Reconstruction
        axes[1, i].imshow(reconstruction_np[i, 0], cmap='gray')
        axes[1, i].set_title(f'Reconstruction {i+1}')
        axes[1, i].axis('off')

    # Handle both numeric epochs and string labels
    if isinstance(epoch, str):
        title_epoch = epoch.capitalize()
        filename_epoch = epoch
    else:
        title_epoch = f"Epoch {epoch}"
        filename_epoch = f"epoch_{epoch}"

    plt.suptitle(f'Triplet Reconstruction - {title_epoch}')
    plt.tight_layout()

    # Save figure with appropriate filename
    filename = f"triplet_reconstruction_{filename_epoch}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Visualization saved to: {filepath}")


def visualize_latent_space(number_z, filter_z, labels, epoch, save_dir="../figures", rotation_labels=None, class_names=None):
    """
    Visualize latent space with t-SNE

    Args:
        number_z: Number encoder latent representations
        filter_z: Filter encoder latent representations
        labels: Digit labels
        epoch: Current epoch
        save_dir: Directory to save figure
        rotation_labels: Rotation labels for coloring (optional)
        class_names: List of class names for the labels (optional)
    """
    try:
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
    except ImportError:
        print("sklearn not available, skipping latent space visualization")
        return

    os.makedirs(save_dir, exist_ok=True)

    # Convert to numpy
    number_z_np = number_z.detach().cpu().numpy()
    filter_z_np = filter_z.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    if rotation_labels is not None:
        rotation_labels_np = rotation_labels.detach().cpu().numpy()
    else:
        rotation_labels_np = None

    # Use PCA for dimensionality reduction if latent dim > 2
    if number_z_np.shape[1] > 2:
        pca = PCA(n_components=2)
        number_z_2d = pca.fit_transform(number_z_np)
        filter_z_2d = pca.fit_transform(filter_z_np)
    else:
        number_z_2d = number_z_np
        filter_z_2d = filter_z_np

    # Determine label type for colorbar
    if class_names is not None:
        label_type = "Class"
        # Create custom colorbar with class names
        unique_labels = np.unique(labels_np)
        colorbar_ticks = unique_labels
        colorbar_labels = [class_names[label] for label in unique_labels]
    else:
        label_type = "Digit"
        colorbar_ticks = None
        colorbar_labels = None

    # Determine subplot layout based on whether rotation labels are provided
    if rotation_labels_np is not None:
        # 2x2 subplot: digit coloring and rotation coloring for both encoders
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Row 1: Digit coloring
        # Number encoder - digit coloring
        scatter1 = axes[0, 0].scatter(number_z_2d[:, 0], number_z_2d[:, 1], c=labels_np, cmap='tab10', s=20, alpha=0.6)
        axes[0, 0].set_title(f'Number Encoder - {label_type} Coloring - Epoch {epoch}')
        axes[0, 0].set_xlabel('Component 1')
        axes[0, 0].set_ylabel('Component 2')
        cbar1 = plt.colorbar(scatter1, ax=axes[0, 0], label=label_type)
        if colorbar_labels is not None:
            cbar1.set_ticks(colorbar_ticks)
            cbar1.set_ticklabels(colorbar_labels)

        # Filter encoder - digit coloring
        scatter2 = axes[0, 1].scatter(filter_z_2d[:, 0], filter_z_2d[:, 1], c=labels_np, cmap='tab10', s=20, alpha=0.6)
        axes[0, 1].set_title(f'Filter Encoder - {label_type} Coloring - Epoch {epoch}')
        axes[0, 1].set_xlabel('Component 1')
        axes[0, 1].set_ylabel('Component 2')
        cbar2 = plt.colorbar(scatter2, ax=axes[0, 1], label=label_type)
        if colorbar_labels is not None:
            cbar2.set_ticks(colorbar_ticks)
            cbar2.set_ticklabels(colorbar_labels)

        # Row 2: Rotation coloring
        # Number encoder - rotation coloring
        scatter3 = axes[1, 0].scatter(number_z_2d[:, 0], number_z_2d[:, 1], c=rotation_labels_np, cmap='viridis', s=20, alpha=0.6)
        axes[1, 0].set_title(f'Number Encoder - Rotation Coloring - Epoch {epoch}')
        axes[1, 0].set_xlabel('Component 1')
        axes[1, 0].set_ylabel('Component 2')
        plt.colorbar(scatter3, ax=axes[1, 0], label='Rotation Angle')

        # Filter encoder - rotation coloring
        scatter4 = axes[1, 1].scatter(filter_z_2d[:, 0], filter_z_2d[:, 1], c=rotation_labels_np, cmap='viridis', s=20, alpha=0.6)
        axes[1, 1].set_title(f'Filter Encoder - Rotation Coloring - Epoch {epoch}')
        axes[1, 1].set_xlabel('Component 1')
        axes[1, 1].set_ylabel('Component 2')
        plt.colorbar(scatter4, ax=axes[1, 1], label='Rotation Angle')

    else:
        # Original 1x2 subplot: only digit coloring
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes = axes.reshape(1, -1)  # Make it 2D for consistent indexing

        # Plot number encoder latent space
        scatter1 = axes[0, 0].scatter(number_z_2d[:, 0], number_z_2d[:, 1], c=labels_np, cmap='tab10', s=20, alpha=0.6)
        axes[0, 0].set_title(f'Number Encoder Latent Space - Epoch {epoch}')
        axes[0, 0].set_xlabel('Component 1')
        axes[0, 0].set_ylabel('Component 2')
        cbar1 = plt.colorbar(scatter1, ax=axes[0, 0], label=label_type)
        if colorbar_labels is not None:
            cbar1.set_ticks(colorbar_ticks)
            cbar1.set_ticklabels(colorbar_labels)

        # Plot filter encoder latent space
        scatter2 = axes[0, 1].scatter(filter_z_2d[:, 0], filter_z_2d[:, 1], c=labels_np, cmap='tab10', s=20, alpha=0.6)
        axes[0, 1].set_title(f'Filter Encoder Latent Space - Epoch {epoch}')
        axes[0, 1].set_xlabel('Component 1')
        axes[0, 1].set_ylabel('Component 2')
        cbar2 = plt.colorbar(scatter2, ax=axes[0, 1], label=label_type)
        if colorbar_labels is not None:
            cbar2.set_ticks(colorbar_ticks)
            cbar2.set_ticklabels(colorbar_labels)

    plt.tight_layout()

    # Save figure
    filename = f"latent_space_epoch_{epoch}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Latent space visualization saved to: {filepath}")


def compute_metrics(reconstruction, target, number_z, filter_z):
    """
    Compute various metrics for evaluation

    Args:
        reconstruction: Reconstructed image
        target: Target image
        number_z: Number encoder latent representations
        filter_z: Filter encoder latent representations

    Returns:
        dict: Dictionary of metrics
    """
    # Reconstruction metrics
    mse = F.mse_loss(reconstruction, target)
    mae = F.l1_loss(reconstruction, target)

    # Latent space metrics
    number_z_std = torch.std(number_z, dim=0).mean()
    filter_z_std = torch.std(filter_z, dim=0).mean()

    # PSNR (Peak Signal-to-Noise Ratio)
    mse_val = mse.item()
    if mse_val > 0:
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    else:
        psnr = torch.tensor(float('inf'))

    metrics = {
        'mse': mse.item(),
        'mae': mae.item(),
        'psnr': psnr.item() if not torch.isinf(psnr) else float('inf'),
        'number_z_std': number_z_std.item(),
        'filter_z_std': filter_z_std.item(),
    }

    return metrics


def print_metrics(metrics, epoch, prefix=""):
    """
    Print metrics in a formatted way

    Args:
        metrics: Dictionary of metrics
        epoch: Current epoch
        prefix: Prefix for the print statement
    """
    print(f"{prefix}Epoch {epoch} - "
          f"MSE: {metrics['mse']:.6f}, "
          f"MAE: {metrics['mae']:.6f}, "
          f"PSNR: {metrics['psnr']:.2f}, "
          f"Number Z std: {metrics['number_z_std']:.4f}, "
          f"Filter Z std: {metrics['filter_z_std']:.4f}")


def test_utils():
    """Test utility functions"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create dummy data
    batch_size = 4
    reconstruction = torch.randn(batch_size, 1, 28, 28).to(device)
    target = torch.randn(batch_size, 1, 28, 28).to(device)
    number_z = torch.randn(batch_size, NUMBER_ENCODER_LATENT_DIM).to(device)
    filter_z = torch.randn(batch_size, FILTER_ENCODER_LATENT_DIM).to(device)
    number_mu = torch.randn(batch_size, NUMBER_ENCODER_LATENT_DIM).to(device)
    number_logvar = torch.randn(batch_size, NUMBER_ENCODER_LATENT_DIM).to(device)
    filter_mu = torch.randn(batch_size, FILTER_ENCODER_LATENT_DIM).to(device)
    filter_logvar = torch.randn(batch_size, FILTER_ENCODER_LATENT_DIM).to(device)

    # Test loss functions
    print("Testing loss functions...")
    recon_loss = reconstruction_loss(reconstruction, target)
    number_kl = kl_divergence_loss(number_mu, number_logvar)
    filter_kl = kl_divergence_loss(filter_mu, filter_logvar)
    total_loss, _, _ = compute_total_loss(reconstruction, target, number_mu, number_logvar,
                                         filter_mu, filter_logvar)

    print(f"Reconstruction loss: {recon_loss:.4f}")
    print(f"Number KL: {number_kl:.4f}")
    print(f"Filter KL: {filter_kl:.4f}")
    print(f"Total loss: {total_loss:.4f}")

    # Test metrics
    print("\nTesting metrics...")
    metrics = compute_metrics(reconstruction, target, number_z, filter_z)
    print_metrics(metrics, 1, "Test ")

    print("\nAll utility tests passed!")


if __name__ == "__main__":
    test_utils()
