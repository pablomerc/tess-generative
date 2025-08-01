"""
Main Training Pipeline for Double Encoder Model

This script implements the complete training pipeline:
1. Creates triplets using TripletCreator
2. Trains the DoubleEncoderDecoder model
3. Uses reconstruction and KL divergence losses
4. Saves models and visualizations
5. Logs to wandb for experiment tracking
"""

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta
import os
import wandb

# Import our modules
from config import *
from triplet_creation import TripletCreator
from decoder import DoubleEncoderDecoder
from utils import (
    compute_total_loss, save_model, load_model,
    visualize_triplet_reconstruction, visualize_latent_space,
    compute_metrics, print_metrics
)


def create_reconstruction_plot_for_wandb(ground_truth, different_digit, same_digit, reconstruction,
                                       original_labels, class_names, epoch):
    """
    Create reconstruction plot for wandb logging
    """
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))

    for i in range(4):
        # Ground truth (target)
        axes[0, i].imshow(ground_truth[i, 0].cpu(), cmap='gray')
        axes[0, i].set_title(f'Ground Truth\nLabel: {class_names[original_labels[i]]}')
        axes[0, i].axis('off')

        # Different digit (filter encoder input)
        axes[1, i].imshow(different_digit[i, 0].cpu(), cmap='gray')
        axes[1, i].set_title(f'Different Digit\nFilter Encoder Input')
        axes[1, i].axis('off')

        # Same digit (number encoder input)
        axes[2, i].imshow(same_digit[i, 0].cpu(), cmap='gray')
        axes[2, i].set_title(f'Same Digit\nNumber Encoder Input')
        axes[2, i].axis('off')

        # Reconstruction
        axes[3, i].imshow(reconstruction[i, 0].cpu(), cmap='gray')
        axes[3, i].set_title(f'Reconstruction\nEpoch {epoch}')
        axes[3, i].axis('off')

    plt.tight_layout()
    return fig


def create_generation_test_for_wandb(model, triplet_creator, epoch):
    """
    Create generation test plot for wandb logging - testing disentanglement
    Shows how the model can swap number and filter components
    """
    model.eval()

    with torch.no_grad():
        # Create a single triplet for the main example
        (ground_truth, different_digit, same_digit, original_label, different_label,
         ground_truth_rotation, ground_truth_scale, same_digit_rotation, same_digit_scale) = \
            triplet_creator.create_triplet(dataset='test')

        # Create a random image for swapping
        (random_gt, random_diff, random_same, random_label, _, _, _, _, _) = \
            triplet_creator.create_triplet(dataset='test')

        # Move to device
        ground_truth = ground_truth.unsqueeze(0).to(device)  # Add batch dimension
        different_digit = different_digit.unsqueeze(0).to(device)
        same_digit = same_digit.unsqueeze(0).to(device)
        random_gt = random_gt.unsqueeze(0).to(device)

        # 1. Normal reconstruction (baseline)
        (reconstruction, number_z, filter_z, _, _, _, _) = model(same_digit, different_digit)

        # 2. Test 1: Same number encoder input, different filter encoder input
        # Use same_digit for number encoder, random_gt for filter encoder
        (reconstruction_test1, _, _, _, _, _, _) = model(same_digit, random_gt)

        # 3. Test 2: Different number encoder input, same filter encoder input
        # Use random_gt for number encoder, different_digit for filter encoder
        (reconstruction_test2, _, _, _, _, _, _) = model(random_gt, different_digit)

    # Create visualization
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    # 1. Ground truth (target)
    axes[0].imshow(ground_truth[0, 0].cpu(), cmap='gray')
    axes[0].set_title(f'1. Ground Truth\nTarget for reconstruction\nLabel: {triplet_creator.class_names[original_label]}')
    axes[0].axis('off')

    # 2. Normal reconstruction
    axes[1].imshow(reconstruction[0, 0].cpu(), cmap='gray')
    axes[1].set_title(f'2. Normal Reconstruction\nNumber: {triplet_creator.class_names[original_label]}\nFilter: {triplet_creator.class_names[different_label]}')
    axes[1].axis('off')

    # 3. Random image (for swapping)
    axes[2].imshow(random_gt[0, 0].cpu(), cmap='gray')
    axes[2].set_title(f'3. Random Image\nFor swapping components\nLabel: {triplet_creator.class_names[random_label]}')
    axes[2].axis('off')

    # 4. Test 1: Same number, different filter
    axes[3].imshow(reconstruction_test1[0, 0].cpu(), cmap='gray')
    axes[3].set_title(f'4. Same Number, Different Filter\nNumber: {triplet_creator.class_names[original_label]}\nFilter: {triplet_creator.class_names[random_label]}')
    axes[3].axis('off')

    # 5. Test 2: Different number, same filter
    axes[4].imshow(reconstruction_test2[0, 0].cpu(), cmap='gray')
    axes[4].set_title(f'5. Different Number, Same Filter\nNumber: {triplet_creator.class_names[random_label]}\nFilter: {triplet_creator.class_names[different_label]}')
    axes[4].axis('off')

    plt.suptitle(f'Disentanglement Test: Swapping Number and Filter Components (Epoch {epoch})', fontsize=14)
    plt.tight_layout()

    return fig


def create_final_training_curves_plot(train_losses, train_recon_losses, train_kl_losses, train_metrics,
                                    val_losses, val_recon_losses, val_kl_losses, val_metrics):
    """
    Create final training curves plot for wandb logging
    """
    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot losses with distinct colors (log scale for better visualization)
    # Primary y-axis for reconstruction and total losses
    ax1 = axes[0, 0]
    ax1.plot(epochs, train_losses, 'b-', label='Train Total Loss', linewidth=2)
    ax1.plot(epochs, train_recon_losses, 'g-', label='Train Recon Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Total Loss', linewidth=2)
    ax1.plot(epochs, val_recon_losses, 'purple', label='Val Recon Loss', linewidth=2)
    ax1.set_title('Training and Validation Losses (Log Scale)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (log scale)')
    ax1.set_yscale('log')
    ax1.grid(True)
    ax1.legend(loc='upper left')

    # Secondary y-axis for KL losses
    ax2 = ax1.twinx()
    ax2.plot(epochs, train_kl_losses, 'orange', label='Train KL Loss', linewidth=2, linestyle='--')
    ax2.plot(epochs, val_kl_losses, 'brown', label='Val KL Loss', linewidth=2, linestyle='--')
    ax2.set_ylabel('KL Loss (log scale)', color='orange')
    ax2.set_yscale('log')
    ax2.tick_params(axis='y', labelcolor='orange')
    ax2.legend(loc='upper right')

    # Plot MSE (linear scale is fine for MSE)
    train_mse = [m['mse'] for m in train_metrics]
    val_mse = [m['mse'] for m in val_metrics]
    axes[0, 1].plot(epochs, train_mse, 'b-', label='Train MSE', linewidth=2)
    axes[0, 1].plot(epochs, val_mse, 'r-', label='Val MSE', linewidth=2)
    axes[0, 1].set_title('Mean Squared Error')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MSE')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Plot PSNR (linear scale is fine for PSNR)
    train_psnr = [m['psnr'] for m in train_metrics if not np.isinf(m['psnr'])]
    val_psnr = [m['psnr'] for m in val_metrics if not np.isinf(m['psnr'])]
    if train_psnr and val_psnr:
        axes[1, 0].plot(epochs[:len(train_psnr)], train_psnr, 'b-', label='Train PSNR', linewidth=2)
        axes[1, 0].plot(epochs[:len(val_psnr)], val_psnr, 'r-', label='Val PSNR', linewidth=2)
        axes[1, 0].set_title('Peak Signal-to-Noise Ratio')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('PSNR (dB)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

    # Plot latent space std (linear scale is fine)
    train_number_z_std = [m['number_z_std'] for m in train_metrics]
    train_filter_z_std = [m['filter_z_std'] for m in train_metrics]
    val_number_z_std = [m['number_z_std'] for m in val_metrics]
    val_filter_z_std = [m['filter_z_std'] for m in val_metrics]

    axes[1, 1].plot(epochs, train_number_z_std, 'b-', label='Train Number Z std', linewidth=2)
    axes[1, 1].plot(epochs, train_filter_z_std, 'g-', label='Train Filter Z std', linewidth=2)
    axes[1, 1].plot(epochs, val_number_z_std, 'r-', label='Val Number Z std', linewidth=2)
    axes[1, 1].plot(epochs, val_filter_z_std, 'purple', label='Val Filter Z std', linewidth=2)
    axes[1, 1].set_title('Latent Space Standard Deviation')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Standard Deviation')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    return fig


def plot_training_curves(train_losses, train_recon_losses, train_kl_losses, train_metrics,
                        val_losses, val_recon_losses, val_kl_losses, val_metrics, model_folder):
    """
    Plot training curves with both training and validation data
    """
    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(9, 6))

    # Plot losses with distinct colors (log scale for better visualization)
    # Primary y-axis for reconstruction and total losses
    ax1 = axes[0, 0]
    ax1.plot(epochs, train_losses, 'b-', label='Train Total Loss', linewidth=2)
    ax1.plot(epochs, train_recon_losses, 'g-', label='Train Recon Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Total Loss', linewidth=2)
    ax1.plot(epochs, val_recon_losses, 'purple', label='Val Recon Loss', linewidth=2)
    ax1.set_title('Training and Validation Losses (Log Scale)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (log scale)')
    ax1.set_yscale('log')
    ax1.grid(True)
    ax1.legend(loc='upper left')

    # Secondary y-axis for KL losses
    ax2 = ax1.twinx()
    ax2.plot(epochs, train_kl_losses, 'orange', label='Train KL Loss', linewidth=2, linestyle='--')
    ax2.plot(epochs, val_kl_losses, 'brown', label='Val KL Loss', linewidth=2, linestyle='--')
    ax2.set_ylabel('KL Loss (log scale)', color='orange')
    ax2.set_yscale('log')
    ax2.tick_params(axis='y', labelcolor='orange')
    ax2.legend(loc='upper right')

    # Plot MSE (linear scale is fine for MSE)
    train_mse = [m['mse'] for m in train_metrics]
    val_mse = [m['mse'] for m in val_metrics]
    axes[0, 1].plot(epochs, train_mse, 'b-', label='Train MSE', linewidth=2)
    axes[0, 1].plot(epochs, val_mse, 'r-', label='Val MSE', linewidth=2)
    axes[0, 1].set_title('Mean Squared Error')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MSE')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Plot PSNR (linear scale is fine for PSNR)
    train_psnr = [m['psnr'] for m in train_metrics if not np.isinf(m['psnr'])]
    val_psnr = [m['psnr'] for m in val_metrics if not np.isinf(m['psnr'])]
    if train_psnr and val_psnr:
        axes[1, 0].plot(epochs[:len(train_psnr)], train_psnr, 'b-', label='Train PSNR', linewidth=2)
        axes[1, 0].plot(epochs[:len(val_psnr)], val_psnr, 'r-', label='Val PSNR', linewidth=2)
        axes[1, 0].set_title('Peak Signal-to-Noise Ratio')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('PSNR (dB)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

    # Plot latent space std (linear scale is fine)
    train_number_z_std = [m['number_z_std'] for m in train_metrics]
    train_filter_z_std = [m['filter_z_std'] for m in train_metrics]
    val_number_z_std = [m['number_z_std'] for m in val_metrics]
    val_filter_z_std = [m['filter_z_std'] for m in val_metrics]

    axes[1, 1].plot(epochs, train_number_z_std, 'b-', label='Train Number Z std', linewidth=2)
    axes[1, 1].plot(epochs, train_filter_z_std, 'g-', label='Train Filter Z std', linewidth=2)
    axes[1, 1].plot(epochs, val_number_z_std, 'r-', label='Val Number Z std', linewidth=2)
    axes[1, 1].plot(epochs, val_filter_z_std, 'purple', label='Val Filter Z std', linewidth=2)
    axes[1, 1].set_title('Latent Space Standard Deviation')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Standard Deviation')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()

    # Save plot in the existing model folder
    filename = f"training_curves.png"
    filepath = os.path.join(model_folder, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"Training curves saved to: {filepath}")


def train_model(model, triplet_creator, optimizer, num_epochs=NUM_EPOCHS,
                batch_size=BATCH_SIZE, save_interval=SAVE_INTERVAL,
                vis_interval=VISUALIZATION_INTERVAL, device=device, start_epoch=0):
    """
    Main training loop

    Args:
        model: DoubleEncoderDecoder model
        triplet_creator: TripletCreator instance
        optimizer: Optimizer
        num_epochs: Number of training epochs
        batch_size: Batch size
        save_interval: Save model every N epochs
        vis_interval: Create visualizations every N epochs
        device: Device to train on
        start_epoch: Starting epoch (for continuing training)
    """

    # Create model-specific folder for all outputs with consistent timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_folder = f"../figures/{DATASET_TYPE}/double_encoder_model_{DATASET_TYPE}_{timestamp}"
    os.makedirs(model_folder, exist_ok=True)
    print(f"All outputs will be saved to: {model_folder}")
    print(f"Using dataset: {DATASET_TYPE}")

    # Training history
    train_losses = []
    train_recon_losses = []
    train_kl_losses = []
    train_metrics = []

    # Validation history
    val_losses = []
    val_recon_losses = []
    val_kl_losses = []
    val_metrics = []

    # Timing setup
    start_time = time.time()
    epoch_times = []

    print(f"Starting training on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Starting from epoch: {start_epoch}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print("="*60)

    for epoch in range(start_epoch, start_epoch + num_epochs):
        epoch_start = time.time()

        # Set model to training mode
        model.train()

        # Training loop
        epoch_losses = []
        epoch_recon_losses = []
        epoch_kl_losses = []
        epoch_metrics = []

        # Calculate number of batches needed to cover the training dataset
        # MNIST has ~60,000 training samples
        total_train_samples = 60000  # Approximate MNIST training set size
        num_batches = total_train_samples // batch_size
        print(f"Epoch {epoch+1}/{start_epoch + num_epochs}: Processing {num_batches} batches ({total_train_samples} samples)")

        for batch_idx in range(num_batches):
            # Create triplet batch
            (ground_truth, different_digit, same_digit, original_labels, different_labels,
             ground_truth_rotations, ground_truth_scales, same_digit_rotations, same_digit_scales) = \
                triplet_creator.create_batch_triplets(batch_size, dataset='train')

            # Move to device
            ground_truth = ground_truth.to(device)
            different_digit = different_digit.to(device)
            same_digit = same_digit.to(device)
            original_labels = original_labels.to(device)
            different_labels = different_labels.to(device)
            ground_truth_rotations = ground_truth_rotations.to(device)
            ground_truth_scales = ground_truth_scales.to(device)
            same_digit_rotations = same_digit_rotations.to(device)
            same_digit_scales = same_digit_scales.to(device)

            # Forward pass
            optimizer.zero_grad()

            (reconstruction, number_z, filter_z,
             number_mu, number_logvar, filter_mu, filter_logvar) = model(same_digit, different_digit)

            # Compute loss
            total_loss, recon_loss, kl_loss = compute_total_loss(
                reconstruction, ground_truth,
                number_mu, number_logvar, filter_mu, filter_logvar
            )

            # Backward pass
            total_loss.backward()
            optimizer.step()

            # Store losses
            epoch_losses.append(total_loss.item())
            epoch_recon_losses.append(recon_loss.item())
            epoch_kl_losses.append(kl_loss.item())

            # Compute metrics
            metrics = compute_metrics(reconstruction, ground_truth, number_z, filter_z)
            epoch_metrics.append(metrics)

            # Print progress every 50 batches
            if batch_idx % 50 == 0:
                avg_loss = np.mean(epoch_losses[-10:]) if len(epoch_losses) >= 10 else np.mean(epoch_losses)
                print(f"Epoch {epoch+1}/{start_epoch + num_epochs}, Batch {batch_idx}/{num_batches}, "
                      f"Avg Loss: {avg_loss:.4f}")

        # Compute training epoch averages
        avg_epoch_loss = np.mean(epoch_losses)
        avg_epoch_recon_loss = np.mean(epoch_recon_losses)
        avg_epoch_kl_loss = np.mean(epoch_kl_losses)

        # Average training metrics across epoch
        avg_train_metrics = {}
        for key in epoch_metrics[0].keys():
            avg_train_metrics[key] = np.mean([m[key] for m in epoch_metrics])

        # Validation - use proper validation set size
        model.eval()
        val_losses_epoch = []
        val_recon_losses_epoch = []
        val_kl_losses_epoch = []
        val_metrics_epoch = []

        with torch.no_grad():
            # Calculate number of validation batches
            total_val_samples = 10000  # Approximate MNIST test set size
            num_val_batches = total_val_samples // batch_size
            print(f"Validation: Processing {num_val_batches} batches ({total_val_samples} samples)")

            for batch_idx in range(num_val_batches):
                # Create triplet batch for validation
                (ground_truth, different_digit, same_digit, original_labels, different_labels,
                 ground_truth_rotations, ground_truth_scales, same_digit_rotations, same_digit_scales) = \
                    triplet_creator.create_batch_triplets(batch_size, dataset='test')

                # Move to device
                ground_truth = ground_truth.to(device)
                different_digit = different_digit.to(device)
                same_digit = same_digit.to(device)
                original_labels = original_labels.to(device)
                different_labels = different_labels.to(device)
                ground_truth_rotations = ground_truth_rotations.to(device)
                ground_truth_scales = ground_truth_scales.to(device)
                same_digit_rotations = same_digit_rotations.to(device)
                same_digit_scales = same_digit_scales.to(device)

                # Forward pass (no gradients)
                (reconstruction, number_z, filter_z,
                 number_mu, number_logvar, filter_mu, filter_logvar) = model(same_digit, different_digit)

                # Compute loss
                total_loss, recon_loss, kl_loss = compute_total_loss(
                    reconstruction, ground_truth,
                    number_mu, number_logvar, filter_mu, filter_logvar
                )

                # Store validation losses
                val_losses_epoch.append(total_loss.item())
                val_recon_losses_epoch.append(recon_loss.item())
                val_kl_losses_epoch.append(kl_loss.item())

                # Compute validation metrics
                metrics = compute_metrics(reconstruction, ground_truth, number_z, filter_z)
                val_metrics_epoch.append(metrics)

        # Compute validation epoch averages
        avg_val_loss = np.mean(val_losses_epoch)
        avg_val_recon_loss = np.mean(val_recon_losses_epoch)
        avg_val_kl_loss = np.mean(val_kl_losses_epoch)

        # Average validation metrics across epoch
        avg_val_metrics = {}
        for key in val_metrics_epoch[0].keys():
            avg_val_metrics[key] = np.mean([m[key] for m in val_metrics_epoch])

        # Store in history
        train_losses.append(avg_epoch_loss)
        train_recon_losses.append(avg_epoch_recon_loss)
        train_kl_losses.append(avg_epoch_kl_loss)
        train_metrics.append(avg_train_metrics)

        val_losses.append(avg_val_loss)
        val_recon_losses.append(avg_val_recon_loss)
        val_kl_losses.append(avg_val_kl_loss)
        val_metrics.append(avg_val_metrics)

        # Log to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train/total_loss": avg_epoch_loss,
            "train/reconstruction_loss": avg_epoch_recon_loss,
            "train/kl_loss": avg_epoch_kl_loss,
            "val/total_loss": avg_val_loss,
            "val/reconstruction_loss": avg_val_recon_loss,
            "val/kl_loss": avg_val_kl_loss,
            "train/mse": avg_train_metrics.get('mse', 0),
            "train/psnr": avg_train_metrics.get('psnr', 0),
            "train/number_z_std": avg_train_metrics.get('number_z_std', 0),
            "train/filter_z_std": avg_train_metrics.get('filter_z_std', 0),
            "val/mse": avg_val_metrics.get('mse', 0),
            "val/psnr": avg_val_metrics.get('psnr', 0),
            "val/number_z_std": avg_val_metrics.get('number_z_std', 0),
            "val/filter_z_std": avg_val_metrics.get('filter_z_std', 0),
            "learning_rate": optimizer.param_groups[0]['lr']
        }, step=epoch + 1)

        # Calculate timing
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        total_time = time.time() - start_time

        # Estimate remaining time
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = (start_epoch + num_epochs) - (epoch + 1)
        estimated_remaining = avg_epoch_time * remaining_epochs

        # Format time strings
        elapsed_str = str(timedelta(seconds=int(total_time)))
        remaining_str = str(timedelta(seconds=int(estimated_remaining)))

        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{start_epoch + num_epochs} Summary:")
        print(f"  Training - Total Loss: {avg_epoch_loss:.4f}, Recon Loss: {avg_epoch_recon_loss:.4f}, KL Loss: {avg_epoch_kl_loss:.4f}")
        print(f"  Validation - Total Loss: {avg_val_loss:.4f}, Recon Loss: {avg_val_recon_loss:.4f}, KL Loss: {avg_val_kl_loss:.4f}")
        print_metrics(avg_train_metrics, epoch+1, "  Train ")
        print_metrics(avg_val_metrics, epoch+1, "  Val  ")
        print(f"  Time: {epoch_time:.1f}s, Elapsed: {elapsed_str}, ETA: {remaining_str}")
        print("-" * 60)

        # Save model
        if (epoch + 1) % save_interval == 0:
            save_model(model, optimizer, epoch + 1, avg_val_loss, timestamp=timestamp)  # Save based on validation loss

        # Create visualizations and log to wandb
        if (epoch + 1) % vis_interval == 0:
            print("Creating visualizations and logging to wandb...")

            # Set model to eval mode for visualization
            model.eval()
            with torch.no_grad():
                # Create a small batch for reconstruction visualization
                vis_batch_size = 4
                (ground_truth, different_digit, same_digit, original_labels, different_labels,
                 ground_truth_rotations, ground_truth_scales, same_digit_rotations, same_digit_scales) = \
                    triplet_creator.create_batch_triplets(vis_batch_size, dataset='train')

                ground_truth = ground_truth.to(device)
                different_digit = different_digit.to(device)
                same_digit = same_digit.to(device)
                original_labels = original_labels.to(device)

                (reconstruction, number_z, filter_z,
                 number_mu, number_logvar, filter_mu, filter_logvar) = model(same_digit, different_digit)

                # Visualize triplet reconstruction and log to wandb
                reconstruction_fig = create_reconstruction_plot_for_wandb(
                    ground_truth, different_digit, same_digit, reconstruction,
                    original_labels, triplet_creator.class_names, epoch + 1
                )
                wandb.log({"reconstructions": wandb.Image(reconstruction_fig)}, step=epoch + 1)
                plt.close(reconstruction_fig)

                # Save local copy
                visualize_triplet_reconstruction(
                    ground_truth, different_digit, same_digit, reconstruction, epoch + 1, model_folder
                )

                # Create generation test and log to wandb
                generation_fig = create_generation_test_for_wandb(
                    model, triplet_creator, epoch + 1
                )
                wandb.log({"generation_test": wandb.Image(generation_fig)}, step=epoch + 1)
                plt.close(generation_fig)

                # Create larger batch for latent space visualization
                latent_batch_size = 4096  # Much larger for better latent space visualization
                (ground_truth_latent, different_digit_latent, same_digit_latent, original_labels_latent, different_labels_latent,
                 ground_truth_rotations_latent, ground_truth_scales_latent, same_digit_rotations_latent, same_digit_scales_latent) = \
                    triplet_creator.create_batch_triplets(latent_batch_size, dataset='train')

                ground_truth_latent = ground_truth_latent.to(device)
                different_digit_latent = different_digit_latent.to(device)
                same_digit_latent = same_digit_latent.to(device)
                original_labels_latent = original_labels_latent.to(device)
                same_digit_rotations_latent = same_digit_rotations_latent.to(device)

                (reconstruction_latent, number_z_latent, filter_z_latent,
                 number_mu_latent, number_logvar_latent, filter_mu_latent, filter_logvar_latent) = model(same_digit_latent, different_digit_latent)

                # Visualize latent space with larger dataset, including rotation labels
                visualize_latent_space(number_z_latent, filter_z_latent, original_labels_latent, epoch + 1, model_folder, same_digit_rotations_latent)

            # Set back to training mode
            model.train()

    # Training completed
    total_training_time = time.time() - start_time
    print(f"\nTraining completed in {str(timedelta(seconds=int(total_training_time)))}")
    print(f"Average epoch time: {sum(epoch_times)/len(epoch_times):.1f} seconds")

    # Save final model
    save_model(model, optimizer, start_epoch + num_epochs, val_losses[-1], model_name="double_encoder_final", timestamp=timestamp)

    # Plot training curves
    plot_training_curves(train_losses, train_recon_losses, train_kl_losses, train_metrics,
                        val_losses, val_recon_losses, val_kl_losses, val_metrics, model_folder)

    return train_losses, train_recon_losses, train_kl_losses, train_metrics, val_losses, val_recon_losses, val_kl_losses, val_metrics, model_folder


def main():
    """
    Main function to run the training pipeline
    """
    print("Double Encoder Model Training Pipeline")
    print("="*50)

    # Set device
    print(f"Using device: {device}")

    # Log scale configuration
    if SCALE_RANGE[0] == SCALE_RANGE[1] and SCALE_RANGE[0] == 1.0:
        print(f"Scale is set to {SCALE_RANGE}, so no scale transformation will be used")
    else:
        print(f"Scale is set to {SCALE_RANGE}, so there will be scale augmentation")

    # Create triplet creator
    print("Initializing triplet creator...")
    triplet_creator = TripletCreator(dataset_type=DATASET_TYPE)

    # Display dataset information
    triplet_creator.get_dataset_info()

    # Create model
    print("Creating model...")
    model = DoubleEncoderDecoder().to(device)

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Print model summary
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Initialize wandb
    wandb.init(
        project="double-encoder-model",
        name=f"double-encoder-{DATASET_TYPE}-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "model_type": "DoubleEncoderDecoder",
            "dataset_type": DATASET_TYPE,
            "latent_dim": LATENT_DIM,
            "number_encoder_latent_dim": NUMBER_ENCODER_LATENT_DIM,
            "filter_encoder_latent_dim": FILTER_ENCODER_LATENT_DIM,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "beta_kl": BETA_KL,
            "reconstruction_weight": RECONSTRUCTION_WEIGHT,
            "rotation_degrees": ROTATION_DEGREES,
            "rotation_step": ROTATION_STEP,
            "scale_range": SCALE_RANGE,
            "min_rotation_diff": MIN_ROTATION_DIFF,
            "max_rotation_diff": MAX_ROTATION_DIFF,
            "min_scale_diff": MIN_SCALE_DIFF,
            "save_interval": SAVE_INTERVAL,
            "visualization_interval": VISUALIZATION_INTERVAL,
            "device": device
        }
    )

    # Log model architecture to wandb
    wandb.watch(model, log="all")

    # Check if we should load a pre-trained model
    load_pretrained = False  # Set to True to continue training
    start_epoch = 0  # Default starting epoch

    if load_pretrained:
        # Specify the path to your pre-trained model
        # /Users/pablom.perez/Desktop/MIT-PhD-macbook/AstroAI-Code/tess-generative/models/double_encoder_model_fashion_mnist_20250729_163634/double_encoder_epoch_30.pth
        # pretrained_path = "../models/double_encoder_model_fashion_mnist_20250729_165755/double_encoder_epoch_60.pth"
        pretrained_path = "../models/double_encoder_model_mnist_20250801_172221/double_encoder_final_epoch_50.pth"
        if os.path.exists(pretrained_path):
            start_epoch, _ = load_model(model, optimizer, pretrained_path)
            print(f"Loaded pre-trained model, starting from epoch {start_epoch}")
            print(f"Continuing training for {NUM_EPOCHS} more epochs...")
        else:
            print(f"Pre-trained model not found at {pretrained_path}")
            print("Starting training from scratch...")
            start_epoch = 0

    # Train the model
    print("\nStarting training...")
    train_losses, train_recon_losses, train_kl_losses, train_metrics, val_losses, val_recon_losses, val_kl_losses, val_metrics, model_folder = train_model(
        model, triplet_creator, optimizer, start_epoch=start_epoch
    )

    print("\nTraining completed successfully!")

    # Test the trained model
    print("\nTesting trained model...")
    test_trained_model(model, triplet_creator, model_folder)

    # Log final training curves to wandb
    final_curves_fig = create_final_training_curves_plot(
        train_losses, train_recon_losses, train_kl_losses, train_metrics,
        val_losses, val_recon_losses, val_kl_losses, val_metrics
    )
    wandb.log({"final_training_curves": wandb.Image(final_curves_fig)}, step=NUM_EPOCHS)
    plt.close(final_curves_fig)

    # Finish wandb run
    wandb.finish()


def test_trained_model(model, triplet_creator, model_folder):
    """
    Test the trained model with some examples
    """
    model.eval()

    with torch.no_grad():
        # Create test triplets
        (ground_truth, different_digit, same_digit, original_labels, different_labels,
         ground_truth_rotations, ground_truth_scales, same_digit_rotations, same_digit_scales) = \
            triplet_creator.create_batch_triplets(8, dataset='test')

        ground_truth = ground_truth.to(device)
        different_digit = different_digit.to(device)
        same_digit = same_digit.to(device)

        # Forward pass
        (reconstruction, number_z, filter_z,
         number_mu, number_logvar, filter_mu, filter_logvar) = model(same_digit, different_digit)

        # Compute metrics
        metrics = compute_metrics(reconstruction, ground_truth, number_z, filter_z)
        print_metrics(metrics, 0, "Test ")

        # Visualize results with distinct filename
        visualize_triplet_reconstruction(
            ground_truth, different_digit, same_digit, reconstruction, "test", model_folder
        )

        # Test generation with different combinations
        print("\nTesting generation with different combinations...")
        test_generation(model, triplet_creator, model_folder)


def test_generation(model, triplet_creator, model_folder):
    """
    Test generation by combining different number and filter encodings
    """
    model.eval()

    with torch.no_grad():
        # Create two different triplets
        (gt1, diff1, same1, labels1, _, _, _, _, _) = triplet_creator.create_batch_triplets(4, dataset='test')
        (gt2, diff2, same2, labels2, _, _, _, _, _) = triplet_creator.create_batch_triplets(4, dataset='test')

        # Move to device
        gt1, diff1, same1 = gt1.to(device), diff1.to(device), same1.to(device)
        gt2, diff2, same2 = gt2.to(device), diff2.to(device), same2.to(device)

        # Get encodings
        number_z1, filter_z1, _, _, _, _ = model.encode_only(same1, diff1)
        number_z2, filter_z2, _, _, _, _ = model.encode_only(same2, diff2)

        # Generate combinations
        # Combination 1: number from triplet 1, filter from triplet 2
        combination1 = model.decode_only(number_z1, filter_z2)

        # Combination 2: number from triplet 2, filter from triplet 1
        combination2 = model.decode_only(number_z2, filter_z1)

        # Visualize combinations
        fig, axes = plt.subplots(3, 4, figsize=(8, 6))

        for i in range(4):
            # Original reconstructions
            axes[0, i].imshow(gt1[i, 0].cpu(), cmap='gray')
            axes[0, i].set_title(f'Original 1\nLabel: {triplet_creator.class_names[labels1[i]]}')
            axes[0, i].axis('off')

            axes[1, i].imshow(gt2[i, 0].cpu(), cmap='gray')
            axes[1, i].set_title(f'Original 2\nLabel: {triplet_creator.class_names[labels2[i]]}')
            axes[1, i].axis('off')

            # Combinations
            axes[2, i].imshow(combination1[i, 0].cpu(), cmap='gray')
            axes[2, i].set_title(f'Combination\nNumber: {triplet_creator.class_names[labels1[i]]}, Filter: {triplet_creator.class_names[labels2[i]]}')
            axes[2, i].axis('off')

        plt.suptitle('Generation Test: Combining Different Number and Filter Encodings')
        plt.tight_layout()

        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"generation_test_{timestamp}.png"
        filepath = os.path.join(model_folder, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.show()

        print(f"Generation test saved to: {filepath}")


if __name__ == "__main__":
    main()
