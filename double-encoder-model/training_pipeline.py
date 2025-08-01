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
