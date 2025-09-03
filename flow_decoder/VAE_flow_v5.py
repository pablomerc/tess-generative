# -*- coding: utf-8 -*-
"""Double Encoder Flow Matching v5 - Using Double Encoder with Flow Matching Decoder

This module implements a model that combines:
1. Double Encoder architecture (NumberEncoder + FilterEncoder) from the double-encoder-model
2. Flow matching decoder with U-Net velocity field
3. Triplet-based training using TripletCreator

The model learns to disentangle digit identity (number encoder) from augmentation style (filter encoder)
and uses flow matching for high-quality generation.

DATASET OPTIONS:
- Set dataset_type = 'mnist' for vanilla MNIST
- Set dataset_type = 'fashion_mnist' for Fashion MNIST
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta
from torch.utils.data import DataLoader
import wandb
import os
import sys

# Add double-encoder-model to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'double-encoder-model'))

# Import our custom architecture
from double_encoder_flow_architecture import DoubleEncoderFlowMatching

# Import triplet creation and utilities
from triplet_creation import TripletCreator
from utils import compute_metrics, print_metrics

# Import config

def normalize_to_flow_range(x):
    """Convert from [0,1] to [-1,1] range for flow matching"""
    return 2.0 * x - 1.0

def to_visualization_range(x):
    """Convert from [-1,1] to [0,1] range for visualization"""
    return (x + 1.0) / 2.0
from config import *
DATASET_TYPE='mnist'

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

# Check if CUDA is available and print GPU info
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
elif torch.backends.mps.is_available():
    print("Using Apple Silicon GPU (MPS)")
else:
    print("Using CPU - training will be slower")

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


def debug_data_normalization(triplet_creator, device):
    """
    Debug function to check data normalization and ranges
    """
    print("\n" + "="*60)
    print("DEBUGGING DATA NORMALIZATION")
    print("="*60)

    # Create a single triplet
    (ground_truth, different_digit, same_digit, original_image, original_label, different_label,
     ground_truth_rotation, ground_truth_scale, same_digit_rotation, same_digit_scale) = \
        triplet_creator.create_triplet(dataset='train')

    # print(f"Original PIL image type: {type(original_image)}")
    # print(f"Original PIL image size: {original_image.size}")
    # print(f"Original PIL image mode: {original_image.mode}")

    # Check tensor properties
    print(f"\nTensor properties:")
    print(f"Ground truth shape: {ground_truth.shape}")
    print(f"Ground truth dtype: {ground_truth.dtype}")
    print(f"Ground truth min: {ground_truth.min().item():.4f}")
    print(f"Ground truth max: {ground_truth.max().item():.4f}")
    print(f"Ground truth mean: {ground_truth.mean().item():.4f}")
    print(f"Ground truth std: {ground_truth.std().item():.4f}")

    # Apply transformation to normalize flow range
    print('\n Transforming to [-1, 1] range \n')
    ground_truth = ground_truth.to(device)
    ground_truth = normalize_to_flow_range(ground_truth)
    print(f"Ground truth min: {ground_truth.min().item():.4f}")
    print(f"Ground truth max: {ground_truth.max().item():.4f}")
    print(f"Ground truth mean: {ground_truth.mean().item():.4f}")
    print(f"Ground truth std: {ground_truth.std().item():.4f}")

    print("="*60)


def create_reconstruction_plot_for_wandb(ground_truth, different_digit, same_digit, reconstruction,
                                       original_labels, class_names, epoch):
    """
    Create reconstruction plot for wandb logging
    """
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))

    for i in range(4):
        # Ground truth (target)
        axes[0, i].imshow(to_visualization_range(ground_truth[i, 0]).cpu(), cmap='gray')
        axes[0, i].set_title(f'Ground Truth\nLabel: {class_names[original_labels[i]]}')
        axes[0, i].axis('off')

        # Different digit (filter encoder input)
        axes[1, i].imshow(to_visualization_range(different_digit[i, 0]).cpu(), cmap='gray')
        axes[1, i].set_title(f'Different Digit\nFilter Encoder Input')
        axes[1, i].axis('off')

        # Same digit (number encoder input)
        axes[2, i].imshow(to_visualization_range(same_digit[i, 0]).cpu(), cmap='gray')
        axes[2, i].set_title(f'Same Digit\nNumber Encoder Input')
        axes[2, i].axis('off')

        # Reconstruction
        axes[3, i].imshow(to_visualization_range(reconstruction[i, 0]).cpu(), cmap='gray')
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

    # Create visualization for 4 different examples
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))

    for example_idx in range(4):
        with torch.no_grad():
            # Create two different triplets
            (gt1, diff1, same1, _, orig_label1, diff_label1, _, _, _, _) = \
                triplet_creator.create_triplet(dataset='test')
            (gt2, diff2, same2, _, orig_label2, diff_label2, _, _, _, _) = \
                triplet_creator.create_triplet(dataset='test')

            # Move to device
            gt1 = gt1.unsqueeze(0).to(device)
            diff1 = diff1.unsqueeze(0).to(device)
            same1 = same1.unsqueeze(0).to(device)
            gt2 = gt2.unsqueeze(0).to(device)
            diff2 = diff2.unsqueeze(0).to(device)
            same2 = same2.unsqueeze(0).to(device)

            # 1. Normal reconstruction (baseline)
            reconstruction1 = model.reconstruct(same1, diff1)

            # 2. Test: Same number encoder input, different filter encoder input
            reconstruction_test1 = model.reconstruct(same1, diff2)

            # 3. Test: Different number encoder input, same filter encoder input
            reconstruction_test2 = model.reconstruct(same2, diff1)

        # Create visualization for this example
        # 1. Ground truth 1
        axes[example_idx, 0].imshow(to_visualization_range(gt1[0, 0]).cpu(), cmap='gray')
        axes[example_idx, 0].set_title(f'GT1\nLabel: {triplet_creator.class_names[orig_label1]}')
        axes[example_idx, 0].axis('off')

        # 2. Normal reconstruction
        axes[example_idx, 1].imshow(to_visualization_range(reconstruction1[0, 0]).cpu(), cmap='gray')
        axes[example_idx, 1].set_title(f'Normal Recon\nNum: {triplet_creator.class_names[orig_label1]}\nFilt: {triplet_creator.class_names[diff_label1]}')
        axes[example_idx, 1].axis('off')

        # 3. Ground truth 2
        axes[example_idx, 2].imshow(to_visualization_range(gt2[0, 0]).cpu(), cmap='gray')
        axes[example_idx, 2].set_title(f'GT2\nLabel: {triplet_creator.class_names[orig_label2]}')
        axes[example_idx, 2].axis('off')

        # 4. Test 1: Same number, different filter
        axes[example_idx, 3].imshow(to_visualization_range(reconstruction_test1[0, 0]).cpu(), cmap='gray')
        axes[example_idx, 3].set_title(f'Same Num, Diff Filt\nNum: {triplet_creator.class_names[orig_label1]}\nFilt: {triplet_creator.class_names[diff_label2]}')
        axes[example_idx, 3].axis('off')

        # 5. Test 2: Different number, same filter
        axes[example_idx, 4].imshow(to_visualization_range(reconstruction_test2[0, 0]).cpu(), cmap='gray')
        axes[example_idx, 4].set_title(f'Diff Num, Same Filt\nNum: {triplet_creator.class_names[orig_label2]}\nFilt: {triplet_creator.class_names[diff_label1]}')
        axes[example_idx, 4].axis('off')

    plt.suptitle(f'Disentanglement Test: Swapping Number and Filter Components (Epoch {epoch})', fontsize=16)
    plt.tight_layout()

    return fig


def train_double_encoder_flow(model, triplet_creator, num_epochs=1, lr=1e-4, plots_dir="reconstruction_plots"):
    """Training function for Double Encoder Flow Matching model"""

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Add learning rate scheduler for stability
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Add gradient clipping to prevent explosion - more aggressive for flow matching
    max_grad_norm = 0.1  # Much more aggressive clipping for stability

    # Training history
    train_losses = []
    test_losses = []

    # Timing setup
    start_time = time.time()

    print(f"Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Training phase
        model.train()
        train_loss = 0
        num_batches = 0

        # Calculate number of batches needed to cover the training dataset
        total_train_samples = 60000  # Approximate MNIST training set size
        num_batches_epoch = total_train_samples // BATCH_SIZE
        print(f"Epoch {epoch+1}/{num_epochs}: Processing {num_batches_epoch} batches ({total_train_samples} samples)")

        for batch_idx in range(num_batches_epoch):
            # Create triplet batch
            (ground_truth, different_digit, same_digit, original_labels, different_labels,
             ground_truth_rotations, ground_truth_scales, same_digit_rotations, same_digit_scales) = \
                triplet_creator.create_batch_triplets(BATCH_SIZE, dataset='train')

            # Move to device
            ground_truth = ground_truth.to(device)
            ground_truth = normalize_to_flow_range(ground_truth)
            different_digit = different_digit.to(device)
            different_digit = normalize_to_flow_range(different_digit)
            same_digit = same_digit.to(device)
            same_digit = normalize_to_flow_range(same_digit)
            original_labels = original_labels.to(device)
            different_labels = different_labels.to(device)

            optimizer.zero_grad()

            # Forward pass - get flow loss
            flow_loss, number_z, filter_z, number_mu, number_logvar, filter_mu, filter_logvar = \
                model.get_flow_loss(same_digit, different_digit, ground_truth)

            # Check for loss explosion BEFORE backward pass
            if torch.isnan(flow_loss) or torch.isinf(flow_loss) or flow_loss > 1000:
                print(f"WARNING: Loss explosion detected at batch index {batch_idx}. Loss: {flow_loss.item()}")
                print("Skipping this batch and continuing...")
                continue

            # Backward pass
            flow_loss.backward()

            # Check gradients BEFORE clipping
            total_grad_norm = 0
            max_grad = 0
            nan_grads = 0
            inf_grads = 0

            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.data.norm(2)
                    total_grad_norm += grad_norm.item() ** 2
                    max_grad = max(max_grad, param.grad.data.abs().max().item())

                    if torch.isnan(param.grad).any():
                        nan_grads += 1
                        print(f"NaN gradient in {name}")
                    if torch.isinf(param.grad).any():
                        inf_grads += 1
                        print(f"Inf gradient in {name}")

            total_grad_norm = total_grad_norm ** 0.5

            # Log gradient statistics every few batches
            if batch_idx % 50 == 0:
                print(f"Batch {batch_idx}: Loss={flow_loss.item():.4f}, "
                      f"GradNorm={total_grad_norm:.4f}, MaxGrad={max_grad:.4f}, "
                      f"NaN grads: {nan_grads}, Inf grads: {inf_grads}")

            # Check for gradient explosion
            if total_grad_norm > 100 or max_grad > 50 or nan_grads > 0 or inf_grads > 0:
                print(f"WARNING: Gradient explosion at batch {batch_idx}!")
                print(f"  Total grad norm: {total_grad_norm:.4f}")
                print(f"  Max gradient: {max_grad:.4f}")
                print(f"  NaN gradients: {nan_grads}")
                print(f"  Inf gradients: {inf_grads}")
                print("Skipping this batch and continuing...")
                optimizer.zero_grad()  # Clear gradients
                continue

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

            # Accumulate losses
            train_loss += flow_loss.item()
            num_batches += 1

            if batch_idx % 50 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}/{num_batches_epoch}, '
                      f'Flow Loss: {flow_loss.item():.4f}')

        # Calculate average loss (only for valid batches)
        if num_batches > 0:
            avg_train_loss = train_loss / num_batches
        else:
            avg_train_loss = float('inf')

        # Check for epoch-level explosion
        if avg_train_loss > 1000:
            print(f"WARNING: Epoch {epoch+1} has very high loss: {avg_train_loss:.4f}")
            print("Consider reducing learning rate or checking model stability")

        # Validation phase
        model.eval()
        test_loss = 0
        valid_test_batches = 0

        with torch.no_grad():
            # Calculate number of validation batches
            total_val_samples = 10000  # Approximate MNIST test set size
            num_val_batches = total_val_samples // BATCH_SIZE
            print(f"Validation: Processing {num_val_batches} batches ({total_val_samples} samples)")

            for batch_idx in range(num_val_batches):
                # Create triplet batch for validation
                (ground_truth, different_digit, same_digit, original_labels, different_labels,
                 ground_truth_rotations, ground_truth_scales, same_digit_rotations, same_digit_scales) = \
                    triplet_creator.create_batch_triplets(BATCH_SIZE, dataset='test')

                # Move to device
                ground_truth = ground_truth.to(device)
                ground_truth = normalize_to_flow_range(ground_truth)
                different_digit = different_digit.to(device)
                different_digit = normalize_to_flow_range(different_digit)
                same_digit = same_digit.to(device)
                same_digit = normalize_to_flow_range(same_digit)
                original_labels = original_labels.to(device)
                different_labels = different_labels.to(device)

                # Forward pass (no gradients)
                flow_loss, number_z, filter_z, number_mu, number_logvar, filter_mu, filter_logvar = \
                    model.get_flow_loss(same_digit, different_digit, ground_truth)

                # Check for loss explosion in validation
                if not (torch.isnan(flow_loss) or torch.isinf(flow_loss) or flow_loss > 1000):
                    test_loss += flow_loss.item()
                    valid_test_batches += 1

        # Calculate average test loss
        if valid_test_batches > 0:
            avg_test_loss = test_loss / valid_test_batches
        else:
            avg_test_loss = float('inf')

        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)

        # Update learning rate scheduler
        scheduler.step(avg_train_loss)

        # Log to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "test_loss": avg_test_loss,
            "learning_rate": lr,
            "flow_loss": flow_loss.item() if not (torch.isnan(flow_loss) or torch.isinf(flow_loss)) else 0
        }, step=epoch + 1)

        # Generate and log reconstruction plots every 2 epochs
        if (epoch + 1) % 5 == 0:
            print(f"Creating reconstruction plot for epoch {epoch + 1}...")
            try:
                # Set model to eval mode for visualization
                model.eval()
                with torch.no_grad():
                    # Create a small batch for reconstruction visualization
                    vis_batch_size = 4
                    (ground_truth, different_digit, same_digit, original_labels, different_labels,
                     ground_truth_rotations, ground_truth_scales, same_digit_rotations, same_digit_scales) = \
                        triplet_creator.create_batch_triplets(vis_batch_size, dataset='train')

                    ground_truth = ground_truth.to(device)
                    ground_truth = normalize_to_flow_range(ground_truth)
                    different_digit = different_digit.to(device)
                    different_digit = normalize_to_flow_range(different_digit)
                    same_digit = same_digit.to(device)
                    same_digit = normalize_to_flow_range(same_digit)
                    original_labels = original_labels.to(device)

                    # Get reconstruction
                    reconstruction = model.reconstruct(same_digit, different_digit)

                    # Visualize triplet reconstruction and log to wandb
                    reconstruction_fig = create_reconstruction_plot_for_wandb(
                        ground_truth, different_digit, same_digit, reconstruction,
                        original_labels, triplet_creator.class_names, epoch + 1
                    )
                    wandb.log({"reconstructions": wandb.Image(reconstruction_fig)}, step=epoch + 1)
                    plt.close(reconstruction_fig)

                    # Create generation test and log to wandb
                    generation_fig = create_generation_test_for_wandb(
                        model, triplet_creator, epoch + 1
                    )
                    wandb.log({"generation_test": wandb.Image(generation_fig)}, step=epoch + 1)
                    plt.close(generation_fig)

                print(f"Successfully logged reconstruction plot for epoch {epoch + 1}")
            except Exception as e:
                print(f"Error creating reconstruction plot: {e}")
                import traceback
                traceback.print_exc()

        # Timing
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time

        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Test Loss: {avg_test_loss:.4f}, '
              f'Time: {epoch_time:.1f}s, '
              f'Total: {str(timedelta(seconds=int(total_time)))}')

    print(f"Training completed in {str(timedelta(seconds=int(time.time() - start_time)))}")

    # Save the trained model
    model_save_path = os.path.join(plots_dir, f"double_encoder_flow_model_{DATASET_TYPE}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'test_losses': test_losses,
        'config': {
            'number_latent_dim': model.number_latent_dim,
            'filter_latent_dim': model.filter_latent_dim,
            'image_size': model.image_size,
            'dataset_type': DATASET_TYPE
        }
    }, model_save_path)
    print(f"Model saved to: {model_save_path}")

    return train_losses, test_losses


def visualize_results(model, triplet_creator, num_examples=8):
    """Visualize reconstruction and generation results"""

    model.eval()

    with torch.no_grad():
        # Create test triplets
        (ground_truth, different_digit, same_digit, original_labels, different_labels,
         ground_truth_rotations, ground_truth_scales, same_digit_rotations, same_digit_scales) = \
            triplet_creator.create_batch_triplets(num_examples, dataset='test')

        ground_truth = ground_truth.to(device)
        ground_truth = normalize_to_flow_range(ground_truth)
        different_digit = different_digit.to(device)
        different_digit = normalize_to_flow_range(different_digit)
        same_digit = same_digit.to(device)
        same_digit = normalize_to_flow_range(same_digit)
        original_labels = original_labels.to(device)

        # Reconstructions
        reconstruction = model.reconstruct(same_digit, different_digit)

    # Convert to numpy for plotting
    # Convert to visualization range [0,1] for plotting
    ground_truth_np = to_visualization_range(ground_truth).cpu().numpy()
    different_digit_np = to_visualization_range(different_digit).cpu().numpy()
    same_digit_np = to_visualization_range(same_digit).cpu().numpy()
    reconstruction_np = to_visualization_range(reconstruction).cpu().numpy()

    # Remove the old conversion lines

    # Create visualization
    fig, axes = plt.subplots(4, num_examples, figsize=(2*num_examples, 8))

    for i in range(num_examples):
        # Ground truth
        axes[0, i].imshow(ground_truth_np[i, 0], cmap='gray')
        axes[0, i].set_title(f'Ground Truth\nLabel: {triplet_creator.class_names[original_labels[i]]}')
        axes[0, i].axis('off')

        # Different digit (filter encoder input)
        axes[1, i].imshow(different_digit_np[i, 0], cmap='gray')
        axes[1, i].set_title(f'Different Digit\nFilter Input')
        axes[1, i].axis('off')

        # Same digit (number encoder input)
        axes[2, i].imshow(same_digit_np[i, 0], cmap='gray')
        axes[2, i].set_title(f'Same Digit\nNumber Input')
        axes[2, i].axis('off')

        # Reconstruction
        axes[3, i].imshow(reconstruction_np[i, 0], cmap='gray')
        axes[3, i].set_title(f'Reconstruction')
        axes[3, i].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Model parameters - more conservative for stability
    number_latent_dim = NUMBER_ENCODER_LATENT_DIM
    filter_latent_dim = FILTER_ENCODER_LATENT_DIM
    num_epochs = 200
    learning_rate = 1e-4  # Reduced learning rate
    batch_size = 256  # Smaller batch size for stability

    # Override the global BATCH_SIZE with our local batch_size
    BATCH_SIZE = batch_size

    print(f"Double Encoder Flow Matching v5")
    print(f"Dataset: {DATASET_TYPE}")
    print(f"Device: {device}")
    print(f"Number latent dim: {number_latent_dim}")
    print(f"Filter latent dim: {filter_latent_dim}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")

    # Create triplet creator
    print("\nInitializing triplet creator...")
    triplet_creator = TripletCreator(dataset_type=DATASET_TYPE)
    triplet_creator.get_dataset_info()

    # Debug data normalization
    debug_data_normalization(triplet_creator, device)

    # Create folder for reconstruction plots
    plots_dir = f"reconstruction_plots_v5_{DATASET_TYPE}"
    os.makedirs(plots_dir, exist_ok=True)
    print(f"Created plots directory: {plots_dir}")

    # Initialize wandb
    wandb.init(
        project="tess-generative",
        name=f"double-encoder-flow-{DATASET_TYPE}-v5-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "model_type": "DoubleEncoderFlowMatching",
            "dataset_type": DATASET_TYPE,
            "number_latent_dim": number_latent_dim,
            "filter_latent_dim": filter_latent_dim,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "velocity_field_type": "unet",
            "unet_channels": [32, 64, 128],
            "num_residual_layers": 2,
            "t_embed_dim": 40,
            "z_embed_dim": 40,
            "device": device
        }
    )

    # Create model
    print(f"\nCreating Double Encoder Flow Matching model...")
    model = DoubleEncoderFlowMatching(
        number_latent_dim=number_latent_dim,
        filter_latent_dim=filter_latent_dim,
        image_size=28,
        velocity_field_type="unet",
        n_integration_steps=50,  # Good balance of quality and stability
        unet_channels=[32, 64, 128],  # Back to original size for better quality
        num_residual_layers=2,  # Back to original for better quality
        t_embed_dim=40,  # Back to original size
        z_embed_dim=40
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Log model architecture to wandb
    wandb.watch(model, log="all")

    # Optional: Test the model with a single batch
    test_model=False

    if test_model:
        print("\nTesting model with a single batch...")
        model.eval()

        # Test different batch sizes
        batch_sizes = [8, 16, 32, 64, 128, 256]
        n_tests_per_batch = 5  # Number of tests per batch size

        print(f"Testing model stability across different batch sizes...")
        print(f"Running {n_tests_per_batch} tests per batch size")
        print()

        # Store results
        results = []

        for batch_size in batch_sizes:
            print(f"Testing batch size: {batch_size}")
            flow_loss_list = []

            for test_idx in range(n_tests_per_batch):
                with torch.no_grad():
                    (ground_truth, different_digit, same_digit, original_labels, different_labels,
                    ground_truth_rotations, ground_truth_scales, same_digit_rotations, same_digit_scales) = \
                        triplet_creator.create_batch_triplets(batch_size, dataset='train')

                    ground_truth = ground_truth.to(device)
                    ground_truth = normalize_to_flow_range(ground_truth)
                    different_digit = different_digit.to(device)
                    different_digit = normalize_to_flow_range(different_digit)
                    same_digit = same_digit.to(device)
                    same_digit = normalize_to_flow_range(same_digit)

                    # Test forward pass
                    combined_z, number_z, filter_z, number_mu, number_logvar, filter_mu, filter_logvar = \
                        model.forward(same_digit, different_digit)

                    # Test flow loss
                    flow_loss, _, _, _, _, _, _ = model.get_flow_loss(same_digit, different_digit, ground_truth)
                    flow_loss_list.append(flow_loss.item())

            # Calculate statistics for this batch size
            mean_loss = np.mean(flow_loss_list)
            std_loss = np.std(flow_loss_list)

            results.append((batch_size, mean_loss, std_loss))
            print(f"  Mean: {mean_loss:.4f} ± {std_loss:.4f}")

        # Print results table
        print(f"\n{'='*50}")
        print(f"BATCH SIZE STABILITY TEST RESULTS")
        print(f"{'='*50}")
        print(f"{'Batch Size':<12} | {'Mean Loss ± Std':<20}")
        print(f"{'-'*12} | {'-'*20}")

        for batch_size, mean_loss, std_loss in results:
            print(f"{batch_size:<12} | {mean_loss:.4f} ± {std_loss:.4f}")

        print(f"{'='*50}")

        # Find best and worst batch sizes
        best_idx = np.argmin([mean for _, mean, _ in results])
        worst_idx = np.argmax([mean for _, mean, _ in results])

        best_batch, best_mean, best_std = results[best_idx]
        worst_batch, worst_mean, worst_std = results[worst_idx]

        print(f"Best batch size: {best_batch} (loss: {best_mean:.4f} ± {best_std:.4f})")
        print(f"Worst batch size: {worst_batch} (loss: {worst_mean:.4f} ± {worst_std:.4f})")

        # Check for stability issues
        all_means = [mean for _, mean, _ in results]
        all_stds = [std for _, std, _ in results]

        if max(all_stds) > 1.0:
            print("⚠️  WARNING: High variance detected in some batch sizes")
        if max(all_means) > 10.0:
            print("⚠️  WARNING: Very high flow loss detected - potential numerical issues")

        print("ℹ️  Flow loss is relatively high - this is normal for untrained models")


    # Train model
    train_losses, test_losses = train_double_encoder_flow(
        model, triplet_creator,
        num_epochs=num_epochs, lr=learning_rate, plots_dir=plots_dir
    )

    # Visualize results
    print("\nVisualizing results...")
    visualize_results(model, triplet_creator)

    # Plot training curves and log to wandb
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Flow Loss')
    plt.title('Training Curves')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss', alpha=0.7)
    plt.plot(test_losses, label='Test Loss', alpha=0.7)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Flow Loss (log scale)')
    plt.title('Training Curves (Log Scale)')
    plt.legend()

    plt.tight_layout()

    # Log the training curves plot to wandb
    wandb.log({"training_curves": wandb.Image(plt)}, step=num_epochs)
    plt.show()

    print("Training and evaluation completed!")

    # Finish wandb run
    wandb.finish()
