# -*- coding: utf-8 -*-
"""Conditional Autoencoder with Flow Matching Decoder v4 - Using U-Net Velocity Field

This module implements a Conditional Autoencoder with a CNN encoder and a flow matching decoder
that uses a U-Net architecture for the velocity field. Unlike a VAE, this model predicts latent
representations directly without variational components, relying on flow matching for generation.
The U-Net is specifically designed for image data and should provide better performance for
MNIST generation compared to simple MLPs.

DATASET OPTIONS:
- Set dataset_type = 'mnist' for vanilla MNIST
- Set dataset_type = 'augmented_mnist' for augmented MNIST with rotations
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

# Import the FlowMatchingDecoder from carol_decoder.py
from carol_decoder import FlowMatchingDecoder

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

def load_dataset(dataset_type='mnist', batch_size=128):
    """Load either vanilla MNIST or augmented MNIST dataset"""

    if dataset_type == 'mnist':
        print("Loading vanilla MNIST data...")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1] like in lab_three
        ])

        train_dataset = torchvision.datasets.MNIST(
            root='../data',
            train=True,
            transform=transform,
            download=True
        )

        test_dataset = torchvision.datasets.MNIST(
            root='../data',
            train=False,
            transform=transform,
            download=True
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        print(f"Training samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")

        return train_loader, test_loader

    elif dataset_type == 'augmented_mnist':
        print("Loading augmented MNIST data...")

        # Load augmented data
        train_images = np.load('../data/augmented/rotation_v2_augmented_mnist_train_images.npy')
        train_labels = np.load('../data/augmented/rotation_v2_augmented_mnist_train_labels.npy')
        test_images = np.load('../data/augmented/rotation_v2_augmented_mnist_test_images.npy')
        test_labels = np.load('../data/augmented/rotation_v2_augmented_mnist_test_labels.npy')

        print(f"Augmented training images shape: {train_images.shape}")
        print(f"Augmented training labels shape: {train_labels.shape}")
        print(f"Augmented test images shape: {test_images.shape}")
        print(f"Augmented test labels shape: {test_labels.shape}")

        # Convert to torch tensors and normalize to [-1, 1]
        train_images = torch.from_numpy(train_images).float() / 255.0 * 2.0 - 1.0
        test_images = torch.from_numpy(test_images).float() / 255.0 * 2.0 - 1.0
        train_labels = torch.from_numpy(train_labels).long()
        test_labels = torch.from_numpy(test_labels).long()

        # Add channel dimension if not present (should be [batch, channels, height, width])
        if train_images.dim() == 3:
            train_images = train_images.unsqueeze(1)  # Add channel dimension
        if test_images.dim() == 3:
            test_images = test_images.unsqueeze(1)  # Add channel dimension

        print(f"Final tensor shapes - train_images: {train_images.shape}, test_images: {test_images.shape}")

        # Create datasets
        train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
        test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        print(f"Augmented training samples: {len(train_dataset)}")
        print(f"Augmented test samples: {len(test_dataset)}")

        return train_loader, test_loader

    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. Supported: 'mnist', 'augmented_mnist'")

# Load dataset based on configuration
dataset_type = 'mnist'  # Change this to 'augmented_mnist' to use augmented data
train_loader, test_loader = load_dataset(dataset_type=dataset_type, batch_size=128)


def one_hot(labels, num_classes=10):
    """Convert labels to one-hot encoding"""
    return F.one_hot(labels, num_classes=num_classes).float()


class CNNEncoder(nn.Module):
    """CNN Encoder that predicts latent z directly (no VAE)"""

    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim

        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),  # (B,32,14,14)
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), # (B,64,7,7)
            nn.ReLU(),
            nn.Flatten()
        )
        # Direct prediction of z instead of mu/logvar
        self.fc_z = nn.Linear(64*7*7, latent_dim)

    def encode(self, x):
        """Encode input to latent z directly"""
        h = self.enc(x)
        z = self.fc_z(h)
        return z

    def forward(self, x):
        """Forward pass through encoder"""
        z = self.encode(x)
        return z


class ConditionalAutoencoderWithUNetFlowMatching(nn.Module):
    """Conditional Autoencoder with U-Net Flow Matching decoder (no VAE)"""

    def __init__(self, latent_dim=32, num_classes=10, image_size=28):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.image_size = image_size
        self.image_dim = image_size * image_size

        # Encoder (now predicts z directly)
        self.encoder = CNNEncoder(latent_dim)

        # U-Net Flow Matching Decoder from carol_decoder.py
        # input_dim = latent_dim + num_classes (conditioning on z + label)
        # output_dim = image_dim (flattened image)
        self.decoder = FlowMatchingDecoder(
            input_dim=latent_dim + num_classes,
            output_dim=self.image_dim,
            velocity_field_type="unet",  # Use U-Net instead of MLP
            n_integration_steps=100,
            # U-Net specific parameters
            image_size=image_size,
            unet_channels=[32, 64, 128],  # U-Net channel dimensions
            num_residual_layers=2,        # Number of residual layers per block
            t_embed_dim=40,               # Time embedding dimension
            z_embed_dim=40                # Conditioning embedding dimension
        )

    def forward(self, x, labels):
        """Forward pass through the entire model"""
        # Encode to get z directly
        z = self.encoder(x)

        # Create conditioning vector: concatenate latent z with one-hot labels
        labels_onehot = one_hot(labels, self.num_classes).to(x.device)
        condition = torch.cat([z, labels_onehot], dim=1)

        # Flatten input images for flow matching
        x_flat = x.view(x.size(0), -1)  # (B, 784)

        # Get flow matching loss (this is now the only loss)
        flow_loss = self.decoder.get_loss(x_flat, condition)

        return flow_loss, z

    def sample(self, labels, z=None, num_samples=1):
        """Generate samples conditioned on labels"""
        batch_size = labels.shape[0]

        if z is None:
            # Sample from prior - match the batch size of labels
            z = torch.randn(batch_size, self.latent_dim).to(next(self.parameters()).device)

        # Create conditioning vector
        labels_onehot = one_hot(labels, self.num_classes).to(z.device)
        condition = torch.cat([z, labels_onehot], dim=1)

        # Sample from flow matching decoder
        samples_flat = self.decoder.sample(condition, num_samples)

        # Reshape to image format
        if num_samples == 1:
            samples = samples_flat.view(-1, 1, self.image_size, self.image_size)
        else:
            samples = samples_flat.view(num_samples, -1, 1, self.image_size, self.image_size)

        return samples

    def reconstruct(self, x, labels):
        """Reconstruct input images"""
        # Encode to get z directly
        z = self.encoder(x)

        # Create conditioning vector
        labels_onehot = one_hot(labels, self.num_classes).to(x.device)
        condition = torch.cat([z, labels_onehot], dim=1)

        # Sample from flow matching decoder
        samples_flat = self.decoder.sample(condition, 1)

        # Reshape to image format
        reconstructions = samples_flat.view(-1, 1, self.image_size, self.image_size)

        return reconstructions


def train_autoencoder_unet_flow(model, train_loader, test_loader, num_epochs=30, lr=1e-3, plots_dir="reconstruction_plots"):
    """Training function for Conditional Autoencoder with U-Net flow matching"""

    import os
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Add gradient clipping to prevent explosion
    max_grad_norm = 1.0

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

        for batch_idx, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            flow_loss, z = model(data, labels)

            # Check for loss explosion
            if torch.isnan(flow_loss) or torch.isinf(flow_loss) or flow_loss > 1000:
                print(f"WARNING: Loss explosion detected! Loss: {flow_loss.item()}")
                print("Skipping this batch and continuing...")
                continue

            # Backward pass
            flow_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

            # Accumulate losses
            train_loss += flow_loss.item()
            num_batches += 1

            if batch_idx % 50 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Total Loss: {flow_loss.item():.4f}')

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
            for data, labels in test_loader:
                data = data.to(device)
                labels = labels.to(device)

                flow_loss, z = model(data, labels)

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

        # Log to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "test_loss": avg_test_loss,
            "learning_rate": lr,
            "flow_loss": flow_loss.item() if not (torch.isnan(flow_loss) or torch.isinf(flow_loss)) else 0
        }, step=epoch + 1)

        # Generate and log reconstruction plots every 2 epochs
        if (epoch + 1) % 2 == 0:
            print(f"Creating reconstruction plot for epoch {epoch + 1}...")
            try:
                reconstruction_plot = create_reconstruction_plot(model, test_loader, num_examples=8)
                print(f"Plot created, attempting to log to wandb...")

                # Log to wandb
                wandb.log({"reconstructions": wandb.Image(reconstruction_plot)}, step=epoch + 1)

                # Also save as file
                plot_filename = os.path.join(plots_dir, f"reconstruction_epoch_{epoch+1:03d}.png")
                reconstruction_plot.savefig(plot_filename, dpi=150, bbox_inches='tight')
                wandb.log({"reconstructions_file": wandb.Image(plot_filename)}, step=epoch + 1)
                print(f"Saved plot to: {plot_filename}")

                print(f"Successfully logged reconstruction plot for epoch {epoch + 1}")
                plt.close(reconstruction_plot)  # Close to free memory
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
    model_save_path = os.path.join(plots_dir, f"autoencoder_unet_flow_model_{dataset_type}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'test_losses': test_losses,
        'config': {
            'latent_dim': model.latent_dim,
            'num_classes': model.num_classes,
            'image_size': model.image_size,
            'dataset_type': dataset_type
        }
    }, model_save_path)
    print(f"Model saved to: {model_save_path}")

    return train_losses, test_losses


def create_reconstruction_plot(model, test_loader, num_examples=8):
    """Create reconstruction plot for wandb logging"""

    print("Starting to create reconstruction plot...")
    model.eval()

    # Get some test examples
    examples, labels = next(iter(test_loader))
    examples = examples[:num_examples].to(device)
    labels = labels[:num_examples].to(device)
    print(f"Got {num_examples} examples with labels: {labels.cpu().numpy()}")

    with torch.no_grad():
        # Reconstructions
        print("Creating reconstructions...")
        reconstructions = model.reconstruct(examples, labels)
        print(f"Reconstructions shape: {reconstructions.shape}")

        # Generated samples (conditioned on the same labels)
        print("Creating generated samples...")
        generated = model.sample(labels, num_samples=1)
        print(f"Generated samples shape: {generated.shape}")

    # Convert to numpy for plotting
    examples = examples.cpu().numpy()
    reconstructions = reconstructions.cpu().numpy()
    generated = generated.cpu().numpy()
    labels = labels.cpu().numpy()
    print(f"Converted to numpy - examples: {examples.shape}, reconstructions: {reconstructions.shape}, generated: {generated.shape}")

    # Create visualization
    fig, axes = plt.subplots(3, num_examples, figsize=(2*num_examples, 6))

    for i in range(num_examples):
        # Original
        axes[0, i].imshow(examples[i, 0], cmap='gray')
        axes[0, i].set_title(f'Original {labels[i]}')
        axes[0, i].axis('off')

        # Reconstruction
        axes[1, i].imshow(reconstructions[i, 0], cmap='gray')
        axes[1, i].set_title(f'Reconstruction {labels[i]}')
        axes[1, i].axis('off')

        # Generated
        axes[2, i].imshow(generated[i, 0], cmap='gray')
        axes[2, i].set_title(f'Generated {labels[i]}')
        axes[2, i].axis('off')

    plt.tight_layout()
    print("Reconstruction plot created successfully!")
    return fig


def visualize_results(model, test_loader, num_examples=8):
    """Visualize reconstruction and generation results"""

    model.eval()

    # Get some test examples
    examples, labels = next(iter(test_loader))
    examples = examples[:num_examples].to(device)
    labels = labels[:num_examples].to(device)

    with torch.no_grad():
        # Reconstructions
        reconstructions = model.reconstruct(examples, labels)

        # Generated samples (conditioned on the same labels)
        generated = model.sample(labels, num_samples=1)

    # Convert to numpy for plotting
    examples = examples.cpu().numpy()
    reconstructions = reconstructions.cpu().numpy()
    generated = generated.cpu().numpy()
    labels = labels.cpu().numpy()

    # Create visualization
    fig, axes = plt.subplots(3, num_examples, figsize=(2*num_examples, 6))

    for i in range(num_examples):
        # Original
        axes[0, i].imshow(examples[i, 0], cmap='gray')
        axes[0, i].set_title(f'Original {labels[i]}')
        axes[0, i].axis('off')

        # Reconstruction
        axes[1, i].imshow(reconstructions[i, 0], cmap='gray')
        axes[1, i].set_title(f'Reconstruction {labels[i]}')
        axes[1, i].axis('off')

        # Generated
        axes[2, i].imshow(generated[i, 0], cmap='gray')
        axes[2, i].set_title(f'Generated {labels[i]}')
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.show()


def interpolate_latent_space(model, test_loader, num_interpolations=10):
    """Interpolate in latent space"""

    model.eval()

    # Get two random examples
    examples, labels = next(iter(test_loader))
    example1 = examples[0:1].to(device)
    example2 = examples[1:2].to(device)
    label1 = labels[0:1].to(device)
    label2 = labels[1:2].to(device)

    with torch.no_grad():
        # Encode both examples
        z1 = model.encoder(example1)
        z2 = model.encoder(example2)

        # Interpolate in latent space
        interpolations = []
        for alpha in np.linspace(0, 1, num_interpolations):
            z_interp = alpha * z1 + (1 - alpha) * z2
            # Use label1 for conditioning (could also interpolate labels)
            interp_image = model.sample(label1, z=z_interp)
            interpolations.append(interp_image.cpu().numpy())

    # Visualize interpolation
    fig, axes = plt.subplots(1, num_interpolations, figsize=(2*num_interpolations, 2))

    for i, interp in enumerate(interpolations):
        axes[i].imshow(interp[0, 0], cmap='gray')
        axes[i].set_title(f'α={i/(num_interpolations-1):.1f}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def generate_conditioned_samples(model, num_samples_per_digit=5):
    """Generate samples conditioned on each digit"""

    model.eval()

    fig, axes = plt.subplots(10, num_samples_per_digit, figsize=(num_samples_per_digit*1.5, 15))

    with torch.no_grad():
        for digit in range(10):
            # Create labels for this digit
            labels = torch.full((num_samples_per_digit,), digit).to(device)

            # Generate samples
            samples = model.sample(labels, num_samples=num_samples_per_digit)

            for i in range(num_samples_per_digit):
                axes[digit, i].imshow(samples[i, 0].cpu().numpy(), cmap='gray')
                if i == 0:
                    axes[digit, i].set_ylabel(f'Digit {digit}', fontsize=12)
                axes[digit, i].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Model parameters
    latent_dim = 32
    num_classes = 10
    num_epochs = 300
    learning_rate = 1e-4  # Reduced from 1e-3
    dataset_type = 'augmented_mnist'  # Change to 'augmented_mnist' to use augmented data
    batch_size = 128

    # Load dataset based on configuration
    print(f"Loading dataset: {dataset_type}")
    train_loader, test_loader = load_dataset(dataset_type=dataset_type, batch_size=batch_size)

    # Create folder for reconstruction plots
    import os
    plots_dir = f"reconstruction_plots_v4_{dataset_type}_rotonly"
    os.makedirs(plots_dir, exist_ok=True)
    print(f"Created plots directory: {plots_dir}")

    # Set API key if needed (uncomment and add your key)
    # import os
    # os.environ["WANDB_API_KEY"] = "your-api-key-here"

    # Initialize wandb
    wandb.init(
        project="tess-generative",
        # entity="pmercader",  # Comment out to use personal space
        name=f"conditional-autoencoder-unet-flow-{dataset_type}-v4-rotonly-reduced-lr",
        config={
            "latent_dim": latent_dim,
            "num_classes": num_classes,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "model_type": "ConditionalAutoencoderWithUNetFlowMatching",
            "velocity_field_type": "unet",
            "unet_channels": [32, 64, 128],
            "num_residual_layers": 2,
            "t_embed_dim": 40,
            "z_embed_dim": 40,
            "dataset": dataset_type.upper(),
            "device": device
        }
    )

    # Create model
    print(f"Creating Conditional Autoencoder with U-Net Flow Matching model for {dataset_type}...")
    model = ConditionalAutoencoderWithUNetFlowMatching(
        latent_dim=latent_dim,
        num_classes=num_classes,
        image_size=28
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Log model architecture to wandb
    wandb.watch(model, log="all")

    # Test wandb image logging with a simple plot
    print("Testing wandb image logging...")
    test_fig, test_ax = plt.subplots(1, 1, figsize=(4, 4))
    test_ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
    test_ax.set_title("Test Plot")
    wandb.log({"test_plot": wandb.Image(test_fig)}, step=0)
    plt.close(test_fig)
    print("Test plot logged successfully!")

    # Train model
    train_losses, test_losses = train_autoencoder_unet_flow(
        model, train_loader, test_loader,
        num_epochs=num_epochs, lr=learning_rate, plots_dir=plots_dir
    )

    # Visualize results
    print("\nVisualizing results...")
    visualize_results(model, test_loader)

    # Generate conditioned samples
    print("\nGenerating samples conditioned on each digit...")
    generate_conditioned_samples(model)

    # Latent space interpolation
    print("\nGenerating latent space interpolation...")
    interpolate_latent_space(model, test_loader)

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

    # Log final reconstruction plot
    final_reconstruction_plot = create_reconstruction_plot(model, test_loader, num_examples=8)
    wandb.log({"final_reconstructions": wandb.Image(final_reconstruction_plot)}, step=num_epochs)

    # Save final plot to folder
    final_plot_filename = os.path.join(plots_dir, "final_reconstruction.png")
    final_reconstruction_plot.savefig(final_plot_filename, dpi=150, bbox_inches='tight')
    print(f"Saved final plot to: {final_plot_filename}")
    plt.close(final_reconstruction_plot)

    print("Training and evaluation completed!")

    # Finish wandb run
    wandb.finish()
