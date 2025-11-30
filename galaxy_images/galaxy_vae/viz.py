"""
Visualization utilities for Galaxy VAE models.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from .config import *


def create_reconstruction_plot(model, triplet_creator, device, save_path=None, num_examples=8):
    """
    Create a visualization comparing ground truth, inputs, and reconstructions.

    Args:
        model: Trained VAE model
        triplet_creator: TripletCreator instance
        device: Device to run on
        save_path: Path to save the figure
        num_examples: Number of examples to show
    """
    model.eval()

    with torch.no_grad():
        # Get a batch of triplets
        batch = triplet_creator.create_batch_triplets(
            batch_size=num_examples,
            anchor_survey="legacysurvey",
            use_object_mask=USE_OBJECT_MASK
        )

        ground_truth = batch["ground_truth"]["tensor"].to(device)
        cross_instrument = batch["cross_instrument"]["tensor"].to(device)
        different_galaxy = batch["different_galaxy"]["tensor"].to(device)

        # Get reconstruction
        reconstruction, _, _, _, _, _, _ = model(cross_instrument, different_galaxy)

        # Move to CPU and convert to numpy
        ground_truth = ground_truth.cpu().numpy()
        cross_instrument = cross_instrument.cpu().numpy()
        different_galaxy = different_galaxy.cpu().numpy()
        reconstruction = reconstruction.cpu().numpy()

        # Create figure
        fig, axes = plt.subplots(4, num_examples, figsize=(2*num_examples, 8))
        if num_examples == 1:
            axes = axes.reshape(-1, 1)

        for i in range(num_examples):
            # Ground truth (anchor image)
            # Show first band's flux channel (channel 0)
            axes[0, i].imshow(ground_truth[i, 0], cmap='viridis', origin='lower')
            axes[0, i].set_title('Ground Truth' if i == 0 else '')
            axes[0, i].axis('off')

            # Cross instrument (same galaxy, different instrument)
            axes[1, i].imshow(cross_instrument[i, 0], cmap='viridis', origin='lower')
            axes[1, i].set_title('Cross Instrument\n(Same Galaxy)' if i == 0 else '')
            axes[1, i].axis('off')

            # Different galaxy (different galaxy, same instrument)
            axes[2, i].imshow(different_galaxy[i, 0], cmap='viridis', origin='lower')
            axes[2, i].set_title('Different Galaxy\n(Same Instrument)' if i == 0 else '')
            axes[2, i].axis('off')

            # Reconstruction
            axes[3, i].imshow(reconstruction[i, 0], cmap='viridis', origin='lower')
            axes[3, i].set_title('Reconstruction' if i == 0 else '')
            axes[3, i].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved reconstruction plot to {save_path}")

        return fig


def visualize_latent_space(model, triplet_creator, device, save_path=None):
    """
    Visualize the latent space using t-SNE or PCA.
    """
    try:
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
    except ImportError:
        print("sklearn not available, skipping latent space visualization")
        return None

    model.eval()

    # Collect latent representations
    number_latents = []
    filter_latents = []
    labels = []

    with torch.no_grad():
        for _ in range(100):  # Sample 100 triplets
            try:
                batch = triplet_creator.create_batch_triplets(
                    batch_size=1,
                    anchor_survey="legacysurvey",
                    use_object_mask=USE_OBJECT_MASK
                )

                cross_instrument = batch["cross_instrument"]["tensor"].to(device)
                different_galaxy = batch["different_galaxy"]["tensor"].to(device)

                number_z, filter_z, _, _, _, _ = model.encode_only(cross_instrument, different_galaxy)

                number_latents.append(number_z.cpu().numpy())
                filter_latents.append(filter_z.cpu().numpy())
                labels.append(0)  # Could add actual labels if available
            except Exception:
                continue

    number_latents = np.vstack(number_latents)
    filter_latents = np.vstack(filter_latents)

    # Reduce dimensionality for visualization
    pca = PCA(n_components=2)
    number_2d = pca.fit_transform(number_latents)
    filter_2d = pca.fit_transform(filter_latents)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.scatter(number_2d[:, 0], number_2d[:, 1], alpha=0.5)
    ax1.set_title('Number Encoder Latent Space (PCA)')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')

    ax2.scatter(filter_2d[:, 0], filter_2d[:, 1], alpha=0.5)
    ax2.set_title('Filter Encoder Latent Space (PCA)')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved latent space visualization to {save_path}")

    return fig
