"""
Visualize random images from HSCDataLoader to see what they look like.

This script shows:
1. Images as loaded from HSCDataLoader (preprocessed but not normalized)
2. Images after normalization (as used in training)
3. Statistics about the value ranges
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

# Add parent directory to path
current_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_path))
sys.path.insert(0, parent_dir)

from galaxy_images.galaxy_flow.hsc_data_loader import HSCDataLoader
from galaxy_images.galaxy_flow import unconditional_config as cfg


def visualize_images(loader, num_images=8, save_path=None):
    """
    Visualize random images from the data loader.

    Shows both:
    - Images as loaded (preprocessed but not normalized)
    - Images after normalization (as used in training)
    """
    # Get a batch of random images
    batch = loader.get_batch(num_images)  # (B, 4, 96, 96)

    # Convert to numpy for visualization
    images_np = batch.numpy()  # (B, 4, 96, 96)

    # Print statistics about raw images
    print(f"\n{'='*60}")
    print("IMAGE STATISTICS (as loaded from HSCDataLoader)")
    print(f"{'='*60}")
    print(f"Shape: {images_np.shape}")
    print(f"Min value: {images_np.min():.6f}")
    print(f"Max value: {images_np.max():.6f}")
    print(f"Mean value: {images_np.mean():.6f}")
    print(f"Std value: {images_np.std():.6f}")
    print(f"Value range: [{images_np.min():.6f}, {images_np.max():.6f}]")

    # Apply normalization (as done in training)
    images_tensor = torch.from_numpy(images_np).float()
    min_val = images_tensor.min()
    max_val = images_tensor.max()

    if not torch.isclose(max_val, min_val):
        images_norm = (images_tensor - min_val) / (max_val - min_val)  # [0, 1]
        images_norm = 2.0 * images_norm - 1.0  # [-1, 1]
    else:
        images_norm = torch.zeros_like(images_tensor)

    images_norm_np = images_norm.numpy()

    print(f"\n{'='*60}")
    print("IMAGE STATISTICS (after normalization to [-1, 1])")
    print(f"{'='*60}")
    print(f"Min value: {images_norm_np.min():.6f}")
    print(f"Max value: {images_norm_np.max():.6f}")
    print(f"Mean value: {images_norm_np.mean():.6f}")
    print(f"Std value: {images_norm_np.std():.6f}")
    print(f"Value range: [{images_norm_np.min():.6f}, {images_norm_np.max():.6f}]")

    # Create visualization
    fig = plt.figure(figsize=(20, 10))

    # Plot raw images (before normalization)
    for i in range(num_images):
        ax = plt.subplot(2, num_images, i + 1)

        # Convert 4-channel (g, r, i, z) to RGB using first 3 channels (g, r, i)
        rgb = np.stack([
            images_np[i, 0],  # g -> R
            images_np[i, 1],  # r -> G
            images_np[i, 2]   # i -> B
        ], axis=-1)  # (H, W, 3)

        # Normalize each channel to [0, 1] for visualization
        for c in range(3):
            ch = rgb[:, :, c]
            ch_min, ch_max = ch.min(), ch.max()
            if ch_max > ch_min:
                rgb[:, :, c] = (ch - ch_min) / (ch_max - ch_min)
            else:
                rgb[:, :, c] = 0

        ax.imshow(rgb, vmin=0, vmax=1)
        ax.set_title(f'Raw Image {i+1}\n[{images_np[i].min():.3f}, {images_np[i].max():.3f}]', fontsize=10)
        ax.axis('off')

    # Plot normalized images (after normalization)
    for i in range(num_images):
        ax = plt.subplot(2, num_images, num_images + i + 1)

        # Convert 4-channel (g, r, i, z) to RGB using first 3 channels (g, r, i)
        rgb = np.stack([
            images_norm_np[i, 0],  # g -> R
            images_norm_np[i, 1],  # r -> G
            images_norm_np[i, 2]   # i -> B
        ], axis=-1)  # (H, W, 3)

        # Normalize each channel to [0, 1] for visualization
        for c in range(3):
            ch = rgb[:, :, c]
            ch_min, ch_max = ch.min(), ch.max()
            if ch_max > ch_min:
                rgb[:, :, c] = (ch - ch_min) / (ch_max - ch_min)
            else:
                rgb[:, :, c] = 0

        ax.imshow(rgb, vmin=0, vmax=1)
        ax.set_title(f'Normalized Image {i+1}\n[{images_norm_np[i].min():.3f}, {images_norm_np[i].max():.3f}]', fontsize=10)
        ax.axis('off')

    plt.suptitle('Top: Images as loaded from HSCDataLoader (preprocessed but not normalized)\n'
                 'Bottom: Images after normalization to [-1, 1] (as used in training)',
                 fontsize=12, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved visualization to {save_path}")
    else:
        plt.show()

    plt.close()

    return images_np, images_norm_np


if __name__ == "__main__":
    # Path to preprocessed HDF5 file
    hdf5_path = getattr(cfg, 'PREPROCESSED_HDF5_PATH',
                        '/mnt/scratch/legacysurvey_hsc_crossmatched/preprocessed_hsc.h5')

    if not os.path.exists(hdf5_path):
        print(f"Error: HDF5 file not found at {hdf5_path}")
        print("Please run preprocess_hsc_images.py first to create the preprocessed dataset.")
        sys.exit(1)

    print(f"Loading images from {hdf5_path}...")
    loader = HSCDataLoader(hdf5_path=hdf5_path, seed=42, load_to_memory=True)
    print(f"Loaded {len(loader)} images")

    # Visualize 8 random images
    print("\nVisualizing 8 random images...")
    save_path = "hsc_images_visualization.png"
    raw_images, norm_images = visualize_images(loader, num_images=8, save_path=save_path)

    loader.close()
    print("\nVisualization complete!")
