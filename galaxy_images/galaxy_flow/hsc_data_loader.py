"""
Fast data loader for preprocessed HSC images stored in HDF5 format.

This loader provides a drop-in replacement for TripletCreator when using
preprocessed HSC images for unconditional flow matching training.
"""

import os
import random
import h5py
import numpy as np
import torch
from typing import Optional


class HSCDataLoader:
    """
    Fast data loader for preprocessed HSC images from HDF5 format.

    This loader is optimized for unconditional flow matching training where
    we only need HSC images (no triplets, no cross-instrument matching).
    """

    def __init__(
        self,
        hdf5_path: str,
        seed: int = 42,
        load_to_memory: bool = True,
    ):
        """
        Initialize the HSC data loader.

        Args:
            hdf5_path: Path to preprocessed HDF5 file
            seed: Random seed for sampling
            load_to_memory: If True, load all images into memory at initialization (default: True)
        """
        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

        self.hdf5_path = hdf5_path
        self.rng = random.Random(seed)
        self.load_to_memory = load_to_memory

        # Open HDF5 file to read metadata and optionally load data
        with h5py.File(hdf5_path, 'r') as h5_file:
            # Get metadata
            self.num_images = h5_file.attrs['num_images']
            self.crop_size = h5_file.attrs['crop_size']
            self.num_channels = h5_file.attrs['num_channels']

            if load_to_memory:
                # Load all images into memory at once
                print(f"Loading all {self.num_images} images into memory...")
                self.images = torch.from_numpy(h5_file['images'][:]).float()  # (num_images, 4, 96, 96)
                self.h5_file = None  # Close HDF5 file
                print(f"Loaded {self.num_images} images into memory, "
                      f"shape: ({self.num_images}, {self.num_channels}, {self.crop_size}, {self.crop_size})")
                print(f"Memory usage: ~{self.images.numel() * 4 / (1024**3):.3f} GB")
            else:
                # Keep HDF5 file open for on-demand reading
                self.h5_file = h5py.File(hdf5_path, 'r')
                self.images_dset = self.h5_file['images']
                self.images = None
                print(f"Opened HDF5 dataset: {self.num_images} images, "
                      f"shape: ({self.num_channels}, {self.crop_size}, {self.crop_size})")
                print("Images will be loaded on-demand from disk.")

    def __len__(self):
        """Return the number of images in the dataset."""
        return self.num_images

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a single preprocessed image.

        Args:
            idx: Index of the image to retrieve

        Returns:
            Preprocessed image tensor of shape (num_channels, crop_size, crop_size)
        """
        if idx < 0 or idx >= self.num_images:
            raise IndexError(f"Index {idx} out of range [0, {self.num_images})")

        if self.load_to_memory:
            # Read from memory
            return self.images[idx]  # (4, 96, 96)
        else:
            # Read from HDF5 (already preprocessed, but not normalized)
            image = self.images_dset[idx]  # (4, 96, 96)
            return torch.from_numpy(image).float()

    def get_batch(self, batch_size: int) -> torch.Tensor:
        """
        Get a batch of randomly sampled preprocessed images.

        Args:
            batch_size: Number of images to sample

        Returns:
            Batch tensor of shape (batch_size, num_channels, crop_size, crop_size)
        """
        # Sample random indices
        indices = [self.rng.randint(0, self.num_images - 1) for _ in range(batch_size)]

        if self.load_to_memory:
            # Read from memory (much faster) - use tensor indexing
            indices_tensor = torch.tensor(indices, dtype=torch.long)
            return self.images[indices_tensor]  # (batch_size, 4, 96, 96)
        else:
            # Read images from HDF5
            images = []
            for idx in indices:
                image = self.images_dset[idx]  # (4, 96, 96)
                images.append(torch.from_numpy(image).float())
            # Stack into batch
            return torch.stack(images, dim=0)  # (batch_size, 4, 96, 96)

    def create_batch_triplets(
        self,
        batch_size: int,
        anchor_survey: str = "hsc",  # Only HSC is available
        use_object_mask: bool = False,  # Not available in preprocessed data
    ) -> dict:
        """
        Create a batch of images (compatible with TripletCreator API).

        For unconditional training, we only need the ground_truth images.
        This method returns a dict with the same structure as TripletCreator
        for compatibility, but only populates the ground_truth field.

        Args:
            batch_size: Number of images in the batch
            anchor_survey: Ignored (always uses HSC)
            use_object_mask: Ignored (not available in preprocessed data)

        Returns:
            dict with 'ground_truth' key containing batched tensors
        """
        # Get batch of images
        images = self.get_batch(batch_size)  # (batch_size, 4, 96, 96)

        # Flatten for flow matching model: (batch_size, 4*96*96)
        images_flat = images.flatten(1)  # (batch_size, 36864)

        # Return in compatible format
        return {
            "ground_truth": {
                "tensor": images,  # (batch_size, 4, 96, 96) - keep original shape for compatibility
            },
            # Dummy values for compatibility (not used in unconditional training)
            "different_galaxy": {
                "tensor": images,  # Same as ground_truth for compatibility
            },
            "cross_instrument": {
                "tensor": images,  # Same as ground_truth for compatibility
            },
            "anchor_indices": torch.zeros(batch_size, dtype=torch.long),
            "different_indices": torch.zeros(batch_size, dtype=torch.long),
        }

    def close(self):
        """Close the HDF5 file (if still open)."""
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None

    def __del__(self):
        """Cleanup: close HDF5 file when object is destroyed."""
        self.close()


def create_hsc_loader(
    hdf5_path: str = "/mnt/scratch/legacysurvey_hsc_crossmatched/preprocessed_hsc.h5",
    seed: int = 42,
) -> HSCDataLoader:
    """
    Convenience function to create an HSCDataLoader.

    Args:
        hdf5_path: Path to preprocessed HDF5 file
        seed: Random seed for sampling

    Returns:
        HSCDataLoader instance
    """
    return HSCDataLoader(hdf5_path=hdf5_path, seed=seed)


if __name__ == "__main__":
    # Test the data loader
    import sys
    import os

    # Add parent directory to path
    current_path = os.path.abspath(__file__)
    parent_dir = os.path.dirname(os.path.dirname(current_path))
    sys.path.insert(0, parent_dir)

    from galaxy_images.galaxy_flow import unconditional_config as cfg

    # Create loader
    hdf5_path = "/mnt/scratch/legacysurvey_hsc_crossmatched/preprocessed_hsc.h5"

    if not os.path.exists(hdf5_path):
        print(f"Error: HDF5 file not found at {hdf5_path}")
        print("Please run preprocess_hsc_images.py first to create the preprocessed dataset.")
        sys.exit(1)

    loader = create_hsc_loader(hdf5_path=hdf5_path, seed=42)

    # Test getting a batch
    print("\nTesting batch creation...")
    batch = loader.create_batch_triplets(batch_size=4)
    print(f"Batch shape: {batch['ground_truth']['tensor'].shape}")
    print(f"Batch range: [{batch['ground_truth']['tensor'].min():.4f}, "
          f"{batch['ground_truth']['tensor'].max():.4f}]")

    # Test flattening (as done in training)
    images_flat = batch['ground_truth']['tensor'].flatten(1)
    print(f"Flattened shape: {images_flat.shape}")
    print(f"Expected output_dim: {cfg.OUTPUT_DIM}")
    print(f"Match: {images_flat.shape[1] == cfg.OUTPUT_DIM}")

    loader.close()
    print("\nTest completed successfully!")
