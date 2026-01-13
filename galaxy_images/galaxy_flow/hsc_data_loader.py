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
        max_samples: Optional[int] = None,
    ):
        """
        Initialize the HSC data loader.

        Args:
            hdf5_path: Path to preprocessed HDF5 file
            seed: Random seed for sampling
            load_to_memory: If True, load all images into memory at initialization (default: True)
            max_samples: Optional limit on number of images to use. If None, uses all available images.
        """
        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

        self.hdf5_path = hdf5_path
        self.rng = random.Random(seed)
        self.load_to_memory = load_to_memory

        # Open HDF5 file to read metadata and optionally load data
        with h5py.File(hdf5_path, 'r') as h5_file:
            # Get metadata
            total_images = h5_file.attrs['num_images']
            self.crop_size = h5_file.attrs['crop_size']
            self.num_channels = h5_file.attrs['num_channels']

            # Apply max_samples limit if specified
            if max_samples is not None:
                self.num_images = min(max_samples, total_images)
                if self.num_images < total_images:
                    print(f"Limiting dataset to {self.num_images} images (out of {total_images} available)")
            else:
                self.num_images = total_images

            if load_to_memory:
                # Load images into memory (only up to max_samples if specified)
                print(f"Loading {self.num_images} images into memory...")
                self.images = torch.from_numpy(h5_file['images'][:self.num_images]).float()  # (num_images, 4, 96, 96)
                self.h5_file = None  # Close HDF5 file
                print(f"Loaded {self.num_images} images into memory, "
                      f"shape: ({self.num_images}, {self.num_channels}, {self.crop_size}, {self.crop_size})")
                print(f"Memory usage: ~{self.images.numel() * 4 / (1024**3):.3f} GB")
            else:
                # Keep HDF5 file open for on-demand reading
                self.h5_file = h5py.File(hdf5_path, 'r')
                self.images_dset = self.h5_file['images']
                self.images = None
                print(f"Opened HDF5 dataset: {self.num_images} images (out of {total_images} available), "
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
    max_samples: Optional[int] = None,
) -> HSCDataLoader:
    """
    Convenience function to create an HSCDataLoader.

    Args:
        hdf5_path: Path to preprocessed HDF5 file
        seed: Random seed for sampling
        max_samples: Optional limit on number of images to use. If None, uses all available images.

    Returns:
        HSCDataLoader instance
    """
    return HSCDataLoader(hdf5_path=hdf5_path, seed=seed, max_samples=max_samples)


class HSC_Legacy_DataLoader_OneHot(HSCDataLoader):
    """
    Data loader for HSC and Legacy Survey images with one-hot conditioning.

    Loads from HDF5 file with 'hsc_images' and 'legacy_images' datasets.
    """
    def __init__(
        self,
        hdf5_path: str = "/mnt/scratch/legacysurvey_hsc_crossmatched/preprocessed_hsc_legacy.h5",
        seed: int = 42,
        load_to_memory: bool = True,
        max_samples: Optional[int] = None):
        """
        Initialize the HSC and Legacy Survey data loader with one-hot conditioning.

        Args:
            hdf5_path: Path to preprocessed HDF5 file with 'hsc_images' and 'legacy_images' datasets
            seed: Random seed for sampling
            load_to_memory: If True, load all images into memory at initialization (default: True)
            max_samples: Optional limit on number of image pairs to use. If None, uses all available pairs.
        """
        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

        self.hdf5_path = hdf5_path
        self.rng = random.Random(seed)
        self.load_to_memory = load_to_memory

        # Open HDF5 file to read metadata and optionally load data
        with h5py.File(hdf5_path, 'r') as h5_file:
            # Get metadata
            total_images = h5_file.attrs['num_images']
            self.crop_size = h5_file.attrs['crop_size']
            self.num_channels = h5_file.attrs['num_channels']

            # Verify required datasets exist
            if 'hsc_images' not in h5_file:
                raise KeyError(f"HDF5 file missing 'hsc_images' dataset: {hdf5_path}")
            if 'legacy_images' not in h5_file:
                raise KeyError(f"HDF5 file missing 'legacy_images' dataset: {hdf5_path}")

            # Apply max_samples limit if specified
            if max_samples is not None:
                self.num_images = min(max_samples, total_images)
                if self.num_images < total_images:
                    print(f"Limiting dataset to {self.num_images} image pairs (out of {total_images} available)")
            else:
                self.num_images = total_images

            if load_to_memory:
                # Load images into memory (only up to max_samples if specified)
                print(f"Loading {self.num_images} image pairs into memory...")
                self.hsc_images = torch.from_numpy(h5_file['hsc_images'][:self.num_images]).float()  # (num_images, 4, 96, 96)
                self.legacy_images = torch.from_numpy(h5_file['legacy_images'][:self.num_images]).float()  # (num_images, 4, 96, 96)
                self.h5_file = None  # Close HDF5 file
                print(f"Loaded {self.num_images} HSC images into memory, "
                      f"shape: ({self.num_images}, {self.num_channels}, {self.crop_size}, {self.crop_size})")
                print(f"Loaded {self.num_images} Legacy Survey images into memory, "
                      f"shape: ({self.num_images}, {self.num_channels}, {self.crop_size}, {self.crop_size})")
                total_memory = (self.hsc_images.numel() + self.legacy_images.numel()) * 4 / (1024**3)
                print(f"Total memory usage: ~{total_memory:.3f} GB")
            else:
                # Keep HDF5 file open for on-demand reading
                self.h5_file = h5py.File(hdf5_path, 'r')
                self.hsc_images_dset = self.h5_file['hsc_images']
                self.legacy_images_dset = self.h5_file['legacy_images']
                self.hsc_images = None
                self.legacy_images = None
                print(f"Opened HDF5 dataset: {self.num_images} image pairs (out of {total_images} available), "
                      f"shape: ({self.num_channels}, {self.crop_size}, {self.crop_size})")
                print("Images will be loaded on-demand from disk.")

    def __len__(self):
        """Return the number of image pairs in the dataset."""
        return self.num_images

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single image pair (HSC and Legacy Survey) and a pair of one-hot conditioning vectors.

        Args:
            idx: Index of the image pair to retrieve

        Returns:
            tuple: (hsc_image, legacy_image, hsc_one_hot, legacy_one_hot)
                - hsc_image: HSC image tensor, shape (4, 96, 96)
                - legacy_image: Legacy Survey image tensor, shape (4, 96, 96)
                - hsc_one_hot: One-hot vector [1, 0] for HSC, shape (2,)
                - legacy_one_hot: One-hot vector [0, 1] for Legacy Survey, shape (2,)
        """
        if idx < 0 or idx >= self.num_images:
            raise IndexError(f"Index {idx} out of range [0, {self.num_images})")

        if self.load_to_memory:
            # Read from memory
            image_hsc = self.hsc_images[idx]  # (4, 96, 96)
            image_legacy = self.legacy_images[idx]  # (4, 96, 96)
        else:
            # Read from HDF5 (already preprocessed, but not normalized)
            hsc_data = self.hsc_images_dset[idx]  # (4, 96, 96)
            legacy_data = self.legacy_images_dset[idx]  # (4, 96, 96)
            image_hsc = torch.from_numpy(hsc_data).float()
            image_legacy = torch.from_numpy(legacy_data).float()

        # Create one-hot conditioning vectors
        one_hot_vector_hsc = torch.tensor([1.0, 0.0], dtype=torch.float32)  # HSC
        one_hot_vector_legacy = torch.tensor([0.0, 1.0], dtype=torch.float32)  # Legacy Survey

        return image_hsc, image_legacy, one_hot_vector_hsc, one_hot_vector_legacy

    def get_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a batch of randomly sampled image pairs with one-hot conditioning.

        Args:
            batch_size: Number of image pairs to sample

        Returns:
            tuple: (hsc_images, legacy_images, hsc_one_hots, legacy_one_hots)
                - hsc_images: HSC image batch, shape (batch_size, 4, 96, 96)
                - legacy_images: Legacy Survey image batch, shape (batch_size, 4, 96, 96)
                - hsc_one_hots: One-hot vectors for HSC, shape (batch_size, 2)
                - legacy_one_hots: One-hot vectors for Legacy Survey, shape (batch_size, 2)
        """
        # Sample random indices
        indices = [self.rng.randint(0, self.num_images - 1) for _ in range(batch_size)]

        if self.load_to_memory:
            # Read from memory (much faster) - use tensor indexing
            indices_tensor = torch.tensor(indices, dtype=torch.long)
            hsc_batch = self.hsc_images[indices_tensor]  # (batch_size, 4, 96, 96)
            legacy_batch = self.legacy_images[indices_tensor]  # (batch_size, 4, 96, 96)
        else:
            # Read images from HDF5
            hsc_images_list = []
            legacy_images_list = []
            for idx in indices:
                hsc_data = self.hsc_images_dset[idx]  # (4, 96, 96)
                legacy_data = self.legacy_images_dset[idx]  # (4, 96, 96)
                hsc_images_list.append(torch.from_numpy(hsc_data).float())
                legacy_images_list.append(torch.from_numpy(legacy_data).float())
            # Stack into batches
            hsc_batch = torch.stack(hsc_images_list, dim=0)  # (batch_size, 4, 96, 96)
            legacy_batch = torch.stack(legacy_images_list, dim=0)  # (batch_size, 4, 96, 96)

        # Create one-hot conditioning vectors
        hsc_one_hots = torch.tensor([[1.0, 0.0]] * batch_size, dtype=torch.float32)  # (batch_size, 2)
        legacy_one_hots = torch.tensor([[0.0, 1.0]] * batch_size, dtype=torch.float32)  # (batch_size, 2)

        return hsc_batch, legacy_batch, hsc_one_hots, legacy_one_hots

    def get_training_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a mixed batch of images (HSC and Legacy Survey) with their one-hot conditioning vectors.

        This is the main method for training - it returns a mixed batch where each image
        has its corresponding one-hot conditioning vector.

        Args:
            batch_size: Number of images to sample (will be split between HSC and Legacy)

        Returns:
            tuple: (images, conditioning_vectors)
                - images: Mixed batch of images, shape (batch_size, 4, 96, 96)
                - conditioning_vectors: One-hot vectors, shape (batch_size, 2)
                  [1, 0] for HSC images, [0, 1] for Legacy Survey images
        """
        # Sample random indices
        indices = [self.rng.randint(0, self.num_images - 1) for _ in range(batch_size)]

        # Randomly assign each sample to HSC or Legacy Survey
        images_list = []
        conditioning_list = []

        if self.load_to_memory:
            indices_tensor = torch.tensor(indices, dtype=torch.long)
            hsc_images = self.hsc_images[indices_tensor]  # (batch_size, 4, 96, 96)
            legacy_images = self.legacy_images[indices_tensor]  # (batch_size, 4, 96, 96)

            for i in range(batch_size):
                # Randomly choose HSC or Legacy Survey for each sample
                if self.rng.random() < 0.5:
                    images_list.append(hsc_images[i])
                    conditioning_list.append(torch.tensor([1.0, 0.0], dtype=torch.float32))  # HSC
                else:
                    images_list.append(legacy_images[i])
                    conditioning_list.append(torch.tensor([0.0, 1.0], dtype=torch.float32))  # Legacy Survey
        else:
            for idx in indices:
                # Randomly choose HSC or Legacy Survey for each sample
                if self.rng.random() < 0.5:
                    hsc_data = self.hsc_images_dset[idx]  # (4, 96, 96)
                    images_list.append(torch.from_numpy(hsc_data).float())
                    conditioning_list.append(torch.tensor([1.0, 0.0], dtype=torch.float32))  # HSC
                else:
                    legacy_data = self.legacy_images_dset[idx]  # (4, 96, 96)
                    images_list.append(torch.from_numpy(legacy_data).float())
                    conditioning_list.append(torch.tensor([0.0, 1.0], dtype=torch.float32))  # Legacy Survey

        # Stack into batches
        images_batch = torch.stack(images_list, dim=0)  # (batch_size, 4, 96, 96)
        conditioning_batch = torch.stack(conditioning_list, dim=0)  # (batch_size, 2)

        return images_batch, conditioning_batch


class HSC_Legacy_Triplets_Dataloader(HSCDataLoader):
    '''
    Dataloader for training a model with double encoder.
    Gives triplets of 1 target image, 1 example from the other instrument and N examples from the same instrument

    Note: This class is incomplete and needs to be implemented.
    '''

    def __init__(
        self,
        anchor_survey: str = 'hsc',
        N_same_instr: int = 5,
    ):
        # TODO: Implement triplet dataloader
        super().__init__()




def create_hsc_legacy_loader(
    hdf5_path: str = "/mnt/scratch/legacysurvey_hsc_crossmatched/preprocessed_hsc_legacy.h5",
    seed: int = 42,
    load_to_memory: bool = True,
    max_samples: Optional[int] = None
) -> HSC_Legacy_DataLoader_OneHot:
    """
    Convenience function to create an HSC_Legacy_DataLoader_OneHot.

    Args:
        hdf5_path: Path to preprocessed HDF5 file with 'hsc_images' and 'legacy_images' datasets
        seed: Random seed for sampling
        load_to_memory: If True, load all images into memory at initialization (default: True)
        max_samples: Optional limit on number of image pairs to use. If None, uses all available pairs.

    Returns:
        HSC_Legacy_DataLoader_OneHot instance
    """
    return HSC_Legacy_DataLoader_OneHot(
        hdf5_path=hdf5_path,
        seed=seed,
        load_to_memory=load_to_memory,
        max_samples=max_samples
    )


if __name__ == "__main__":
    # Test the HSC+Legacy data loader with one-hot conditioning
    import sys
    import os

    # Add parent directory to path
    current_path = os.path.abspath(__file__)
    parent_dir = os.path.dirname(os.path.dirname(current_path))
    sys.path.insert(0, parent_dir)

    from galaxy_images.galaxy_flow import unconditional_config as cfg

    # Test HSC_Legacy_DataLoader_OneHot
    print("=" * 60)
    print("Testing HSC_Legacy_DataLoader_OneHot")
    print("=" * 60)

    # Try different possible paths
    possible_paths = [
        "/Users/pablom.perez/Desktop/data/legacysurvey_hsc_crossmatched/preprocessed_hsc_legacy.h5",
        "/mnt/scratch/legacysurvey_hsc_crossmatched/preprocessed_hsc_legacy.h5",
    ]

    hdf5_path = None
    for path in possible_paths:
        if os.path.exists(path):
            hdf5_path = path
            break

    if hdf5_path is None:
        print(f"Error: HDF5 file not found at any of these paths:")
        for path in possible_paths:
            print(f"  - {path}")
        print("\nPlease run preprocess_hsc_legacy_images.py first to create the preprocessed dataset.")
        sys.exit(1)

    print(f"\nUsing HDF5 file: {hdf5_path}")

    # Create loader
    print("\n1. Creating HSC+Legacy data loader...")
    loader = create_hsc_legacy_loader(hdf5_path=hdf5_path, seed=42, load_to_memory=True)
    print(f"   Loaded {len(loader)} image pairs")

    # Test __getitem__
    print("\n2. Testing __getitem__ (single image pair)...")
    hsc_img, legacy_img, hsc_onehot, legacy_onehot = loader[0]
    print(f"   HSC image shape: {hsc_img.shape}")
    print(f"   Legacy image shape: {legacy_img.shape}")
    print(f"   HSC one-hot: {hsc_onehot}")
    print(f"   Legacy one-hot: {legacy_onehot}")
    print(f"   HSC image range: [{hsc_img.min():.4f}, {hsc_img.max():.4f}]")
    print(f"   Legacy image range: [{legacy_img.min():.4f}, {legacy_img.max():.4f}]")

    # Test get_training_batch (main method for training)
    print("\n3. Testing get_training_batch() (main training method)...")
    batch_size = 16
    images, conditioning = loader.get_training_batch(batch_size=batch_size)
    print(f"   Images batch shape: {images.shape}")
    print(f"   Conditioning batch shape: {conditioning.shape}")
    print(f"   Images range: [{images.min():.4f}, {images.max():.4f}]")

    # Count HSC vs Legacy Survey in the batch
    hsc_count = (conditioning[:, 0] == 1.0).sum().item()
    legacy_count = (conditioning[:, 1] == 1.0).sum().item()
    print(f"   HSC images in batch: {hsc_count}")
    print(f"   Legacy Survey images in batch: {legacy_count}")
    print(f"   Conditioning vectors:")
    for i in range(min(8, batch_size)):
        cond_str = "HSC" if conditioning[i, 0] == 1.0 else "Legacy"
        print(f"     Sample {i}: {conditioning[i].tolist()} ({cond_str})")

    # Test flattening (as done in training)
    print("\n4. Testing flattening for training compatibility...")
    images_flat = images.flatten(1)  # (batch_size, 4*96*96)
    print(f"   Flattened shape: {images_flat.shape}")
    print(f"   Expected output_dim: {cfg.OUTPUT_DIM}")
    print(f"   Match: {images_flat.shape[1] == cfg.OUTPUT_DIM}")

    # Test multiple batches to verify randomness
    print("\n5. Testing multiple batches (checking randomness)...")
    for i in range(3):
        images_batch, cond_batch = loader.get_training_batch(batch_size=64)
        hsc_count = (cond_batch[:, 0] == 1.0).sum().item()
        legacy_count = (cond_batch[:, 1] == 1.0).sum().item()
        print(f"   Batch {i+1}: {hsc_count} HSC, {legacy_count} Legacy Survey")

        #for each batch show the range of values
        print(f"   Batch {i+1} range: [{images_batch.min():.4f}, {images_batch.max():.4f}]")
        print(f"   Batch {i+1} mean: {images_batch.mean():.4f}")
        print(f"   Batch {i+1} std: {images_batch.std():.4f}")

    global_min = float('inf')
    global_max = float('-inf')
    for i in range(256):
        images_batch, cond_batch = loader.get_training_batch(batch_size=64)
        print(f"   Batch {i+1} range: [{images_batch.min():.4f}, {images_batch.max():.4f}]")

        global_min = min(global_min, images_batch.min().item())
        global_max = max(global_max, images_batch.max().item())

    print(f"   Global range: [{global_min:.4f}, {global_max:.4f}]")



    # Test get_batch (separate batches)
    print("\n6. Testing get_batch() (separate HSC and Legacy batches)...")
    hsc_batch, legacy_batch, hsc_onehots, legacy_onehots = loader.get_batch(batch_size=4)
    print(f"   HSC batch shape: {hsc_batch.shape}")
    print(f"   Legacy batch shape: {legacy_batch.shape}")
    print(f"   HSC one-hots shape: {hsc_onehots.shape}")
    print(f"   Legacy one-hots shape: {legacy_onehots.shape}")
    print(f"   All HSC one-hots are [1, 0]: {torch.all(hsc_onehots == torch.tensor([1.0, 0.0]))}")
    print(f"   All Legacy one-hots are [0, 1]: {torch.all(legacy_onehots == torch.tensor([0.0, 1.0]))}")

    loader.close()
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
