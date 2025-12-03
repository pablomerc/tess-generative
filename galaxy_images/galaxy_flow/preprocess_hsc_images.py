"""
Preprocessing script to load all HSC images once and store them in a faster format.

This script:
1. Loads all HSC images from parquet files
2. Applies the full preprocessing pipeline (crop, clamp, rescale, range compress)
3. Saves to HDF5 format for fast random access during training

Note: Normalization to [-1, 1] range is handled in the training code, not here.
"""

import os
import sys
import h5py
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
import glob

# Add parent directory to path
current_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_path))
sys.path.insert(0, parent_dir)

from galaxy_images.image_preprocessing import preprocess_image
from galaxy_images.galaxy_triplets import TripletCreator


def preprocess_all_hsc_images(
    dataset_path: str = "/mnt/scratch/legacysurvey_hsc_crossmatched/data",
    output_path: str = "/mnt/scratch/legacysurvey_hsc_crossmatched/preprocessed_hsc.h5",
    crop_size: int = 96,
    batch_size: int = 100,
    files_to_use: int = 40,  # Default to 40 files for testing
):
    """
    Load all HSC images, preprocess them, and save to HDF5 format.

    Args:
        dataset_path: Path to directory containing parquet files
        output_path: Path to output HDF5 file
        crop_size: Size to crop images to (default: 96)
        batch_size: Number of images to process in memory at once
        files_to_use: Number of parquet files to use (None = all)
    """
    print(f"Loading dataset from {dataset_path}")

    # Load dataset using TripletCreator's logic
    pattern = os.path.join(dataset_path, "*.parquet")
    all_files = sorted(glob.glob(pattern))
    if not all_files:
        raise FileNotFoundError(f"No parquet files found under {pattern}")

    if files_to_use is not None and files_to_use > 0:
        all_files = all_files[:files_to_use]

    print(f"Found {len(all_files)} parquet files")

    # Load dataset
    dataset = load_dataset("parquet", data_files=all_files, split="train")
    dataset = dataset.with_format("numpy")
    num_rows = len(dataset)
    print(f"Total number of images: {num_rows}")

    # HSC band names (first 4 bands: G, R, I, Z)
    hsc_bands = ['HSC-G', 'HSC-R', 'HSC-I', 'HSC-Z']

    # Preprocess and save to HDF5
    print(f"\nPreprocessing and saving to {output_path}...")

    # Create HDF5 file
    with h5py.File(output_path, 'w') as f:
        # Create dataset for preprocessed images
        # Shape: (num_images, 4, crop_size, crop_size)
        dset = f.create_dataset(
            'images',
            shape=(num_rows, 4, crop_size, crop_size),
            dtype=np.float32,
            compression='gzip',
            compression_opts=4,
            chunks=(min(batch_size, num_rows), 4, crop_size, crop_size)
        )

        # Create dataset for indices (to track which original image each preprocessed image corresponds to)
        indices_dset = f.create_dataset(
            'indices',
            shape=(num_rows,),
            dtype=np.int64
        )

        # Process and save images
        valid_count = 0
        for idx in tqdm(range(num_rows), desc="Preprocessing"):
            try:
                record = dataset[idx]
                hsc_image = record['hsc_image']

                # Extract flux channels (first 4 bands)
                flux = np.array(hsc_image['flux'])[:4, :, :]  # (4, H, W)

                # Convert to tensor and add batch dimension
                flux_tensor = torch.from_numpy(flux).float().unsqueeze(0)  # (1, 4, H, W)

                # Apply preprocessing (crop, clamp, rescale, range compress)
                # Note: normalization to [-1, 1] is done in training code, not here
                processed = preprocess_image(
                    flux_tensor,
                    bands=hsc_bands,
                    crop_size=crop_size,
                    apply_range_compression=True
                )  # (1, 4, 96, 96)

                # Remove batch dimension and convert to numpy
                processed_np = processed.squeeze(0).numpy()  # (4, 96, 96)

                # Save to HDF5
                dset[valid_count] = processed_np
                indices_dset[valid_count] = idx
                valid_count += 1

            except Exception as e:
                print(f"Error processing image {idx}: {e}")
                continue

        # Resize datasets to actual number of valid images
        if valid_count < num_rows:
            print(f"\nResizing datasets from {num_rows} to {valid_count} valid images...")
            dset.resize((valid_count, 4, crop_size, crop_size))
            indices_dset.resize((valid_count,))

        # Save metadata
        f.attrs['num_images'] = valid_count
        f.attrs['crop_size'] = crop_size
        f.attrs['num_channels'] = 4
        f.attrs['bands'] = ','.join(hsc_bands)

    print(f"\nPreprocessing complete!")
    print(f"Saved {valid_count} preprocessed images to {output_path}")
    print(f"File size: {os.path.getsize(output_path) / (1024**3):.2f} GB")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess HSC images and save to HDF5")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/mnt/scratch/legacysurvey_hsc_crossmatched/data",
        help="Path to directory containing parquet files"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/mnt/scratch/legacysurvey_hsc_crossmatched/preprocessed_hsc.h5",
        help="Path to output HDF5 file"
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        default=96,
        help="Size to crop images to (default: 96)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Batch size for processing (default: 100)"
    )
    parser.add_argument(
        "--files_to_use",
        type=int,
        default=4,
        help="Number of parquet files to use (default: 40, use 0 for all files)"
    )

    args = parser.parse_args()

    # Convert 0 to None (meaning use all files)
    files_to_use = None if args.files_to_use == 0 else args.files_to_use

    preprocess_all_hsc_images(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        crop_size=args.crop_size,
        batch_size=args.batch_size,
        files_to_use=files_to_use,
    )
