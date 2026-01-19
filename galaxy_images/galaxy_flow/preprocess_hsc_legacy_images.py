"""
Preprocessing script to load all HSC and Legacy Survey images once and store them in a faster format.

This script:
1. Loads all HSC and Legacy Survey images from parquet files
2. Applies the full preprocessing pipeline (crop, clamp, rescale, range compress)
3. Saves both HSC and Legacy Survey images to HDF5 format for fast random access during training

Note: Normalization to [-1, 1] range is handled in the training code, not here.
"""

import os
import sys

# Set HuggingFace cache to a writable location BEFORE importing datasets
# (datasets library reads these env vars at import time)
hf_cache_dir = "/data/vision/billf/scratch/pablomer/.cache/huggingface"
os.makedirs(hf_cache_dir, exist_ok=True)
os.environ['HF_HOME'] = hf_cache_dir
os.environ['HF_DATASETS_CACHE'] = os.path.join(hf_cache_dir, 'datasets')

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


def preprocess_all_hsc_legacy_images(
    # dataset_path: str = "/mnt/scratch/legacysurvey_hsc_crossmatched/data",
    # dataset_path: str = "/Users/pablom.perez/Desktop/data/legacysurvey_hsc_crossmatched/",
    dataset_path: str = "/data/vision/billf/scratch/pablomer/legacysurvey_hsc/data/",
    # output_path: str = "/mnt/scratch/legacysurvey_hsc_crossmatched/preprocessed_hsc_legacy.h5",
    # output_path: str = "/Users/pablom.perez/Desktop/data/legacysurvey_hsc_crossmatched/preprocessed_hsc_legacy.h5",
    output_path: str = "/data/vision/billf/scratch/pablomer/legacysurvey_hsc/preprocessed_hsc_legacy_48x48_all.h5",
    crop_size: int = 96,
    batch_size: int = 100,
    files_to_use: int = 4,  # Default to 4 files for testing
):
    """
    Load all HSC and Legacy Survey images, preprocess them, and save to HDF5 format.

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

    total_files_found = len(all_files)
    print(f"Total parquet files found: {total_files_found}")

    if files_to_use is not None and files_to_use > 0:
        print(f"Limiting to {files_to_use} files (from {total_files_found} total)")
        all_files = all_files[:files_to_use]
    else:
        print(f"Using all {total_files_found} files")

    print(f"Processing {len(all_files)} parquet files")

    # Load dataset
    dataset = load_dataset("parquet", data_files=all_files, split="train")
    dataset = dataset.with_format("numpy")
    num_rows = len(dataset)
    print(f"Total number of images: {num_rows}")

    # Band names (first 4 bands: G, R, I, Z)
    hsc_bands = ['HSC-G', 'HSC-R', 'HSC-I', 'HSC-Z']
    legacy_bands = ['DES-G', 'DES-R', 'DES-I', 'DES-Z']

    # Preprocess and save to HDF5
    print(f"\nPreprocessing and saving to {output_path}...")

    # Create HDF5 file
    with h5py.File(output_path, 'w') as f:
        # Create dataset for preprocessed HSC images
        # Shape: (num_images, 4, crop_size, crop_size)
        hsc_dset = f.create_dataset(
            'hsc_images',
            shape=(num_rows, 4, crop_size, crop_size),
            dtype=np.float32,
            compression='gzip',
            compression_opts=4,
            chunks=(min(batch_size, num_rows), 4, crop_size, crop_size)
        )

        # Create dataset for preprocessed Legacy Survey images
        # Shape: (num_images, 4, crop_size, crop_size)
        legacy_dset = f.create_dataset(
            'legacy_images',
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

                # Process HSC image
                hsc_image = record['hsc_image']
                hsc_flux = np.array(hsc_image['flux'])[:4, :, :]  # (4, H, W)
                hsc_flux_tensor = torch.from_numpy(hsc_flux).float().unsqueeze(0)  # (1, 4, H, W)

                hsc_processed = preprocess_image(
                    hsc_flux_tensor,
                    bands=hsc_bands,
                    crop_size=crop_size,
                    apply_range_compression=True
                )  # (1, 4, 96, 96)
                hsc_processed_np = hsc_processed.squeeze(0).numpy()  # (4, 96, 96)

                # Process Legacy Survey image
                legacy_image = record['legacysurvey_image']
                legacy_flux = np.array(legacy_image['flux'])[:4, :, :]  # (4, H, W)
                legacy_flux_tensor = torch.from_numpy(legacy_flux).float().unsqueeze(0)  # (1, 4, H, W)

                legacy_processed = preprocess_image(
                    legacy_flux_tensor,
                    bands=legacy_bands,
                    crop_size=crop_size,
                    apply_range_compression=True
                )  # (1, 4, 96, 96)
                legacy_processed_np = legacy_processed.squeeze(0).numpy()  # (4, 96, 96)

                # Save to HDF5
                hsc_dset[valid_count] = hsc_processed_np
                legacy_dset[valid_count] = legacy_processed_np
                indices_dset[valid_count] = idx
                valid_count += 1

            except Exception as e:
                print(f"Error processing image {idx}: {e}")
                continue

        # Resize datasets to actual number of valid images
        if valid_count < num_rows:
            print(f"\nResizing datasets from {num_rows} to {valid_count} valid images...")
            hsc_dset.resize((valid_count, 4, crop_size, crop_size))
            legacy_dset.resize((valid_count, 4, crop_size, crop_size))
            indices_dset.resize((valid_count,))

        # Save metadata
        f.attrs['num_images'] = valid_count
        f.attrs['crop_size'] = crop_size
        f.attrs['num_channels'] = 4
        f.attrs['hsc_bands'] = ','.join(hsc_bands)
        f.attrs['legacy_bands'] = ','.join(legacy_bands)

    print(f"\nPreprocessing complete!")
    print(f"Saved {valid_count} preprocessed images to {output_path}")
    print(f"File size: {os.path.getsize(output_path) / (1024**3):.2f} GB")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess HSC and Legacy Survey images and save to HDF5")
    parser.add_argument(
        "--dataset_path",
        type=str,
        # default="/mnt/scratch/legacysurvey_hsc_crossmatched/data",
        # default="/Users/pablom.perez/Desktop/data/legacysurvey_hsc_crossmatched/",
        default="/data/vision/billf/scratch/pablomer/legacysurvey_hsc/data/",
        help="Path to directory containing parquet files"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        # default="/mnt/scratch/legacysurvey_hsc_crossmatched/preprocessed_hsc_legacy.h5",
        # default="/Users/pablom.perez/Desktop/data/legacysurvey_hsc_crossmatched/preprocessed_hsc_legacy.h5",
        default="/data/vision/billf/scratch/pablomer/legacysurvey_hsc/preprocessed_hsc_legacy_48x48_all.h5",
        help="Path to output HDF5 file"
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        default=48,
        help="Size to crop images to (default: 48)"
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
        default=0,
        help="Number of parquet files to use (default: 40, use 0 for all files)"
    )

    args = parser.parse_args()

    # Convert 0 to None (meaning use all files)
    files_to_use = None if args.files_to_use == 0 else args.files_to_use

    preprocess_all_hsc_legacy_images(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        crop_size=args.crop_size,
        batch_size=args.batch_size,
        files_to_use=files_to_use,
    )
