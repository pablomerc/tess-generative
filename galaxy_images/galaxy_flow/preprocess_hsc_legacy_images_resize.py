"""
Preprocessing script to load all HSC and Legacy Survey images once and store them in a faster format.

This script:
1. Loads all HSC and Legacy Survey images from parquet files
2. For HSC images: Applies preprocessing to 96x96, then crops to SxS and resizes to final_size x final_size
3. For Legacy images: Applies full preprocessing pipeline directly to final_size x final_size
4. Extracts all metadata columns (excluding rgb, blobmodel, object_mask)
5. Saves both HSC and Legacy Survey images and metadata to HDF5 format for fast random access during training

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
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
import glob
import json

# Add parent directory to path
current_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_path))
sys.path.insert(0, parent_dir)

from galaxy_images.image_preprocessing import preprocess_image


# All columns to extract (excluding rgb, blobmodel, object_mask)
METADATA_COLUMNS = [
    'legacysurvey_object_id',
    'hsc_object_id',
    'legacysurvey_image',
    'hsc_image',
    'catalog',
    'z_cmodel_mag',
    'z_sdssshape_shape12',
    'a_g',
    'a_r',
    'y_sdssshape_psf_shape11',
    'FLUX_R',
    'FLUX_W3',
    'i_cmodel_mag',
    'z_sdssshape_shape22',
    'r_sdssshape_psf_shape11',
    'z_extendedness_value',
    'r_cmodel_mag',
    'i_sdssshape_shape11',
    'i_sdssshape_shape12',
    'y_sdssshape_psf_shape12',
    'g_sdssshape_shape11',
    'a_i',
    'y_cmodel_mag',
    'g_sdssshape_psf_shape11',
    'g_sdssshape_shape12',
    'y_sdssshape_psf_shape22',
    'y_sdssshape_shape11',
    'a_y',
    'i_sdssshape_psf_shape12',
    'g_sdssshape_shape22',
    'i_sdssshape_psf_shape11',
    'r_sdssshape_psf_shape22',
    'FLUX_W1',
    'a_z',
    'g_cmodel_mag',
    'g_sdssshape_psf_shape12',
    'SHAPE_E1',
    'r_sdssshape_psf_shape12',
    'EBV',
    'g_extendedness_value',
    'r_sdssshape_shape11',
    'FLUX_I',
    'i_sdssshape_shape22',
    'y_sdssshape_shape22',
    'y_extendedness_value',
    'i_cmodel_magerr',
    'z_cmodel_magerr',
    'z_sdssshape_shape11',
    'y_cmodel_magerr',
    'r_extendedness_value',
    'SHAPE_R',
    'SHAPE_E2',
    'r_cmodel_magerr',
    'i_sdssshape_psf_shape22',
    'FLUX_W4',
    'FLUX_Z',
    'y_sdssshape_shape12',
    'FLUX_W2',
    'g_cmodel_magerr',
    'r_sdssshape_shape12',
    'z_sdssshape_psf_shape12',
    'i_extendedness_value',
    'z_sdssshape_psf_shape11',
    'r_sdssshape_shape22',
    'FLUX_G',
    'g_sdssshape_psf_shape22',
    'z_sdssshape_psf_shape22',
]


def get_dtype_and_shape(value):
    """
    Determine appropriate HDF5 dtype and shape for a value.

    Returns:
        (dtype, shape, is_string) tuple
    """
    if value is None:
        return np.float32, (), False

    value_array = np.asarray(value)

    # Check if it's a string or bytes
    if value_array.dtype.kind in ['U', 'S', 'O']:
        # For strings, we'll store as variable-length UTF-8
        return h5py.string_dtype(encoding='utf-8'), (), True

    # For numeric arrays
    if value_array.dtype.kind in ['f', 'i', 'u']:
        return value_array.dtype, value_array.shape, False

    # For other types, try to convert to float
    try:
        value_float = np.asarray(value, dtype=np.float32)
        return np.float32, value_float.shape, False
    except:
        # Fallback: store as string
        return h5py.string_dtype(encoding='utf-8'), (), True


def extract_value(record, key):
    """Extract a value from a record, handling nested structures."""
    value = record.get(key)

    if value is None:
        return None

    # Handle nested dictionaries (like catalog, legacysurvey_image, hsc_image)
    if isinstance(value, dict):
        # For image dictionaries, they might have 'flux' arrays which are large
        # Store the entire structure as JSON string
        # Note: This may be inefficient for very large image arrays
        try:
            # Convert numpy arrays in dict to lists for JSON serialization
            def convert_for_json(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.floating)):
                    return obj.item()
                elif isinstance(obj, dict):
                    return {k: convert_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_for_json(item) for item in obj]
                return obj
            converted_value = convert_for_json(value)
            return json.dumps(converted_value)
        except (TypeError, ValueError) as e:
            # Fallback: convert to string representation
            return str(value)

    # Handle numpy arrays
    if isinstance(value, np.ndarray):
        # For very large arrays (like image flux), we might want to skip or compress
        # But since user requested these columns, we'll store them
        return value

    # Handle lists
    if isinstance(value, (list, tuple)):
        return np.asarray(value)

    return value


def crop_and_resize(img, S=78, out_size=48):
    """
    Crop and resize function from compare_embedding_different_sizes.ipynb

    Args:
        img: Input image tensor with shape:
            (H, W)
            (C, H, W)
            (B, H, W)
            (B, C, H, W)
        S: Size to crop to before resizing (default: 78)
        out_size: Final output size (default: 48)

    Returns:
        Resized image with same shape structure, but spatial dims -> out_size x out_size
    """
    # Normalize shape to (B, C, H, W)
    if img.ndim == 2:                  # (H, W)
        img = img.unsqueeze(0).unsqueeze(0)
    elif img.ndim == 3:
        if img.shape[0] <= 4:          # (C, H, W)
            img = img.unsqueeze(0)
        else:                          # (B, H, W)
            img = img.unsqueeze(1)
    elif img.ndim != 4:
        raise ValueError(f"Unsupported shape {img.shape}")

    B, C, H, W = img.shape
    assert H >= S and W >= S

    top = (H - S) // 2
    left = (W - S) // 2

    crop = img[:, :, top:top+S, left:left+S]

    resized = F.interpolate(
        crop,
        size=(out_size, out_size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )

    # Squeeze back if needed
    if resized.shape[0] == 1:
        resized = resized.squeeze(0)
    if resized.shape[0] == 1 and resized.ndim == 3:
        resized = resized.squeeze(0)

    return resized


def preprocess_all_hsc_legacy_images(
    # dataset_path: str = "/mnt/scratch/legacysurvey_hsc_crossmatched/data",
    # dataset_path: str = "/Users/pablom.perez/Desktop/data/legacysurvey_hsc_crossmatched/",
    dataset_path: str = "/data/vision/billf/scratch/pablomer/legacysurvey_hsc/data/",
    # output_path: str = "/mnt/scratch/legacysurvey_hsc_crossmatched/preprocessed_hsc_legacy.h5",
    # output_path: str = "/Users/pablom.perez/Desktop/data/legacysurvey_hsc_crossmatched/preprocessed_hsc_legacy.h5",
    output_path: str = "/data/vision/billf/scratch/pablomer/legacysurvey_hsc/preprocessed_hsc_legacy_48x48_resized_all.h5",
    final_size: int = 48,
    batch_size: int = 100,
    files_to_use: int = 4,  # Default to 4 files for testing
    crop_resize_S: int = 78,  # Size to crop to before resizing HSC images
):
    """
    Load all HSC and Legacy Survey images, preprocess them, and save to HDF5 format.

    HSC images: Preprocessed to 96x96, then cropped to SxS and resized to final_size x final_size
    Legacy images: Preprocessed directly to final_size x final_size

    Args:
        dataset_path: Path to directory containing parquet files
        output_path: Path to output HDF5 file
        final_size: Final output size for images (default: 48)
        batch_size: Number of images to process in memory at once
        files_to_use: Number of parquet files to use (None = all)
        crop_resize_S: Size to crop HSC images to before resizing (default: 78)
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

    # Verify metadata columns exist in the dataset
    available_columns = set(dataset.column_names)
    columns_to_extract = [col for col in METADATA_COLUMNS if col in available_columns]
    missing_columns = [col for col in METADATA_COLUMNS if col not in available_columns]

    if missing_columns:
        print(f"Warning: {len(missing_columns)} metadata columns not found in dataset: {missing_columns}")

    print(f"Extracting {len(columns_to_extract)} metadata columns")

    # Preprocess and save to HDF5
    print(f"\nPreprocessing and saving to {output_path}...")

    # First pass: determine data types and shapes for each metadata column
    print("Determining metadata column types and shapes...")
    column_info = {}
    sample_record = dataset[0]

    for col in columns_to_extract:
        try:
            value = extract_value(sample_record, col)
            dtype, shape, is_string = get_dtype_and_shape(value)
            column_info[col] = {
                'dtype': dtype,
                'shape': shape,
                'is_string': is_string,
                'is_vlen_string': is_string  # Variable-length strings
            }
        except Exception as e:
            print(f"Warning: Could not determine type for column {col}: {e}")
            # Default to float32 scalar
            column_info[col] = {
                'dtype': np.float32,
                'shape': (),
                'is_string': False,
                'is_vlen_string': False
            }

    # Create HDF5 file
    with h5py.File(output_path, 'w') as f:
        # Create dataset for preprocessed HSC images
        # Shape: (num_images, 4, final_size, final_size)
        hsc_dset = f.create_dataset(
            'hsc_images',
            shape=(num_rows, 4, final_size, final_size),
            dtype=np.float32,
            compression='gzip',
            compression_opts=4,
            chunks=(min(batch_size, num_rows), 4, final_size, final_size)
        )

        # Create dataset for preprocessed Legacy Survey images
        # Shape: (num_images, 4, final_size, final_size)
        legacy_dset = f.create_dataset(
            'legacy_images',
            shape=(num_rows, 4, final_size, final_size),
            dtype=np.float32,
            compression='gzip',
            compression_opts=4,
            chunks=(min(batch_size, num_rows), 4, final_size, final_size)
        )

        # Create dataset for indices (to track which original image each preprocessed image corresponds to)
        indices_dset = f.create_dataset(
            'indices',
            shape=(num_rows,),
            dtype=np.int64
        )

        # Create datasets for each metadata column
        column_datasets = {}
        for col in columns_to_extract:
            info = column_info[col]
            shape = info['shape']

            if info['is_vlen_string']:
                # Variable-length string dataset
                dt = h5py.string_dtype(encoding='utf-8')
                dset = f.create_dataset(
                    col,
                    shape=(num_rows,),
                    dtype=dt,
                    compression='gzip',
                    compression_opts=4,
                    chunks=(min(batch_size, num_rows),)
                )
            else:
                # Numeric dataset
                if len(shape) == 0:
                    # Scalar
                    dset_shape = (num_rows,)
                else:
                    # Array
                    dset_shape = (num_rows,) + shape

                dset = f.create_dataset(
                    col,
                    shape=dset_shape,
                    dtype=info['dtype'],
                    compression='gzip',
                    compression_opts=4,
                    chunks=(min(batch_size, num_rows),) + (shape if len(shape) > 0 else ())
                )

            column_datasets[col] = dset

        # Process and save images
        valid_count = 0
        for idx in tqdm(range(num_rows), desc="Preprocessing"):
            try:
                record = dataset[idx]

                # Process HSC image: first preprocess to 96x96, then crop and resize to final_size
                hsc_image = record['hsc_image']
                hsc_flux = np.array(hsc_image['flux'])[:4, :, :]  # (4, H, W)
                hsc_flux_tensor = torch.from_numpy(hsc_flux).float().unsqueeze(0)  # (1, 4, H, W)

                # Step 1: Preprocess to 96x96
                hsc_processed_96 = preprocess_image(
                    hsc_flux_tensor,
                    bands=hsc_bands,
                    crop_size=96,
                    apply_range_compression=True
                )  # (1, 4, 96, 96)

                # Step 2: Crop to SxS and resize to final_size x final_size
                hsc_processed = crop_and_resize(
                    hsc_processed_96,
                    S=crop_resize_S,
                    out_size=final_size
                )  # (1, 4, final_size, final_size) or (4, final_size, final_size)

                # Ensure it's (4, final_size, final_size) for saving
                if hsc_processed.ndim == 4:
                    hsc_processed = hsc_processed.squeeze(0)
                hsc_processed_np = hsc_processed.numpy()  # (4, final_size, final_size)

                # Process Legacy Survey image: preprocess directly to final_size
                legacy_image = record['legacysurvey_image']
                legacy_flux = np.array(legacy_image['flux'])[:4, :, :]  # (4, H, W)
                legacy_flux_tensor = torch.from_numpy(legacy_flux).float().unsqueeze(0)  # (1, 4, H, W)

                legacy_processed = preprocess_image(
                    legacy_flux_tensor,
                    bands=legacy_bands,
                    crop_size=final_size,
                    apply_range_compression=True
                )  # (1, 4, final_size, final_size)
                legacy_processed_np = legacy_processed.squeeze(0).numpy()  # (4, final_size, final_size)

                # Save images to HDF5
                hsc_dset[valid_count] = hsc_processed_np
                legacy_dset[valid_count] = legacy_processed_np
                indices_dset[valid_count] = idx

                # Extract and save metadata
                for col in columns_to_extract:
                    try:
                        value = extract_value(record, col)
                        dset = column_datasets[col]

                        if value is None:
                            # Fill with NaN or empty string
                            if column_info[col]['is_string']:
                                dset[valid_count] = ""
                            else:
                                dset[valid_count] = np.nan if dset.dtype.kind == 'f' else 0
                        else:
                            if column_info[col]['is_vlen_string']:
                                # Store as string
                                if isinstance(value, (dict, list)):
                                    dset[valid_count] = json.dumps(value)
                                else:
                                    dset[valid_count] = str(value)
                            else:
                                # Store as numeric array
                                value_array = np.asarray(value, dtype=column_info[col]['dtype'])
                                if len(column_info[col]['shape']) == 0:
                                    # Scalar
                                    dset[valid_count] = value_array.item() if value_array.size == 1 else value_array.flat[0]
                                else:
                                    # Array
                                    dset[valid_count] = value_array
                    except Exception as e:
                        print(f"Error processing metadata column {col} for record {idx}: {e}")
                        # Fill with default value
                        dset[valid_count] = np.nan if not column_info[col]['is_string'] else ""

                valid_count += 1

            except Exception as e:
                print(f"Error processing image {idx}: {e}")
                continue

        # Resize datasets to actual number of valid images
        if valid_count < num_rows:
            print(f"\nResizing datasets from {num_rows} to {valid_count} valid images...")
            hsc_dset.resize((valid_count, 4, final_size, final_size))
            legacy_dset.resize((valid_count, 4, final_size, final_size))
            indices_dset.resize((valid_count,))
            # Resize metadata datasets
            for col in columns_to_extract:
                dset = column_datasets[col]
                if len(column_info[col]['shape']) == 0:
                    dset.resize((valid_count,))
                else:
                    dset.resize((valid_count,) + column_info[col]['shape'])

        # Save metadata
        f.attrs['num_images'] = valid_count
        f.attrs['final_size'] = final_size
        f.attrs['hsc_crop_resize_S'] = crop_resize_S
        f.attrs['num_channels'] = 4
        f.attrs['hsc_bands'] = ','.join(hsc_bands)
        f.attrs['legacy_bands'] = ','.join(legacy_bands)
        f.attrs['num_metadata_columns'] = len(columns_to_extract)
        f.attrs['metadata_columns'] = ','.join(columns_to_extract)
        f.attrs['excluded_columns'] = 'rgb,blobmodel,object_mask'

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
        default="/data/vision/billf/scratch/pablomer/legacysurvey_hsc/preprocessed_hsc_legacy_48x48_resized_all.h5",
        help="Path to output HDF5 file"
    )
    parser.add_argument(
        "--final_size",
        type=int,
        default=48,
        help="Final output size for images (default: 48)"
    )
    parser.add_argument(
        "--crop_resize_S",
        type=int,
        default=78,
        help="Size to crop HSC images to before resizing (default: 78)"
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
        final_size=args.final_size,
        batch_size=args.batch_size,
        files_to_use=files_to_use,
        crop_resize_S=args.crop_resize_S,
    )
