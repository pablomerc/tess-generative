"""
Preprocessing script to extract and save metadata columns from HSC and Legacy Survey dataset.

This script:
1. Loads all HSC and Legacy Survey data from parquet files (same order as preprocess_hsc_legacy_images.py)
2. Extracts all specified metadata columns (excluding rgb, blobmodel, object_mask)
3. Saves metadata to HDF5 format for fast random access during training

The data is stored in the same order as in preprocess_hsc_legacy_images.py to ensure alignment.
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
from tqdm import tqdm
from datasets import load_dataset
import glob
import json

# Add parent directory to path
current_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_path))
sys.path.insert(0, parent_dir)


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


def preprocess_all_hsc_legacy_metadata(
    dataset_path: str = "/data/vision/billf/scratch/pablomer/legacysurvey_hsc/data/",
    output_path: str = "/data/vision/billf/scratch/pablomer/legacysurvey_hsc/preprocessed_hsc_legacy_metadata.h5",
    batch_size: int = 100,
    files_to_use: int = None,
    max_examples: int = None,
):
    """
    Load all metadata columns from HSC and Legacy Survey dataset and save to HDF5 format.

    This function processes data in the same order as preprocess_hsc_legacy_images.py
    to ensure alignment between metadata and preprocessed images.

    Args:
        dataset_path: Path to directory containing parquet files
        output_path: Path to output HDF5 file
        batch_size: Number of records to process in memory at once (for chunking)
        files_to_use: Number of parquet files to use (None = all)
        max_examples: Maximum number of examples to process (None = all)
    """
    print(f"Loading dataset from {dataset_path}")

    # Load dataset using the same logic as preprocess_hsc_legacy_images.py
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
    total_num_rows = len(dataset)

    # Limit number of examples if specified
    if max_examples is not None and max_examples > 0:
        num_rows = min(max_examples, total_num_rows)
        print(f"Total number of records: {total_num_rows}")
        print(f"Limiting to first {num_rows} examples")
    else:
        num_rows = total_num_rows
        print(f"Total number of records: {num_rows}")

    # Verify all columns exist in the dataset
    available_columns = set(dataset.column_names)
    columns_to_extract = [col for col in METADATA_COLUMNS if col in available_columns]
    missing_columns = [col for col in METADATA_COLUMNS if col not in available_columns]

    if missing_columns:
        print(f"Warning: {len(missing_columns)} columns not found in dataset: {missing_columns}")

    print(f"Extracting {len(columns_to_extract)} metadata columns")
    print(f"Columns: {columns_to_extract}")

    # Preprocess and save to HDF5
    print(f"\nPreprocessing and saving to {output_path}...")

    # First pass: determine data types and shapes for each column
    print("Determining data types and shapes...")
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
        # Create dataset for indices (to track which original record each metadata corresponds to)
        indices_dset = f.create_dataset(
            'indices',
            shape=(num_rows,),
            dtype=np.int64
        )

        # Create datasets for each column
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

        # Process and save metadata
        valid_count = 0
        for idx in tqdm(range(num_rows), desc="Preprocessing metadata"):
            try:
                record = dataset[idx]

                # Extract and save each column
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
                        print(f"Error processing column {col} for record {idx}: {e}")
                        # Fill with default value
                        dset[valid_count] = np.nan if not column_info[col]['is_string'] else ""

                # Save index
                indices_dset[valid_count] = idx
                valid_count += 1

            except Exception as e:
                print(f"Error processing record {idx}: {e}")
                continue

        # Resize datasets to actual number of valid records
        if valid_count < num_rows:
            print(f"\nResizing datasets from {num_rows} to {valid_count} valid records...")
            indices_dset.resize((valid_count,))
            for col in columns_to_extract:
                dset = column_datasets[col]
                if len(column_info[col]['shape']) == 0:
                    dset.resize((valid_count,))
                else:
                    dset.resize((valid_count,) + column_info[col]['shape'])

        # Save metadata
        f.attrs['num_records'] = valid_count
        f.attrs['num_columns'] = len(columns_to_extract)
        f.attrs['columns'] = ','.join(columns_to_extract)
        f.attrs['excluded_columns'] = 'rgb,blobmodel,object_mask'

    print(f"\nPreprocessing complete!")
    print(f"Saved {valid_count} metadata records to {output_path}")
    print(f"File size: {os.path.getsize(output_path) / (1024**3):.2f} GB")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract and save HSC and Legacy Survey metadata to HDF5")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/data/vision/billf/scratch/pablomer/legacysurvey_hsc/data/",
        help="Path to directory containing parquet files"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/data/vision/billf/scratch/pablomer/legacysurvey_hsc/preprocessed_hsc_legacy_metadata.h5",
        help="Path to output HDF5 file"
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
        help="Number of parquet files to use (use 0 for all files)"
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=8192,
        help="Maximum number of examples to process (default: 512, use 0 for all)"
    )

    args = parser.parse_args()

    # Convert 0 to None (meaning use all files/examples)
    files_to_use = None if args.files_to_use == 0 else args.files_to_use
    max_examples = None if args.max_examples == 0 else args.max_examples

    # Modify output_path to include number of examples if max_examples is set
    output_path = args.output_path
    if max_examples is not None:
        # Insert the number before the file extension
        base, ext = os.path.splitext(output_path)
        output_path = f"{base}_{max_examples}{ext}"

    preprocess_all_hsc_legacy_metadata(
        dataset_path=args.dataset_path,
        output_path=output_path,
        batch_size=args.batch_size,
        files_to_use=files_to_use,
        max_examples=max_examples,
    )
