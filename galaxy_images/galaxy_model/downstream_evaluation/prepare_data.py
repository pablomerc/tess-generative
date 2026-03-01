"""
Script to generate embeddings and load metadata for downstream prediction tasks.

This script:
1. Loads 8k image examples
2. Generates embeddings using the double-encoder model
3. Loads metadata for those images
4. Creates a dataset with embeddings (features) and metadata (labels) separated
5. Saves the dataset to an HDF5 file for easy loading in downstream tasks
"""

import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import importlib.util
# from double_train_fm import ConditionalFlowMatchingModule
# Import module with hyphen in name using importlib
spec = importlib.util.spec_from_file_location(
    "double_train_fm_no_attn",
    Path(__file__).parent.parent / "double_train_fm_no-attn.py"
)
double_train_fm_no_attn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(double_train_fm_no_attn)
ConditionalFlowMatchingModule = double_train_fm_no_attn.ConditionalFlowMatchingModule


from torch.utils.data import DataLoader
from data import HSCLegacyDatasetZoom
import numpy as np
import h5py

# ============================================================================
# Configuration
# ============================================================================
checkpoint_path = '/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy-flow-matching/wu1csh99/checkpoints/latest-step=step=75000.ckpt'
# # attn, geom loss 7.5e-4, zdim=16

# checkpoint_path = '/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy-flow-matching/srj4opub/checkpoints/latest-step=step=75000.ckpt'
# class cond

hdf5_path = '/data/vision/billf/scratch/pablomer/legacysurvey_hsc/preprocessed_hsc_legacy_48x48_all.h5'
metadata_path = '/data/vision/billf/scratch/pablomer/legacysurvey_hsc/preprocessed_hsc_legacy_metadata_8192.h5'
output_path = '/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy_images/galaxy_model/downstream_evaluation/downstream_dataset.h5'
num_examples = 8192
batch_size = 512

# All metadata parameters to load
# metadata_params = [
#     'EBV', 'FLUX_G', 'FLUX_I', 'FLUX_R', 'FLUX_W1', 'FLUX_W2', 'FLUX_W3', 'FLUX_W4', 'FLUX_Z',
#     'SHAPE_E1', 'SHAPE_E2', 'SHAPE_R', 'a_g', 'a_i', 'a_r', 'a_y', 'a_z',
#     'g_cmodel_mag', 'g_cmodel_magerr', 'g_extendedness_value',
#     'g_sdssshape_psf_shape11', 'g_sdssshape_psf_shape12', 'g_sdssshape_psf_shape22',
#     'g_sdssshape_shape11', 'g_sdssshape_shape12', 'g_sdssshape_shape22',
#     'i_cmodel_mag', 'i_cmodel_magerr', 'i_extendedness_value',
#     'i_sdssshape_psf_shape11', 'i_sdssshape_psf_shape12', 'i_sdssshape_psf_shape22',
#     'i_sdssshape_shape11', 'i_sdssshape_shape12', 'i_sdssshape_shape22',
#     'r_cmodel_mag', 'r_cmodel_magerr', 'r_extendedness_value',
#     'r_sdssshape_psf_shape11', 'r_sdssshape_psf_shape12', 'r_sdssshape_psf_shape22',
#     'r_sdssshape_shape11', 'r_sdssshape_shape12', 'r_sdssshape_shape22',
#     'y_cmodel_mag', 'y_cmodel_magerr', 'y_extendedness_value',
#     'y_sdssshape_psf_shape11', 'y_sdssshape_psf_shape12', 'y_sdssshape_psf_shape22',
#     'y_sdssshape_shape11', 'y_sdssshape_shape12', 'y_sdssshape_shape22',
#     'z_cmodel_mag', 'z_cmodel_magerr', 'z_extendedness_value',
#     'z_sdssshape_psf_shape11', 'z_sdssshape_psf_shape12', 'z_sdssshape_psf_shape22',
#     'z_sdssshape_shape11', 'z_sdssshape_shape12', 'z_sdssshape_shape22'
# ]

metadata_params = [ # from keep list
    'EBV',
    'FLUX_G',
    'FLUX_I',
    'FLUX_R',
    'FLUX_W1',
    'FLUX_W2',
    'FLUX_W3',
    'FLUX_W4',
    'FLUX_Z',
    'SHAPE_E1',
    'SHAPE_E2',
    'SHAPE_R',
    'a_g',
    'a_i',
    'a_r',
    'a_y',
    'a_z',
    'g_cmodel_mag',
    'g_cmodel_magerr',
    'g_sdssshape_psf_shape11',
    'g_sdssshape_psf_shape12',
    'g_sdssshape_psf_shape22',
    'i_cmodel_mag',
    'i_cmodel_magerr',
    'i_extendedness_value',
    'i_sdssshape_psf_shape11',
    'i_sdssshape_psf_shape12',
    'i_sdssshape_psf_shape22',
    'r_cmodel_mag',
    'r_sdssshape_psf_shape11',
    'r_sdssshape_psf_shape12',
    'r_sdssshape_psf_shape22',
    'y_cmodel_mag',
    'y_cmodel_magerr',
    'y_extendedness_value',
    'z_cmodel_mag',
    'z_sdssshape_psf_shape11',
    'z_sdssshape_psf_shape12',
    'z_sdssshape_psf_shape22',
]


# metadata_params = ['EBV', 'FLUX_G', 'FLUX_I', 'FLUX_R', 'FLUX_W1', 'FLUX_W2', 'FLUX_W3', 'FLUX_W4', 'FLUX_Z',
#     'SHAPE_E1', 'SHAPE_E2', 'SHAPE_R',]


# metadata_params = ['FLUX_W2', 'FLUX_W3']

# ============================================================================
# Helper Functions
# ============================================================================
def get_device():
    """Determine and return the best available device."""
    device = torch.device('cpu')
    if torch.cuda.is_available():
        for gpu_id in range(torch.cuda.device_count()):
            try:
                test_tensor = torch.tensor([1.0], device=f'cuda:{gpu_id}')
                del test_tensor
                torch.cuda.empty_cache()
                device = torch.device(f'cuda:{gpu_id}')
                print(f"Using GPU {gpu_id}")
                return device
            except RuntimeError:
                continue
    print("No working GPU found, using CPU")
    return device


def load_metadata(metadata_path, idx_list, metadata_params):
    """
    Load metadata for specified indices.

    Args:
        metadata_path: Path to metadata HDF5 file
        idx_list: List of dataset indices to load
        metadata_params: List of parameter names to load

    Returns:
        metadata_dict: Dictionary mapping parameter names to numpy arrays
        param_names: List of valid parameter names that were found
    """
    print(f"\nLoading metadata from: {metadata_path}")
    metadata_dict = {}
    idx_to_metadata_idx = None

    with h5py.File(metadata_path, 'r') as f:
        # Check if there's an indices field to map metadata to dataset indices
        if 'indices' in f:
            metadata_indices = f['indices'][:]
            print(f"Found indices mapping: shape={metadata_indices.shape}")
            idx_to_metadata_idx = {dataset_idx: meta_idx for meta_idx, dataset_idx in enumerate(metadata_indices)}
        else:
            print("No indices field found, assuming metadata is in order")

        # Load all parameters
        valid_params = []
        for param in metadata_params:
            if param in f:
                param_values_all = f[param][:].astype(np.float32)

                # Map to correct indices if needed
                if idx_to_metadata_idx is not None:
                    metadata_idx_list = [idx_to_metadata_idx[idx] for idx in idx_list]
                    param_values = param_values_all[metadata_idx_list]
                else:
                    param_values = param_values_all[idx_list]

                metadata_dict[param] = param_values
                valid_params.append(param)
                print(f"  Loaded {param}: shape={param_values.shape}, "
                      f"range=[{param_values.min():.4f}, {param_values.max():.4f}]")
            else:
                print(f"  Warning: {param} not found in metadata file")

    print(f"\nLoaded {len(valid_params)} valid parameters")
    return metadata_dict, valid_params


def generate_embeddings(model, dataset, device, batch_size):
    """
    Generate embeddings for all images in the dataset.

    Args:
        model: The trained model with encoder_1 and encoder_2
        dataset: Dataset containing HSC and Legacy image pairs
        device: Device to run inference on
        batch_size: Batch size for processing

    Returns:
        embeddings_1: Embeddings from encoder_1 (first 8k are HSC, next 8k are Legacy, flattened)
        embeddings_2: Embeddings from encoder_2 (first 8k are HSC, next 8k are Legacy, flattened)
    """
    print(f"\nGenerating embeddings for {len(dataset)} examples...")
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    hsc_embeddings_1_list = []
    hsc_embeddings_2_list = []
    legacy_embeddings_1_list = []
    legacy_embeddings_2_list = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(train_loader):
            hsc_images = batch[0].to(device)
            legacy_images = batch[1].to(device)

            # Generate embeddings from both encoders
            hsc_emb_1 = model.encoder_1(hsc_images)
            legacy_emb_1 = model.encoder_1(legacy_images)
            hsc_emb_2 = model.encoder_2(hsc_images)
            legacy_emb_2 = model.encoder_2(legacy_images)

            # Store HSC and Legacy embeddings separately
            hsc_embeddings_1_list.append(hsc_emb_1.cpu())
            hsc_embeddings_2_list.append(hsc_emb_2.cpu())
            legacy_embeddings_1_list.append(legacy_emb_1.cpu())
            legacy_embeddings_2_list.append(legacy_emb_2.cpu())

            if (batch_idx + 1) % 4 == 0:
                print(f"  Processed {batch_idx + 1} batches...")

    # Concatenate all HSC embeddings first, then all Legacy embeddings, then flatten
    # Structure: [all HSC embeddings (8k), all Legacy embeddings (8k)]
    embeddings_1 = torch.cat(hsc_embeddings_1_list + legacy_embeddings_1_list, dim=0).flatten(start_dim=1)
    embeddings_2 = torch.cat(hsc_embeddings_2_list + legacy_embeddings_2_list, dim=0).flatten(start_dim=1)

    print(f"  Embeddings 1 shape: {embeddings_1.shape}")
    print(f"  Embeddings 2 shape: {embeddings_2.shape}")

    return embeddings_1, embeddings_2


def save_dataset(output_path, embeddings_1, embeddings_2, metadata_array, param_names):
    """
    Save embeddings and metadata to an HDF5 file.

    Args:
        output_path: Path to save the HDF5 file
        embeddings_1: Embeddings from encoder_1 (numpy array)
        embeddings_2: Embeddings from encoder_2 (numpy array)
        metadata_array: Metadata array (numpy array)
        param_names: List of parameter names for metadata
    """
    print(f"\nSaving dataset to: {output_path}")

    # Create output directory if it doesn't exist
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        # Save embeddings (features)
        f.create_dataset('embeddings_1', data=embeddings_1, compression='gzip', compression_opts=4)
        f.create_dataset('embeddings_2', data=embeddings_2, compression='gzip', compression_opts=4)

        # Save metadata (labels)
        f.create_dataset('metadata', data=metadata_array, compression='gzip', compression_opts=4)

        # Save parameter names as a dataset of strings
        param_names_bytes = [name.encode('utf-8') for name in param_names]
        f.create_dataset('param_names', data=param_names_bytes)

        # Save metadata as attributes for easy access
        f.attrs['num_examples'] = embeddings_1.shape[0]
        f.attrs['embedding_1_dim'] = embeddings_1.shape[1]
        f.attrs['embedding_2_dim'] = embeddings_2.shape[1]
        f.attrs['num_params'] = len(param_names)
        f.attrs['description'] = 'Downstream prediction dataset: embeddings (features) and metadata (labels)'

    print(f"  Saved embeddings_1: shape={embeddings_1.shape}")
    print(f"  Saved embeddings_2: shape={embeddings_2.shape}")
    print(f"  Saved metadata: shape={metadata_array.shape}")
    print(f"  Saved {len(param_names)} parameter names")


# ============================================================================
# Main Script
# ============================================================================
if __name__ == "__main__":
    # Setup device
    device = get_device()

    # Load model
    print(f"\nLoading model from: {checkpoint_path}")
    model = ConditionalFlowMatchingModule.load_from_checkpoint(checkpoint_path, map_location='cpu')
    model.eval()
    torch.set_grad_enabled(False)
    model = model.to(device)

    # Create index list
    idx_list = list(range(num_examples))

    # Load dataset
    print(f"\nLoading dataset with {len(idx_list)} examples...")
    dataset = HSCLegacyDatasetZoom(hdf5_path=hdf5_path, idx_list=idx_list)

    # Generate embeddings (features)
    embeddings_1, embeddings_2 = generate_embeddings(model, dataset, device, batch_size)

    # Load metadata (labels)
    metadata_dict, valid_param_names = load_metadata(metadata_path, idx_list, metadata_params)

    # Convert embeddings to numpy
    embeddings_1_np = embeddings_1.numpy()
    embeddings_2_np = embeddings_2.numpy()

    # Create metadata array (num_examples, num_params)
    # Note: We have 2x num_examples embeddings (HSC + Legacy), so we duplicate metadata
    metadata_array = np.stack([metadata_dict[param] for param in valid_param_names], axis=1)
    # Duplicate metadata to match embeddings (HSC and Legacy)
    metadata_array = np.repeat(metadata_array, 2, axis=0)

    print(f"\n{'='*60}")
    print("Dataset Summary")
    print(f"{'='*60}")
    print(f"Embeddings 1 (features): shape={embeddings_1_np.shape}, dtype={embeddings_1_np.dtype}")
    print(f"Embeddings 2 (features): shape={embeddings_2_np.shape}, dtype={embeddings_2_np.dtype}")
    print(f"Metadata (labels): shape={metadata_array.shape}, dtype={metadata_array.dtype}")
    print(f"Number of metadata parameters: {len(valid_param_names)}")
    print(f"Parameter names: {valid_param_names}")

    # Save dataset to file
    save_dataset(output_path, embeddings_1_np, embeddings_2_np, metadata_array, valid_param_names)

    print(f"\n{'='*60}")
    print("Dataset saved successfully!")
    print(f"  Output file: {output_path}")
    print(f"  - Use 'embeddings_1' or 'embeddings_2' as features (X)")
    print(f"  - Use 'metadata' as labels (y)")
    print(f"  - Parameter names stored in 'param_names'")
    print(f"{'='*60}")
