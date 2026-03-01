"""
Prepare MMU downstream dataset: use preprocessed H5 (48x48) and metadata H5,
extract embeddings (trained + untrained) with same model types as HSC ProvaBGS,
save to HDF5.

Uses HSCLegacyDatasetZoom and metadata from preprocessed_hsc_legacy_metadata_8192.h5.

Run from galaxy_model/ or downstream_evaluation/:
  python downstream_evaluation/prepare_mmu.py test   # dataloader only
  python downstream_evaluation/prepare_mmu.py prepare  # full pipeline
"""
import sys
from pathlib import Path

# Allow importing from galaxy_model when run from downstream_evaluation/
_here = Path(__file__).resolve().parent
_src = _here.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import h5py
import numpy as np
import torch

from data import HSCLegacyDatasetZoom

# MMU data paths (same as prepare_data.py)
HDF5_PATH = "/data/vision/billf/scratch/pablomer/legacysurvey_hsc/preprocessed_hsc_legacy_48x48_all.h5"
METADATA_PATH = "/data/vision/billf/scratch/pablomer/legacysurvey_hsc/preprocessed_hsc_legacy_metadata_8192.h5"

NUM_EXAMPLES = 8192
IMAGE_SIZE = 48

# Three model configs: (output suffix, checkpoint path, module filename for model class)
# Same as prepare_hsc_provabgs.py.
HSC_MODEL_CONFIGS = [
    ("zdim16_geom_neighbors", "/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy-flow-matching-neighbours/03mwpijx/checkpoints/latest-step=step=75000.ckpt", "double_train_fm_neighbors.py"),
    ("zdim16_nogeom_neighbors", "/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy-flow-matching-neighbours/g2g9kvr4/checkpoints/latest-step=step=75000.ckpt", "double_train_fm_neighbors.py"),
    ("zdim16_geom_old_dataloader", "/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy-flow-matching/wu1csh99/checkpoints/latest-step=step=75000.ckpt", "double_train_fm.py"),
]

OUTPUT_DOWNSTREAM_UNTRAINED_H5 = _here / "downstream_mmu_untrained.h5"

# Metadata parameters to load (from prepare_data.py keep list)
METADATA_PARAMS = [
    "EBV",
    "FLUX_G",
    "FLUX_I",
    "FLUX_R",
    "FLUX_W1",
    "FLUX_W2",
    "FLUX_W3",
    "FLUX_W4",
    "FLUX_Z",
    "SHAPE_E1",
    "SHAPE_E2",
    "SHAPE_R",
    "a_g",
    "a_i",
    "a_r",
    "a_y",
    "a_z",
    "g_cmodel_mag",
    "g_cmodel_magerr",
    "g_sdssshape_psf_shape11",
    "g_sdssshape_psf_shape12",
    "g_sdssshape_psf_shape22",
    "i_cmodel_mag",
    "i_cmodel_magerr",
    "i_extendedness_value",
    "i_sdssshape_psf_shape11",
    "i_sdssshape_psf_shape12",
    "i_sdssshape_psf_shape22",
    "r_cmodel_mag",
    "r_sdssshape_psf_shape11",
    "r_sdssshape_psf_shape12",
    "r_sdssshape_psf_shape22",
    "y_cmodel_mag",
    "y_cmodel_magerr",
    "y_extendedness_value",
    "z_cmodel_mag",
    "z_sdssshape_psf_shape11",
    "z_sdssshape_psf_shape12",
    "z_sdssshape_psf_shape22",
]


def load_metadata(metadata_path, idx_list, metadata_params):
    """
    Load metadata for specified indices (from prepare_data.py).

    Returns:
        metadata_dict: dict mapping param names to numpy arrays
        valid_params: list of parameter names that were found
    """
    print(f"\nLoading metadata from: {metadata_path}")
    metadata_dict = {}
    idx_to_metadata_idx = None

    with h5py.File(metadata_path, "r") as f:
        if "indices" in f:
            metadata_indices = f["indices"][:]
            print(f"Found indices mapping: shape={metadata_indices.shape}")
            idx_to_metadata_idx = {
                int(dataset_idx): meta_idx
                for meta_idx, dataset_idx in enumerate(metadata_indices)
            }
        else:
            print("No indices field found, assuming metadata is in order")

        valid_params = []
        for param in metadata_params:
            if param in f:
                param_values_all = f[param][:].astype(np.float32)
                if idx_to_metadata_idx is not None:
                    metadata_idx_list = [idx_to_metadata_idx[idx] for idx in idx_list]
                    param_values = param_values_all[metadata_idx_list]
                else:
                    param_values = param_values_all[idx_list]
                metadata_dict[param] = param_values
                valid_params.append(param)
                print(
                    f"  Loaded {param}: shape={param_values.shape}, "
                    f"range=[{param_values.min():.4f}, {param_values.max():.4f}]"
                )
            else:
                print(f"  Warning: {param} not found in metadata file")

    print(f"\nLoaded {len(valid_params)} valid parameters")
    return metadata_dict, valid_params


def _load_model(checkpoint_path, module_filename):
    """Load the double-encoder model from checkpoint. Module is double_train_fm_neighbors.py or double_train_fm.py."""
    import importlib.util

    module_name = module_filename.replace(".py", "").replace(".", "_")
    spec = importlib.util.spec_from_file_location(
        module_name,
        _src / module_filename,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.ConditionalFlowMatchingModule.load_from_checkpoint(
        checkpoint_path, map_location="cpu"
    )
    model.eval()
    torch.set_grad_enabled(False)
    return model


def _load_untrained_model():
    """Create an untrained model with same architecture as double_train_fm_neighbors.py (random init)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "double_train_fm_neighbors",
        _src / "double_train_fm_neighbors.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.ConditionalFlowMatchingModule(
        in_channels=4,
        cond_channels=4,
        image_size=IMAGE_SIZE,
        model_channels=128,
        channel_mult=(1, 2, 4, 4),
        cross_attention_dim=16,
        pretrained_encoder=False,
        concat_conditioning=False,
        lr=1e-4,
        num_sample_images=10,
        num_mse_images=32,
        num_integration_steps=250,
        lambda_generative=1.0,
        lambda_geometric=0.3,
        mask_center=False,
    )
    model.eval()
    torch.set_grad_enabled(False)
    return model


def generate_embeddings(model, dataset, device, batch_size=256):
    """
    Run encoder_1 and encoder_2 on HSC and Legacy; return 6 embedding arrays
    (same layout as prepare_hsc_provabgs / prepare_legacy_provabgs).
    """
    from torch.utils.data import DataLoader as TorchDataLoader

    loader = TorchDataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    hsc_1, hsc_2, leg_1, leg_2 = [], [], [], []
    with torch.no_grad():
        for hsc_im, leg_im in loader:
            h, l = hsc_im.to(device), leg_im.to(device)
            hsc_1.append(model.encoder_1(h).cpu())
            hsc_2.append(model.encoder_2(h).cpu())
            leg_1.append(model.encoder_1(l).cpu())
            leg_2.append(model.encoder_2(l).cpu())
    e1 = torch.cat(hsc_1, dim=0).flatten(start_dim=1)
    e2 = torch.cat(hsc_2, dim=0).flatten(start_dim=1)
    e3 = torch.cat(leg_1, dim=0).flatten(start_dim=1)
    e4 = torch.cat(leg_2, dim=0).flatten(start_dim=1)
    e5 = torch.cat(
        [
            torch.cat([h.flatten(start_dim=1), l.flatten(start_dim=1)], dim=1)
            for h, l in zip(hsc_1, leg_1)
        ],
        dim=0,
    )
    e6 = torch.cat(
        [
            torch.cat([h.flatten(start_dim=1), l.flatten(start_dim=1)], dim=1)
            for h, l in zip(hsc_2, leg_2)
        ],
        dim=0,
    )
    return (
        e1.numpy(),
        e2.numpy(),
        e3.numpy(),
        e4.numpy(),
        e5.numpy(),
        e6.numpy(),
    )


EMBEDDING_NAMES = (
    "hsc_encoder1",
    "hsc_encoder2",
    "legacy_encoder1",
    "legacy_encoder2",
    "hsc_legacy_encoder1",
    "hsc_legacy_encoder2",
)


def _labels_from_metadata(metadata_dict, valid_param_names):
    """Build labels dict for save_downstream_h5 from metadata arrays (all float32)."""
    return {col: metadata_dict[col] for col in valid_param_names}


def save_downstream_h5(path, embeddings_dict, labels_n, label_columns):
    """Save 6 embedding datasets and labels to HDF5."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for name in EMBEDDING_NAMES:
            arr = embeddings_dict[name]
            f.create_dataset(name, data=arr, compression="gzip", compression_opts=4)
        for col in label_columns:
            f.create_dataset(
                f"labels/{col}", data=labels_n[col], compression="gzip", compression_opts=4
            )
        f.attrs["embedding_names"] = list(EMBEDDING_NAMES)
        f.attrs["num_examples"] = len(labels_n[list(labels_n)[0]])
        f.attrs["label_columns"] = list(label_columns)
    print(f"Saved: {path} (6 embeddings + labels)")


def prepare_downstream_dataset(batch_size=256):
    """
    Generate embeddings with the pretrained (and untrained) model for MMU 8k subset,
    use metadata from metadata H5 as labels, and save to downstream_mmu_*.h5.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    idx_list = list(range(NUM_EXAMPLES))
    print(f"Loading dataset with {len(idx_list)} examples (HSCLegacyDatasetZoom)...")
    dataset = HSCLegacyDatasetZoom(hdf5_path=HDF5_PATH, idx_list=idx_list)

    print("Loading metadata (labels)...")
    metadata_dict, valid_param_names = load_metadata(
        METADATA_PATH, idx_list, METADATA_PARAMS
    )
    labels_n = _labels_from_metadata(metadata_dict, valid_param_names)
    label_columns = valid_param_names

    # Trained models: one H5 per config
    saved_paths = []
    for suffix, checkpoint_path, module_filename in HSC_MODEL_CONFIGS:
        output_h5 = _here / f"downstream_mmu_{suffix}.h5"
        print(f"Loading model: {suffix} from {checkpoint_path}")
        model = _load_model(checkpoint_path, module_filename).to(device)
        print(f"Generating embeddings for {suffix}...")
        (e1, e2, e3, e4, e5, e6) = generate_embeddings(
            model, dataset, device, batch_size
        )
        embeddings_dict = {
            "hsc_encoder1": e1,
            "hsc_encoder2": e2,
            "legacy_encoder1": e3,
            "legacy_encoder2": e4,
            "hsc_legacy_encoder1": e5,
            "hsc_legacy_encoder2": e6,
        }
        N = len(idx_list)
        assert embeddings_dict["hsc_encoder1"].shape[0] == N
        assert embeddings_dict["hsc_legacy_encoder1"].shape[0] == N
        save_downstream_h5(
            output_h5,
            embeddings_dict=embeddings_dict,
            labels_n=labels_n,
            label_columns=label_columns,
        )
        saved_paths.append(output_h5)

    # Untrained model
    print("Loading untrained model (same architecture)...")
    model_untrained = _load_untrained_model().to(device)
    print("Generating embeddings with untrained model...")
    (e1, e2, e3, e4, e5, e6) = generate_embeddings(
        model_untrained, dataset, device, batch_size
    )
    embeddings_untrained = {
        "hsc_encoder1": e1,
        "hsc_encoder2": e2,
        "legacy_encoder1": e3,
        "legacy_encoder2": e4,
        "hsc_legacy_encoder1": e5,
        "hsc_legacy_encoder2": e6,
    }
    save_downstream_h5(
        OUTPUT_DOWNSTREAM_UNTRAINED_H5,
        embeddings_dict=embeddings_untrained,
        labels_n=labels_n,
        label_columns=label_columns,
    )

    return saved_paths, OUTPUT_DOWNSTREAM_UNTRAINED_H5


def test_dataloader():
    """Quick test: shapes, dtypes, and one batch via torch DataLoader."""
    print("\n--- Testing MMU dataloader (HSCLegacyDatasetZoom) ---")
    idx_list = list(range(min(100, NUM_EXAMPLES)))
    dataset = HSCLegacyDatasetZoom(hdf5_path=HDF5_PATH, idx_list=idx_list)
    hsc, leg = dataset[0]
    assert hsc.shape == leg.shape, f"Shape mismatch: hsc {hsc.shape} vs legacy {leg.shape}"
    assert hsc.dtype == torch.float32 and leg.dtype == torch.float32
    assert len(hsc.shape) == 3, f"Expected (C,H,W), got {hsc.shape}"
    print(f"  Single sample: hsc {hsc.shape}, legacy {leg.shape}, dtypes ok")

    from torch.utils.data import DataLoader as TorchDataLoader

    batch_loader = TorchDataLoader(
        dataset, batch_size=4, shuffle=False, num_workers=0
    )
    h_batch, l_batch = next(iter(batch_loader))
    assert h_batch.shape == (4, *hsc.shape), f"Batch shape: {h_batch.shape}"
    print(f"  Batch (size 4): hsc {h_batch.shape}, legacy {l_batch.shape}")
    print("--- All dataloader tests passed ---\n")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="MMU dataloader and downstream prep (HSCLegacyDatasetZoom + metadata H5)"
    )
    p.add_argument(
        "mode",
        nargs="?",
        default="prepare",
        choices=("test", "prepare"),
        help="test=dataloader only; prepare=embeddings+metadata labels -> H5",
    )
    args = p.parse_args()
    if args.mode == "test":
        test_dataloader()
    else:
        prepare_downstream_dataset()
