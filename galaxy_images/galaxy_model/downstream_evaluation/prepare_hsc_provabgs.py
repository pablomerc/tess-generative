"""
Prepare HSC ProvaBGS downstream dataset: match HSC catalog to preprocessed H5,
extract embeddings (trained + untrained), align FITS labels, save to HDF5.

Run from galaxy_model/ or downstream_evaluation/:
  python downstream_evaluation/prepare_hsc_provabgs.py test   # dataloader only
  python downstream_evaluation/prepare_hsc_provabgs.py prepare  # full pipeline
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
import pandas as pd
import torch

from data import zoom_legacy_image

# HSC: single overlap CSV (train only; no eval split for HSC ProvaBGS catalog)
OVERLAP_CSV = _here / "hsc_train_overlap_df.csv"
train_overlap_df = pd.read_csv(OVERLAP_CSV)
print("HSC overlap (train):")
print(train_overlap_df.head())

processed_h5_path = "/data/vision/billf/scratch/pablomer/legacysurvey_hsc/preprocessed_hsc_legacy_48x48_all.h5"

with h5py.File(processed_h5_path, "r") as f:
    h5_indices = f["indices"][:]
    print("H5 keys:", list(f.keys()))
    train_mask = np.isin(h5_indices, train_overlap_df["abs_index"])
    train_actual_h5_rows = np.where(train_mask)[0]
    print(f"Train matches: {train_actual_h5_rows.shape[0]}")

NORM_DICT = {
    "hsc": [0.022, 0.05],
    "legacy": [0.023, 0.063],
    "legacy_zoom": [0.045, 0.078],
    "hsc96": [0.00897, 0.0312],
    "legacy96": [0.0108, 0.050],
    "legacy96_zoom": [0.0173, 0.053],
}


class DataLoader:
    """Dataset that loads HSC+Legacy pairs by custom H5 row indices (same as prepare_legacy_provabgs)."""

    def __init__(self, actual_h5_rows, norm_dict=None):
        self.actual_h5_rows = np.asarray(actual_h5_rows)
        self.norm_dict = norm_dict if norm_dict is not None else NORM_DICT

    def __len__(self):
        return len(self.actual_h5_rows)

    def __getitem__(self, idx):
        index = int(self.actual_h5_rows[idx])
        with h5py.File(processed_h5_path, "r") as f:
            hsc_image = f["hsc_images"][index][:]
            legacy_image = f["legacy_images"][index][:]
        hsc_image = torch.from_numpy(hsc_image).float()
        legacy_image = torch.from_numpy(legacy_image).float()
        mean_hsc, std_hsc = self.norm_dict["hsc"]
        hsc_image = (hsc_image - mean_hsc) / std_hsc
        legacy_image = zoom_legacy_image(legacy_image)
        mean_legacy_zoom, std_legacy_zoom = self.norm_dict["legacy_zoom"]
        legacy_image = (legacy_image - mean_legacy_zoom) / std_legacy_zoom
        return hsc_image, legacy_image


train_loader = DataLoader(train_actual_h5_rows)

# FITS path for HSC ProvaBGS labels (scalar columns only)
FITS_TRAIN_PATH = "/data/vision/billf/scratch/pablomer/data/provabgs_hsc_train_v2.fits"

# Three model configs: (output suffix, checkpoint path, module filename for model class)
# Neighbors models use double_train_fm_neighbors.py; old dataloader uses double_train_fm.py.
HSC_MODEL_CONFIGS = [
    ("zdim16_geom_neighbors", "/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy-flow-matching-neighbours/03mwpijx/checkpoints/latest-step=step=66000.ckpt", "double_train_fm_neighbors.py"),
    # ("zdim16_nogeom_neighbors", "/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy-flow-matching-neighbours/g2g9kvr4/checkpoints/latest-step=step=75000.ckpt", "double_train_fm_neighbors.py"),
    # ("zdim16_geom_old_dataloader", "/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy-flow-matching/wu1csh99/checkpoints/latest-step=step=75000.ckpt", "double_train_fm.py"),
]

OUTPUT_DOWNSTREAM_UNTRAINED_H5 = _here / "downstream_hsc_provabgs_untrained.h5"
IMAGE_SIZE = 48

# Columns to drop when loading FITS (array-valued); HSC has tok_image_hsc
FITS_DROP_COLS = {
    "rgb",
    "tok_image",
    "tok_image_hsc",
    "tok_spectrum_desi",
    "PROVABGS_MCMC",
    "PROVABGS_THETA_BF",
    "PROVABGS_LOGMSTAR",
}


def _to_native_byteorder(arr):
    """Convert array to native byte order (avoids Big-endian buffer errors on little-endian)."""
    arr = np.asarray(arr)
    if arr.dtype.byteorder not in ("=", "|"):
        arr = np.ascontiguousarray(arr.astype(arr.dtype.newbyteorder("=")))
    return arr


def load_fits_scalar_columns(path):
    """Load FITS table keeping only scalar columns (ndim==1), matching hsc_provabgs / read_fits."""
    from astropy.io import fits

    with fits.open(path, memmap=True) as hdul:
        d = hdul[1].data
        names = d.columns.names
        keep = [c for c in names if (c not in FITS_DROP_COLS) and (d[c].ndim == 1)]
        df = pd.DataFrame({c: _to_native_byteorder(d[c].copy()) for c in keep})
    for c in df.columns:
        if df[c].dtype == np.int8 and c.startswith("IS_"):
            df[c] = df[c].astype(bool)
    return df


def get_labels_in_dataloader_order(overlap_df, actual_h5_rows, h5_indices, fits_df):
    """
    Return a DataFrame of FITS labels in the same order as the dataloader.
    overlap_df has TARGETID (from CSV; same values as hsc_object_id) and abs_index.
    Force same ID column name and int64 type so the merge does not fail silently.
    """
    id_col = "hsc_object_id"
    abs_indices = h5_indices[actual_h5_rows]

    # CSV stores the ID as TARGETID; ensure we use it with consistent type
    overlap_ids = overlap_df["TARGETID"].astype(np.int64)
    overlap_df = overlap_df.copy()
    overlap_df[id_col] = overlap_ids
    overlap_by_abs = overlap_df.set_index("abs_index")[id_col]
    target_ids = overlap_by_abs.reindex(abs_indices).values

    if pd.isna(target_ids).any():
        bad = np.where(pd.isna(target_ids))[0]
        raise ValueError(f"Some abs_indices have no TARGETID in overlap: {bad[:5]}...")
    target_ids = np.asarray(target_ids, dtype=np.int64)

    # Ensure FITS ID column is int64 (FITS/astropy may give uint64 or other)
    fits_df = fits_df.copy()
    fits_df[id_col] = fits_df[id_col].astype(np.int64)

    # Debug: verify IDs overlap before merge
    n_overlap = len(set(target_ids) & set(fits_df[id_col]))
    if n_overlap == 0:
        raise ValueError(
            "CSV and FITS IDs do not overlap (check types: CSV TARGETID vs FITS hsc_object_id). "
            f"CSV TARGETID dtype: {overlap_df[id_col].dtype}, FITS {id_col} dtype: {fits_df[id_col].dtype}."
        )
    if n_overlap < len(target_ids):
        print(f"Warning: only {n_overlap}/{len(target_ids)} IDs from overlap CSV found in FITS.")

    order_df = pd.DataFrame({id_col: target_ids})
    labels_df = order_df.merge(fits_df, on=id_col, how="left")

    # Keep only rows that found a match in FITS (drop rows with any NaN from failed merge)
    valid = ~labels_df.isna().any(axis=1)
    n_missing = (~valid).sum()
    if n_missing > 0:
        print(f"  Proceeding with {valid.sum()}/{len(labels_df)} rows (dropping {n_missing} without FITS match).")
    labels_df = labels_df[valid].reset_index(drop=True)
    kept_mask = valid.to_numpy()
    return labels_df, kept_mask


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
    """Create an untrained model with same architecture as double_train_fm.py (random init)."""
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
    (same layout as prepare_legacy_provabgs).
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


def _serialize_labels_df(labels_df):
    """Convert labels DataFrame to dict of arrays suitable for HDF5."""
    out = {}
    for col in labels_df.columns:
        arr = labels_df[col].values
        if arr.dtype.kind in "OU" or arr.dtype == object:
            arr = np.array(
                [str(x).encode("utf-8") if pd.notna(x) else b"" for x in arr],
                dtype="S",
            )
        elif arr.dtype == bool:
            arr = arr.astype(np.int8)
        out[col] = arr
    return out


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
    Generate embeddings with the pretrained (and untrained) model for HSC ProvaBGS overlap,
    align labels from FITS, and save to downstream_hsc_provabgs.h5.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Loading FITS labels (scalar columns)...")
    fits_train = load_fits_scalar_columns(FITS_TRAIN_PATH)
    print(f"  Train FITS: {fits_train.shape}")

    print("Aligning labels to dataloader order...")
    labels_train, kept_mask = get_labels_in_dataloader_order(
        train_overlap_df, train_actual_h5_rows, h5_indices, fits_train
    )
    # Use only H5 rows that have a FITS label (keep embeddings and labels aligned)
    train_actual_h5_rows_filtered = train_actual_h5_rows[kept_mask]
    train_loader_filtered = DataLoader(train_actual_h5_rows_filtered)
    labels_n = _serialize_labels_df(labels_train)

    # Trained models: one H5 per config
    saved_paths = []
    for suffix, checkpoint_path, module_filename in HSC_MODEL_CONFIGS:
        output_h5 = _here / f"downstream_hsc_provabgs_{suffix}.h5"
        print(f"Loading model: {suffix} from {checkpoint_path}")
        model = _load_model(checkpoint_path, module_filename).to(device)
        print(f"Generating embeddings (train) for {suffix}...")
        (e1, e2, e3, e4, e5, e6) = generate_embeddings(
            model, train_loader_filtered, device, batch_size
        )
        embeddings_dict = {
            "hsc_encoder1": e1,
            "hsc_encoder2": e2,
            "legacy_encoder1": e3,
            "legacy_encoder2": e4,
            "hsc_legacy_encoder1": e5,
            "hsc_legacy_encoder2": e6,
        }
        N = len(labels_train)
        assert embeddings_dict["hsc_encoder1"].shape[0] == N
        assert embeddings_dict["hsc_legacy_encoder1"].shape[0] == N
        save_downstream_h5(
            output_h5,
            embeddings_dict=embeddings_dict,
            labels_n=labels_n,
            label_columns=list(labels_train.columns),
        )
        saved_paths.append(output_h5)

    # Untrained model
    print("Loading untrained model (same architecture)...")
    model_untrained = _load_untrained_model().to(device)
    print("Generating embeddings with untrained model (train)...")
    (e1, e2, e3, e4, e5, e6) = generate_embeddings(
        model_untrained, train_loader_filtered, device, batch_size
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
        label_columns=list(labels_train.columns),
    )

    return saved_paths, OUTPUT_DOWNSTREAM_UNTRAINED_H5


def test_dataloader():
    """Quick test: shapes, dtypes, and one batch via torch DataLoader."""
    print("\n--- Testing HSC dataloader ---")
    hsc, leg = train_loader[0]
    assert hsc.shape == leg.shape, f"Shape mismatch: hsc {hsc.shape} vs legacy {leg.shape}"
    assert hsc.dtype == torch.float32 and leg.dtype == torch.float32
    assert len(hsc.shape) == 3, f"Expected (C,H,W), got {hsc.shape}"
    print(f"  Single sample: hsc {hsc.shape}, legacy {leg.shape}, dtypes ok")

    from torch.utils.data import DataLoader as TorchDataLoader

    batch_loader = TorchDataLoader(
        train_loader, batch_size=4, shuffle=False, num_workers=0
    )
    h_batch, l_batch = next(iter(batch_loader))
    assert h_batch.shape == (4, *hsc.shape), f"Batch shape: {h_batch.shape}"
    print(f"  Batch (size 4): hsc {h_batch.shape}, legacy {l_batch.shape}")
    print("--- All dataloader tests passed ---\n")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="HSC ProvaBGS dataloader and downstream prep"
    )
    p.add_argument(
        "mode",
        nargs="?",
        default="prepare",
        choices=("test", "prepare"),
        help="test=dataloader only; prepare=embeddings+FITS labels -> H5",
    )
    args = p.parse_args()
    if args.mode == "test":
        test_dataloader()
    else:
        prepare_downstream_dataset()
