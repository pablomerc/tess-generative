"""
End-to-end downstream prediction for "ours" (ConditionalFlowMatchingModule).

Combines:
  Stage 1 (Prepare) – extract embeddings (real + untrained + random) from a
    single checkpoint, build downstream H5 files.
  Stage 2 (Predict) – train MLP regressors on the embeddings, evaluate per-target
    R², produce CSV + per-objective bar plots.

Model architecture is ConditionalFlowMatchingModule from double_train_fm_neighbors.py
with encoder_1 (physics, cross_attention_dim=16) and encoder_2 (instrument, zdim=4).
Trained by neighbours_train.py.

Usage:
  python predict_all_ours_instrument4x4.py --suffix ours_instrument4x4
  python predict_all_ours_instrument4x4.py --suffix ours_instrument4x4 --skip-prepare
  python predict_all_ours_instrument4x4.py --suffix ours_instrument4x4 --prepare-only
  python predict_all_ours_instrument4x4.py --suffix ours_instrument4x4 --datasets mmu,neighbors
"""

import argparse
import csv
import inspect
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset, DataLoader, Subset
from torch.optim import AdamW
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_here = Path(__file__).resolve().parent
_galaxy_model = _here.parent
_project_root = _galaxy_model.parent.parent  # .../tess-generative
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_galaxy_model) not in sys.path:
    sys.path.insert(0, str(_galaxy_model))

from galaxy_images.galaxy_model.data import HSCLegacyDatasetZoom, zoom_legacy_image
from galaxy_images.galaxy_model.neighbors import NeighborsSimpleDataset

# ============================================================================
# Default checkpoint
# ============================================================================
DEFAULT_CHECKPOINT = (
    "/data/vision/billf/scratch/pablomer/projects/tess-generative/"
    "galaxy-flow-matching-neighbours/t1k3q5kh/checkpoints/"
    "latest-step=step=32000.ckpt"
)
MODULE_FILENAME = "double_train_fm_neighbors.py"

# ============================================================================
# Data paths (shared with existing prepare scripts)
# ============================================================================
# MMU
HDF5_PATH_MMU = "/data/vision/billf/scratch/pablomer/legacysurvey_hsc/preprocessed_hsc_legacy_48x48_all.h5"
METADATA_PATH_MMU = "/data/vision/billf/scratch/pablomer/legacysurvey_hsc/preprocessed_hsc_legacy_metadata_8192.h5"
NUM_EXAMPLES_MMU = 8192
IMAGE_SIZE = 48
METADATA_PARAMS_MMU = [
    "EBV", "FLUX_G", "FLUX_I", "FLUX_R", "FLUX_W1", "FLUX_W2", "FLUX_W3", "FLUX_W4", "FLUX_Z",
    "SHAPE_E1", "SHAPE_E2", "SHAPE_R",
    "a_g", "a_i", "a_r", "a_y", "a_z",
    "g_cmodel_mag", "g_cmodel_magerr", "g_sdssshape_psf_shape11", "g_sdssshape_psf_shape12", "g_sdssshape_psf_shape22",
    "i_cmodel_mag", "i_cmodel_magerr", "i_extendedness_value", "i_sdssshape_psf_shape11", "i_sdssshape_psf_shape12", "i_sdssshape_psf_shape22",
    "r_cmodel_mag", "r_sdssshape_psf_shape11", "r_sdssshape_psf_shape12", "r_sdssshape_psf_shape22",
    "y_cmodel_mag", "y_cmodel_magerr", "y_extendedness_value",
    "z_cmodel_mag", "z_sdssshape_psf_shape11", "z_sdssshape_psf_shape12", "z_sdssshape_psf_shape22",
]

# Legacy ProvaBGS
OVERLAP_TRAIN_CSV = _galaxy_model / "util_notebooks" / "leagcy_train_overlap_df.csv"
OVERLAP_EVAL_CSV = _galaxy_model / "util_notebooks" / "leagcy_eval_overlap_df.csv"
PROCESSED_H5_LEGACY = "/data/vision/billf/scratch/pablomer/legacysurvey_hsc/preprocessed_hsc_legacy_48x48_all.h5"
FITS_TRAIN_PATH_LEGACY = "/data/vision/billf/scratch/pablomer/data/provabgs_legacysurvey_train_v2.fits"
FITS_EVAL_PATH_LEGACY = "/data/vision/billf/scratch/pablomer/data/provabgs_legacysurvey_eval_v2.fits"

# HSC ProvaBGS
OVERLAP_CSV_HSC = _here / "hsc_train_overlap_df.csv"
FITS_TRAIN_PATH_HSC = "/data/vision/billf/scratch/pablomer/data/provabgs_hsc_train_v2.fits"

# Neighbors
NEIGHBORS_HDF5 = "/data/vision/billf/scratch/pablomer/data/neighbours_v2.h5"
NUM_EXAMPLES_NEIGHBORS = 4096
NEIGHBORS_SEED = 42
SHUFFLE_NEIGHBORS = True

FITS_DROP_COLS = {"rgb", "tok_image", "tok_image_hsc", "tok_spectrum_desi", "PROVABGS_MCMC", "PROVABGS_THETA_BF", "PROVABGS_LOGMSTAR"}

EMBEDDING_NAMES = (
    "hsc_encoder1",
    "hsc_encoder2",
    "legacy_encoder1",
    "legacy_encoder2",
    "hsc_legacy_encoder1",
    "hsc_legacy_encoder2",
)

# ============================================================================
# Prediction config (mirroring predict_all_contrastive.py)
# ============================================================================
KEY1 = "hsc_legacy_encoder1"
KEY2 = "hsc_legacy_encoder2"
EMBEDDING_VARIANTS = ("real", "untrained", "random")

physics_mmu = ["SHAPE_E1", "SHAPE_E2", "SHAPE_R"]
instrument_mmu_legacy = ["EBV"]
instrument_mmu_hsc = ["a_g", "a_i", "a_r", "a_y", "a_z"]

instrument_neighbors_legacy = [
    "legacy_PSFSIZE_G", "legacy_PSFSIZE_I", "legacy_PSFSIZE_R", "legacy_PSFSIZE_Z",
    "legacy_PSFDEPTH_G", "legacy_PSFDEPTH_R", "legacy_PSFDEPTH_I", "legacy_PSFDEPTH_Z",
    "legacy_GALDEPTH_G", "legacy_GALDEPTH_R", "legacy_GALDEPTH_I", "legacy_GALDEPTH_Z",
    "legacy_NOBS_G", "legacy_NOBS_R", "legacy_NOBS_I", "legacy_NOBS_Z",
]
instrument_neighbors_hsc = ["hsc_i_variance_value", "hsc_r_variance_value", "hsc_z_variance_value"]
instrument_neighbors_hsc_psf_fwhm = ["hsc_g_psf_fwhm", "hsc_i_psf_fwhm", "hsc_r_psf_fwhm", "hsc_z_psf_fwhm"]

physics_provabgs = ["desi_Z", "LOG_MSTAR", "TAGE_MW", "LOG_Z_MW", "sSFR", "hsc_g_extendedness_value", "DEC"]
physics_hsc_provabgs = ["sSFR", "LOG_Z_MW", "TAGE_MW", "LOG_MSTAR", "desi_Z"]

TASKS = [
    ("mmu", "physics_mmu", "mmu", 1, physics_mmu),
    ("mmu", "instrument_mmu", "mmu", 2, instrument_mmu_legacy + instrument_mmu_hsc),
    ("legacy_provabgs", "physics_legacy_provabgs", "legacy_provabgs", 1, physics_provabgs),
    ("neighbors", "instrument_neighbors_legacy", "neighbors", 2, instrument_neighbors_legacy),
    ("neighbors", "instrument_neighbors_hsc", "neighbors", 2, instrument_neighbors_hsc + instrument_neighbors_hsc_psf_fwhm),
    ("hsc_provabgs", "physics_hsc_provabgs", "hsc_provabgs", 1, physics_hsc_provabgs),
]

MLP_HIDDEN = (256, 128)
BATCH_SIZE = 64
TRAIN_FRAC = 0.9


# ============================================================================
# Stage 1: Prepare — model loading & embedding extraction
# ============================================================================

def _load_model(checkpoint_path):
    import importlib.util
    module_name = MODULE_FILENAME.replace(".py", "").replace(".", "_")
    spec = importlib.util.spec_from_file_location(module_name, _galaxy_model / MODULE_FILENAME)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.ConditionalFlowMatchingModule.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model.eval()
    torch.set_grad_enabled(False)
    return model


def _load_untrained_model(checkpoint_path):
    import importlib.util
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    hparams = ckpt.get("hyper_parameters", {})

    module_name = MODULE_FILENAME.replace(".py", "").replace(".", "_")
    spec = importlib.util.spec_from_file_location(module_name, _galaxy_model / MODULE_FILENAME)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    valid_keys = set(inspect.signature(mod.ConditionalFlowMatchingModule.__init__).parameters.keys())
    init_kwargs = {k: v for k, v in hparams.items() if k in valid_keys}
    model = mod.ConditionalFlowMatchingModule(**init_kwargs)
    model.eval()
    torch.set_grad_enabled(False)
    return model


def _encode_hsc_legacy(model, hsc_batch, legacy_batch):
    h_1 = model.encoder_1(hsc_batch)
    h_2 = model.encoder_2(hsc_batch)
    l_1 = model.encoder_1(legacy_batch)
    l_2 = model.encoder_2(legacy_batch)
    return h_1, h_2, l_1, l_2


def generate_embeddings_hsc_legacy(model, dataset, device, batch_size=256):
    loader = TorchDataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    hsc_1, hsc_2, leg_1, leg_2 = [], [], [], []
    with torch.no_grad():
        for hsc_im, leg_im in loader:
            h, l = hsc_im.to(device), leg_im.to(device)
            e1, e2, e3, e4 = _encode_hsc_legacy(model, h, l)
            hsc_1.append(e1.cpu())
            hsc_2.append(e2.cpu())
            leg_1.append(e3.cpu())
            leg_2.append(e4.cpu())
    e1 = torch.cat(hsc_1, dim=0).flatten(start_dim=1)
    e2 = torch.cat(hsc_2, dim=0).flatten(start_dim=1)
    e3 = torch.cat(leg_1, dim=0).flatten(start_dim=1)
    e4 = torch.cat(leg_2, dim=0).flatten(start_dim=1)
    e5 = torch.cat([torch.cat([h.flatten(start_dim=1), l.flatten(start_dim=1)], dim=1) for h, l in zip(hsc_1, leg_1)], dim=0)
    e6 = torch.cat([torch.cat([h.flatten(start_dim=1), l.flatten(start_dim=1)], dim=1) for h, l in zip(hsc_2, leg_2)], dim=0)
    return e1.numpy(), e2.numpy(), e3.numpy(), e4.numpy(), e5.numpy(), e6.numpy()


def collate_neighbors_simple(batch):
    hsc = torch.stack([b[0] for b in batch])
    leg = torch.stack([b[1] for b in batch])
    meta = [b[2] for b in batch]
    return hsc, leg, meta


def generate_embeddings_neighbors(model, dataset, device, batch_size=256, shuffle=False, seed=None):
    loader_kw = dict(batch_size=batch_size, num_workers=0, collate_fn=collate_neighbors_simple)
    if shuffle:
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)
        loader_kw["shuffle"] = True
        loader_kw["generator"] = g
    else:
        loader_kw["shuffle"] = False
    loader = TorchDataLoader(dataset, **loader_kw)

    hsc_1, hsc_2, leg_1, leg_2 = [], [], [], []
    metadata_collected = []
    with torch.no_grad():
        for hsc_im, leg_im, meta_list in loader:
            metadata_collected.extend(meta_list)
            h, l = hsc_im.to(device), leg_im.to(device)
            e1, e2, e3, e4 = _encode_hsc_legacy(model, h, l)
            hsc_1.append(e1.cpu())
            hsc_2.append(e2.cpu())
            leg_1.append(e3.cpu())
            leg_2.append(e4.cpu())

    e1 = torch.cat(hsc_1, dim=0).flatten(start_dim=1)
    e2 = torch.cat(hsc_2, dim=0).flatten(start_dim=1)
    e3 = torch.cat(leg_1, dim=0).flatten(start_dim=1)
    e4 = torch.cat(leg_2, dim=0).flatten(start_dim=1)
    e5 = torch.cat([torch.cat([h.flatten(start_dim=1), l.flatten(start_dim=1)], dim=1) for h, l in zip(hsc_1, leg_1)], dim=0)
    e6 = torch.cat([torch.cat([h.flatten(start_dim=1), l.flatten(start_dim=1)], dim=1) for h, l in zip(hsc_2, leg_2)], dim=0)
    return (e1.numpy(), e2.numpy(), e3.numpy(), e4.numpy(), e5.numpy(), e6.numpy()), metadata_collected


def make_random_embeddings_like(real_embeddings_dict, seed=42):
    rng = np.random.default_rng(seed)
    return {k: rng.standard_normal(v.shape).astype(np.float32) for k, v in real_embeddings_dict.items()}


def _to_native_byteorder(arr):
    arr = np.asarray(arr)
    if arr.dtype.byteorder not in ("=", "|"):
        arr = np.ascontiguousarray(arr.astype(arr.dtype.newbyteorder("=")))
    return arr


def load_fits_scalar_columns(path):
    from astropy.io import fits
    with fits.open(path, memmap=True) as hdul:
        d = hdul[1].data
        keep = [c for c in d.columns.names if (c not in FITS_DROP_COLS) and (d[c].ndim == 1)]
        df = pd.DataFrame({c: _to_native_byteorder(d[c].copy()) for c in keep})
    for c in df.columns:
        if df[c].dtype == np.int8 and c.startswith("IS_"):
            df[c] = df[c].astype(bool)
    return df


def _serialize_labels_df(labels_df):
    out = {}
    for col in labels_df.columns:
        arr = labels_df[col].values
        if arr.dtype.kind in "OU" or arr.dtype == object:
            arr = np.array([str(x).encode("utf-8") if pd.notna(x) else b"" for x in arr], dtype="S")
        elif arr.dtype == bool:
            arr = arr.astype(np.int8)
        out[col] = arr
    return out


def _labels_from_metadata_list(metadata_list):
    if not metadata_list:
        return {}, []
    keys = list(metadata_list[0].keys())
    labels_n, valid_columns = {}, []
    for col in keys:
        try:
            vals = [m[col] for m in metadata_list]
            arr = np.array(vals)
            if arr.dtype.kind in "iuflb" and arr.shape[0] == len(metadata_list):
                labels_n[col] = arr.astype(np.float32) if arr.dtype.kind == "f" else arr
                valid_columns.append(col)
        except (TypeError, ValueError):
            continue
    return labels_n, valid_columns


class H5PairRowsDataset(torch.utils.data.Dataset):
    def __init__(self, actual_h5_rows, h5_path=PROCESSED_H5_LEGACY):
        self.actual_h5_rows = np.asarray(actual_h5_rows)
        self.h5_path = h5_path
        self.norm_hsc = (0.022, 0.05)
        self.norm_legacy_zoom = (0.045, 0.078)

    def __len__(self):
        return len(self.actual_h5_rows)

    def __getitem__(self, idx):
        row = int(self.actual_h5_rows[idx])
        with h5py.File(self.h5_path, "r") as f:
            hsc_image = torch.from_numpy(f["hsc_images"][row][:]).float()
            legacy_image = torch.from_numpy(f["legacy_images"][row][:]).float()
        mean_hsc, std_hsc = self.norm_hsc
        hsc_image = (hsc_image - mean_hsc) / std_hsc
        legacy_image = zoom_legacy_image(legacy_image)
        mean_legacy, std_legacy = self.norm_legacy_zoom
        legacy_image = (legacy_image - mean_legacy) / std_legacy
        return hsc_image, legacy_image


def _fit_labels_by_abs(overlap_df, actual_h5_rows, h5_indices, fits_df, overlap_id_col, target_id_col):
    overlap = overlap_df.copy()
    overlap[target_id_col] = overlap[overlap_id_col]
    abs_indices = h5_indices[actual_h5_rows]
    overlap_by_abs = overlap.set_index("abs_index")[target_id_col]
    target_ids = overlap_by_abs.reindex(abs_indices).values
    if pd.isna(target_ids).any():
        raise ValueError("Some abs_index entries have no matching ID in overlap CSV.")
    order_df = pd.DataFrame({target_id_col: target_ids})
    fits_cast = fits_df.copy()
    if np.issubdtype(order_df[target_id_col].dtype, np.number) and target_id_col in fits_cast.columns:
        fits_cast[target_id_col] = pd.to_numeric(fits_cast[target_id_col], errors="coerce")
    labels_df = order_df.merge(fits_cast, on=target_id_col, how="left")
    valid = ~labels_df.isna().any(axis=1)
    labels_df = labels_df[valid].reset_index(drop=True)
    kept_mask = valid.to_numpy()
    return labels_df, kept_mask


def save_downstream_h5_all_variants(path, real_dict, untrained_dict, random_dict, labels_n, label_columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for name in EMBEDDING_NAMES:
            f.create_dataset(name, data=real_dict[name], compression="gzip", compression_opts=4)
            f.create_dataset(name + "_untrained", data=untrained_dict[name], compression="gzip", compression_opts=4)
            f.create_dataset(name + "_random", data=random_dict[name], compression="gzip", compression_opts=4)
        for col in label_columns:
            f.create_dataset(f"labels/{col}", data=labels_n[col], compression="gzip", compression_opts=4)
        f.attrs["embedding_names"] = list(EMBEDDING_NAMES)
        f.attrs["num_examples"] = real_dict[EMBEDDING_NAMES[0]].shape[0]
        f.attrs["label_columns"] = list(label_columns)
    print(f"Saved: {path}")


def _run_model_variants(get_real_emb_fn, checkpoint_path, seed):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = _load_model(checkpoint_path).to(device)
    real = get_real_emb_fn(model)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model_untrained = _load_untrained_model(checkpoint_path).to(device)
    untrained = get_real_emb_fn(model_untrained)
    del model_untrained
    if device.type == "cuda":
        torch.cuda.empty_cache()

    random_emb = make_random_embeddings_like(real, seed=seed)
    return real, untrained, random_emb


def _load_metadata_mmu(metadata_path, idx_list, metadata_params):
    metadata_dict = {}
    with h5py.File(metadata_path, "r") as f:
        idx_to_meta = None
        if "indices" in f:
            meta_indices = f["indices"][:]
            idx_to_meta = {int(dataset_idx): meta_idx for meta_idx, dataset_idx in enumerate(meta_indices)}
        valid = []
        for param in metadata_params:
            if param not in f:
                continue
            arr = f[param][:].astype(np.float32)
            if idx_to_meta is not None:
                arr = arr[[idx_to_meta[i] for i in idx_list]]
            else:
                arr = arr[idx_list]
            metadata_dict[param] = arr
            valid.append(param)
    return metadata_dict, valid


def prepare_mmu(checkpoint_path, output_dir, batch_size, seed, suffix):
    idx_list = list(range(NUM_EXAMPLES_MMU))
    dataset = HSCLegacyDatasetZoom(hdf5_path=HDF5_PATH_MMU, idx_list=idx_list)
    metadata_dict, label_columns = _load_metadata_mmu(METADATA_PATH_MMU, idx_list, METADATA_PARAMS_MMU)
    labels_n = {col: metadata_dict[col] for col in label_columns}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _emb_fn(model):
        arrs = generate_embeddings_hsc_legacy(model, dataset, device, batch_size)
        return dict(zip(EMBEDDING_NAMES, arrs))

    real, untrained, random_emb = _run_model_variants(_emb_fn, checkpoint_path, seed)
    out_path = output_dir / f"downstream_mmu_{suffix}.h5"
    save_downstream_h5_all_variants(out_path, real, untrained, random_emb, labels_n, label_columns)
    return out_path


def prepare_legacy_provabgs(checkpoint_path, output_dir, batch_size, seed, suffix):
    train_overlap_df = pd.read_csv(OVERLAP_TRAIN_CSV)
    eval_overlap_df = pd.read_csv(OVERLAP_EVAL_CSV)
    with h5py.File(PROCESSED_H5_LEGACY, "r") as f:
        h5_indices = f["indices"][:]
        train_rows = np.where(np.isin(h5_indices, train_overlap_df["abs_index"]))[0]
        eval_rows = np.where(np.isin(h5_indices, eval_overlap_df["abs_index"]))[0]

    fits_train = load_fits_scalar_columns(FITS_TRAIN_PATH_LEGACY)
    fits_eval = load_fits_scalar_columns(FITS_EVAL_PATH_LEGACY)
    labels_train, kept_train = _fit_labels_by_abs(
        train_overlap_df, train_rows, h5_indices, fits_train, overlap_id_col="TARGETID", target_id_col="legacy_object_id"
    )
    labels_eval, kept_eval = _fit_labels_by_abs(
        eval_overlap_df, eval_rows, h5_indices, fits_eval, overlap_id_col="TARGETID", target_id_col="legacy_object_id"
    )
    train_rows = train_rows[kept_train]
    eval_rows = eval_rows[kept_eval]
    train_ds = H5PairRowsDataset(train_rows)
    eval_ds = H5PairRowsDataset(eval_rows)
    labels_all = pd.concat([labels_train, labels_eval], axis=0, ignore_index=True)
    labels_n = _serialize_labels_df(labels_all)
    label_columns = list(labels_all.columns)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _emb_fn(model):
        e_tr = generate_embeddings_hsc_legacy(model, train_ds, device, batch_size)
        e_ev = generate_embeddings_hsc_legacy(model, eval_ds, device, batch_size)
        return {name: np.concatenate([e_tr[i], e_ev[i]], axis=0) for i, name in enumerate(EMBEDDING_NAMES)}

    real, untrained, random_emb = _run_model_variants(_emb_fn, checkpoint_path, seed)
    out_path = output_dir / f"downstream_legacy_provabgs_{suffix}.h5"
    save_downstream_h5_all_variants(out_path, real, untrained, random_emb, labels_n, label_columns)
    return out_path


def prepare_hsc_provabgs(checkpoint_path, output_dir, batch_size, seed, suffix):
    overlap_df = pd.read_csv(OVERLAP_CSV_HSC)
    with h5py.File(PROCESSED_H5_LEGACY, "r") as f:
        h5_indices = f["indices"][:]
        rows = np.where(np.isin(h5_indices, overlap_df["abs_index"]))[0]

    fits_train = load_fits_scalar_columns(FITS_TRAIN_PATH_HSC)
    labels_df, kept = _fit_labels_by_abs(
        overlap_df, rows, h5_indices, fits_train, overlap_id_col="TARGETID", target_id_col="hsc_object_id"
    )
    rows = rows[kept]
    dataset = H5PairRowsDataset(rows)
    labels_n = _serialize_labels_df(labels_df)
    label_columns = list(labels_df.columns)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _emb_fn(model):
        arrs = generate_embeddings_hsc_legacy(model, dataset, device, batch_size)
        return dict(zip(EMBEDDING_NAMES, arrs))

    real, untrained, random_emb = _run_model_variants(_emb_fn, checkpoint_path, seed)
    out_path = output_dir / f"downstream_hsc_provabgs_{suffix}.h5"
    save_downstream_h5_all_variants(out_path, real, untrained, random_emb, labels_n, label_columns)
    return out_path


def prepare_neighbors(checkpoint_path, output_dir, batch_size, seed, suffix):
    full_dataset = NeighborsSimpleDataset(hdf5_path=NEIGHBORS_HDF5)
    n_use = min(NUM_EXAMPLES_NEIGHBORS, len(full_dataset))
    dataset = Subset(full_dataset, range(n_use))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    meta_cache = {"labels_n": None, "label_columns": None}

    def _emb_fn(model):
        e_arrays, metadata_collected = generate_embeddings_neighbors(
            model, dataset, device, batch_size, shuffle=SHUFFLE_NEIGHBORS, seed=NEIGHBORS_SEED
        )
        if meta_cache["labels_n"] is None:
            labels_n, label_columns = _labels_from_metadata_list(metadata_collected)
            meta_cache["labels_n"] = labels_n
            meta_cache["label_columns"] = label_columns
        return dict(zip(EMBEDDING_NAMES, e_arrays))

    real, untrained, random_emb = _run_model_variants(_emb_fn, checkpoint_path, seed)
    out_path = output_dir / f"downstream_neighbors_{suffix}.h5"
    save_downstream_h5_all_variants(
        out_path, real, untrained, random_emb, meta_cache["labels_n"], meta_cache["label_columns"]
    )
    return out_path


# ============================================================================
# Stage 2: Predict — MLP training & evaluation (from predict_all_contrastive.py)
# ============================================================================

def compute_hsc_psf_seeing(shape11, shape22):
    pixel_scale_hsc = 0.168
    return 2.355 * np.sqrt((shape11 + shape22) / 2) * pixel_scale_hsc


def load_h5_variants(path, key1, key2, add_neighbors_derived=False):
    with h5py.File(path, "r") as f:
        raw_cols = f.attrs.get("label_columns", [])
        label_columns = [c.decode("utf-8") if isinstance(c, bytes) else c for c in (raw_cols if isinstance(raw_cols, (list, tuple)) else list(raw_cols))]
        if not label_columns:
            label_columns = [k.replace("labels/", "") for k in f.keys() if k.startswith("labels/")]

        meta_list = []
        param_names = []
        for col in label_columns:
            key = "labels/" + col
            if key not in f:
                continue
            arr = np.array(f[key][:])
            if arr.dtype.kind not in "fiu":
                try:
                    arr = arr.astype(np.float64)
                except Exception:
                    continue
            if arr.ndim == 1:
                meta_list.append(arr)
                param_names.append(col)
            elif arr.ndim == 2:
                for j in range(arr.shape[1]):
                    meta_list.append(arr[:, j].astype(np.float64))
                    param_names.append(f"{col}_{j}")

        if not meta_list:
            raise ValueError(f"No numeric label columns in {path}")
        n = meta_list[0].shape[0]

        if add_neighbors_derived:
            for band in ("g", "i", "r", "z"):
                name_11 = f"hsc_{band}_sdssshape_psf_shape11"
                name_22 = f"hsc_{band}_sdssshape_psf_shape22"
                if name_11 in param_names and name_22 in param_names:
                    idx11 = param_names.index(name_11)
                    idx22 = param_names.index(name_22)
                    psf_fwhm = compute_hsc_psf_seeing(meta_list[idx11], meta_list[idx22])
                    meta_list.append(psf_fwhm.astype(np.float64))
                    param_names.append(f"hsc_{band}_psf_fwhm")

        metadata = np.stack(meta_list, axis=1)

        out = {}
        for variant in EMBEDDING_VARIANTS:
            suf = "" if variant == "real" else f"_{variant}"
            k1, k2 = key1 + suf, key2 + suf
            if k1 not in f or k2 not in f:
                raise ValueError(f"Missing {k1} or {k2} in {path}")
            emb1 = np.array(f[k1][:])
            emb2 = np.array(f[k2][:])
            if emb1.shape[0] != n or emb2.shape[0] != n:
                raise ValueError(f"Length mismatch in {path}: embeddings vs labels")
            out[variant] = (emb1, emb2, metadata.copy(), list(param_names))
    return out


def _standardize_with_stats(data, mean, std):
    std = np.where(std == 0, 1.0, std)
    return (np.asarray(data, dtype=np.float64) - mean) / (std + 1e-8)


class MLPRegressor(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=None, dropout=0.2):
        super().__init__()
        hidden = hidden or MLP_HIDDEN
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LitRegressor(pl.LightningModule):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.save_hyperparameters()
        self.model = MLPRegressor(in_dim=in_dim, out_dim=out_dim)
        self.loss_fn = nn.SmoothL1Loss(beta=1.0)

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, stage):
        x, y = batch
        y_hat = torch.nan_to_num(self(x), nan=0.0)
        loss = self.loss_fn(y_hat, y)
        self.log(f"{stage}/loss", loss, on_epoch=True, prog_bar=True, batch_size=x.size(0))
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def configure_optimizers(self):
        opt = AdamW(self.parameters(), lr=1e-3, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
        return {"optimizer": opt, "lr_scheduler": sched}

    def on_before_optimizer_step(self, optimizer):
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)


def evaluate_per_target(model, loader, param_names, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in loader:
            y_hat = model(x.to(device))
            all_preds.append(y_hat.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    results = []
    for i in range(all_targets.shape[1]):
        y_true = all_targets[:, i]
        y_pred = all_preds[:, i]
        valid = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_t, y_p = y_true[valid], y_pred[valid]
        if len(y_t) == 0:
            r2, mae, rmse = np.nan, np.nan, np.nan
        else:
            r2 = np.nan if np.std(y_t) < 1e-6 else r2_score(y_t, y_p)
            mae = mean_absolute_error(y_t, y_p)
            rmse = np.sqrt(mean_squared_error(y_t, y_p))
        name = param_names[i] if i < len(param_names) else f"target_{i}"
        results.append({"target": name, "r2": r2, "mae": mae, "rmse": rmse, "n_valid": len(y_t)})
    return results


def evaluate_mean_baseline(meta, param_names, tr_idx, va_idx):
    mean_tr = meta[tr_idx].mean(axis=0)
    meta_val = meta[va_idx]
    results = []
    for i in range(meta_val.shape[1]):
        y_true = meta_val[:, i]
        y_pred = np.full_like(y_true, mean_tr[i])
        valid = ~np.isnan(y_true)
        y_t, y_p = y_true[valid], y_pred[valid]
        if len(y_t) == 0:
            r2, mae, rmse = np.nan, np.nan, np.nan
        else:
            r2 = np.nan if np.std(y_t) < 1e-6 else r2_score(y_t, y_p)
            mae = mean_absolute_error(y_t, y_p)
            rmse = np.sqrt(mean_squared_error(y_t, y_p))
        name = param_names[i] if i < len(param_names) else f"target_{i}"
        results.append({"target": name, "r2": r2, "mae": mae, "rmse": rmse, "n_valid": len(y_t)})
    return results


def train_and_eval_one(emb, meta, param_names, tr_idx, va_idx, use_embedding, use_gpu):
    mean = meta[tr_idx].mean(axis=0)
    std = meta[tr_idx].std(axis=0)
    meta_s = _standardize_with_stats(meta, mean, std)

    x_all = torch.tensor(emb[use_embedding - 1], dtype=torch.float32)
    m = torch.tensor(meta_s, dtype=torch.float32)
    in_dim = x_all.shape[1]
    print(f"  train_and_eval_one: use_embedding={use_embedding}, in_dim={in_dim}, n={x_all.shape[0]}")

    train_ds = TensorDataset(x_all[tr_idx], m[tr_idx])
    val_ds = TensorDataset(x_all[va_idx], m[va_idx])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = LitRegressor(in_dim=in_dim, out_dim=meta_s.shape[1])
    ckpt = ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1, save_last=False)
    trainer = Trainer(
        max_epochs=100,
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        callbacks=[
            ckpt,
            EarlyStopping(monitor="val/loss", mode="min", patience=20),
            LearningRateMonitor(logging_interval="epoch"),
        ],
        log_every_n_steps=20,
        enable_progress_bar=True,
    )
    trainer.fit(model, train_loader, val_loader)
    best = LitRegressor.load_from_checkpoint(ckpt.best_model_path) if ckpt.best_model_path else model
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    return evaluate_per_target(best.to(device), val_loader, param_names, device)


def run_task(task_name, h5_stem, default_use_embedding, target_list, h5_path, seed, use_gpu):
    add_derived = h5_stem == "neighbors"
    data = load_h5_variants(h5_path, KEY1, KEY2, add_neighbors_derived=add_derived)
    _, _, metadata, param_names = data["real"]

    col_set = set(target_list)
    indices = [i for i, n in enumerate(param_names) if n in col_set]
    missing = sorted(col_set - {param_names[i] for i in indices})
    if missing:
        print(f"  Warning {task_name}: missing targets skipped: {missing}")
    if not indices:
        raise ValueError(f"No requested target columns found for {task_name}")

    param_names = [param_names[i] for i in indices]
    meta = metadata[:, indices].astype(np.float64)
    finite = np.isfinite(meta).all(axis=1)
    if finite.sum() == 0:
        raise ValueError(f"No finite labels for {task_name}")
    meta = meta[finite]

    rng = np.random.default_rng(seed)
    idx = np.arange(meta.shape[0])
    rng.shuffle(idx)
    n_train = int(TRAIN_FRAC * len(idx))
    tr_idx, va_idx = idx[:n_train], idx[n_train:]

    out = {}
    for variant in EMBEDDING_VARIANTS:
        emb1, emb2, _, _ = data[variant]
        emb = (
            np.nan_to_num(emb1[finite], nan=0.0, posinf=0.0, neginf=0.0),
            np.nan_to_num(emb2[finite], nan=0.0, posinf=0.0, neginf=0.0),
        )
        if variant == "real":
            out["real_physics"] = train_and_eval_one(emb, meta, param_names, tr_idx, va_idx, use_embedding=1, use_gpu=use_gpu)
            out["real_instrument"] = train_and_eval_one(emb, meta, param_names, tr_idx, va_idx, use_embedding=2, use_gpu=use_gpu)
        else:
            out[variant] = train_and_eval_one(
                emb, meta, param_names, tr_idx, va_idx, use_embedding=default_use_embedding, use_gpu=use_gpu
            )

    out["mean"] = evaluate_mean_baseline(meta, param_names, tr_idx, va_idx)
    return out, param_names


def save_results_csv(all_results, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "objective", "task", "target",
        "r2_physics", "r2_instrument", "r2_untrained", "r2_mean",
        "mae_physics", "mae_instrument", "mae_untrained", "mae_mean",
    ]
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for objective, task_name, results_per_variant, param_names in all_results:
            for t in param_names:
                row = {"objective": objective, "task": task_name, "target": t}
                mapping = [
                    ("real_physics", "physics"),
                    ("real_instrument", "instrument"),
                    ("untrained", "untrained"),
                    ("mean", "mean"),
                ]
                for key, short in mapping:
                    rec = next((x for x in results_per_variant.get(key, []) if x["target"] == t), None)
                    if rec is None:
                        row[f"r2_{short}"] = ""
                        row[f"mae_{short}"] = ""
                    else:
                        row[f"r2_{short}"] = rec["r2"] if not np.isnan(rec["r2"]) else ""
                        row[f"mae_{short}"] = rec["mae"] if not np.isnan(rec["mae"]) else ""
                w.writerow(row)
    print(f"Results CSV saved: {output_path}")


def plot_objective(df_obj, objective, out_path):
    df_plot = df_obj.copy().sort_values(["task", "target"]).reset_index(drop=True)
    if len(df_plot) == 0:
        return
    x = np.arange(len(df_plot))
    width = 0.25

    vals_phys = pd.to_numeric(df_plot["r2_physics"], errors="coerce").to_numpy()
    vals_inst = pd.to_numeric(df_plot["r2_instrument"], errors="coerce").to_numpy()
    vals_untr = pd.to_numeric(df_plot["r2_untrained"], errors="coerce").to_numpy()

    fig, ax = plt.subplots(figsize=(max(12, len(df_plot) * 0.5), 6))
    ax.bar(x - width, np.nan_to_num(vals_phys, nan=0.0), width, label="Physics latents", color="#8AC3EE")
    ax.bar(x, np.nan_to_num(vals_inst, nan=0.0), width, label="Instrument latents", color="#E5254E")
    ax.bar(x + width, np.nan_to_num(vals_untr, nan=0.0), width, label="Untrained", color="#B19221")

    labels = [f"{t}\n({task})" for t, task in zip(df_plot["target"], df_plot["task"])]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("R²")
    ax.set_title(f"Downstream Objective: {objective}")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="upper left")
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {out_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser(
        description="End-to-end downstream evaluation for 'ours' model (prepare + predict)."
    )
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                   help="Path to ours model checkpoint (.ckpt)")
    p.add_argument("--suffix", default=None,
                   help="Output suffix. Default: derived from checkpoint path")
    p.add_argument("--output-dir", type=Path, default=_here,
                   help="Directory for H5 files and results")
    p.add_argument("--batch-size", type=int, default=256,
                   help="Batch size for embedding extraction")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-gpu", action="store_true")
    p.add_argument("--skip-prepare", action="store_true",
                   help="Skip Stage 1 (prepare); assume H5 files exist")
    p.add_argument("--prepare-only", action="store_true",
                   help="Run Stage 1 (prepare) only; skip prediction")
    p.add_argument("--datasets", default="mmu,legacy_provabgs,neighbors,hsc_provabgs",
                   help="Comma-separated subset of: mmu,legacy_provabgs,neighbors,hsc_provabgs")
    args = p.parse_args()

    suffix = args.suffix or Path(args.checkpoint).stem
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = [s.strip() for s in args.datasets.split(",") if s.strip()]
    use_gpu = (not args.no_gpu) and torch.cuda.is_available()

    h5_paths = {
        "mmu": output_dir / f"downstream_mmu_{suffix}.h5",
        "legacy_provabgs": output_dir / f"downstream_legacy_provabgs_{suffix}.h5",
        "neighbors": output_dir / f"downstream_neighbors_{suffix}.h5",
        "hsc_provabgs": output_dir / f"downstream_hsc_provabgs_{suffix}.h5",
    }

    # --- Stage 1: Prepare ---
    if not args.skip_prepare:
        print("=" * 60)
        print("Stage 1: Prepare downstream H5 files")
        print("=" * 60)
        if "mmu" in datasets:
            print("\n[MMU]")
            prepare_mmu(args.checkpoint, output_dir, args.batch_size, args.seed, suffix)
        if "legacy_provabgs" in datasets:
            print("\n[Legacy ProvaBGS]")
            prepare_legacy_provabgs(args.checkpoint, output_dir, args.batch_size, args.seed, suffix)
        if "neighbors" in datasets:
            print("\n[Neighbors]")
            prepare_neighbors(args.checkpoint, output_dir, args.batch_size, args.seed, suffix)
        if "hsc_provabgs" in datasets:
            print("\n[HSC ProvaBGS]")
            prepare_hsc_provabgs(args.checkpoint, output_dir, args.batch_size, args.seed, suffix)
        print("\nStage 1 complete.")

    if args.prepare_only:
        print("Done (prepare-only).")
        return

    # --- Stage 2: Predict ---
    print("\n" + "=" * 60)
    print("Stage 2: Train MLPs & evaluate")
    print("=" * 60)

    for k in datasets:
        if k in h5_paths and not h5_paths[k].exists():
            raise FileNotFoundError(f"Missing required H5 for {k}: {h5_paths[k]}")

    print(f"GPU enabled: {use_gpu}")

    all_results = []
    for objective, task_name, h5_stem, default_use_embedding, targets in TASKS:
        if h5_stem not in datasets:
            continue
        if not targets:
            continue
        print(f"\n--- {objective} :: {task_name} ({len(targets)} targets) ---")
        results_per_variant, param_names = run_task(
            task_name, h5_stem, default_use_embedding, targets,
            h5_paths[h5_stem], args.seed, use_gpu,
        )
        all_results.append((objective, task_name, results_per_variant, param_names))

    csv_path = output_dir / f"predict_all_ours_{suffix}.csv"
    save_results_csv(all_results, csv_path)

    df = pd.read_csv(csv_path)
    for objective in sorted(df["objective"].unique()):
        df_obj = df[df["objective"] == objective].copy()
        plot_path = output_dir / f"predict_all_ours_{suffix}_{objective}.png"
        plot_objective(df_obj, objective, plot_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
