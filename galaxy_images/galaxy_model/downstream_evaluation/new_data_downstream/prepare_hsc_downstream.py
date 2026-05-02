"""
Stage 1: Prepare HSC downstream embeddings for ConditionalFlowMatchingModule.

Reads images from hsc_downstream/hsc_flux.bin (raw nJy, float16, 5-band, 160x160),
preprocesses (center-crop 160->48, arcsinh+normalize, drop y-band), and encodes with
encoder_1 (physics) and encoder_2 (instrument).

Output: downstream_hsc_new_{suffix}.h5
  - hsc_encoder1{,_untrained,_random}   shape (N, D1)
  - hsc_encoder2{,_untrained,_random}   shape (N, D2)
  - labels/{col}                         shape (N,) per label column
"""

import argparse
import importlib.util
import inspect
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# ROCm workaround: hipBLASLt is buggy on MI210 for certain matrix shapes.
if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "preferred_blas_library"):
    torch.backends.cuda.preferred_blas_library("hipblas")

_here = Path(__file__).resolve().parent
_project_root = _here.parents[3]   # .../tess-generative
_galaxy_model  = _project_root / "galaxy_images" / "galaxy_model"
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from galaxy_images.image_preprocessing import preprocess_image_v2

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

HSC_DATA_DIR        = Path("/work1/jeroenaudenaert/pablomer/data/hsc_downstream")
DEFAULT_CATALOG     = HSC_DATA_DIR / "catalog.parquet"
DEFAULT_IMAGES_BIN  = HSC_DATA_DIR / "hsc_flux.bin"
HSC_BANDS           = 5
HSC_HEIGHT          = 160
HSC_WIDTH           = 160
HSC_BYTES_PER_GALAXY = HSC_BANDS * HSC_HEIGHT * HSC_WIDTH * 2  # float16 = 2 bytes


def _detect_n_galaxies(images_bin: Path) -> int:
    """Infer number of complete galaxy records from file size."""
    n = os.path.getsize(images_bin) // HSC_BYTES_PER_GALAXY
    if n == 0:
        raise ValueError(f"File too small to contain any complete galaxies: {images_bin}")
    return n

HSC_NORM_MEAN = 0.022
HSC_NORM_STD  = 0.05

EMBEDDING_NAMES = ("hsc_encoder1", "hsc_encoder2")

# Catalog columns that are identifiers, constants, or non-numeric
SKIP_COLS = frozenset({
    "object_id", "image_idx", "healpix",
    "tract", "patch",     # HSC field/patch identifiers
    "image_scale",        # list col, always 0.168 arcsec/px — constant
    "sugohi_grade",       # string
    "sugohi_name",        # string
    "desi_targetid",      # int64 identifier, not a prediction target
    "is_bgs_bright", "is_bgs_faint",  # boolean flags — low variance, not useful for regression
})


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HSCBinaryDataset(Dataset):
    """
    Reads hsc_flux.bin via file seek — avoids mapping the full 14+ GB file into
    virtual address space, which fails on login nodes with ulimit -v restrictions.
    One file handle per dataset instance (num_workers=0 assumed).
    """

    def __init__(self, images_bin: Path, indices: np.ndarray, n_total: int):
        self._images_bin = str(images_bin)
        self.indices     = indices
        self._file       = None   # opened lazily

    def __len__(self):
        return len(self.indices)

    def __del__(self):
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass

    def _get_file(self):
        if self._file is None:
            self._file = open(self._images_bin, "rb")
        return self._file

    def __getitem__(self, idx):
        byte_offset = int(self.indices[idx]) * HSC_BYTES_PER_GALAXY
        f = self._get_file()
        f.seek(byte_offset)
        buf = f.read(HSC_BYTES_PER_GALAXY)
        raw = np.frombuffer(buf, dtype=np.float16).copy().reshape(HSC_BANDS, HSC_HEIGHT, HSC_WIDTH).astype(np.float32)
        img = torch.from_numpy(raw)
        img = preprocess_image_v2(img, crop_size=48, survey="hsc")     # (5, 48, 48)
        img = (img - HSC_NORM_MEAN) / HSC_NORM_STD
        return img[:4]   # drop y-band → (4, 48, 48)


# ---------------------------------------------------------------------------
# Label extraction
# ---------------------------------------------------------------------------

def load_catalog_labels(catalog_path: Path, indices: np.ndarray):
    """Return (labels_n, label_columns) for the given row indices."""
    df = pd.read_parquet(catalog_path)
    df = df.iloc[indices].reset_index(drop=True)

    # Expand list column image_psf_fwhm → 5 scalar columns
    if "image_psf_fwhm" in df.columns:
        try:
            psf = np.stack(df["image_psf_fwhm"].values).astype(np.float32)
        except Exception:
            psf = np.array(df["image_psf_fwhm"].tolist(), dtype=np.float32)
        for i, band in enumerate(["g", "r", "i", "z", "y"]):
            df[f"psf_fwhm_{band}"] = psf[:, i]
        df = df.drop(columns=["image_psf_fwhm"])

    labels_n, label_columns = {}, []
    for col in df.columns:
        if col in SKIP_COLS:
            continue
        series = df[col]
        if series.dtype.kind in "iufc":   # int, uint, float, complex
            arr = series.to_numpy(dtype=np.float32, na_value=np.nan)
            labels_n[col] = arr
            label_columns.append(col)

    return labels_n, label_columns


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_module_class(module_filename: str, model_class: str):
    mod_name = module_filename.replace(".py", "").replace("/", "_").replace(".", "_")
    spec = importlib.util.spec_from_file_location(mod_name, _galaxy_model / module_filename)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, model_class):
        available = [k for k in dir(mod) if not k.startswith("_")]
        raise AttributeError(
            f"Class '{model_class}' not found in {module_filename}. "
            f"Available names: {available}"
        )
    return getattr(mod, model_class)


def load_trained_model(checkpoint_path: str, module_filename: str, model_class: str):
    cls   = _load_module_class(module_filename, model_class)
    model = cls.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model.eval()
    torch.set_grad_enabled(False)
    return model


def load_untrained_model(checkpoint_path: str, module_filename: str, model_class: str):
    """Same architecture as checkpoint, but random (untrained) weights."""
    cls    = _load_module_class(module_filename, model_class)
    ckpt   = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hparams = ckpt.get("hyper_parameters", {})
    valid  = set(inspect.signature(cls.__init__).parameters.keys()) - {"self"}
    kwargs = {k: v for k, v in hparams.items() if k in valid}
    model  = cls(**kwargs)
    model.eval()
    torch.set_grad_enabled(False)
    return model


# ---------------------------------------------------------------------------
# Embedding generation
# ---------------------------------------------------------------------------

def generate_embeddings(model, dataset: Dataset, device, batch_size: int):
    """Run encoder_1 and encoder_2 on all images; return (e1, e2) numpy arrays."""
    loader   = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                          num_workers=0, pin_memory=(device.type == "cuda"))
    e1_parts, e2_parts = [], []
    total = 0
    with torch.no_grad():
        for imgs in loader:
            imgs = imgs.to(device)
            e1_parts.append(model.encoder_1(imgs).flatten(start_dim=1).cpu())
            e2_parts.append(model.encoder_2(imgs).flatten(start_dim=1).cpu())
            total += imgs.size(0)
            if total % 10_000 < batch_size:
                print(f"  encoded {total}/{len(dataset)}")
    e1 = torch.cat(e1_parts, dim=0).numpy()
    e2 = torch.cat(e2_parts, dim=0).numpy()
    return e1, e2


def make_random_embeddings_like(real_dict: dict, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    return {k: rng.standard_normal(v.shape).astype(np.float32) for k, v in real_dict.items()}


# ---------------------------------------------------------------------------
# HDF5 output
# ---------------------------------------------------------------------------

def save_h5(path: Path, real, untrained, random_emb, labels_n, label_columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for name in EMBEDDING_NAMES:
            f.create_dataset(name,                  data=real[name],      compression="gzip", compression_opts=4)
            f.create_dataset(name + "_untrained",   data=untrained[name], compression="gzip", compression_opts=4)
            f.create_dataset(name + "_random",      data=random_emb[name],compression="gzip", compression_opts=4)
        for col in label_columns:
            f.create_dataset(f"labels/{col}", data=labels_n[col], compression="gzip", compression_opts=4)
        f.attrs["embedding_names"] = list(EMBEDDING_NAMES)
        f.attrs["num_examples"]    = real[EMBEDDING_NAMES[0]].shape[0]
        f.attrs["label_columns"]   = list(label_columns)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Prepare HSC downstream embeddings (flow matching model).")
    p.add_argument("--checkpoint",   required=True,  help="Path to .ckpt file")
    p.add_argument("--module",       default="double_train_fm_neighbors.py",
                   help="Module filename relative to galaxy_model/ (e.g. 'hierarchical_attention/double_train_fm_neighbors_hier_global_ins.py')")
    p.add_argument("--model-class",  default="ConditionalFlowMatchingModule",
                   help="LightningModule class name inside --module (default: ConditionalFlowMatchingModule)")
    p.add_argument("--suffix",       default=None,   help="Output suffix (default: checkpoint stem)")
    p.add_argument("--output-dir",   type=Path, default=_here)
    p.add_argument("--batch-size",   type=int,  default=256)
    p.add_argument("--num-examples", type=int,  default=None,
                   help="Limit to first N galaxies (default: all 474954)")
    p.add_argument("--seed",         type=int,  default=42)
    p.add_argument("--catalog",      type=Path, default=DEFAULT_CATALOG)
    p.add_argument("--images-bin",   type=Path, default=DEFAULT_IMAGES_BIN)
    args = p.parse_args()

    suffix     = args.suffix or Path(args.checkpoint).stem
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    n_complete = _detect_n_galaxies(args.images_bin)
    n_total    = min(args.num_examples or n_complete, n_complete)
    indices    = np.arange(n_total, dtype=np.int64)
    print(f"Complete galaxies in binary: {n_complete:,}  |  Processing: {n_total:,}")

    print("Loading catalog labels...")
    labels_n, label_columns = load_catalog_labels(args.catalog, indices)
    print(f"  {len(label_columns)} label columns: {label_columns[:5]}...")

    dataset = HSCBinaryDataset(args.images_bin, indices, n_complete)

    def _emb_fn(model):
        e1, e2 = generate_embeddings(model, dataset, device, args.batch_size)
        return {"hsc_encoder1": e1, "hsc_encoder2": e2}

    print("\n[1/3] Encoding with trained model...")
    model = load_trained_model(args.checkpoint, args.module, args.model_class).to(device)
    real  = _emb_fn(model)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("\n[2/3] Encoding with untrained model...")
    model_u  = load_untrained_model(args.checkpoint, args.module, args.model_class).to(device)
    untrained = _emb_fn(model_u)
    del model_u
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("\n[3/3] Generating random embeddings...")
    random_emb = make_random_embeddings_like(real, seed=args.seed)

    out_path = output_dir / f"downstream_hsc_new_{suffix}.h5"
    save_h5(out_path, real, untrained, random_emb, labels_n, label_columns)
    print("Done.")


if __name__ == "__main__":
    main()
