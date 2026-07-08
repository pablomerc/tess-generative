"""
Stage 2 — embed three subsets (overlap / hsc_only / legacy_only) with a single
checkpoint and write one HDF5 with the layout below.

CLI shape (locked — predict_combined.py and the slurm wrappers depend on it):

    python prepare_combined.py \
        --checkpoint   <path-to-snapshot.ckpt> \
        --module       <module-relative-to-galaxy_model/> \
        --model-class  <ClassName> \
        --index-dir    outputs/index \
        --out          outputs/embeddings/<ckpt_name>.h5 \
        [--hsc-dir    /path/to/hsc_downstream] \
        [--legacy-dir /path/to/legacy_downstream_full/full_1M] \
        [--batch-size 256] \
        [--smoke]              # if set, take first 512 rows of each subset

Per-checkpoint H5 layout:

    /overlap/
        hsc_e1, hsc_e2                  (N, D1), (N, D2)   encoder_1/2 on HSC image
        legacy_e1, legacy_e2            (N, D1), (N, D2)   encoder_1/2 on Legacy image
        combined_e1, combined_e2        (N, 2*D1), (N, 2*D2)
        labels/<col>                    catalog labels prefixed "hsc_"/"legacy_"
    /hsc_only/
        hsc_e1, hsc_e2
        labels/<col>
    /legacy_only/
        legacy_e1, legacy_e2
        labels/<col>

Single-encoder variant (SingleEncoderFlowMatchingModule) writes only *_e1.
Hierarchical variants use the global_concat physics latent (see
hierarchical_latent_experiments/common/latents.py) for *_e1 and the
instrument flat for *_e2.
"""

import argparse
import importlib.util
import inspect
import os
import sys
import time
from pathlib import Path

# ROCm workaround — harmless on H100/CUDA.
try:
    import torch
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "preferred_blas_library"):
        torch.backends.cuda.preferred_blas_library("hipblas")
except Exception:
    pass

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

_here = Path(__file__).resolve().parent
_project_root = _here.parents[3]            # .../tess-generative
_galaxy_model = _project_root / "galaxy_images" / "galaxy_model"
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from galaxy_images.galaxy_model.neighbors import preprocess_raw_image  # noqa: E402

HSC_BANDS, HSC_H, HSC_W = 5, 160, 160
LEGACY_BANDS, LEGACY_H, LEGACY_W = 4, 160, 160
HSC_BYTES = HSC_BANDS * HSC_H * HSC_W * 2     # float16
LEGACY_BYTES = LEGACY_BANDS * LEGACY_H * LEGACY_W * 2

# Catalog columns we never want to use as labels (identifiers / strings / constants).
SKIP_COLS = frozenset({
    "image_idx", "object_id", "healpix",
    "tract", "patch",
    "image_scale", "image_psf_fwhm",   # list cols; PSF is exploded below
    "sugohi_grade", "sugohi_name",
    "desi_targetid",
    "is_bgs_bright", "is_bgs_faint",
    # Legacy-side string / sparse-string identifiers
    "BRICKID", "BRICKNAME", "OBJID", "RELEASE", "gid",
    "TYPE", "MASKBITS",
    "euclid_grade", "euclid_id_str", "euclid_subset",
})


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class _BinaryImageDataset(Dataset):
    """Reads images by file-seek from a `.bin` of float16 records."""

    def __init__(self, bin_path: Path, indices: np.ndarray, bands: int, h: int, w: int, survey: str):
        self._bin = str(bin_path)
        self._indices = np.asarray(indices, dtype=np.int64)
        self._bytes = bands * h * w * 2
        self._shape = (bands, h, w)
        self._survey = survey
        self._fh = None

    def __len__(self):
        return len(self._indices)

    def _file(self):
        if self._fh is None:
            self._fh = open(self._bin, "rb")
        return self._fh

    def __del__(self):
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass

    def __getitem__(self, i):
        offset = int(self._indices[i]) * self._bytes
        fh = self._file()
        fh.seek(offset)
        buf = fh.read(self._bytes)
        raw = np.frombuffer(buf, dtype=np.float16).copy().reshape(self._shape).astype(np.float32)
        img = torch.from_numpy(raw)
        img = preprocess_raw_image(img, survey=self._survey, crop_size=48)
        if self._survey == "hsc":
            img = img[:4]   # drop y-band → (4, 48, 48)
        return img


# ---------------------------------------------------------------------------
# Model loading and embedding
# ---------------------------------------------------------------------------

def _load_module_class(module_filename: str, class_name: str):
    mod_name = module_filename.replace("/", "_").replace(".py", "")
    spec = importlib.util.spec_from_file_location(mod_name, _galaxy_model / module_filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, class_name):
        raise AttributeError(
            f"{class_name} not found in {module_filename}. "
            f"Available: {[n for n in dir(mod) if not n.startswith('_')]}"
        )
    return getattr(mod, class_name)


def _load_model(checkpoint: Path, module: str, model_class: str, device: torch.device):
    cls = _load_module_class(module, model_class)
    model = cls.load_from_checkpoint(str(checkpoint), map_location="cpu")
    model.eval()
    torch.set_grad_enabled(False)
    return model.to(device)


def _is_hierarchical(model) -> bool:
    return hasattr(model, "encode_image")


def _is_single_encoder(model) -> bool:
    return hasattr(model, "encoder") and not hasattr(model, "encoder_1")


@torch.no_grad()
def _embed_batch(model, imgs: torch.Tensor, has_e2: bool):
    """Return (e1, e2) for a batch. e2 is None if the model has only one encoder."""
    if _is_hierarchical(model):
        enc = model.encode_image(imgs)
        p = enc["physics"]
        e1 = torch.cat([p["global_vec"], p["spatial_flat"]], dim=1)   # global_concat
        e2 = enc["instrument"]["flat"]
    elif _is_single_encoder(model):
        e1 = model.encoder(imgs).flatten(start_dim=1)
        e2 = None
    else:
        e1 = model.encoder_1(imgs).flatten(start_dim=1)
        e2 = model.encoder_2(imgs).flatten(start_dim=1)
    return e1.float().cpu(), (e2.float().cpu() if e2 is not None else None)


@torch.no_grad()
def _encode_dataset(model, dataset: Dataset, device: torch.device, batch_size: int, has_e2: bool, log_every: int = 10_000):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=(device.type == "cuda"))
    e1_chunks, e2_chunks = [], []
    n_done = 0
    last_log = 0
    t0 = time.time()
    for imgs in loader:
        imgs = imgs.to(device, non_blocking=(device.type == "cuda"))
        e1, e2 = _embed_batch(model, imgs, has_e2)
        e1_chunks.append(e1)
        if has_e2:
            e2_chunks.append(e2)
        n_done += imgs.size(0)
        if n_done - last_log >= log_every:
            dt = time.time() - t0
            print(f"    encoded {n_done:>7d}/{len(dataset)}  ({n_done/dt:.1f} img/s)")
            last_log = n_done
    e1_arr = torch.cat(e1_chunks, dim=0).numpy().astype(np.float32)
    e2_arr = torch.cat(e2_chunks, dim=0).numpy().astype(np.float32) if has_e2 else None
    return e1_arr, e2_arr


# ---------------------------------------------------------------------------
# Catalog labels
# ---------------------------------------------------------------------------

def _expand_psf_list(df: pd.DataFrame, survey: str):
    """Expand list-valued image_psf_fwhm into per-band scalar columns."""
    if "image_psf_fwhm" not in df.columns:
        return df
    bands = ["g", "r", "i", "z", "y"] if survey == "hsc" else ["g", "r", "i", "z"]
    try:
        psf = np.stack([np.asarray(v, dtype=np.float32) for v in df["image_psf_fwhm"].values])
    except Exception:
        psf = np.array(df["image_psf_fwhm"].tolist(), dtype=np.float32)
    if psf.ndim == 2 and psf.shape[1] == len(bands):
        for j, b in enumerate(bands):
            df[f"psf_fwhm_{b}"] = psf[:, j]
    return df.drop(columns=["image_psf_fwhm"])


def _extract_labels(catalog_path: Path, indices: np.ndarray, survey: str, prefix: str):
    """Return dict[col -> 1-d float array] for catalog rows at `indices`. Columns are prefixed."""
    df = pd.read_parquet(catalog_path)
    df = df.iloc[indices].reset_index(drop=True)
    df = _expand_psf_list(df, survey)
    out = {}
    for col in df.columns:
        if col in SKIP_COLS:
            continue
        s = df[col]
        if s.dtype.kind in "iufcb":   # int, unsigned, float, complex, bool
            arr = s.to_numpy(dtype=np.float32, na_value=np.nan)
            out[f"{prefix}{col}"] = arr
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--module", required=True,
                   help="Module filename relative to galaxy_model/ (e.g. 'double_train_fm_neighbors.py')")
    p.add_argument("--model-class", required=True,
                   help="LightningModule class name inside --module")
    p.add_argument("--index-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--hsc-dir", type=Path,
                   default=Path("/home/pablomer/orcd/scratch/hsc_downstream"))
    p.add_argument("--legacy-dir", type=Path,
                   default=Path("/home/pablomer/orcd/scratch/legacy_downstream_full/full_1M"))
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--smoke", action="store_true", help="Use only the first 512 rows of each subset")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Module/class: {args.module} :: {args.model_class}")

    model = _load_model(args.checkpoint, args.module, args.model_class, device)
    has_e2 = not _is_single_encoder(model)
    arch_tag = (
        "hier" if _is_hierarchical(model) else
        ("single" if _is_single_encoder(model) else "base")
    )
    print(f"Architecture: {arch_tag}  has_e2={has_e2}")

    # Load index parquets (output of build_overlap_dataset.py).
    idx_dir = args.index_dir
    overlap_path = idx_dir / "overlap.parquet"
    hsc_only_path = idx_dir / "hsc_only.parquet"
    legacy_only_path = idx_dir / "legacy_only.parquet"

    overlap_df = pd.read_parquet(overlap_path) if overlap_path.exists() else pd.DataFrame()
    hsc_only_df = pd.read_parquet(hsc_only_path) if hsc_only_path.exists() else pd.DataFrame()
    legacy_only_df = pd.read_parquet(legacy_only_path) if legacy_only_path.exists() else pd.DataFrame()

    if args.smoke:
        if len(overlap_df):     overlap_df    = overlap_df.head(512).reset_index(drop=True)
        if len(hsc_only_df):    hsc_only_df   = hsc_only_df.head(512).reset_index(drop=True)
        if len(legacy_only_df): legacy_only_df = legacy_only_df.head(512).reset_index(drop=True)

    print(
        f"Subsets: overlap={len(overlap_df)}  "
        f"hsc_only={len(hsc_only_df)}  legacy_only={len(legacy_only_df)}"
    )

    hsc_bin = args.hsc_dir / "hsc_flux.bin"
    leg_bin = args.legacy_dir / "legacy_flux.bin"
    hsc_cat = args.hsc_dir / "catalog.parquet"
    leg_cat = args.legacy_dir / "catalog.parquet"

    n_hsc_bin = os.path.getsize(hsc_bin) // HSC_BYTES
    n_leg_bin = os.path.getsize(leg_bin) // LEGACY_BYTES
    print(f"HSC bin rows: {n_hsc_bin:,}   Legacy bin rows: {n_leg_bin:,}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out, "w") as fout:
        fout.attrs["checkpoint_path"] = str(args.checkpoint)
        fout.attrs["checkpoint_name"] = args.checkpoint.parent.name
        fout.attrs["architecture"] = arch_tag
        fout.attrs["has_e2"] = bool(has_e2)
        per_subset_n = {}
        per_subset_labels = {}
        embedding_names = ["e1"] if not has_e2 else ["e1", "e2"]

        # --- overlap subset: encode HSC AND Legacy images on the same model ---
        if len(overlap_df):
            print("\n[overlap] encoding HSC images")
            hsc_indices = overlap_df["hsc_image_idx"].to_numpy(dtype=np.int64)
            assert (hsc_indices < n_hsc_bin).all(), "overlap.hsc_image_idx exceeds HSC bin"
            hsc_ds = _BinaryImageDataset(hsc_bin, hsc_indices, HSC_BANDS, HSC_H, HSC_W, "hsc")
            hsc_e1, hsc_e2 = _encode_dataset(model, hsc_ds, device, args.batch_size, has_e2)

            print("[overlap] encoding Legacy images")
            leg_indices = overlap_df["legacy_image_idx"].to_numpy(dtype=np.int64)
            assert (leg_indices < n_leg_bin).all(), "overlap.legacy_image_idx exceeds Legacy bin"
            leg_ds = _BinaryImageDataset(leg_bin, leg_indices, LEGACY_BANDS, LEGACY_H, LEGACY_W, "legacy")
            leg_e1, leg_e2 = _encode_dataset(model, leg_ds, device, args.batch_size, has_e2)

            grp = fout.create_group("overlap")
            grp.create_dataset("hsc_e1", data=hsc_e1, compression="gzip", compression_opts=4)
            grp.create_dataset("legacy_e1", data=leg_e1, compression="gzip", compression_opts=4)
            grp.create_dataset("combined_e1", data=np.concatenate([hsc_e1, leg_e1], axis=1),
                               compression="gzip", compression_opts=4)
            if has_e2:
                grp.create_dataset("hsc_e2", data=hsc_e2, compression="gzip", compression_opts=4)
                grp.create_dataset("legacy_e2", data=leg_e2, compression="gzip", compression_opts=4)
                grp.create_dataset("combined_e2", data=np.concatenate([hsc_e2, leg_e2], axis=1),
                                   compression="gzip", compression_opts=4)

            print("[overlap] writing labels")
            hsc_labels = _extract_labels(hsc_cat, hsc_indices, "hsc", "hsc_")
            leg_labels = _extract_labels(leg_cat, leg_indices, "legacy", "legacy_")
            for k, v in {**hsc_labels, **leg_labels}.items():
                grp.create_dataset(f"labels/{k}", data=v, compression="gzip", compression_opts=4)
            per_subset_n["overlap"] = len(overlap_df)
            per_subset_labels["overlap"] = list(hsc_labels.keys()) + list(leg_labels.keys())
        else:
            print("\n[overlap] empty — skipped")

        # --- hsc_only subset ---
        if len(hsc_only_df):
            print("\n[hsc_only] encoding HSC images")
            hsc_indices = hsc_only_df["hsc_image_idx"].to_numpy(dtype=np.int64)
            assert (hsc_indices < n_hsc_bin).all(), "hsc_only.hsc_image_idx exceeds HSC bin"
            hsc_ds = _BinaryImageDataset(hsc_bin, hsc_indices, HSC_BANDS, HSC_H, HSC_W, "hsc")
            hsc_e1, hsc_e2 = _encode_dataset(model, hsc_ds, device, args.batch_size, has_e2)

            grp = fout.create_group("hsc_only")
            grp.create_dataset("hsc_e1", data=hsc_e1, compression="gzip", compression_opts=4)
            if has_e2:
                grp.create_dataset("hsc_e2", data=hsc_e2, compression="gzip", compression_opts=4)
            print("[hsc_only] writing labels")
            hsc_labels = _extract_labels(hsc_cat, hsc_indices, "hsc", "hsc_")
            for k, v in hsc_labels.items():
                grp.create_dataset(f"labels/{k}", data=v, compression="gzip", compression_opts=4)
            per_subset_n["hsc_only"] = len(hsc_only_df)
            per_subset_labels["hsc_only"] = list(hsc_labels.keys())
        else:
            print("\n[hsc_only] empty — skipped")

        # --- legacy_only subset ---
        if len(legacy_only_df):
            print("\n[legacy_only] encoding Legacy images")
            leg_indices = legacy_only_df["legacy_image_idx"].to_numpy(dtype=np.int64)
            assert (leg_indices < n_leg_bin).all(), "legacy_only.legacy_image_idx exceeds Legacy bin"
            leg_ds = _BinaryImageDataset(leg_bin, leg_indices, LEGACY_BANDS, LEGACY_H, LEGACY_W, "legacy")
            leg_e1, leg_e2 = _encode_dataset(model, leg_ds, device, args.batch_size, has_e2)

            grp = fout.create_group("legacy_only")
            grp.create_dataset("legacy_e1", data=leg_e1, compression="gzip", compression_opts=4)
            if has_e2:
                grp.create_dataset("legacy_e2", data=leg_e2, compression="gzip", compression_opts=4)
            print("[legacy_only] writing labels")
            leg_labels = _extract_labels(leg_cat, leg_indices, "legacy", "legacy_")
            for k, v in leg_labels.items():
                grp.create_dataset(f"labels/{k}", data=v, compression="gzip", compression_opts=4)
            per_subset_n["legacy_only"] = len(legacy_only_df)
            per_subset_labels["legacy_only"] = list(leg_labels.keys())
        else:
            print("\n[legacy_only] empty — skipped")

        fout.attrs["embedding_names"] = embedding_names
        # h5py struggles with dict attrs — store as JSON strings.
        import json
        fout.attrs["num_examples_per_subset"] = json.dumps(per_subset_n)
        fout.attrs["label_columns_per_subset"] = json.dumps(per_subset_labels)

    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
