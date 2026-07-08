"""
Encode the engaging downstream subsets (overlap / hsc_only / legacy_only) with
AION-base and write one HDF5 in the same layout as prepare_combined.py so that
predict_combined.py + makeplot.py can consume it as just another checkpoint.

CLI shape:
    python prepare_aion.py \
        --index-dir outputs/index \
        --out       outputs/embeddings/aion-base.h5 \
        [--hsc-dir /path] [--legacy-dir /path] [--batch-size 16] [--smoke]

Layout written:
    /overlap/{hsc_e1, legacy_e1, combined_e1, labels/<col>}
    /hsc_only/{hsc_e1, labels/<col>}
    /legacy_only/{legacy_e1, labels/<col>}
    attrs: checkpoint_name="aion-base", architecture="aion", has_e2=False

Embeddings are AION encoder outputs mean-pooled across token dimension (768-d).
Raw 160x160 float16 flux is fed directly to AION's codec without our preprocessing.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

# AION 0.x has a known typo (RescaleToLegacySurvey._reverse_zeropoint); patch it.
try:
    from aion.codecs.preprocessing.image import RescaleToLegacySurvey
    if not hasattr(RescaleToLegacySurvey, "_reverse_zeropoint"):
        RescaleToLegacySurvey._reverse_zeropoint = RescaleToLegacySurvey.reverse_zeropoint
except Exception:
    pass

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from aion import AION
from aion.codecs import CodecManager
from aion.modalities import LegacySurveyImage, HSCImage

HSC_BANDS, HSC_H, HSC_W = 5, 160, 160
LEG_BANDS, LEG_H, LEG_W = 4, 160, 160
HSC_BYTES = HSC_BANDS * HSC_H * HSC_W * 2
LEG_BYTES = LEG_BANDS * LEG_H * LEG_W * 2

NUM_TOKENS_SINGLE = 600
NUM_TOKENS_COMBINED = 1200

HSC_BANDS_NAMES = ["HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y"]
LEG_BANDS_NAMES = ["DES-G", "DES-R", "DES-I", "DES-Z"]

# Same as prepare_combined.py — keep in sync if that file ever changes.
SKIP_COLS = frozenset({
    "image_idx", "object_id", "healpix",
    "tract", "patch",
    "image_scale", "image_psf_fwhm",
    "sugohi_grade", "sugohi_name",
    "desi_targetid",
    "is_bgs_bright", "is_bgs_faint",
    "BRICKID", "BRICKNAME", "OBJID", "RELEASE", "gid",
    "TYPE", "MASKBITS",
    "euclid_grade", "euclid_id_str", "euclid_subset",
})


class _RawBinaryImageDataset(Dataset):
    """Like prepare_combined._BinaryImageDataset but WITHOUT preprocess_raw_image —
    AION's codec consumes raw flux."""

    def __init__(self, bin_path: Path, indices: np.ndarray, bands: int, h: int, w: int):
        self._bin = str(bin_path)
        self._indices = np.asarray(indices, dtype=np.int64)
        self._bytes = bands * h * w * 2
        self._shape = (bands, h, w)
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
        return torch.from_numpy(raw)


@torch.no_grad()
def _encode_single(model, codec_manager, ds, kind: str, batch_size: int, device):
    """Mean-pool AION encoder output for a single-modality dataset → (N, D)."""
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=(device.type == "cuda"))
    chunks = []
    n_total = len(ds)
    t0 = time.time()
    n_done = 0
    for imgs in loader:
        imgs = imgs.to(device, non_blocking=(device.type == "cuda"))
        if kind == "hsc":
            obj = HSCImage(flux=imgs, bands=HSC_BANDS_NAMES)
        else:
            obj = LegacySurveyImage(flux=imgs, bands=LEG_BANDS_NAMES)
        tokens = codec_manager.encode(obj)
        emb = model.encode(tokens, num_encoder_tokens=NUM_TOKENS_SINGLE)
        chunks.append(emb.mean(dim=1).float().cpu().numpy().astype(np.float32))
        n_done += imgs.size(0)
        if n_done % (batch_size * 20) == 0 or n_done >= n_total:
            dt = time.time() - t0
            print(f"    [{kind}] encoded {n_done:>6d}/{n_total} ({n_done/max(dt,1e-6):.1f} img/s)")
    return np.concatenate(chunks, axis=0)


@torch.no_grad()
def _encode_combined(model, codec_manager, hsc_ds, leg_ds, batch_size: int, device):
    """Mean-pool AION encoder output for paired (HSC, Legacy) → (N, D)."""
    assert len(hsc_ds) == len(leg_ds), "overlap subsets must have matching length"
    loader_h = DataLoader(hsc_ds, batch_size=batch_size, shuffle=False,
                          num_workers=0, pin_memory=(device.type == "cuda"))
    loader_l = DataLoader(leg_ds, batch_size=batch_size, shuffle=False,
                          num_workers=0, pin_memory=(device.type == "cuda"))
    chunks = []
    n_total = len(hsc_ds)
    t0 = time.time()
    n_done = 0
    for img_h, img_l in zip(loader_h, loader_l):
        img_h = img_h.to(device, non_blocking=(device.type == "cuda"))
        img_l = img_l.to(device, non_blocking=(device.type == "cuda"))
        obj_h = HSCImage(flux=img_h, bands=HSC_BANDS_NAMES)
        obj_l = LegacySurveyImage(flux=img_l, bands=LEG_BANDS_NAMES)
        tokens = codec_manager.encode(obj_h, obj_l)
        emb = model.encode(tokens, num_encoder_tokens=NUM_TOKENS_COMBINED)
        chunks.append(emb.mean(dim=1).float().cpu().numpy().astype(np.float32))
        n_done += img_h.size(0)
        if n_done % (batch_size * 20) == 0 or n_done >= n_total:
            dt = time.time() - t0
            print(f"    [combined] encoded {n_done:>6d}/{n_total} ({n_done/max(dt,1e-6):.1f} img/s)")
    return np.concatenate(chunks, axis=0)


def _expand_psf_list(df: pd.DataFrame, survey: str):
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
    df = pd.read_parquet(catalog_path)
    df = df.iloc[indices].reset_index(drop=True)
    df = _expand_psf_list(df, survey)
    out = {}
    for col in df.columns:
        if col in SKIP_COLS:
            continue
        s = df[col]
        if s.dtype.kind in "iufcb":
            arr = s.to_numpy(dtype=np.float32, na_value=np.nan)
            out[f"{prefix}{col}"] = arr
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--hsc-dir", type=Path, default=Path("/home/pablomer/orcd/scratch/hsc_downstream"))
    p.add_argument("--legacy-dir", type=Path, default=Path("/home/pablomer/orcd/scratch/legacy_downstream_full/full_1M"))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--smoke", action="store_true", help="Use only first 64 rows of each subset")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Loading polymathic-ai/aion-base ...")
    model = AION.from_pretrained("polymathic-ai/aion-base").to(device)
    model.eval()
    codec_manager = CodecManager(device=device)

    overlap_path = args.index_dir / "overlap.parquet"
    hsc_only_path = args.index_dir / "hsc_only.parquet"
    legacy_only_path = args.index_dir / "legacy_only.parquet"
    overlap_df = pd.read_parquet(overlap_path) if overlap_path.exists() else pd.DataFrame()
    hsc_only_df = pd.read_parquet(hsc_only_path) if hsc_only_path.exists() else pd.DataFrame()
    legacy_only_df = pd.read_parquet(legacy_only_path) if legacy_only_path.exists() else pd.DataFrame()

    if args.smoke:
        if len(overlap_df):     overlap_df    = overlap_df.head(64).reset_index(drop=True)
        if len(hsc_only_df):    hsc_only_df   = hsc_only_df.head(64).reset_index(drop=True)
        if len(legacy_only_df): legacy_only_df = legacy_only_df.head(64).reset_index(drop=True)

    print(f"Subsets: overlap={len(overlap_df)}  hsc_only={len(hsc_only_df)}  legacy_only={len(legacy_only_df)}")

    hsc_bin = args.hsc_dir / "hsc_flux.bin"
    leg_bin = args.legacy_dir / "legacy_flux.bin"
    hsc_cat = args.hsc_dir / "catalog.parquet"
    leg_cat = args.legacy_dir / "catalog.parquet"
    n_hsc_bin = os.path.getsize(hsc_bin) // HSC_BYTES
    n_leg_bin = os.path.getsize(leg_bin) // LEG_BYTES
    print(f"HSC bin rows: {n_hsc_bin:,}   Legacy bin rows: {n_leg_bin:,}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Write to /tmp first; H5 file locking is unreliable on the network FS.
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".h5", prefix="aion_base_", dir=tempfile.gettempdir())
    os.close(tmp_fd)
    try:
        with h5py.File(tmp_path, "w") as fout:
            fout.attrs["checkpoint_name"] = "aion-base"
            fout.attrs["checkpoint_path"] = "polymathic-ai/aion-base"
            fout.attrs["architecture"] = "aion"
            fout.attrs["has_e2"] = False
            fout.attrs["embedding_names"] = ["e1"]

            per_subset_n = {}
            per_subset_labels = {}

            # --- overlap ---
            if len(overlap_df):
                hsc_idx = overlap_df["hsc_image_idx"].to_numpy(dtype=np.int64)
                leg_idx = overlap_df["legacy_image_idx"].to_numpy(dtype=np.int64)
                assert (hsc_idx < n_hsc_bin).all(), "overlap.hsc_image_idx exceeds HSC bin"
                assert (leg_idx < n_leg_bin).all(), "overlap.legacy_image_idx exceeds Legacy bin"
                hsc_ds = _RawBinaryImageDataset(hsc_bin, hsc_idx, HSC_BANDS, HSC_H, HSC_W)
                leg_ds = _RawBinaryImageDataset(leg_bin, leg_idx, LEG_BANDS, LEG_H, LEG_W)

                print("\n[overlap] AION encoding HSC images")
                e_hsc = _encode_single(model, codec_manager, hsc_ds, "hsc", args.batch_size, device)
                print("[overlap] AION encoding Legacy images")
                e_leg = _encode_single(model, codec_manager, leg_ds, "legacy", args.batch_size, device)
                print("[overlap] AION encoding paired HSC+Legacy")
                e_combined = _encode_combined(model, codec_manager, hsc_ds, leg_ds, args.batch_size, device)

                grp = fout.create_group("overlap")
                grp.create_dataset("hsc_e1", data=e_hsc, compression="gzip", compression_opts=4)
                grp.create_dataset("legacy_e1", data=e_leg, compression="gzip", compression_opts=4)
                grp.create_dataset("combined_e1", data=e_combined, compression="gzip", compression_opts=4)

                print("[overlap] writing labels")
                hsc_labels = _extract_labels(hsc_cat, hsc_idx, "hsc", "hsc_")
                leg_labels = _extract_labels(leg_cat, leg_idx, "legacy", "legacy_")
                for k, v in {**hsc_labels, **leg_labels}.items():
                    grp.create_dataset(f"labels/{k}", data=v, compression="gzip", compression_opts=4)
                per_subset_n["overlap"] = len(overlap_df)
                per_subset_labels["overlap"] = list(hsc_labels.keys()) + list(leg_labels.keys())

            # --- hsc_only ---
            if len(hsc_only_df):
                hsc_idx = hsc_only_df["hsc_image_idx"].to_numpy(dtype=np.int64)
                assert (hsc_idx < n_hsc_bin).all(), "hsc_only.hsc_image_idx exceeds HSC bin"
                hsc_ds = _RawBinaryImageDataset(hsc_bin, hsc_idx, HSC_BANDS, HSC_H, HSC_W)
                print("\n[hsc_only] AION encoding HSC images")
                e_hsc = _encode_single(model, codec_manager, hsc_ds, "hsc", args.batch_size, device)
                grp = fout.create_group("hsc_only")
                grp.create_dataset("hsc_e1", data=e_hsc, compression="gzip", compression_opts=4)
                hsc_labels = _extract_labels(hsc_cat, hsc_idx, "hsc", "hsc_")
                for k, v in hsc_labels.items():
                    grp.create_dataset(f"labels/{k}", data=v, compression="gzip", compression_opts=4)
                per_subset_n["hsc_only"] = len(hsc_only_df)
                per_subset_labels["hsc_only"] = list(hsc_labels.keys())

            # --- legacy_only ---
            if len(legacy_only_df):
                leg_idx = legacy_only_df["legacy_image_idx"].to_numpy(dtype=np.int64)
                assert (leg_idx < n_leg_bin).all(), "legacy_only.legacy_image_idx exceeds Legacy bin"
                leg_ds = _RawBinaryImageDataset(leg_bin, leg_idx, LEG_BANDS, LEG_H, LEG_W)
                print("\n[legacy_only] AION encoding Legacy images")
                e_leg = _encode_single(model, codec_manager, leg_ds, "legacy", args.batch_size, device)
                grp = fout.create_group("legacy_only")
                grp.create_dataset("legacy_e1", data=e_leg, compression="gzip", compression_opts=4)
                leg_labels = _extract_labels(leg_cat, leg_idx, "legacy", "legacy_")
                for k, v in leg_labels.items():
                    grp.create_dataset(f"labels/{k}", data=v, compression="gzip", compression_opts=4)
                per_subset_n["legacy_only"] = len(legacy_only_df)
                per_subset_labels["legacy_only"] = list(leg_labels.keys())

            fout.attrs["num_examples_per_subset"] = json.dumps(per_subset_n)
            fout.attrs["label_columns_per_subset"] = json.dumps(per_subset_labels)
        shutil.move(tmp_path, args.out)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise

    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
