"""
Baseline (no model): same neighbor dataset as prepare_all.py, but instead of training
we compute simple features from the images: mean and std per channel (first 4 channels;
drop 5th on HSC). Saves features (B, 8) for HSC, (B, 8) for Legacy, (B, 16) for
Legacy+HSC concatenated, plus labels, to an .h5 file compatible with downstream eval.

Run from galaxy_model/ or downstream_evaluation/final/:
  python downstream_evaluation/final/baseline_all.py
"""
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
_src = _here.parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import argparse
import json
import h5py
import numpy as np
# Reuse neighbor config and label building from prepare_all
from prepare_all import (
    NEIGHBORS_HDF5,
    NUM_EXAMPLES_NEIGHBORS,
    NEIGHBORS_SEED,
    SHUFFLE_NEIGHBORS,
    _labels_from_metadata_list,
)

# Same exclude keys as NeighborsSimpleDataset (neighbors.py)
NEIGHBORS_SIMPLE_EXCLUDE_KEYS = frozenset({
    "source_type",
    "images_hsc",
    "images_legacy",
    "neighbor_idx_hsc",
    "neighbor_idx_legacy",
    "neighbor_dist_hsc",
    "neighbor_dist_legacy",
})


def _metadata_value_from_h5(val):
    """Convert an HDF5/dataset slice to a Python type for metadata (scalars and small arrays)."""
    if hasattr(val, "shape") and val.shape == ():
        return val.item()
    if np.isscalar(val):
        if isinstance(val, (np.floating, np.float32, np.float64)):
            return float(val)
        if isinstance(val, (np.integer, np.int64, np.int32)):
            return int(val)
        if isinstance(val, np.bool_):
            return bool(val)
    if hasattr(val, "tolist"):
        return val.tolist()
    return val


def channel_mean_std(img, n_channels=4):
    """
    img: (C, H, W) or (C, H, W) with C >= n_channels.
    Use first n_channels, compute mean and std over spatial dims.
    Returns (n_channels * 2,) = [mean_0, std_0, mean_1, std_1, ...].
    """
    img = np.asarray(img, dtype=np.float64)
    if img.ndim != 3:
        raise ValueError(f"Expected (C,H,W), got shape {img.shape}")
    img = img[:n_channels]  # (n_channels, H, W)
    # axis=(-2,-1) = spatial
    mu = np.mean(img, axis=(-2, -1))   # (n_channels,)
    sd = np.std(img, axis=(-2, -1))    # (n_channels,)
    return np.stack([mu, sd], axis=1).flatten().astype(np.float32)  # (n_channels*2,)


def prepare_baseline_neighbors(output_dir):
    """
    Same example set and order as prepare_all's neighbors: indexes_mmu, subset to
    NUM_EXAMPLES_NEIGHBORS, shuffle with NEIGHBORS_SEED if SHUFFLE_NEIGHBORS.
    For each example load raw images from H5, take first 4 channels for HSC (drop 5th),
    compute per-channel mean and std -> (8,) per image. Save features + labels to H5.
    """
    with h5py.File(NEIGHBORS_HDF5, "r") as f:
        sources = f["source_type"][:]
        indexes_mmu = np.where(sources == 0)[0]
        meta_keys = [k for k in f.keys() if k not in NEIGHBORS_SIMPLE_EXCLUDE_KEYS]
        n_total = len(indexes_mmu)
        n_use = min(NUM_EXAMPLES_NEIGHBORS, n_total)

    # Same ordering as prepare_all: Subset(dataset, range(n_use)) then DataLoader(shuffle=SHUFFLE_NEIGHBORS, generator=seed)
    rng = np.random.default_rng(NEIGHBORS_SEED)
    indices = np.arange(n_use)
    if SHUFFLE_NEIGHBORS:
        indices = rng.permutation(indices)

    # Pre-allocate feature arrays: (n_use, 8) and (n_use, 16)
    features_hsc = np.zeros((n_use, 8), dtype=np.float32)
    features_legacy = np.zeros((n_use, 8), dtype=np.float32)
    metadata_list = []

    with h5py.File(NEIGHBORS_HDF5, "r") as f:
        for k in range(n_use):
            dataset_idx = int(indices[k])
            index_mmu = int(indexes_mmu[dataset_idx])
            # Raw images: HSC may be (5, H, W), Legacy (4, H, W)
            img_hsc = f["images_hsc"][index_mmu]
            img_legacy = f["images_legacy"][index_mmu]
            # First 4 channels only; mean/std over spatial
            features_hsc[k] = channel_mean_std(img_hsc, n_channels=4)
            features_legacy[k] = channel_mean_std(img_legacy, n_channels=4)
            # Metadata in same format as NeighborsSimpleDataset
            meta = {"idx": dataset_idx, "index_mmu": index_mmu}
            for key in meta_keys:
                val = f[key][index_mmu]
                meta[key] = _metadata_value_from_h5(val)
            metadata_list.append(meta)

    features_hsc_legacy = np.concatenate([features_hsc, features_legacy], axis=1)  # (n_use, 16)
    labels_n, label_columns = _labels_from_metadata_list(metadata_list)

    out_path = Path(output_dir) / "downstream_neighbors_baseline.h5"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("features/hsc", data=features_hsc, compression="gzip", compression_opts=4)
        f.create_dataset("features/legacy", data=features_legacy, compression="gzip", compression_opts=4)
        f.create_dataset("features/hsc_legacy", data=features_hsc_legacy, compression="gzip", compression_opts=4)
        for col in label_columns:
            f.create_dataset(f"labels/{col}", data=labels_n[col], compression="gzip", compression_opts=4)
        f.attrs["num_examples"] = n_use
        f.attrs["label_columns"] = list(label_columns)
        # HDF5 attrs can't store dicts; use JSON string (tuples become lists when read back)
        f.attrs["feature_shapes"] = json.dumps({"hsc": [8], "legacy": [8], "hsc_legacy": [16]})

    print(f"Saved: {out_path}")
    print(f"  features/hsc: {features_hsc.shape}, features/legacy: {features_legacy.shape}, features/hsc_legacy: {features_hsc_legacy.shape}")
    print(f"  labels: {len(label_columns)} columns")
    return out_path


def main():
    p = argparse.ArgumentParser(description="Build baseline (mean/std) features from neighbors dataset, same as prepare_all.")
    p.add_argument("--output-dir", type=Path, default=_here, help="Output directory for H5 file")
    args = p.parse_args()
    prepare_baseline_neighbors(args.output_dir)


if __name__ == "__main__":
    main()
