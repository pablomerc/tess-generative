"""
Datasets for the JEPA ablation (see PLAN.md).

All read the compact 48x48 f16 store built by contrastive_baseline/build_ram48.py
(/orcd/pool/007/pablomer/efficient_neighs_48). Preprocessing is baked in at build
time, so __getitem__ only indexes + casts.

Two new dataset classes on top of the existing NeighborsRAM48Dataset:

  * HSC48PoolDataset   — the 366,706 rows with a valid HSC stamp
                         (source_type in {0,1} == the paper anomaly pool).
                         HSC side only; used by the jepa-hsc-* arms.
  * AnchorPairs48Dataset — the 103,741 anchors (source_type == 0), returning the
                         (hsc, legacy) pair of the same galaxy; used by jepa-x.

The jepa-cf arm reuses NeighborsRAM48Dataset + collate_neighbors unchanged.

NOTE: loading uses np.fromfile (a true copy into RAM) — np.array/ascontiguousarray
on an already-contiguous memmap is a NO-OP that leaves data on disk (Ceph-bound).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

DEFAULT_DATA_DIR = "/orcd/pool/007/pablomer/efficient_neighs_48"


def _load_meta(data_dir: Path, crop_size: int):
    with open(data_dir / "meta.json") as f:
        meta = json.load(f)
    n = int(meta["n_records"])
    return n, (n, 4, crop_size, crop_size)


def _load_bin(path: Path, shp, load_into_ram: bool):
    if load_into_ram:
        return np.fromfile(path, dtype="float16").reshape(shp)
    return np.memmap(path, dtype="float16", mode="r", shape=shp)


class HSC48PoolDataset(Dataset):
    """HSC stamps of the 367k paper pool (source_type in {0,1}). One image per item."""

    def __init__(
        self,
        data_dir: str | Path = DEFAULT_DATA_DIR,
        crop_size: int = 48,
        load_into_ram: bool = True,
    ):
        data_dir = Path(data_dir)
        _, shp = _load_meta(data_dir, crop_size)
        self.hsc = _load_bin(data_dir / "hsc48.bin", shp, load_into_ram)

        cat = pd.read_parquet(data_dir / "catalog.parquet", columns=["source_type"])
        self.rows = cat.index[cat["source_type"].isin([0, 1])].to_numpy().astype(np.int64)
        print(
            f"[HSC48PoolDataset] {len(self.rows):,} HSC rows (source_type 0+1) | "
            f"in_ram={load_into_ram}",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> torch.Tensor:
        row = int(self.rows[idx])
        return torch.from_numpy(self.hsc[row].astype(np.float32))


class AnchorPairs48Dataset(Dataset):
    """(hsc, legacy) stamps of the same galaxy, for the 103,741 anchors."""

    def __init__(
        self,
        data_dir: str | Path = DEFAULT_DATA_DIR,
        crop_size: int = 48,
        load_into_ram: bool = True,
    ):
        data_dir = Path(data_dir)
        _, shp = _load_meta(data_dir, crop_size)
        self.hsc = _load_bin(data_dir / "hsc48.bin", shp, load_into_ram)
        self.leg = _load_bin(data_dir / "legacy48.bin", shp, load_into_ram)

        cat = pd.read_parquet(data_dir / "catalog.parquet", columns=["source_type"])
        self.rows = cat.index[cat["source_type"] == 0].to_numpy().astype(np.int64)
        print(
            f"[AnchorPairs48Dataset] {len(self.rows):,} anchor pairs | "
            f"in_ram={load_into_ram}",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = int(self.rows[idx])
        hsc = torch.from_numpy(self.hsc[row].astype(np.float32))
        leg = torch.from_numpy(self.leg[row].astype(np.float32))
        return hsc, leg


class AllStamps48Dataset(Dataset):
    """Every valid stamp from BOTH surveys as independent single images:
    HSC side of source_type {0,1} (366,706) + Legacy side of source_type {0,2}
    (205,232) = 571,938 items. Used by the multi-vit arm (round 3, PLAN.md A1).

    No survey label is returned — conditioning lives in predictors only, and a
    shared encoder can infer the domain from the image itself."""

    def __init__(
        self,
        data_dir: str | Path = DEFAULT_DATA_DIR,
        crop_size: int = 48,
        load_into_ram: bool = True,
    ):
        data_dir = Path(data_dir)
        _, shp = _load_meta(data_dir, crop_size)
        self.hsc = _load_bin(data_dir / "hsc48.bin", shp, load_into_ram)
        self.leg = _load_bin(data_dir / "legacy48.bin", shp, load_into_ram)

        cat = pd.read_parquet(data_dir / "catalog.parquet", columns=["source_type"])
        self.rows_hsc = cat.index[cat["source_type"].isin([0, 1])].to_numpy().astype(np.int64)
        self.rows_leg = cat.index[cat["source_type"].isin([0, 2])].to_numpy().astype(np.int64)
        print(
            f"[AllStamps48Dataset] {len(self.rows_hsc):,} HSC + {len(self.rows_leg):,} Legacy "
            f"= {len(self):,} stamps | in_ram={load_into_ram}",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.rows_hsc) + len(self.rows_leg)

    def __getitem__(self, idx: int) -> torch.Tensor:
        n_h = len(self.rows_hsc)
        if idx < n_h:
            return torch.from_numpy(self.hsc[int(self.rows_hsc[idx])].astype(np.float32))
        return torch.from_numpy(self.leg[int(self.rows_leg[idx - n_h])].astype(np.float32))


def load_umap_pairs(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    n: int = 512,
    seed: int = 42,
    crop_size: int = 48,
):
    """Fixed (hsc, legacy) tensors of n random anchors, for survey-colored UMAP panels.

    Uses memmap (not fromfile): only n random rows are touched, one time — no need
    to pull the full 17 GB into RAM just for a validation figure.
    """
    data_dir = Path(data_dir)
    _, shp = _load_meta(data_dir, crop_size)
    hsc = np.memmap(data_dir / "hsc48.bin", dtype="float16", mode="r", shape=shp)
    leg = np.memmap(data_dir / "legacy48.bin", dtype="float16", mode="r", shape=shp)

    cat = pd.read_parquet(data_dir / "catalog.parquet", columns=["source_type"])
    anchors = cat.index[cat["source_type"] == 0].to_numpy().astype(np.int64)
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(anchors, size=min(n, len(anchors)), replace=False))

    hsc_t = torch.from_numpy(hsc[rows].astype(np.float32))
    leg_t = torch.from_numpy(leg[rows].astype(np.float32))
    return hsc_t, leg_t
