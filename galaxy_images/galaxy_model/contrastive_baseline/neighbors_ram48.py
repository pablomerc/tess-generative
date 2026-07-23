"""
In-RAM 48x48 neighbors dataset — the fast Engaging data path.

Reads the compact store built by build_ram48.py (hsc48.bin / legacy48.bin, each
float16 (N,4,48,48), ~8GB apiece) fully into RAM, so training is not bound by the
Ceph random-read latency that made the 160x160 mmap store ~0.35 batch/s.

Output contract is IDENTICAL to NeighborsEfficientDataset: __getitem__ returns
(target, samegal, sameins, metadata) — use collate_neighbors. Preserves the
`random_neighbors` flag (random rows drawn from the full pool) so the ablation
variant is unchanged. Preprocessing was already baked in at build time, so
__getitem__ only indexes + casts + stacks.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class NeighborsRAM48Dataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        crop_size: int = 48,          # accepted for API-compat; store is already 48
        max_neighbors: int = 5,
        random_neighbors: bool = False,
        load_into_ram: bool = True,
    ):
        data_dir = Path(data_dir)
        with open(data_dir / "meta.json") as f:
            meta = json.load(f)
        n = int(meta["n_records"])
        shp = (n, 4, crop_size, crop_size)

        if load_into_ram:
            # Sequential read of ~16GB off Ceph into RAM (~1 min), then all random
            # access during training is memory-speed. Forked workers share via COW.
            # NOTE: use np.fromfile (a true copy) — np.array/ascontiguousarray on an
            # already-contiguous memmap is a NO-OP and leaves data on disk.
            self.hsc = np.fromfile(data_dir / "hsc48.bin", dtype="float16").reshape(shp)
            self.leg = np.fromfile(data_dir / "legacy48.bin", dtype="float16").reshape(shp)
        else:
            self.hsc = np.memmap(data_dir / "hsc48.bin", dtype="float16", mode="r", shape=shp)
            self.leg = np.memmap(data_dir / "legacy48.bin", dtype="float16", mode="r", shape=shp)

        cat = pd.read_parquet(data_dir / "catalog.parquet")
        self.anchor_indices = cat.index[cat["source_type"] == 0].to_numpy().astype(np.int64)
        self.neighbor_idx_hsc = np.stack(cat["neighbor_idx_hsc"].to_numpy()).astype(np.int64)
        self.neighbor_idx_legacy = np.stack(cat["neighbor_idx_legacy"].to_numpy()).astype(np.int64)

        self.max_neighbors = max_neighbors
        self.random_neighbors = bool(random_neighbors)
        self._n_total_rows = n
        print(
            f"[NeighborsRAM48Dataset] {len(self.anchor_indices):,} anchors / {n:,} rows | "
            f"in_ram={load_into_ram} | max_neighbors={max_neighbors} | "
            f"random_neighbors={self.random_neighbors}",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.anchor_indices)

    def __getitem__(self, idx: int):
        row = int(self.anchor_indices[idx])
        anchor_is_hsc = (idx % 2 == 0)

        if anchor_is_hsc:
            target = torch.from_numpy(self.hsc[row].astype(np.float32))
            samegal = torch.from_numpy(self.leg[row].astype(np.float32))
            anchor_survey = "hsc"
            neighbor_row_ids = self.neighbor_idx_hsc[row]
            pool = self.hsc
        else:
            target = torch.from_numpy(self.leg[row].astype(np.float32))
            samegal = torch.from_numpy(self.hsc[row].astype(np.float32))
            anchor_survey = "legacy"
            neighbor_row_ids = self.neighbor_idx_legacy[row]
            pool = self.leg

        if self.random_neighbors:
            valid_ids = np.random.randint(0, self._n_total_rows,
                                          size=self.max_neighbors, dtype=np.int64)
        else:
            valid_ids = neighbor_row_ids[neighbor_row_ids >= 0][: self.max_neighbors]

        sameins_list = [torch.from_numpy(pool[int(nid)].astype(np.float32)) for nid in valid_ids]
        if sameins_list:
            sameins = torch.stack(sameins_list, dim=0)          # (N, 4, 48, 48)
        else:
            sameins = torch.empty(0, 4, self.hsc.shape[2], self.hsc.shape[3])

        metadata = {"anchor_survey": anchor_survey, "idx": row,
                    "num_same_instrument": len(sameins_list)}
        return target, samegal, sameins, metadata
