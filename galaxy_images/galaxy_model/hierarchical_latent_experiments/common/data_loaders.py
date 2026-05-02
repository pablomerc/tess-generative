"""HSC/Legacy pair loader for the hierarchical-latent experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

NEIGHBORS_HDF5_DEFAULT = "/work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5"


def make_pair_dataset(hdf5_path: str = NEIGHBORS_HDF5_DEFAULT, n: int | None = None,
                      shuffle: bool = False, seed: int = 42):
    """Returns a torch Subset of NeighborsSimpleDataset yielding (hsc, legacy, metadata)."""
    from galaxy_images.galaxy_model.neighbors import NeighborsSimpleDataset
    base = NeighborsSimpleDataset(hdf5_path=hdf5_path)
    total = len(base)
    if n is None:
        n_use = total
        indices = np.arange(total)
    else:
        n_use = min(int(n), total)
        if shuffle:
            rng = np.random.default_rng(seed)
            indices = rng.permutation(total)[:n_use]
        else:
            indices = np.arange(n_use)
    return Subset(base, indices.tolist())


def collate_pair(batch):
    hsc = torch.stack([b[0] for b in batch])
    leg = torch.stack([b[1] for b in batch])
    raw_indices = np.array([b[2]["index_mmu"] for b in batch], dtype=np.int64)
    metadata = [b[2] for b in batch]
    return hsc, leg, raw_indices, metadata


def make_loader(dataset, batch_size: int = 128, num_workers: int = 2, pin_memory: bool = True):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_pair, pin_memory=pin_memory,
    )
