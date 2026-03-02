from __future__ import annotations

from typing import Callable, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from galaxy_images.galaxy_model.config import ExperimentConfig
from galaxy_images.galaxy_model.neighbors import (
    NeighborsDataset,
    NeighborsPrecomputedDataset,
    collate_neighbors,
    simple_collate,
)


def _collate_for_neighbors(batch):
    # NeighborsDataset already returns variable-length neighbors; this collate pads and returns masks.
    return collate_neighbors(batch)


def _collate_for_precomputed(batch):
    # Precomputed samples are already padded and include masks.
    return simple_collate(batch)


def build_neighbors_dataloaders(config: ExperimentConfig, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    dataset: Dataset
    collate_fn: Callable

    if config.data.mode == "precomputed":
        dataset = NeighborsPrecomputedDataset(config.data.precomputed_h5)
        collate_fn = _collate_for_precomputed
    elif config.data.mode == "neighbors":
        dataset = NeighborsDataset(
            hdf5_path=config.data.neighbors_h5,
            max_neighbors=config.data.max_neighbors,
            crop_size=config.model.image_size,
        )
        collate_fn = _collate_for_neighbors
    else:
        raise ValueError(f"Unsupported data mode: {config.data.mode}")

    total_size = len(dataset)
    val_size = int(total_size * config.data.val_ratio)
    train_size = total_size - val_size
    generator = torch.Generator().manual_seed(config.trainer.seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        collate_fn=collate_fn,
        persistent_workers=config.data.num_workers > 0,
        pin_memory=config.data.pin_memory,
        drop_last=config.data.drop_last_train,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=collate_fn,
        persistent_workers=config.data.num_workers > 0,
        pin_memory=config.data.pin_memory,
    )
    return train_loader, val_loader
