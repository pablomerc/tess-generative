from __future__ import annotations

from pathlib import Path
from typing import Callable, Tuple

import h5py
import numpy as np
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


def _save_heldout_validation_subset(
    dataset: Dataset,
    val_indices: list[int],
    config: ExperimentConfig,
    batch_size: int,
) -> None:
    if not config.data.save_heldout_validation or not config.data.heldout_validation_dir:
        return

    num_samples = min(len(val_indices), config.data.heldout_num_batches * batch_size)
    if num_samples <= 0:
        return

    out_dir = Path(config.data.heldout_validation_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    file_name = config.data.heldout_file_name or (
        f"heldout_val_seed{config.trainer.seed}_b{config.data.heldout_num_batches}_bs{batch_size}.h5"
    )
    out_path = out_dir / file_name
    if out_path.exists():
        print(f"[heldout-val] existing file found, skipping export: {out_path}")
        return

    selected_indices = list(val_indices[:num_samples])
    print(f"[heldout-val] saving {len(selected_indices)} validation samples to {out_path} ...")

    targets = []
    samegals = []
    sameins = []
    masks = []
    meta_idx = []
    meta_survey = []
    meta_num_same = []

    for idx in selected_indices:
        target, samegal, samein, mask, metadata = dataset[idx]
        targets.append(target.detach().cpu())
        samegals.append(samegal.detach().cpu())
        sameins.append(samein.detach().cpu())
        masks.append(mask.detach().cpu())
        meta_idx.append(metadata.get("idx", -1))
        meta_survey.append(str(metadata.get("anchor_survey", "unknown")).encode("utf-8"))
        meta_num_same.append(metadata.get("num_same_instrument", 0))

    with h5py.File(out_path, "w") as f:
        f.create_dataset("targets", data=torch.stack(targets).numpy(), compression="lzf")
        f.create_dataset("samegals", data=torch.stack(samegals).numpy(), compression="lzf")
        f.create_dataset("sameins", data=torch.stack(sameins).numpy(), compression="lzf")
        f.create_dataset("neighbor_masks", data=torch.stack(masks).numpy(), compression="lzf")
        f.create_dataset("meta_idx", data=np.asarray(meta_idx))
        f.create_dataset("meta_survey", data=np.asarray(meta_survey, dtype="S"))
        f.create_dataset("meta_num_same_instrument", data=np.asarray(meta_num_same))
        f.attrs["seed"] = config.trainer.seed
        f.attrs["batch_size"] = batch_size
        f.attrs["num_batches"] = config.data.heldout_num_batches
        f.attrs["num_samples"] = len(selected_indices)
        f.attrs["source_mode"] = config.data.mode

    print(f"[heldout-val] saved {out_path}")


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
    if hasattr(val_ds, "indices"):
        _save_heldout_validation_subset(dataset, list(val_ds.indices), config, batch_size)

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
