from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from galaxy_images.galaxy_model.config import ExperimentConfig
from galaxy_images.galaxy_model.neighbors import (
    NeighborsDataset,
    NeighborsPrecomputedDataset,
    collate_neighbors,
    simple_collate,
)


def _resolve_lens_anchor_positions(
    lens_h5: str | Path,
    lens_indices_zero_based: List[int],
    efficient_data_dir: str | Path,
) -> List[int]:
    """Map lens rows -> their position within NeighborsEfficientDataset.anchor_indices.

    Uses object_id_hsc as the join key because the efficient catalog row order
    differs from neighbours_v2.h5's row order (so h5_index from lens_h5 is not
    a position in the efficient dataset).
    """
    import pandas as pd

    lens_h5 = Path(lens_h5)
    catalog_path = Path(efficient_data_dir) / "catalog.parquet"
    if not lens_h5.exists():
        raise FileNotFoundError(f"lens_h5 not found: {lens_h5}")
    if not catalog_path.exists():
        raise FileNotFoundError(f"catalog.parquet not found: {catalog_path}")

    with h5py.File(lens_h5, "r") as lf:
        all_obj_ids = lf["object_id_hsc"][:]
    selected_obj_ids = [all_obj_ids[i].decode("utf-8") for i in lens_indices_zero_based]

    catalog = pd.read_parquet(catalog_path, columns=["object_id_hsc", "source_type"])
    catalog_obj = catalog["object_id_hsc"].astype(str)
    anchor_indices = catalog.index[catalog["source_type"] == 0].to_numpy()
    anchor_pos_lookup = {int(row): int(pos) for pos, row in enumerate(anchor_indices)}

    positions: List[int] = []
    missing: List[str] = []
    for oid in selected_obj_ids:
        rows = catalog.index[catalog_obj == oid].tolist()
        if not rows:
            missing.append(oid)
            continue
        row = int(rows[0])
        if row not in anchor_pos_lookup:
            missing.append(oid)
            continue
        positions.append(anchor_pos_lookup[row])

    if missing:
        raise ValueError(
            f"Could not find {len(missing)} lens object_ids in efficient catalog "
            f"as source_type==0 anchors: {missing}"
        )
    return positions


def _normalize_legacy_id(raw: str) -> str:
    """Strip the legacy `b'...'` byte-repr wrapper that the parquet stores as a string."""
    s = str(raw)
    if len(s) >= 3 and s.startswith("b'") and s.endswith("'"):
        return s[2:-1]
    if len(s) >= 3 and s.startswith('b"') and s.endswith('"'):
        return s[2:-1]
    return s


def _resolve_downstream_holdout_positions(
    holdout_txt: str | Path,
    efficient_data_dir: str | Path,
) -> List[int]:
    """Return anchor positions whose object_id_legacy is in the downstream-holdout list."""
    import pandas as pd

    holdout_txt = Path(holdout_txt)
    if not holdout_txt.exists():
        raise FileNotFoundError(f"downstream holdout file not found: {holdout_txt}")

    with open(holdout_txt, "r") as f:
        holdout_ids = {line.strip() for line in f if line.strip()}

    catalog_path = Path(efficient_data_dir) / "catalog.parquet"
    catalog = pd.read_parquet(catalog_path, columns=["object_id_legacy", "source_type"])
    legacy_norm = catalog["object_id_legacy"].astype(str).map(_normalize_legacy_id)

    in_holdout = legacy_norm.isin(holdout_ids)
    is_anchor = catalog["source_type"] == 0
    matched_mask = in_holdout & is_anchor
    matched_rows = catalog.index[matched_mask].to_numpy()

    anchor_indices = catalog.index[is_anchor].to_numpy()
    anchor_pos_lookup = {int(row): int(pos) for pos, row in enumerate(anchor_indices)}
    positions = sorted(anchor_pos_lookup[int(r)] for r in matched_rows)

    print(
        f"[downstream-holdout] holdout list: {len(holdout_ids):,} unique legacy IDs | "
        f"matched anchors: {len(positions):,} / {len(anchor_indices):,}"
    )
    if not positions:
        print(
            "[downstream-holdout] WARNING: 0 anchor matches — check object_id_legacy "
            "encoding in the catalog vs the holdout file."
        )
    return positions


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


def build_neighbors_dataloaders(
    config: ExperimentConfig, batch_size: int
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
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
    elif config.data.mode == "efficient":
        from galaxy_images.galaxy_model.neighbors_efficient import NeighborsEfficientDataset
        dataset = NeighborsEfficientDataset(
            data_dir=config.data.efficient_data_dir,
            max_neighbors=config.data.max_neighbors,
            crop_size=config.model.image_size,
            random_neighbors=config.data.random_neighbors,
        )
        collate_fn = _collate_for_neighbors
    elif config.data.mode == "ram48":
        from galaxy_images.galaxy_model.contrastive_baseline.neighbors_ram48 import (
            NeighborsRAM48Dataset,
        )
        dataset = NeighborsRAM48Dataset(
            data_dir=config.data.efficient_data_dir,
            max_neighbors=config.data.max_neighbors,
            crop_size=config.model.image_size,
            random_neighbors=config.data.random_neighbors,
        )
        collate_fn = _collate_for_neighbors
    else:
        raise ValueError(f"Unsupported data mode: {config.data.mode}")

    total_size = len(dataset)

    # Resolve lens anchor positions (always built so we can also create the
    # lens validation loader, even if exclude_from_train is False).
    lens_positions: List[int] = []
    if config.lens_val.enabled and config.data.mode == "efficient":
        lens_positions = _resolve_lens_anchor_positions(
            lens_h5=config.lens_val.lens_h5,
            lens_indices_zero_based=config.lens_val.lens_indices_zero_based,
            efficient_data_dir=config.data.efficient_data_dir,
        )
        print(
            f"[lens-val] identified {len(lens_positions)} lens anchors at positions: "
            f"{lens_positions}"
        )

    # Resolve downstream-holdout anchor positions to exclude from train/val.
    holdout_positions: List[int] = []
    if config.data.downstream_holdout_ids_txt and config.data.mode == "efficient":
        holdout_positions = _resolve_downstream_holdout_positions(
            holdout_txt=config.data.downstream_holdout_ids_txt,
            efficient_data_dir=config.data.efficient_data_dir,
        )

    excluded_set: set[int] = set()
    if config.lens_val.enabled and config.lens_val.exclude_from_train:
        excluded_set.update(lens_positions)
    excluded_set.update(holdout_positions)

    print(
        f"[data] anchor totals — total: {total_size:,} | lens-excluded: "
        f"{len(set(lens_positions) & excluded_set):,} | downstream-holdout-excluded: "
        f"{len(set(holdout_positions) & excluded_set):,} | union excluded: "
        f"{len(excluded_set):,}"
    )

    if excluded_set:
        kept_positions = [i for i in range(total_size) if i not in excluded_set]
        kept_dataset: Dataset = Subset(dataset, kept_positions)
    else:
        kept_dataset = dataset

    kept_size = len(kept_dataset)
    val_size = int(kept_size * config.data.val_ratio)
    train_size = kept_size - val_size
    print(
        f"[data] after exclusions: kept={kept_size:,} | "
        f"val_ratio={config.data.val_ratio} -> train={train_size:,}, val={val_size:,}"
    )
    generator = torch.Generator().manual_seed(config.trainer.seed)
    train_ds, val_ds = random_split(kept_dataset, [train_size, val_size], generator=generator)
    if hasattr(val_ds, "indices"):
        _save_heldout_validation_subset(kept_dataset, list(val_ds.indices), config, batch_size)

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

    lens_loader: Optional[DataLoader] = None
    if config.lens_val.enabled and config.data.mode == "efficient" and lens_positions:
        lens_subset = Subset(dataset, lens_positions)
        lens_loader = DataLoader(
            lens_subset,
            batch_size=min(batch_size, len(lens_positions)),
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
            pin_memory=False,
        )

    return train_loader, val_loader, lens_loader
