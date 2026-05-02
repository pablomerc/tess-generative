"""
Efficient neighbors dataset backed by memory-mapped binary files + Parquet catalog.

Drop-in replacement for NeighborsDataset: returns identical
(target, samegal, sameins, metadata) tuples and is compatible with collate_neighbors.

Data directory layout (see neighbors_efficient/README.md):
  hsc_image.bin     float16  (N, 5, 160, 160)  bands: g,r,i,z,y
  legacy_image.bin  float16  (N, 4, 160, 160)  bands: g,r,i,z
  catalog.parquet   one row per galaxy, same row order as binaries
  meta.json         shapes / dtypes (optional; shapes inferred from catalog if absent)

Only rows with source_type == 0 are sampled as anchors. The remaining rows exist
solely so neighbor indices (which span all 468k rows) can be resolved.

HSC images: first 4 bands (g,r,i,z) are kept — y band is dropped.
Legacy images: all 4 bands are kept and zoomed to match HSC pixel scale.

Both surveys therefore produce 4-channel tensors, matching the existing model configs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from galaxy_images.image_preprocessing import preprocess_image_v2
from galaxy_images.galaxy_model.data import zoom_legacy_image
from galaxy_images.galaxy_model.neighbors import NORM_DICT


class NeighborsEfficientDataset(Dataset):
    """
    Triplet dataset (target, same-galaxy, same-instrument neighbors) loaded from
    memory-mapped binary files for fast random access.

    Output contract matches NeighborsDataset exactly — use collate_neighbors.
    """

    def __init__(
        self,
        data_dir: str | Path,
        crop_size: int = 48,
        max_neighbors: int = 15,
        norm_dict: dict = NORM_DICT,
    ):
        data_dir = Path(data_dir)

        hsc_bin = data_dir / "hsc_image.bin"
        legacy_bin = data_dir / "legacy_image.bin"
        catalog_path = data_dir / "catalog.parquet"
        meta_path = data_dir / "meta.json"

        for p in (hsc_bin, legacy_bin, catalog_path):
            if not p.exists():
                raise FileNotFoundError(f"Expected file not found: {p}")

        catalog = pd.read_parquet(catalog_path)
        n_rows = len(catalog)

        # Derive shapes from meta.json if available; otherwise use canonical defaults.
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            hsc_shape = tuple(meta["hsc_image"]["shape"])
            legacy_shape = tuple(meta["legacy_image"]["shape"])
        else:
            hsc_shape = (n_rows, 5, 160, 160)
            legacy_shape = (n_rows, 4, 160, 160)

        self.hsc_mmap = np.memmap(hsc_bin, dtype="float16", mode="r", shape=hsc_shape)
        self.legacy_mmap = np.memmap(legacy_bin, dtype="float16", mode="r", shape=legacy_shape)

        # Anchor rows: only source_type == 0 are sampled.
        self.anchor_indices = catalog.index[catalog["source_type"] == 0].to_numpy().astype(np.int32)

        # Pre-load neighbor index arrays (N_catalog, 20) for O(1) per-row lookup.
        self.neighbor_idx_hsc = np.stack(catalog["neighbor_idx_hsc"].to_numpy()).astype(np.int32)
        self.neighbor_idx_legacy = np.stack(catalog["neighbor_idx_legacy"].to_numpy()).astype(np.int32)

        self.crop_size = crop_size
        self.max_neighbors = max_neighbors
        self.norm_dict = norm_dict

        print(
            f"[NeighborsEfficientDataset] {len(self.anchor_indices):,} anchors "
            f"out of {n_rows:,} total rows | crop_size={crop_size} | max_neighbors={max_neighbors}"
        )

    def __len__(self) -> int:
        return len(self.anchor_indices)

    def _preprocess(self, row_idx: int, survey: str) -> torch.Tensor:
        """Load one image from mmap, apply preprocessing, return (C, H, W) float32 tensor."""
        if survey == "hsc":
            raw = self.hsc_mmap[row_idx].astype(np.float32)   # (5, 160, 160)
        else:
            raw = self.legacy_mmap[row_idx].astype(np.float32) # (4, 160, 160)

        img = torch.from_numpy(raw)
        img = preprocess_image_v2(img, crop_size=self.crop_size, survey=survey)

        if survey == "legacy":
            img = zoom_legacy_image(img)
            mean, std = self.norm_dict["legacy_zoom"]
        else:
            mean, std = self.norm_dict["hsc"]

        img = (img - mean) / std
        return img  # (5, H, W) for HSC or (4, H, W) for Legacy

    def __getitem__(self, idx: int):
        row_idx = int(self.anchor_indices[idx])
        anchor_is_hsc = (idx % 2 == 0)

        hsc_img = self._preprocess(row_idx, "hsc")[:4]  # drop y band → (4, H, W)
        legacy_img = self._preprocess(row_idx, "legacy")  # (4, H, W)

        if anchor_is_hsc:
            target, samegal = hsc_img, legacy_img
            anchor_survey = "hsc"
            neighbor_row_ids = self.neighbor_idx_hsc[row_idx]
            sameins_survey = "hsc"
        else:
            target, samegal = legacy_img, hsc_img
            anchor_survey = "legacy"
            neighbor_row_ids = self.neighbor_idx_legacy[row_idx]
            sameins_survey = "legacy"

        valid_ids = neighbor_row_ids[neighbor_row_ids >= 0][: self.max_neighbors]

        sameins_list: list[torch.Tensor] = []
        for nid in valid_ids:
            nimg = self._preprocess(int(nid), sameins_survey)
            if sameins_survey == "hsc":
                nimg = nimg[:4]  # drop y band
            sameins_list.append(nimg)

        if sameins_list:
            sameins = torch.stack(sameins_list, dim=0)  # (N, 4, H, W)
        else:
            sameins = torch.empty(0, 4, self.crop_size, self.crop_size)

        metadata = {
            "anchor_survey": anchor_survey,
            "idx": row_idx,
            "num_same_instrument": len(sameins_list),
        }

        return target, samegal, sameins, metadata
