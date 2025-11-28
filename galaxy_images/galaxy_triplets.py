"""
Triplet creation utilities for real galaxy cutouts.

This module mirrors the behaviour of ``double-encoder-model/triplet_creation.py``
but sources the data from parquet catalogs (e.g. Legacy Survey x HSC matches).
Each row in the parquet files contains two versions of the same galaxy observed
by different instruments.  We expose helpers that:

* load/stream multiple parquet shards through HuggingFace ``datasets``
* return tensors ready for training a deep metric-learning or reconstruction model
* provide both single-triplet and batch-triplet APIs
"""

from __future__ import annotations

import glob
import os
import random
from typing import List, Optional, Sequence

import numpy as np
import torch
from datasets import load_dataset
import torchvision.transforms.functional as F




class TripletCreator:
    """Build Galaxy triplets from catalog parquet files."""

    SUPPORTED_CATALOGS = {"legacysurvey_hsc_crossmatched"}

    def __init__(
        self,
        dataset_path: str = "/mnt/scratch/legacysurvey_hsc_crossmatched/data",
        catalog_name: str = "legacysurvey_hsc_crossmatched",
        files_to_use: Optional[Sequence[int | str]] = None,
        split: str = "train",
        seed: int = 42,
    ) -> None:
        """
        Args:
            dataset_path: Directory containing ``*.parquet`` shards.
            catalog_name: Currently only supports ``legacysurvey_hsc_crossmatched``.
            files_to_use: ``None`` (all shards), integer count, or explicit list
                of indices / filenames to keep.
            split: Passed to HuggingFace ``load_dataset``.
            seed: RNG seed for deterministic sampling.
        """

        if catalog_name not in self.SUPPORTED_CATALOGS:
            raise ValueError(
                f"Unsupported catalog '{catalog_name}'. "
                f"Supported: {self.SUPPORTED_CATALOGS}"
            )

        self.dataset_path = os.path.abspath(dataset_path)
        self.catalog_name = catalog_name
        self.split = split
        self.rng = random.Random(seed)

        self.files = self._resolve_files(files_to_use)
        self.dataset = self._load_dataset()
        self.num_rows = len(self.dataset)

    # --------------------------------------------------------------------- #
    # Dataset helpers
    # --------------------------------------------------------------------- #
    def _resolve_files(
        self, files_to_use: Optional[Sequence[int | str]]
    ) -> List[str]:
        """Return an ordered list of parquet files we will load."""

        pattern = os.path.join(self.dataset_path, "*.parquet")
        all_files = sorted(glob.glob(pattern))
        if not all_files:
            raise FileNotFoundError(f"No parquet files found under {pattern}")

        # If user passed a count
        if isinstance(files_to_use, int):
            return all_files[: files_to_use]

        if isinstance(files_to_use, str):
            return [
                files_to_use
                if files_to_use.endswith(".parquet")
                else os.path.join(self.dataset_path, files_to_use)
            ]

        # If user passed explicit identifiers (indices or filenames)
        if (
            isinstance(files_to_use, Sequence)
            and files_to_use
            and not isinstance(files_to_use, (str, bytes))
        ):
            resolved: List[str] = []
            for item in files_to_use:
                if isinstance(item, int):
                    resolved.append(all_files[item])
                else:
                    resolved.append(
                        item
                        if item.endswith(".parquet")
                        else os.path.join(self.dataset_path, item)
                    )
            return resolved

        return all_files

    def _load_dataset(self):
        """Load parquet shards through HuggingFace datasets."""

        return load_dataset(
            "parquet",
            data_files=self.files if len(self.files) > 1 else self.files[0],
            split=self.split,
        )

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def create_triplet(self, idx: Optional[int] = None, anchor_survey: str = "legacysurvey", use_object_mask: bool = True):
        """
        Produce a single galaxy triplet.

        Args:
            idx: Optional index into dataset. If None, randomly sampled.
            anchor_survey: Which survey to use as anchor ('legacysurvey' or 'hsc').
            use_object_mask: If True, include object mask in the tensor channels.

        Returns:
            dict with keys:
                - ground_truth: dict with 'tensor', 'psf_fwhm', 'scale', 'instrument_onehot'
                  for the anchor image
                - different_galaxy: dict with same structure for a different galaxy
                  from the same instrument
                - cross_instrument: dict with same structure for the same galaxy
                  from the cross instrument
                - anchor_index: index of the anchor record
                - different_index: index of the different galaxy record
                - include_object_mask: boolean indicating if object mask was included
        """
        if idx is None:
            idx = self.rng.randint(0, self.num_rows - 1)

        anchor_record = self.dataset[idx]

        # Sample a different galaxy (different row, same instrument)
        different_idx = idx
        while different_idx == idx:
            different_idx = self.rng.randint(0, self.num_rows - 1)
        different_record = self.dataset[different_idx]

        # Select which images to use based on anchor_survey
        anchor_image, cross_image = self._select_instruments(anchor_record, anchor_survey)
        different_image, _ = self._select_instruments(different_record, anchor_survey)

        # Extract object masks if requested
        anchor_object_mask = None
        different_object_mask = None
        if use_object_mask:
            anchor_object_mask = self._extract_object_mask(anchor_record)
            different_object_mask = self._extract_object_mask(different_record)

        # Determine cross instrument name
        cross_instrument_name = "hsc" if anchor_survey == "legacysurvey" else "legacysurvey"

        # Extract image data for all three images
        ground_truth_data = self._extract_image_data(anchor_image, anchor_survey, object_mask=anchor_object_mask)
        different_galaxy_data = self._extract_image_data(different_image, anchor_survey, object_mask=different_object_mask)
        cross_instrument_data = self._extract_image_data(cross_image, cross_instrument_name, object_mask=anchor_object_mask)

        return {
            "ground_truth": ground_truth_data,
            "different_galaxy": different_galaxy_data,
            "cross_instrument": cross_instrument_data,
            "anchor_index": idx,
            "different_index": different_idx,
            "include_object_mask": use_object_mask,
        }

    def create_batch_triplets(self, batch_size: int, anchor_survey: str = "legacysurvey", use_object_mask: bool = True) -> dict:
        """
        Stack multiple triplets into batched tensors.

        Returns:
            dict with keys:
                - ground_truth: dict with batched 'tensor', 'psf_fwhm', 'scale', 'instrument_onehot'
                - different_galaxy: dict with batched tensors
                - cross_instrument: dict with batched tensors
                - anchor_indices: tensor of anchor indices
                - different_indices: tensor of different galaxy indices
        """
        ground_truth_tensors = []
        ground_truth_psf = []
        ground_truth_scale = []
        ground_truth_instrument = []

        different_tensors = []
        different_psf = []
        different_scale = []
        different_instrument = []

        cross_inst_tensors = []
        cross_inst_psf = []
        cross_inst_scale = []
        cross_inst_instrument = []

        anchor_indices = []
        different_indices = []

        for _ in range(batch_size):
            triplet = self.create_triplet(anchor_survey=anchor_survey, use_object_mask=use_object_mask)

            # Ground truth
            ground_truth_tensors.append(triplet["ground_truth"]["tensor"])
            ground_truth_psf.append(triplet["ground_truth"]["psf_fwhm"])
            ground_truth_scale.append(triplet["ground_truth"]["scale"])
            ground_truth_instrument.append(triplet["ground_truth"]["instrument_onehot"])

            # Different galaxy
            different_tensors.append(triplet["different_galaxy"]["tensor"])
            different_psf.append(triplet["different_galaxy"]["psf_fwhm"])
            different_scale.append(triplet["different_galaxy"]["scale"])
            different_instrument.append(triplet["different_galaxy"]["instrument_onehot"])

            # Cross instrument
            cross_inst_tensors.append(triplet["cross_instrument"]["tensor"])
            cross_inst_psf.append(triplet["cross_instrument"]["psf_fwhm"])
            cross_inst_scale.append(triplet["cross_instrument"]["scale"])
            cross_inst_instrument.append(triplet["cross_instrument"]["instrument_onehot"])

            anchor_indices.append(triplet["anchor_index"])
            different_indices.append(triplet["different_index"])

        return {
            "ground_truth": {
                "tensor": torch.stack(ground_truth_tensors),
                "psf_fwhm": torch.stack(ground_truth_psf),
                "scale": torch.stack(ground_truth_scale),
                "instrument_onehot": torch.stack(ground_truth_instrument),
            },
            "different_galaxy": {
                "tensor": torch.stack(different_tensors),
                "psf_fwhm": torch.stack(different_psf),
                "scale": torch.stack(different_scale),
                "instrument_onehot": torch.stack(different_instrument),
            },
            "cross_instrument": {
                "tensor": torch.stack(cross_inst_tensors),
                "psf_fwhm": torch.stack(cross_inst_psf),
                "scale": torch.stack(cross_inst_scale),
                "instrument_onehot": torch.stack(cross_inst_instrument),
            },
            "anchor_indices": torch.tensor(anchor_indices, dtype=torch.long),
            "different_indices": torch.tensor(different_indices, dtype=torch.long),
        }

    # --------------------------------------------------------------------- #
    # Utility
    # --------------------------------------------------------------------- #
    def _select_instruments(self, record, anchor_survey: str):
        """Return (anchor_image, cross_image) tensors based on the anchor survey."""

        if anchor_survey == "legacysurvey":
            return record["legacysurvey_image"], record["hsc_image"]
        if anchor_survey == "hsc":
            return record["hsc_image"], record["legacysurvey_image"]
        else:
            raise ValueError("anchor_survey must be 'legacysurvey' or 'hsc'")

    def _extract_image_data(self, image: dict, instrument: str, num_bands: int = 4, object_mask: Optional[torch.Tensor] = None):
        """
        Extract image data (flux, ivar, mask) from an image dict and return
        combined tensor along with auxiliary metadata.

        Args:
            image: Image dict with keys 'flux', 'ivar', 'mask', 'psf_fwhm', 'scale'
            instrument: Instrument name ('legacysurvey' or 'hsc')
            num_bands: Number of bands to extract (default: 4)
            object_mask: Optional object mask tensor of shape (1, H, W) to concatenate

        Returns:
            dict with keys:
                - tensor: Combined tensor of shape (num_bands*3, H, W) or (num_bands*3 + 1, H, W)
                          if object_mask is provided, containing flux, ivar, mask (and optionally
                          object_mask) stacked along channel dimension
                - psf_fwhm: PSF FWHM values for each band, shape (num_bands,)
                - scale: Scale values for each band, shape (num_bands,)
                - instrument_onehot: One-hot encoded instrument vector, shape (2,)
                                    [1, 0] for 'legacysurvey', [0, 1] for 'hsc'
        """
        # Extract flux, ivar, mask arrays
        flux = np.array(image['flux'])  # (N, H, W) where N >= num_bands
        ivar = np.array(image['ivar'])  # (N, H, W)
        mask = np.array(image['mask'])  # (N, H, W)

        # Reduce to num_bands (take first num_bands)
        if flux.shape[0] < num_bands:
            raise ValueError(
                f"Image has {flux.shape[0]} bands, expected at least {num_bands}"
            )

        flux = flux[:num_bands, :, :]
        ivar = ivar[:num_bands, :, :]
        mask = mask[:num_bands, :, :]

        # Convert to tensors and combine along channel dimension
        # Result: (num_bands*3, H, W) = [flux_bands, ivar_bands, mask_bands]
        flux_tensor = torch.from_numpy(flux).float()
        ivar_tensor = torch.from_numpy(ivar).float()
        mask_tensor = torch.from_numpy(mask).float()

        if object_mask is None:
            combined_tensor = torch.cat([flux_tensor, ivar_tensor, mask_tensor], dim=0) # (num_bands*3, H, W)
        else:
            combined_tensor = torch.cat([flux_tensor, ivar_tensor, mask_tensor, object_mask], dim=0) # (num_bands*3 + 1, H, W)

        # Extract auxiliary metadata
        psf_fwhm = np.array(image['psf_fwhm'])[:num_bands]  # (num_bands,)
        scale = np.array(image['scale'][0]) # (1,)

        # Create one-hot encoding for instrument
        if instrument == "legacysurvey":
            instrument_onehot = torch.tensor([1.0, 0.0], dtype=torch.float32)
        elif instrument == "hsc":
            instrument_onehot = torch.tensor([0.0, 1.0], dtype=torch.float32)
        else:
            raise ValueError(f"Unknown instrument: {instrument}")

        return {
            "tensor": combined_tensor,
            "psf_fwhm": torch.from_numpy(psf_fwhm).float(),
            "scale": torch.from_numpy(scale).float(),
            "instrument_onehot": instrument_onehot,
        }

    @staticmethod
    def _normalize_tensor(tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Normalize tensor to [0,1] with numerical stability."""

        min_val = tensor.min()
        max_val = tensor.max()
        if torch.isclose(max_val, min_val):
            return torch.zeros_like(tensor)
        return (tensor - min_val) / (max_val - min_val + eps)

    def _extract_object_mask(self, record):
        """Extract object mask from record."""

        img = record['object_mask']
        tensor_mask = F.to_tensor(img)
        return tensor_mask # (1, 160, 160) for hsc-legacy survey
