"""
Dataset class for HSC and Legacy Survey images. Returns them normalized.
"""

import torch
import h5py
from pathlib import Path
from torch.utils.data import Dataset
import random
import math
# import torch
from torch.utils.data import Sampler
from torch.utils.data._utils.collate import default_collate


NORM_DICT = {
    'hsc': [0.022, 0.05],
    'legacy': [0.023, 0.063],
}

class HSCLegacyDataset(Dataset):
    def __init__(
        self,
        hdf5_path: str,
        norm_dict: dict = NORM_DICT,
        idx_list: list = None,
    ):
        hdf5_path = Path(hdf5_path)
        if not hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
        self.hdf5_path = hdf5_path
        self.norm_dict = norm_dict
        self.idx_list = idx_list
        self.num_images = len(idx_list) if idx_list is not None else None

        with h5py.File(hdf5_path, 'r') as f:
            total_images = f.attrs['num_images']
            self.crop_size = f.attrs['crop_size']
            self.num_channels = f.attrs['num_channels']
            if self.idx_list is not None:
                self.hsc_images = torch.from_numpy(f['hsc_images'][self.idx_list]).float()
                self.legacy_images = torch.from_numpy(f['legacy_images'][self.idx_list]).float()
            else:
                self.hsc_images = torch.from_numpy(f['hsc_images'][:total_images]).float()
                self.legacy_images = torch.from_numpy(f['legacy_images'][:total_images]).float()
        if self.idx_list is None:
            self.num_images = total_images
        print(f"Loaded {self.num_images} images into memory, "
        f"shape: ({self.num_images}, {self.num_channels}, {self.crop_size}, {self.crop_size})")
        print(f"Memory usage: ~{2 * self.hsc_images.numel() * 4 / (1024**3):.3f} GB")

    def __len__(self):

        return self.num_images

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.num_images:
            raise IndexError(f"Index {idx} out of range [0, {self.num_images})")
        hsc_image = self.hsc_images[idx]
        legacy_image = self.legacy_images[idx]
        mean_hsc, std_hsc = self.norm_dict['hsc']
        hsc_image = (hsc_image - mean_hsc) / std_hsc
        mean_legacy, std_legacy = self.norm_dict['legacy']
        legacy_image = (legacy_image - mean_legacy) / std_legacy
        return hsc_image, legacy_image

class HSCLegacyTripletDataset(Dataset):
    def __init__(
        self,
        hdf5_path: str,
        norm_dict: dict = NORM_DICT,
        idx_list: list = None,
    ):

        hdf5_path = Path(hdf5_path)
        if not hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found:{hdf5_path}")
        self.hdf5_path = hdf5_path
        self.norm_dict = norm_dict
        self.idx_list = idx_list
        self.num_images = len(idx_list) if idx_list is not None else None

        with h5py.File(hdf5_path, 'r') as f:
            total_images = f.attrs['num_images']
            self.crop_size = f.attrs['crop_size']
            self.num_channels = f.attrs['num_channels']
            if self.idx_list is not None:
                self.hsc_images = torch.from_numpy(f['hsc_images'][self.idx_list]).float()
                self.legacy_images = torch.from_numpy(f['legacy_images'][self.idx_list]).float()
            else:
                self.hsc_images = torch.from_numpy(f['hsc_images'][:total_images]).float()
                self.legacy_images = torch.from_numpy(f['legacy_images'][:total_images]).float()
        if self.idx_list is None:
            self.num_images = total_images
        print(f"Loaded {self.num_images} images into memory, "
        f"shape: ({self.num_images}, {self.num_channels}, {self.crop_size}, {self.crop_size})")
        print(f"Memory usage: ~{2 * self.hsc_images.numel() * 4 / (1024**3):.3f} GB")

    def __len__(self):

        return self.num_images

    def __getitem__(self, idx):
        """
        Returns an example with anchor image, same galaxy on the other instrument, and k examples of same instrument with different galaxies.

        Args:
            idx: Either an int (dataset index) or a tuple (idx, anchor_survey) when using BalancedAnchorBatchSampler.
                 If tuple, anchor_survey will be used instead of random choice.

        Returns:
            tuple: (anchor_image, same_galaxy, same_instrument, metadata)
                - anchor_image: torch.Tensor, shape (C, H, W) - normalized anchor image
                - same_galaxy: torch.Tensor, shape (C, H, W) - same galaxy from other instrument, normalized
                - same_instrument: torch.Tensor, shape (k, C, H, W) - k different galaxies from same instrument, normalized
                - metadata: dict with keys:
                    - 'anchor_survey': str, either 'hsc' or 'legacy'
                    - 'idx': int, the dataset index used
                    - 'num_same_instrument': int, actual number of same_instrument examples (may be < k for small datasets)
        """
        # Handle tuple from BalancedAnchorBatchSampler: (idx, anchor_survey)
        if isinstance(idx, tuple):
            idx, anchor_survey = idx
        else:
            anchor_survey = None  # Will be randomly chosen below

        if idx < 0 or idx >= self.num_images:
            raise IndexError(f"Index {idx} out of range [0, {self.num_images})")
        hsc_image = self.hsc_images[idx]
        legacy_image = self.legacy_images[idx]
        mean_hsc, std_hsc = self.norm_dict['hsc']
        hsc_image = (hsc_image - mean_hsc) / std_hsc
        mean_legacy, std_legacy = self.norm_dict['legacy']
        legacy_image = (legacy_image - mean_legacy) / std_legacy

        # Use provided anchor_survey or randomly choose
        if anchor_survey is None:
            anchor_survey = random.choice(['hsc', 'legacy'])


        # TODO: Replace this by SNR-based matching
        k = 5
        # Generate enough candidates to ensure we get k unique indices (excluding idx)
        # Use a set to ensure uniqueness, and keep sampling until we have enough
        different_indexes_set = set()
        max_attempts = 100  # Prevent infinite loop
        attempts = 0
        while len(different_indexes_set) < k and attempts < max_attempts:
            candidates = torch.randint(0, self.num_images, (k * 2,)).tolist()
            for cand_idx in candidates:
                if cand_idx != idx:
                    different_indexes_set.add(cand_idx)
                if len(different_indexes_set) >= k:
                    break
            attempts += 1

        if len(different_indexes_set) < k:
            # Fallback: if we can't get k unique indices, use what we have
            # This can happen with very small datasets
            different_indexes = torch.tensor(list(different_indexes_set), dtype=torch.long)
        else:
            different_indexes = torch.tensor(list(different_indexes_set)[:k], dtype=torch.long)

        anchor_image = None
        same_galaxy = None
        same_instrument = None

        if anchor_survey == 'hsc':
            anchor_image = hsc_image
            same_galaxy = legacy_image

            # Normalize same_instrument images
            same_instrument_raw = self.hsc_images[different_indexes]
            same_instrument = (same_instrument_raw - mean_hsc) / std_hsc

        elif anchor_survey == 'legacy':
            anchor_image = legacy_image
            same_galaxy = hsc_image

            # Normalize same_instrument images
            same_instrument_raw = self.legacy_images[different_indexes]
            same_instrument = (same_instrument_raw - mean_legacy) / std_legacy

        # Metadata dictionary for debugging, analysis, and logging
        metadata = {
            'anchor_survey': anchor_survey,
            'idx': idx,
            'num_same_instrument': len(different_indexes),
        }

        return anchor_image, same_galaxy, same_instrument, metadata


class BalancedAnchorBatchSampler(Sampler):
    """
    Yields batches of (idx, anchor_survey) tuples so that each batch is exactly 50/50
    in terms of anchor survey. The anchor_survey assignments are randomly shuffled within
    each batch to avoid systematic patterns.

    This ensures balanced training while maintaining randomness in the assignment.
    """
    def __init__(self, num_samples: int, batch_size: int, drop_last: bool = True, seed: int = 0):
        assert batch_size % 2 == 0, "batch_size must be even for 50/50 split"
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.half = batch_size // 2
        self.drop_last = drop_last
        self.seed = seed

    def __len__(self):
        if self.drop_last:
            return self.num_samples // self.batch_size
        return math.ceil(self.num_samples / self.batch_size)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed)

        # Shuffle indices once per epoch
        perm = torch.randperm(self.num_samples, generator=g).tolist()

        # Calculate number of batches
        n_full = self.num_samples // self.batch_size
        n_batches = n_full if self.drop_last else math.ceil(self.num_samples / self.batch_size)

        cursor = 0
        for _ in range(n_batches):
            if cursor + self.batch_size > self.num_samples:
                if self.drop_last:
                    break
                # If not dropping last, stop here (could pad if needed)
                break

            batch_idxs = perm[cursor:cursor + self.batch_size]
            cursor += self.batch_size

            # Create balanced anchor_survey assignments: half hsc, half legacy
            anchor_surveys = ['hsc'] * self.half + ['legacy'] * self.half
            # Shuffle the anchor_survey assignments within the batch for randomness
            anchor_survey_perm = torch.randperm(self.batch_size, generator=g).tolist()
            anchor_surveys_shuffled = [anchor_surveys[i] for i in anchor_survey_perm]

            # Pair each idx with its randomly assigned anchor_survey
            batch = [(idx, anchor_survey) for idx, anchor_survey in zip(batch_idxs, anchor_surveys_shuffled)]
            yield batch

def custom_collate_fn(batch):
    """
    Custom collate function that handles metadata dictionaries properly.
    """
    # Separate tensors from metadata
    anchor_images = [item[0] for item in batch]
    same_galaxies = [item[1] for item in batch]
    same_instruments = [item[2] for item in batch]
    metadata_list = [item[3] for item in batch]

    # Collate tensors normally
    collated_anchor = default_collate(anchor_images)
    collated_same_galaxy = default_collate(same_galaxies)
    collated_same_instrument = default_collate(same_instruments)

    # Keep metadata as a list of dicts (don't try to collate it)
    return collated_anchor, collated_same_galaxy, collated_same_instrument, metadata_list
