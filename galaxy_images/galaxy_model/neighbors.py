"""
Neighbors dataset for HSC/Legacy triplets (target, same-galaxy, same-instrument neighbors).

Use from a training script:

  from galaxy_images.galaxy_model.neighbors import (
      NeighborsDataset,
      collate_neighbors,
      NORM_DICT,
      worker_init_fn,
  )
  train_dataset = NeighborsDataset(hdf5_path="path/to/neighbours.h5", ...)
  train_loader = DataLoader(
      train_dataset,
      batch_size=64,
      shuffle=True,
      num_workers=4,
      collate_fn=collate_neighbors,
      persistent_workers=True,
      pin_memory=True,
      worker_init_fn=worker_init_fn,
  )
  # Batch: (targets, samegals, padded_neighbors, neighbor_masks, metadata)
"""

import os
import sys

_current_path = os.path.abspath(__file__)
_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(_current_path)))
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from galaxy_images.image_preprocessing import preprocess_image_v2
from galaxy_images.galaxy_model.data import zoom_legacy_image

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

NORM_DICT = {
    "hsc": [0.022, 0.05],
    "legacy": [0.023, 0.063],
    "legacy_zoom": [0.045, 0.078],
    "hsc96": [0.00897, 0.0312],
    "legacy96": [0.0108, 0.050],
    "legacy96_zoom": [0.0173, 0.053],
}


def preprocess_raw_image(image, survey: str = "hsc", crop_size: int = 48, norm_dict: dict = NORM_DICT) -> torch.Tensor:
    """Preprocess raw images: crop, clamp, rescale, range compress; zoom for legacy; then normalize."""
    if not torch.is_tensor(image):
        image = torch.from_numpy(np.asarray(image, dtype=np.float32)).float()
    image = preprocess_image_v2(image, crop_size=crop_size, survey=survey)
    if survey == "legacy":
        image = zoom_legacy_image(image)
        mean, std = norm_dict["legacy_zoom"]
    else:
        mean, std = norm_dict["hsc"]
    image = (image - mean) / std
    return image


def worker_init_fn(worker_id: int) -> None:
    """Use as DataLoader worker_init_fn to avoid thread oversubscription."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


class NeighborsDataset(Dataset):
    def __init__(self, hdf5_path, norm_dict=NORM_DICT, crop_size=48, max_neighbors=15):
        self.hdf5_path = hdf5_path
        self.norm_dict = norm_dict
        self.crop_size = crop_size
        self.max_neighbors = max_neighbors
        self.file = None  # Handle for lazy loading

        # Pre-filter indices once in __init__ to avoid doing logic in __getitem__
        with h5py.File(self.hdf5_path, 'r') as f:
            sources = f['source_type'][:]
            indexes_mmu = np.where(sources == 0)[0]

            neigh_hsc = f["neighbor_idx_hsc"][indexes_mmu]
            neigh_legacy = f["neighbor_idx_legacy"][indexes_mmu]

            # Vectorized check for "good" indices
            good_both = (~np.all(neigh_hsc == -1, axis=1)) & (~np.all(neigh_legacy == -1, axis=1))
            self.indexes_mmu = indexes_mmu[good_both]

            # Pre-cache the neighbor indices for the filtered set to avoid double-reading
            self.cached_neighbor_hsc = neigh_hsc[good_both]
            self.cached_neighbor_legacy = neigh_legacy[good_both]

    def _open_file(self):
        """Opens the HDF5 file once per worker process."""
        if self.file is None:
            self.file = h5py.File(self.hdf5_path, 'r', libver='latest', swmr=True)

    def __len__(self):
        return len(self.indexes_mmu)

    def __getitem__(self, idx):
        self._open_file()
        index_mmu = self.indexes_mmu[idx]

        # Decide survey once
        anchor_is_hsc = (idx % 2 == 0)
        anchor_survey = 'hsc' if anchor_is_hsc else 'legacy'

        # Fetch primary images
        img_hsc = self.file['images_hsc'][index_mmu]
        img_legacy = self.file['images_legacy'][index_mmu]

        if anchor_is_hsc:
            target_raw, samegal_raw = img_hsc, img_legacy
            # Use pre-cached neighbor indices
            neighbor_ids = self.cached_neighbor_hsc[idx]
            images_ds = self.file['images_hsc']
            survey_key, pair_key = "hsc", "legacy"
        else:
            target_raw, samegal_raw = img_legacy, img_hsc
            neighbor_ids = self.cached_neighbor_legacy[idx]
            images_ds = self.file['images_legacy']
            survey_key, pair_key = "legacy", "hsc"

        # Filter neighbors
        neighbor_ids = neighbor_ids[neighbor_ids != -1][:self.max_neighbors]

        # Batch-read neighbor images (one HDF5 slice; h5py requires indices in increasing order)
        if len(neighbor_ids) > 0:
            order = np.argsort(neighbor_ids)
            sorted_ids = neighbor_ids[order]
            neigh_imgs_sorted = images_ds[np.asarray(sorted_ids)]
            sameins_list = [
                preprocess_raw_image(neigh_imgs_sorted[i], survey_key, self.crop_size, self.norm_dict)
                for i in range(len(sorted_ids))
            ]
        else:
            sameins_list = []

        target = preprocess_raw_image(target_raw, survey_key, self.crop_size, self.norm_dict)
        samegal = preprocess_raw_image(samegal_raw, pair_key, self.crop_size, self.norm_dict)

        # Slice to 4 channels if HSC
        if anchor_is_hsc:
            target = target[:4]
            sameins = torch.stack(sameins_list, dim=0)[:, :4] if sameins_list else torch.empty(0, 4, self.crop_size, self.crop_size)
        else:
            samegal = samegal[:4]
            sameins = torch.stack(sameins_list, dim=0) if sameins_list else torch.empty(0, 3, self.crop_size, self.crop_size)

        metadata = {"anchor_survey": anchor_survey, "idx": idx, "num_same_instrument": len(sameins_list)}
        return target, samegal, sameins, metadata


def collate_neighbors(batch):
    """Collate list of (target, samegal, sameins, metadata) into batched tensors and padded sameins."""
    targets = torch.stack([item[0] for item in batch])
    samegals = torch.stack([item[1] for item in batch])

    # Handle the variable length 'sameins'
    neighbor_tensors = [item[2] for item in batch] # List of (N_i, C, H, W)

    # Get the max number of neighbors in THIS batch
    max_n = max(t.size(0) for t in neighbor_tensors)

    padded_neighbors = []
    neighbor_masks = []

    for t in neighbor_tensors:
        n_current = t.size(0)
        # Pad (C, H, W) is static, we only pad the first dimension (N)
        # F.pad expects padding from the last dim backwards: (W, W, H, H, C, C, N_top, N_bottom)
        # Easier to just create a zeros tensor and copy
        pad_size = (max_n, *t.shape[1:])
        padded_t = torch.zeros(pad_size, dtype=t.dtype)
        padded_t[:n_current] = t

        padded_neighbors.append(padded_t)

        # Create a mask (1 for real data, 0 for padding)
        mask = torch.zeros(max_n, dtype=torch.bool)
        mask[:n_current] = True
        neighbor_masks.append(mask)

    # Re-wrap metadata into a list of dicts
    metadata = [item[3] for item in batch]

    return (
        targets,
        samegals,
        torch.stack(padded_neighbors),
        torch.stack(neighbor_masks),
        metadata,
    )


if __name__ == "__main__":
    import time
    from tqdm import tqdm
    from torch.utils.data import DataLoader

    neighbors_dataset = NeighborsDataset(
        hdf5_path="/data/vision/billf/scratch/pablomer/data/neighbours_v2.h5",
    )
    batch_size = 8
    num_warmup = 25
    num_measure = 100

    loader = DataLoader(
        neighbors_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_neighbors,
        persistent_workers=True,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    it = iter(loader)
    for _ in range(num_warmup):
        next(it)
    start = time.time()
    for _ in tqdm(range(num_measure)):
        next(it)
    elapsed = time.time() - start
    print(f"Average time per batch (steady state): {elapsed / num_measure:.4f}s")
