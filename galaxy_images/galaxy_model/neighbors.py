"""
Dataset class for the neighbors dataset.
"""

import os
import sys
from torch.utils.data import DataLoader
import torch.nn.functional as F
# Path setup so galaxy_images package can be imported (same as load_lenses.py)
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
    'hsc': [0.022, 0.05],
    'legacy': [0.023, 0.063],
    'legacy_zoom': [0.045, 0.078],
    'hsc96': [0.00897, 0.0312],
    'legacy96': [0.0108, 0.050],
    'legacy96_zoom': [0.0173, 0.053],
    # 'hsc64': [0.022, 0.05], # TODO: actually measure these (for now using 48x48 stats)
    # 'legacy64': [0.023, 0.063],
    # 'legacy64_zoom': [0.045, 0.078],
}



def preprocess_raw_image(image, survey: str = "hsc", crop_size: int = 48, norm_dict: dict = NORM_DICT) -> torch.Tensor:
    """Preprocess raw images: pipeline (crop, clamp, rescale, range compress), then zoom for legacy, then normalize."""
    if not torch.is_tensor(image):
        image = torch.from_numpy(np.asarray(image, dtype=np.float32)).float()
    # Standard pipeline (Crop, Clamp, Rescale, RangeCompress)
    image = preprocess_image_v2(image, crop_size=crop_size, survey=survey)
    # Zoom + normalization for this model
    if survey == "legacy":
        image = zoom_legacy_image(image)
        mean, std = norm_dict["legacy_zoom"]
    else:
        mean, std = norm_dict["hsc"]
    image = (image - mean) / std
    return image


def preprocess_raw_image_batch(images: np.ndarray, survey: str, crop_size: int, norm_dict: dict) -> torch.Tensor:
    """Batched preprocessing for same-instrument neighbors. Input (N, C, H, W) or (N, H, W, C)."""
    if images.size == 0:
        raise ValueError("images must be non-empty")
    x = torch.from_numpy(np.asarray(images, dtype=np.float32)).float()
    # Ensure (N, C, H, W) for preprocess_image_v2
    if x.ndim == 4 and x.shape[-1] in (3, 4, 5):
        x = x.permute(0, 3, 1, 2)
    x = preprocess_image_v2(x, crop_size=crop_size, survey=survey)
    if survey == "legacy":
        x = zoom_legacy_image(x)
        mean, std = norm_dict["legacy_zoom"]
    else:
        mean, std = norm_dict["hsc"]
    x = (x - mean) / std
    return x



# class NeighborsDataset(Dataset):
#     def __init__(
#         self,
#         hdf5_path: str,
#         norm_dict: dict = NORM_DICT,
#         crop_size: int = 48,
#         max_neighbors: int = 15,
#     ):
#         self.hdf5_path = hdf5_path
#         self.norm_dict = norm_dict
#         self.crop_size = crop_size
#         self.max_neighbors = max_neighbors

#         with h5py.File(hdf5_path, 'r') as f:
#             num_images = f['dec'].shape[0]
#             self.num_images = num_images
#             sources = f['source_type'][:]

#             # Get the right indices to iterate over
#             indexes_mmu = np.where(sources==0)[0]
#             # Pull neighbors only for MMU rows
#             neigh_hsc    = f["neighbor_idx_hsc"][indexes_mmu]    # shape (M, K)
#             neigh_legacy = f["neighbor_idx_legacy"][indexes_mmu] # shape (M, K)
#             # rows that are entirely -1 // they have no neighbors
#             allneg_hsc    = np.all(neigh_hsc == -1, axis=1)      # shape (M,)
#             allneg_legacy = np.all(neigh_legacy == -1, axis=1)   # shape (M,)
#             # "have no all_neg" = NOT all -1
#             good_hsc    = ~allneg_hsc
#             good_legacy = ~allneg_legacy
#             # require BOTH to be good
#             good_both = good_hsc & good_legacy

#             # final indices in the *original file indexing*
#             idx_overlap = indexes_mmu[good_both]
#             self.indexes_mmu = idx_overlap

#         print('Initialized Dataset')

#     def __len__(self):
#         return self.indexes_mmu.shape[0]

#     def __getitem__(self, idx):
#         if idx < 0 or idx >= self.indexes_mmu.shape[0]:
#             raise IndexError(f"Index {idx} out of range [0, {self.indexes_mmu.shape[0]}]")

#         index_mmu = self.indexes_mmu[idx]
#         with h5py.File(self.hdf5_path, 'r') as f:
#             hsc_image = np.array(f['images_hsc'][index_mmu])
#             legacy_image = np.array(f['images_legacy'][index_mmu])

#             anchor_survey = 'hsc' if idx % 2 == 0 else 'legacy'

#             # Match data.py + double_train_fm: all images 4 channels (HSC uses first 4 bands).
#             if anchor_survey == 'hsc':
#                 target = preprocess_raw_image(hsc_image, "hsc", self.crop_size, self.norm_dict)[:4]
#                 samegal = preprocess_raw_image(legacy_image, "legacy", self.crop_size, self.norm_dict)

#                 neighbor_indexes = np.array(f['neighbor_idx_hsc'][index_mmu])
#                 # Boolean indexing: "Select only elements where the value is not -1"
#                 neighbor_indexes = neighbor_indexes[neighbor_indexes != -1]

#                 n_neighbors = min(self.max_neighbors, neighbor_indexes.shape[0])

#                 neighbor_indexes = neighbor_indexes[:n_neighbors]

#                 images_ds = f['images_hsc']
#                 sameins_list = [
#                     preprocess_raw_image(np.array(images_ds[int(i)]), "hsc", self.crop_size, self.norm_dict)[:4]
#                     for i in neighbor_indexes
#                 ]
#                 sameins = torch.stack(sameins_list, dim=0)
#             else:
#                 target = preprocess_raw_image(legacy_image, "legacy", self.crop_size, self.norm_dict)
#                 samegal = preprocess_raw_image(hsc_image, "hsc", self.crop_size, self.norm_dict)[:4]

#                 neighbor_indexes = np.array(f['neighbor_idx_legacy'][index_mmu]) # h
#                 # Boolean indexing: "Select only elements where the value is not -1"
#                 neighbor_indexes = neighbor_indexes[neighbor_indexes != -1]

#                 n_neighbors = min(self.max_neighbors, neighbor_indexes.shape[0])

#                 neighbor_indexes = neighbor_indexes[:n_neighbors]

#                 images_ds = f['images_legacy']
#                 sameins_list = [
#                     preprocess_raw_image(np.array(images_ds[int(i)]), "legacy", self.crop_size, self.norm_dict)[:4]
#                     for i in neighbor_indexes
#                 ]
#                 sameins = torch.stack(sameins_list, dim=0)

#         metadata = {"anchor_survey": anchor_survey, "idx": idx, "num_same_instrument": sameins.shape[0]}
#         return target, samegal, sameins, metadata


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

        # OPTIMIZATION: Use list comprehension but minimize casting
        # If neighbors are sequential, slices are faster, but since they are indices,
        # this is the bottleneck.
        sameins_list = [
            preprocess_raw_image(images_ds[int(i)], survey_key, self.crop_size, self.norm_dict)
            for i in neighbor_ids
        ]

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
    """
    batch: list of tuples (target, samegal, sameins, metadata)
    """
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
        metadata
    )




#TODO: need to handle the different size of examples


# ############ TESTING
# import matplotlib.pyplot as plt
# import torch
# import numpy as np

# def normalize_for_vis(img: torch.Tensor) -> np.ndarray:
#     """
#     Normalize image to [0, 1] for visualization (same as double_train_fm._normalize_for_vis).
#     Input: (C, H, W). Output: (H, W, 3) numpy for imshow.
#     """
#     x = img[:3].clone()
#     x = x - x.min()
#     if x.max() > 0:
#         x = x / x.max()
#     return x.permute(1, 2, 0).cpu().numpy()

# def load_and_plot_neighbors_row_scaled(hdf5_path, max_neighbors_to_plot=5):
#     # 1. Initialize Dataset
#     dataset = NeighborsDataset(hdf5_path=hdf5_path)

#     # 2. Setup Plot
#     num_examples = 8
#     cols = 2 + max_neighbors_to_plot  # Target + SameGal + Neighbors
#     rows = num_examples

#     fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
#     plt.subplots_adjust(wspace=0.1, hspace=0.2)

#     print(f"Loading first {num_examples} examples (per-image normalize_for_vis, same as double_train_fm)...")

#     for i in range(num_examples):
#         # Load data
#         target, samegal, sameins, metadata = dataset[i]
#         survey_name = metadata['anchor_survey'].upper()

#         # Prepare axes for this row
#         ax_row = axes[i] if rows > 1 else axes

#         # --- Plot Target ---
#         target_vis = normalize_for_vis(target)
#         ax_row[0].imshow(target_vis)
#         ax_row[0].set_title(f"Target ({survey_name})\nIdx: {metadata['idx']}")
#         ax_row[0].axis('off')

#         # --- Plot Same Galaxy (Cross-Survey) ---
#         pair_survey = "LEGACY" if survey_name == "HSC" else "HSC"
#         samegal_vis = normalize_for_vis(samegal)
#         ax_row[1].imshow(samegal_vis)
#         ax_row[1].set_title(f"Same Gal ({pair_survey})")
#         ax_row[1].axis('off')

#         # --- Plot Neighbors (Same Instrument) ---
#         num_neighbors = sameins.shape[0]

#         for n_idx in range(max_neighbors_to_plot):
#             ax = ax_row[2 + n_idx]

#             if n_idx < num_neighbors:
#                 neighbor_vis = normalize_for_vis(sameins[n_idx])
#                 ax.imshow(neighbor_vis)
#                 ax.set_title(f"Neigh {n_idx+1}")
#             else:
#                 ax.text(0.5, 0.5, "No Data", ha='center', va='center')

#             ax.axis('off')

#     plt.tight_layout()
#     plt.savefig('neighbors_row_scaled.png')
#     plt.close()


# # Usage:
# load_and_plot_neighbors_row_scaled('/data/vision/billf/scratch/pablomer/data/neighbours_v2.h5')



### LEts measure speed
import time
from tqdm import tqdm
from torch.utils.data import DataLoader

# 1. Setup Dataset and Loader
neighbors_dataset = NeighborsDataset(
    hdf5_path='/data/vision/billf/scratch/pablomer/data/neighbours_v2.h5',
)

neighbors_dataloader = DataLoader(
    neighbors_dataset,
    batch_size=2,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_neighbors # <--- Add this
)
# 1. Setup Parameters
num_warmup = 25         # Batches to "throw away" to let workers start up
num_measure = 100       # Batches to actually time
batch_size = 64

# 2. Timing Loop
print(f"Warming up for {num_warmup} batches...")
iterator = iter(neighbors_dataloader)

# Warmup phase (not timed)
for _ in range(num_warmup):
    _ = next(iterator)

print(f"Starting benchmark for {num_measure} batches...")
start_time = time.time()

for i in tqdm(range(num_measure)):
    batch = next(iterator)
    # Optional: if using GPU, add torch.cuda.synchronize() here

end_time = time.time()

# 3. Results
total_time = end_time - start_time
avg_time_per_batch = total_time / num_measure
print(f"Average time per batch (Steady State): {avg_time_per_batch:.4f}s")
