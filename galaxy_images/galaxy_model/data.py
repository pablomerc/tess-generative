"""
Dataset class for HSC and Legacy Survey images. Returns them normalized.
"""

import torch
import h5py
from pathlib import Path
from torch.utils.data import Dataset


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
