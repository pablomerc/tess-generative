import torch
import h5py
from pathlib import Path
from torch.utils.data import Dataset


NORM_DICT = {
    # IMPORTANT: you should calculate these for your images and scalars too!
    'images': (0.022, 0.05),  # (mean, std) for the target images
    'scalars': None
     #  should calculate these for your 4 scalars too!
    # Example: 'scalars': ([mean1, mean2, mean3, mean4], [std1, std2, std3, std4])
}

# Before you start I would calculate the mean and std of your images and use them to normalize them.

# I would also try to put all your images and scalars in a single HDF5 file
# and make the dataset class to load them to memory at once to avoid loading them one by one.
# Maybe shuffle you csv file once and then put it into an HDF5 file and then
# for you split you can just read the indices like I was doing in the other script

class CustomDataset(Dataset): #Change the name to whatever you want to call your dataset
    def __init__(
        self,
        hdf5_path: str,
        norm_dict: dict = NORM_DICT,
        idx_list: list = None,
    ):
        hdf5_path = Path(hdf5_path)
        if not hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

        self.norm_dict = norm_dict

        with h5py.File(hdf5_path, 'r') as f:
            # Check strictly if dataset exists
            if 'images' not in f or 'parameters' not in f:
                raise KeyError(f"HDF5 file must contain 'images' and 'parameters' keys. Found: {list(f.keys())}")

            # 1. Load Data
            if idx_list is not None:
                # Use sorted indices to optimize HDF5 read speed
                sorted_indices = sorted(idx_list) # Note: idx list needs to be in ascending order. if you call it with range it will automatically be so
                self.images = torch.from_numpy(f['images'][sorted_indices]).float()
                self.scalars = torch.from_numpy(f['parameters'][sorted_indices]).float() # 'parameters' is the key for the 4 scalar values / or however you saved them in the hdf5 file, maybe you even saved in 4 different keys
            else:
                self.images = torch.from_numpy(f['images'][:]).float()
                self.scalars = torch.from_numpy(f['parameters'][:]).float()

        self.num_images = len(self.images)

        # 2. Print Stats - not sure how accurate this will be but anyway it will print once at the beginning of training and you will know the datset was initialized correctly
        img_mem = self.images.element_size() * self.images.numel() / (1024**3)
        scalar_mem = self.scalars.element_size() * self.scalars.numel() / (1024**3)
        print(f"Loaded {self.num_images} items.")
        print(f"Images Shape: {self.images.shape} | Scalars Shape: {self.scalars.shape}")
        print(f"Memory usage: ~{img_mem + scalar_mem:.3f} GB")

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        # 1. Get raw data
        # shape: (C, H, W) / make sure its this way because the model expects (C, H, W ) and not (H, W, C) if i remember correctly (you should doublecheck just in case)
        target_image = self.images[idx]
        scalar_cond = self.scalars[idx] # shape: (4,)

        # 2. Normalize Image
        # norm_dict['images'] is tuple (mean, std)
        if 'images' in self.norm_dict and self.norm_dict['images'] is not None:
            mean, std = self.norm_dict['images']
            target_image = (target_image - mean) / std

        # 3. Normalize Scalars
        if 'scalars' in self.norm_dict and self.norm_dict['scalars'] is not None:
            s_mean = torch.tensor(self.norm_dict['scalars'][0])
            s_std = torch.tensor(self.norm_dict['scalars'][1])
            scalar_cond = (scalar_cond - s_mean) / s_std

        return target_image, scalar_cond
