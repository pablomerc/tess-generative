'''
Script for preprocessing galaxy images.

Following the AION-1 Paper from Parker at al 2025.

'''
from __future__ import annotations
import torch


# Keeps track of the band indices for HSC and DES bands
BAND_TO_INDEX = {
    "HSC-G": 0,
    "HSC-R": 1,
    "HSC-I": 2,
    "HSC-Z": 3,
    "HSC-Y": 4,
    "DES-G": 5,
    "DES-R": 6,
    "DES-I": 7,
    "DES-Z": 8,
}
# Maximum band center values for HSC and DES bands
BAND_CENTER_MAX = {
    "HSC-G": 80,
    "HSC-R": 110,
    "HSC-I": 200,
    "HSC-Z": 330,
    "HSC-Y": 500,
    "DES-G": 6,
    "DES-R": 15,
    "DES-I": 20,
    "DES-Z": 25,
}

class CenterCrop:
    """Formatter that crops the images to have a fixed number of bands."""

    def __init__(self, crop_size: int = 96):
        self.crop_size = crop_size

    def __call__(self, image):
        _, _, height, width = image.shape
        start_x = (width - self.crop_size) // 2
        start_y = (height - self.crop_size) // 2
        return image[
            :, :, start_y : start_y + self.crop_size, start_x : start_x + self.crop_size
        ]


class Clamp:
    """Formatter that clamps the images to a given range."""

    def __init__(self):
        self.clamp_dict = BAND_CENTER_MAX

    def __call__(self, image, bands):
        for i, band in enumerate(bands):
            image[:, i, :, :] = torch.clip(
                image[:, i, :, :], -self.clamp_dict[band], self.clamp_dict[band]
            )
        return image


class RescaleToLegacySurvey:
    """Formatter that rescales the images to have a fixed number of bands."""

    def __init__(self):
        pass

    def convert_zeropoint(self, zp: float) -> float:
        return 10.0 ** ((zp - 22.5) / 2.5)

    def reverse_zeropoint(self, scale: float) -> float:
        return 22.5 - 2.5 * torch.log10(scale)

    def forward(self, image, survey):
        zpscale = self.convert_zeropoint(27.0) if survey == "HSC" else 1.0
        image /= zpscale
        return image

    def backward(self, image, survey):
        zpscale = self._reverse_zeropoint(27.0) if survey == "HSC" else 1.0
        image *= zpscale
        return image



def main():


    import glob
    import os
    import random
    from typing import Iterable, List, Optional, Sequence, Tuple

    import numpy as np
    import torch
    from datasets import load_dataset
    dataset_path = '/mnt/scratch/legacysurvey_hsc_crossmatched/data'
    catalog_name = "legacysurvey_hsc_crossmatched"
    files_to_use = 1
    pattern = os.path.join(dataset_path, "*.parquet")
    all_files = sorted(glob.glob(pattern))
    dataset=load_dataset(
            "parquet",
            data_files=all_files,
            split="train"
        )
    idx = 10
    example_record = dataset[idx]

    # print(example_record.keys() )
    hsc_image = np.array(example_record['hsc_image']['flux'])
    legacysurvey_image = np.array(example_record['legacysurvey_image']['flux'])

    print(hsc_image.shape)
    print(legacysurvey_image.shape)

    hsc_image_tensor = torch.from_numpy(hsc_image).float()
    legacysurvey_image_tensor = torch.from_numpy(legacysurvey_image).float()
    print(hsc_image_tensor.shape)
    print(legacysurvey_image_tensor.shape)


    #Lets test the classes

    cropper = CenterCrop(crop_size=96)
    clamper = Clamp()
    rescaler = RescaleToLegacySurvey()



if __name__ == "__main__":
    main()
