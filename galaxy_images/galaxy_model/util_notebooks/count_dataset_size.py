"""
Quick script to figure out how many examples there are in the dataset.
"""

import h5py

hdf5_path = '/data/vision/billf/scratch/pablomer/legacysurvey_hsc/preprocessed_hsc_legacy_96x96_half.h5'

with h5py.File(hdf5_path, 'r') as f:
    print(len(f['hsc_images']))
