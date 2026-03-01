import h5py
import numpy as np

path = '/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy_images/galaxy_model/downstream_evaluation/downstream_mmu_zdim16_geom_neighbors.h5'

with h5py.File(path, 'r') as f:
    print(f.keys())
    # print(len(f['labels'][0]))
    # print(f['labels'].keys())

# path = '/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy_images/galaxy_model/downstream_evaluation/downstream_legacy_provabgs_zdim16_geom_neighbors.h5
