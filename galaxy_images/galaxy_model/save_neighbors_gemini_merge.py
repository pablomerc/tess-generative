import h5py
import glob
import os
import numpy as np
from tqdm import tqdm

# --- Configuration ---
# Directory containing your 32 shard files
INPUT_DIR = '/data/vision/billf/scratch/pablomer/data/neighbor_batches'
# Directory to save the VDS files
OUTPUT_DIR = '/data/vision/billf/scratch/pablomer/data/'

def create_vds(files, output_path):
    """Creates a Virtual Dataset combining multiple HDF5 files."""
    print(f"Creating VDS {output_path} from {len(files)} files...")

    # 1. Scan files to determine total size and shapes
    total_rows = 0
    sources = [] # List of (filename, rows)

    # We read the first file to get the base shapes/dtypes of all datasets
    with h5py.File(files[0], 'r') as f0:
        keys = list(f0.keys())
        shapes = {k: f0[k].shape[1:] for k in keys} # (C, H, W) etc, skipping N
        dtypes = {k: f0[k].dtype for k in keys}

    # Scan all files to get exact row counts
    for fn in tqdm(files, desc="Scanning files"):
        with h5py.File(fn, 'r') as f:
            rows = f['targets'].shape[0]
            sources.append((fn, rows))
            total_rows += rows

    print(f"Total samples: {total_rows}")

    # 2. Create the Virtual Layouts
    layouts = {}
    for k in keys:
        shape = (total_rows,) + shapes[k]
        layouts[k] = h5py.VirtualLayout(shape=shape, dtype=dtypes[k])

    # 3. Map slices
    current_idx = 0
    for fn, rows in tqdm(sources, desc="Mapping VDS"):
        for k in keys:
            vsource = h5py.VirtualSource(fn, k, shape=(rows,) + shapes[k])
            layouts[k][current_idx : current_idx + rows] = vsource
        current_idx += rows

    # 4. Write the VDS file
    with h5py.File(output_path, 'w', libver='latest') as f:
        for k, layout in layouts.items():
            f.create_virtual_dataset(k, layout)
    print(f"Saved {output_path}")

def main():
    # Get all shard files
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "neighbors_shard_*.h5")))
    if not files:
        raise ValueError("No files found!")

    print(f"Found {len(files)} shards.")

    # Split: 31 Train, 1 Val
    train_files = files[:-1]
    val_files = files[-1:] # Keep as list for consistency

    create_vds(train_files, os.path.join(OUTPUT_DIR, "train_neighbors.vds"))
    create_vds(val_files, os.path.join(OUTPUT_DIR, "val_neighbors.vds"))

if __name__ == "__main__":
    main()
