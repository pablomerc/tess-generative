import os
import sys
import traceback
from datetime import datetime

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Path setup so galaxy_images is importable regardless of cwd
_current_path = os.path.abspath(__file__)
_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(_current_path)))
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from galaxy_images.galaxy_model.neighbors import (
    NeighborsDataset,
    collate_neighbors,
)

# --- Configuration ---
SOURCE_H5 = '/work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5'
OUTPUT_DIR = '/work1/jeroenaudenaert/pablomer/data/neighbor_batches'
BATCH_SIZE = 64
NUM_WORKERS = 8  # Crank this up! We want to brute-force the random reads.
CHUNKS_PER_FILE = 50 # How many batches to save before closing a file (approx 3200 samples)

def save_shard(shard_idx, buffer, save_dir, global_max_neighbors):
    """Writes a buffered list of batches to a single HDF5 file."""
    if not buffer:
        return

    filename = os.path.join(save_dir, f"neighbors_shard_{shard_idx:04d}.h5")

    # Concatenate all batches in the buffer
    # buffer items are tuples: (targets, samegals, sameins, masks, metadata)

    all_targets = torch.cat([b[0] for b in buffer], dim=0).numpy()
    all_samegals = torch.cat([b[1] for b in buffer], dim=0).numpy()

    # Neighbors is tricky: different batches might have padded to different widths.
    # We must pad everything to the GLOBAL max_neighbors defined in the dataset.
    # buffer[i][2] is (Batch, N_neighbors_local, C, H, W)
    all_sameins_list = []
    all_masks_list = []

    for b in buffer:
        neigh_tensor = b[2] # (B, N_loc, C, H, W)
        mask_tensor = b[3]  # (B, N_loc)

        B, N_loc, C, H, W = neigh_tensor.shape

        # Pad to global max
        pad_n = global_max_neighbors - N_loc
        if pad_n > 0:
            # Pad dim 1 (neighbors)
            # F.pad format is (W, W, H, H, C, C, Front, Back) ... tedious for 5D
            # easier to just assign
            new_neigh = torch.zeros((B, global_max_neighbors, C, H, W), dtype=neigh_tensor.dtype)
            new_neigh[:, :N_loc] = neigh_tensor

            new_mask = torch.zeros((B, global_max_neighbors), dtype=mask_tensor.dtype)
            new_mask[:, :N_loc] = mask_tensor

            all_sameins_list.append(new_neigh)
            all_masks_list.append(new_mask)
        else:
            all_sameins_list.append(neigh_tensor)
            all_masks_list.append(mask_tensor)

    all_sameins = torch.cat(all_sameins_list, dim=0).numpy()
    all_masks = torch.cat(all_masks_list, dim=0).numpy() # Boolean

    # Metadata is a list of lists of dicts. Flatten it.
    flat_metadata = [item for b in buffer for item in b[4]]
    # Extract raw columns for HDF5 storage
    meta_idxs = np.array([m["idx"] for m in flat_metadata])
    meta_num_same = np.array([m["num_same_instrument"] for m in flat_metadata], dtype=np.int32)
    meta_survey = np.array([m["anchor_survey"].encode("utf-8") for m in flat_metadata])

    print(f"Saving {filename} | Shape: {all_targets.shape}...")

    with h5py.File(filename, 'w') as f:
        # Enable compression (lzf is fast)
        f.create_dataset('targets', data=all_targets, compression="lzf")
        f.create_dataset('samegals', data=all_samegals, compression="lzf")
        f.create_dataset('sameins', data=all_sameins, compression="lzf")
        f.create_dataset('neighbor_masks', data=all_masks, compression="lzf")

        # Metadata
        f.create_dataset("meta_idx", data=meta_idxs)
        f.create_dataset("meta_num_same_instrument", data=meta_num_same)
        f.create_dataset("meta_survey", data=meta_survey)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Init Source Dataset
    # shuffle=False so we scan in file order (deterministdic shards)
    dataset = NeighborsDataset(hdf5_path=SOURCE_H5, max_neighbors=5)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_neighbors,
        drop_last=False,
        persistent_workers=NUM_WORKERS > 0,
        pin_memory=True,
    )

    buffer = []
    shard_counter = 0

    start_time = datetime.now().isoformat()
    print(f"Starting generation at {start_time}")
    print(f"Output: {OUTPUT_DIR} | Total batches: {len(dataloader)}")

    try:
        for batch in tqdm(dataloader):
            buffer.append(batch)

            if len(buffer) >= CHUNKS_PER_FILE:
                save_shard(shard_counter, buffer, OUTPUT_DIR, dataset.max_neighbors)
                buffer = []
                shard_counter += 1

        # Save remaining
        if buffer:
            save_shard(shard_counter, buffer, OUTPUT_DIR, dataset.max_neighbors)

        print(f"Done at {datetime.now().isoformat()} | Wrote {shard_counter + (1 if buffer else 0)} shards.")
    except Exception as e:
        # Save current buffer so progress is not lost
        if buffer:
            print(f"Error: {e}", file=sys.stderr)
            print(f"Saving {len(buffer)} buffered batches to recovery shard ...", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            save_shard(shard_counter, buffer, OUTPUT_DIR, dataset.max_neighbors)
            normal_path = os.path.join(OUTPUT_DIR, f"neighbors_shard_{shard_counter:04d}.h5")
            recovery_path = os.path.join(OUTPUT_DIR, f"neighbors_shard_{shard_counter:04d}_recovery.h5")
            os.rename(normal_path, recovery_path)
            print(f"Recovery shard saved: {recovery_path}", file=sys.stderr)
        raise

if __name__ == "__main__":
    main()



# OLD ONE
# import h5py
# import glob
# import os
# import numpy as np

# OUTPUT_DIR = '/work1/jeroenaudenaert/pablomer/data/neighbor_batches'
# VDS_PATH = os.path.join(OUTPUT_DIR, 'neighbours_vds.h5')

# files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "neighbors_shard_*.h5")))

# # Calculate layout
# total_rows = 0
# sources = [] # (file_path, rows_in_file)
# for fn in files:
# with h5py.File(fn, 'r') as f:
# rows = f['targets'].shape[0]
# sources.append((fn, rows))
# total_rows += rows

# # Create the VDS file
# with h5py.File(VDS_PATH, 'w', libver='latest') as f_out:
# # Open first file to get shapes
# with h5py.File(files[0], 'r') as f0:
# for k in f0.keys():
# shape = (total_rows,) + f0[k].shape[1:]
# dtype = f0[k].dtype

# # Create the virtual layout
# layout = h5py.VirtualLayout(shape=shape, dtype=dtype)

# current_idx = 0
# for fn, rows in sources:
# vsource = h5py.VirtualSource(fn, k, shape=(rows,) + shape[1:])
# layout[current_idx : current_idx + rows] = vsource
# current_idx += rows

# f_out.create_virtual_dataset(k, layout)

# print(f"Created Virtual Dataset at {VDS_PATH}")
