import torch
import h5py
import time
from pathlib import Path

def calculate_dataset_stats(
    hdf5_path: str,
    batch_size: int = 1000,
):
    """
    Load images and scalars from HDF5, calculate mean/std, and print NORM_DICT.

    Args:
        hdf5_path: Path to HDF5 file containing 'images' and 'parameters'
        batch_size: Batch size for processing
    """
    hdf5_path = Path(hdf5_path)
    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    print(f"Loading data from {hdf5_path}...")
    start_time = time.time()

    with h5py.File(hdf5_path, 'r') as f:
        # Check available keys
        print(f"Keys found in file: {list(f.keys())}")

        # 1. LOAD IMAGES
        # Assuming images are stored as 'images' or similar key
        if 'images' in f:
            images = torch.from_numpy(f['images'][:]).float()
            print(f"Loaded images: {images.shape}")
        else:
            raise KeyError("Could not find 'images' key in HDF5 file.")

        # 2. LOAD SCALARS
        # Assuming scalars are stored as 'parameters'
        if 'parameters' in f:
            scalars = torch.from_numpy(f['parameters'][:]).float()
            print(f"Loaded scalars: {scalars.shape}")
        else:
            raise KeyError("Could not find 'parameters' key in HDF5 file.")

    load_time = time.time() - start_time
    print(f"Data loaded in {load_time:.2f} seconds")

    # --- CALCULATE IMAGE STATS ---
    print(f"\n=== CALCULATING IMAGE STATS ===")
    stats_start = time.time()

    # Global mean/std (across all images, pixels, and channels usually)
    img_mean = images.mean().item()
    img_std = images.std().item()

    print(f"Image Mean: {img_mean:.6f}")
    print(f"Image Std:  {img_std:.6f}")

    # --- CALCULATE SCALAR STATS ---
    print(f"\n=== CALCULATING SCALAR STATS ===")
    scalar_mean = scalars.mean(dim=0)
    scalar_std = scalars.std(dim=0)

    print(f"Scalar Means: {scalar_mean.tolist()}")
    print(f"Scalar Stds:  {scalar_std.tolist()}")

    stats_time = time.time() - stats_start
    print(f"Stats calculated in {stats_time:.2f} seconds")

    # --- PRINT FINAL DICTIONARY ---
    print(f"\n{'='*20} COPY THIS TO YOUR CODE {'='*20}")
    print("NORM_DICT = {")
    print(f"    'images': ({img_mean:.4f}, {img_std:.4f}),")

    # Formatting lists for cleaner output
    mean_str = "[" + ", ".join([f"{x:.4f}" for x in scalar_mean]) + "]"
    std_str = "[" + ", ".join([f"{x:.4f}" for x in scalar_std]) + "]"

    print(f"    'scalars': ({mean_str},")
    print(f"                {std_str})")
    print("}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    # Update this path to your actual data file
    data_path = "/data/vision/billf/scratch/pablomer/data/neighbours_v2.h5"

    calculate_dataset_stats(data_path)
