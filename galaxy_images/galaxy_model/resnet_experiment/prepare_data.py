"""
Preprocess HSC parquet files into a single H5 for ResNet ellipticity training.

For each row in each parquet file:
  - Extracts 4-channel HSC image (G, R, I, Z bands), center-crops to 48x48
  - Stores raw flux (no normalization — done on-the-fly in the dataset)
  - Stores SHAPE_E1, SHAPE_E2 labels
  - Stores legacysurvey_object_id (used for Stage 2 crossmatch)

Output: resnet_data.h5 with datasets:
  hsc_images   (N, 4, 48, 48)  float32   raw flux
  shape_e1     (N,)             float32
  shape_e2     (N,)             float32
  ls_object_id (N,)             bytes     legacysurvey_object_id

Run from this directory or from project root:
  python galaxy_images/galaxy_model/resnet_experiment/prepare_data.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import h5py

_here = Path(__file__).resolve().parent

PARQUET_DIR = Path("/data/vision/billf/scratch/pablomer/legacysurvey_hsc/data")
OUTPUT_H5 = _here / "resnet_data.h5"
N_PARQUET_FILES = 396
CROP_SIZE = 48  # center crop size in pixels


def extract_hsc_image(hsc_image_dict):
    """
    Extract 4-channel (G, R, I, Z) image from the hsc_image dict in a parquet row.

    The parquet stores each band's flux as a list of 160 1D arrays of length 160,
    forming a 160x160 image.  We skip band 4 (Y) to get 4 channels matching the
    existing preprocessing convention (see data.py).

    Returns:
        np.ndarray of shape (4, CROP_SIZE, CROP_SIZE), dtype float32
        or None if the image is invalid.
    """
    flux = hsc_image_dict["flux"]
    if flux is None or len(flux) < 4:
        return None

    bands = []
    for b in range(4):  # G, R, I, Z  (skip Y = band 4)
        rows = flux[b]
        if rows is None:
            return None
        band_2d = np.stack(rows).astype(np.float32)  # (160, 160)
        bands.append(band_2d)

    img = np.stack(bands, axis=0)  # (4, 160, 160)

    # Check for NaN/inf
    if not np.all(np.isfinite(img)):
        return None

    # Center crop to CROP_SIZE x CROP_SIZE
    h, w = img.shape[1], img.shape[2]
    sh = (h - CROP_SIZE) // 2
    sw = (w - CROP_SIZE) // 2
    img = img[:, sh : sh + CROP_SIZE, sw : sw + CROP_SIZE]

    return img


def process_parquets():
    parquet_files = sorted(PARQUET_DIR.glob("train-*-of-00396.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {PARQUET_DIR}")
    print(f"Found {len(parquet_files)} parquet files")

    all_images = []
    all_e1 = []
    all_e2 = []
    all_ls_ids = []
    skipped = 0

    for file_idx, fpath in enumerate(parquet_files):
        if file_idx % 50 == 0:
            print(f"  Processing file {file_idx}/{len(parquet_files)}  "
                  f"(collected {len(all_images)} so far, skipped {skipped})")

        df = pd.read_parquet(fpath)

        for _, row in df.iterrows():
            hsc_dict = row["hsc_image"]
            if hsc_dict is None:
                skipped += 1
                continue

            img = extract_hsc_image(hsc_dict)
            if img is None:
                skipped += 1
                continue

            all_images.append(img)
            all_e1.append(float(row["SHAPE_E1"]))
            all_e2.append(float(row["SHAPE_E2"]))
            # legacysurvey_object_id is an int stored as string/bytes
            ls_id = row.get("legacysurvey_object_id", b"")
            if ls_id is None:
                ls_id = b""
            if isinstance(ls_id, str):
                ls_id = ls_id.encode()
            elif isinstance(ls_id, (int, np.integer)):
                ls_id = str(int(ls_id)).encode()
            all_ls_ids.append(ls_id)

    print(f"Done: collected {len(all_images)} examples, skipped {skipped}")

    # Convert to arrays
    images_np = np.stack(all_images, axis=0)  # (N, 4, 48, 48)
    e1_np = np.array(all_e1, dtype=np.float32)
    e2_np = np.array(all_e2, dtype=np.float32)
    ids_np = np.array(all_ls_ids, dtype=object)

    print(f"images shape: {images_np.shape}  dtype: {images_np.dtype}")
    print(f"e1 range: [{e1_np.min():.4f}, {e1_np.max():.4f}]")
    print(f"e2 range: [{e2_np.min():.4f}, {e2_np.max():.4f}]")

    print(f"Saving to {OUTPUT_H5}")
    with h5py.File(OUTPUT_H5, "w") as f:
        f.create_dataset("hsc_images", data=images_np, compression="gzip", compression_opts=4)
        f.create_dataset("shape_e1", data=e1_np, compression="gzip", compression_opts=4)
        f.create_dataset("shape_e2", data=e2_np, compression="gzip", compression_opts=4)
        f.create_dataset("ls_object_id", data=ids_np, compression="gzip", compression_opts=4)
        f.attrs["num_examples"] = len(all_images)
        f.attrs["crop_size"] = CROP_SIZE
        f.attrs["bands"] = "HSC-G,HSC-R,HSC-I,HSC-Z"
        f.attrs["normalization"] = "none (raw flux — normalize in dataset)"
        f.attrs["norm_mean"] = 0.022
        f.attrs["norm_std"] = 0.05
    print(f"Saved {OUTPUT_H5}")


if __name__ == "__main__":
    process_parquets()
