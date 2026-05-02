"""
Extract per-pixel HSC ivar maps (cropped to 48x48, bands g/r/i/z) for the rows
already present in hsc_ivar_psf_stats.h5. Saved with hdf5_row_idx in identical
row order to the stats file so the two HDF5s line up index-for-index.

Output: /work1/jeroenaudenaert/pablomer/data/hsc_ivar_maps.h5
  - hdf5_row_idx   (N,)      int32
  - hsc_ivar_maps  (N,4,48,48) float32  (g/r/i/z)
"""

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from discord_notify import notify as _notify

PARQUET_DIR    = Path("/work1/jeroenaudenaert/pablomer/data/raw_mmu/data")
NEIGHBORS_HDF5 = "/work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5"
STATS_HDF5     = Path("/work1/jeroenaudenaert/pablomer/data/hsc_ivar_psf_stats.h5")
OUTPUT_DEFAULT = Path("/work1/jeroenaudenaert/pablomer/data/hsc_ivar_maps.h5")

PARQUET_COLS = ["hsc_object_id", "hsc_image"]
CROP_SIZE = 48
N_BANDS = 4  # g/r/i/z


def build_join_map(hdf5_path: str) -> dict:
    with h5py.File(hdf5_path, "r") as f:
        oids = f["object_id_hsc"][:]
    return {oid.decode(): i for i, oid in enumerate(oids)}


def _band_to_2d(band_data) -> np.ndarray:
    raw = np.asarray(band_data)
    if raw.dtype == object:
        return np.stack([np.asarray(r, dtype=np.float32) for r in raw])
    return raw.astype(np.float32)


def process_file(args):
    parquet_path, join_map, target_rows = args
    try:
        df = pd.read_parquet(parquet_path, columns=PARQUET_COLS, engine="pyarrow")
    except Exception as e:
        print(f"  ERROR reading {parquet_path}: {e}")
        return []

    out = []
    for row in df.itertuples(index=False):
        hdf5_row = join_map.get(row.hsc_object_id)
        if hdf5_row is None or hdf5_row not in target_rows:
            continue

        ivar = np.stack([_band_to_2d(row.hsc_image["ivar"][b]) for b in range(N_BANDS)])
        H, W = ivar.shape[1], ivar.shape[2]
        y0, x0 = (H - CROP_SIZE) // 2, (W - CROP_SIZE) // 2
        crop = ivar[:, y0:y0 + CROP_SIZE, x0:x0 + CROP_SIZE].astype(np.float32)
        out.append((hdf5_row, crop))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-dir", type=Path, default=PARQUET_DIR)
    parser.add_argument("--hdf5-path",   default=NEIGHBORS_HDF5)
    parser.add_argument("--stats-path",  type=Path, default=STATS_HDF5)
    parser.add_argument("--output",      type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--n-files",     type=int, default=None)
    parser.add_argument("--n-workers",   type=int, default=8)
    args = parser.parse_args()

    print(f"=== Loading target rows from {args.stats_path} ===")
    with h5py.File(args.stats_path, "r") as f:
        stats_rows = np.array(f["hdf5_row_idx"], dtype=np.int64)
        # stats file is sorted by hdf5_row_idx (extract_hsc_stats.py:137)
        ref_per_band_mean = np.array(f["hsc_ivar_per_band"], dtype=np.float64)  # (N,5)
    target_rows = set(int(r) for r in stats_rows)
    print(f"  {len(target_rows):,} target rows")

    print(f"=== Building join map from {args.hdf5_path} ===")
    join_map = build_join_map(args.hdf5_path)
    print(f"  join map: {len(join_map):,} entries")

    files = sorted(args.parquet_dir.glob("train-*.parquet"))
    if args.n_files is not None:
        files = files[:args.n_files]
    print(f"=== Processing {len(files)} parquet file(s) with {args.n_workers} workers ===")

    _notify(f"📥 ivar-map extract: starting parquet pass over {len(files)} files, {len(target_rows):,} target rows")

    pool_args = [(str(f), join_map, target_rows) for f in files]
    batches = []
    notify_every = max(1, len(files) // 10)
    with mp.Pool(args.n_workers) as pool:
        for i, batch in enumerate(pool.imap_unordered(process_file, pool_args), start=1):
            batches.append(batch)
            if i % notify_every == 0 or i == len(files):
                matched = sum(len(b) for b in batches)
                _notify(f"⏳ ivar-map extract: {i}/{len(files)} files done, {matched:,} rows matched")

    records = [r for batch in batches for r in batch]
    print(f"  Matched {len(records):,} / {len(target_rows):,} target rows")
    _notify(f"📊 ivar-map extract: matched {len(records):,}/{len(target_rows):,} rows; saving HDF5...")
    if not records:
        raise RuntimeError("No records extracted — check join key.")

    # sort by hdf5_row_idx (mirrors extract_hsc_stats.py:137)
    records.sort(key=lambda r: r[0])

    rows  = np.array([r[0] for r in records], dtype=np.int32)
    maps  = np.stack([r[1] for r in records], axis=0).astype(np.float32)
    print(f"  ivar_maps shape={maps.shape}  dtype={maps.dtype}")

    # alignment check
    if not np.array_equal(rows.astype(np.int64), stats_rows):
        diff = np.setdiff1d(stats_rows, rows.astype(np.int64))
        raise RuntimeError(
            f"Row alignment mismatch: {len(diff)} stats rows missing from extraction"
        )
    print("  Row alignment OK (matches stats file row-for-row)")

    # cross-check per-band means against existing scalar means
    extracted_per_band_mean = maps.mean(axis=(2, 3))  # (N, 4)
    diff_max = np.abs(extracted_per_band_mean - ref_per_band_mean[:, :4]).max()
    diff_med = np.median(np.abs(extracted_per_band_mean - ref_per_band_mean[:, :4]))
    print(f"  per-band-mean cross-check: max|Δ|={diff_max:.4g}  median|Δ|={diff_med:.4g}")
    if diff_max > 1e-2:
        print("  WARNING: cross-check Δ unexpectedly large — verify band ordering")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as f:
        f.create_dataset("hdf5_row_idx",  data=rows)
        f.create_dataset(
            "hsc_ivar_maps", data=maps,
            chunks=(min(len(rows), 256), N_BANDS, CROP_SIZE, CROP_SIZE),
            compression="gzip", compression_opts=4,
        )
        f.attrs["bands"] = ["g", "r", "i", "z"]
        f.attrs["n_examples"] = len(rows)
    print(f"\nSaved {args.output}  ({len(rows):,} rows)")
    _notify(f"💾 ivar-map extract: saved {args.output.name} ({len(rows):,} rows, max|Δ|={diff_max:.3g})")


if __name__ == "__main__":
    main()
