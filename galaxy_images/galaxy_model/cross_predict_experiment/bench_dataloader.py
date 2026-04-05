"""
Benchmark the new lean CrossPredictDataset.
Measures: init time, single __getitem__, and batch iteration throughput.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

_script_dir = Path(__file__).resolve().parent
_model_dir = _script_dir.parent
_project_root = _model_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from cross_predict import CrossPredictDataset, NEIGHBORS_HDF5

N_BATCHES = 20
BATCH_SIZE = 64


def time_getitem(dataset, n_samples=20):
    times = []
    for i in range(n_samples):
        t0 = time.time()
        _ = dataset[i]
        times.append(time.time() - t0)
    return times


def time_dataloader(dataset, n_batches=N_BATCHES):
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    it = iter(loader)

    t0 = time.time()
    next(it)
    warmup_time = time.time() - t0

    t0 = time.time()
    count = 0
    for _ in range(n_batches):
        try:
            next(it)
            count += 1
        except StopIteration:
            break
    elapsed = time.time() - t0
    return warmup_time, elapsed, count


def main():
    print("=" * 60)
    print("LEAN DATALOADER BENCHMARK")
    print("=" * 60)

    for direction in ["hsc_to_legacy", "legacy_to_hsc"]:
        print(f"\n--- Direction: {direction} ---")

        t0 = time.time()
        ds = CrossPredictDataset(hdf5_path=NEIGHBORS_HDF5, direction=direction)
        init_time = time.time() - t0
        print(f"  Init: {init_time:.2f}s, {len(ds)} samples, {len(ds.all_target_names)} targets")

        n_subset = min(5000, len(ds))
        subset = Subset(ds, list(range(n_subset)))

        # Single item
        times = time_getitem(ds, 20)
        print(f"  __getitem__ (20 calls): mean={np.mean(times)*1000:.1f}ms, "
              f"min={np.min(times)*1000:.1f}ms, max={np.max(times)*1000:.1f}ms")

        # Batch throughput
        warmup, elapsed, count = time_dataloader(subset, n_batches=N_BATCHES)
        sps = (count * BATCH_SIZE) / elapsed if elapsed > 0 else 0
        print(f"  DataLoader ({count} batches): warmup={warmup:.2f}s, "
              f"{elapsed:.2f}s, {sps:.0f} samples/sec")

        # Epoch estimate
        n_train = int(len(ds) * 0.9)
        epoch_batches = n_train // BATCH_SIZE
        est_epoch_sec = epoch_batches * (elapsed / max(count, 1))
        print(f"  Estimated epoch time: {est_epoch_sec/60:.1f} min "
              f"({epoch_batches} batches, {n_train} train samples)")

    print("\n" + "=" * 60)
    print("DONE")


if __name__ == "__main__":
    main()
