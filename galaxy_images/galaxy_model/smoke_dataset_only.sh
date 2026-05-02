#!/bin/bash
# Dataset-only smoke test: exercises the mmap + parquet layout and the
# NeighborsEfficientDataset wrapper (no model, no GPU needed).
#
# Usage:
#   bash galaxy_images/galaxy_model/smoke_dataset_only.sh

set -eo pipefail

source ~/.bashrc
conda activate torchenv

cd /work1/jeroenaudenaert/pablomer/tess-generative
export PYTHONPATH=/work1/jeroenaudenaert/pablomer/tess-generative:$PYTHONPATH

echo "=== [1/2] Standalone dataset smoke test (shipped with the data) ==="
python /work1/jeroenaudenaert/pablomer/data/neighbors_efficient/triplet_dataset.py \
    /work1/jeroenaudenaert/pablomer/data/neighbors_efficient

echo ""
echo "=== [2/2] NeighborsEfficientDataset wrapper smoke test ==="
python - <<'PY'
import time
import torch
from torch.utils.data import DataLoader
from galaxy_images.galaxy_model.neighbors_efficient import NeighborsEfficientDataset
from galaxy_images.galaxy_model.neighbors import collate_neighbors

ds = NeighborsEfficientDataset(
    data_dir="/work1/jeroenaudenaert/pablomer/data/neighbors_efficient",
    crop_size=48,
    max_neighbors=5,
)
print(f"Dataset length: {len(ds):,}")

target, samegal, sameins, meta = ds[0]
print(f"Sample 0: target={tuple(target.shape)} samegal={tuple(samegal.shape)} "
      f"sameins={tuple(sameins.shape)} meta={meta}")

loader = DataLoader(ds, batch_size=4, num_workers=2, collate_fn=collate_neighbors)
t0 = time.time()
batch = next(iter(loader))
dt = time.time() - t0
print(f"First batch ready in {dt:.2f}s")
print(f"Batch keys: {list(batch.keys()) if isinstance(batch, dict) else type(batch)}")
PY

echo "Dataset smoke test completed at $(date)"
