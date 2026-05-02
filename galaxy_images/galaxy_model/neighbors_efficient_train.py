"""
Training entry point for the efficient-dataset (mmap + Parquet) neighbors variant.

You can run:
  python galaxy_images/galaxy_model/neighbors_efficient_train.py

Internally this delegates to:
  python -m galaxy_images.galaxy_model.train --config galaxy_images/galaxy_model/configs/neighbors_efficient.json

Pass any extra --set key=value overrides as normal CLI arguments, e.g.:
  python neighbors_efficient_train.py --set trainer.num_steps=100
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
# Force regular hipBLAS instead of hipBLASLt — hipBLASLt is buggy on MI210
# for certain matrix shapes and causes HIPBLAS_STATUS_INVALID_VALUE at runtime.
torch.backends.cuda.preferred_blas_library("hipblas")

from galaxy_images.galaxy_model.train import main as unified_train_main


def main() -> None:
    default_config = Path(__file__).resolve().parent / "configs" / "neighbors_efficient.json"
    argv = ["--config", str(default_config), *sys.argv[1:]]
    unified_train_main(argv)


if __name__ == "__main__":
    main()
