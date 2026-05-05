"""
Generic training entry point: forwards --config (and any --set overrides) to
galaxy_images.galaxy_model.train.

Usage:
    python -u galaxy_images/galaxy_model/generic_train.py \
        --config galaxy_images/galaxy_model/configs/<your_config>.json
"""

from __future__ import annotations

import sys

import torch
# Force regular hipBLAS instead of hipBLASLt - hipBLASLt is buggy on MI210
# for certain matrix shapes and causes HIPBLAS_STATUS_INVALID_VALUE at runtime.
torch.backends.cuda.preferred_blas_library("hipblas")

from galaxy_images.galaxy_model.train import main as unified_train_main


def main() -> None:
    unified_train_main(sys.argv[1:])


if __name__ == "__main__":
    main()
