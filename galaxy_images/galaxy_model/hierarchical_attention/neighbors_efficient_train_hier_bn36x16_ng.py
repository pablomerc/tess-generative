"""
Training entry point for the bn_36x16 hierarchical variant with the
global physics conditioning path disabled.

Delegates to:
  python -m galaxy_images.galaxy_model.train --config galaxy_images/galaxy_model/configs/neighbors_efficient_hier_bn36x16_ng.json
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
# Force regular hipBLAS instead of hipBLASLt - hipBLASLt is buggy on MI210
# for certain matrix shapes and causes HIPBLAS_STATUS_INVALID_VALUE at runtime.
torch.backends.cuda.preferred_blas_library("hipblas")

from galaxy_images.galaxy_model.train import main as unified_train_main


def main() -> None:
    default_config = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "neighbors_efficient_hier_bn36x16_ng.json"
    )
    argv = ["--config", str(default_config), *sys.argv[1:]]
    unified_train_main(argv)


if __name__ == "__main__":
    main()
