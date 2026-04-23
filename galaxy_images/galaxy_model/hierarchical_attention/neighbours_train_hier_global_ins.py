"""
Backward-compatible wrapper around the unified trainer for the hierarchical
global-instrument neighbors variant.

You can run:
  python galaxy_images/galaxy_model/hierarchical_attention/neighbours_train_hier_global_ins.py

Internally this delegates to:
  python -m galaxy_images.galaxy_model.train --config galaxy_images/galaxy_model/configs/neighbors_hier_global_ins.json
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

torch.backends.cuda.preferred_blas_library("hipblas")

from galaxy_images.galaxy_model.train import main as unified_train_main


def main() -> None:
    default_config = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "neighbors_hier_global_ins.json"
    )
    argv = ["--config", str(default_config), *sys.argv[1:]]
    unified_train_main(argv)


if __name__ == "__main__":
    main()
