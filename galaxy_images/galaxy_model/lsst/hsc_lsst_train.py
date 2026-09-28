"""Entry point for the HSC x LSST dual-encoder run.

Thin wrapper, same pattern as galaxy_images/galaxy_model/neighbors_efficient_train.py:
apply the AMD hipBLASLt workaround (must happen in the Python process, the env var
in the sbatch only covers child processes), then delegate to the unified trainer
with configs/hsc_lsst_cosmos.json. Any --set key=value overrides pass through:

    PYTHONPATH=. python -m galaxy_images.galaxy_model.lsst.hsc_lsst_train \
        --set trainer.num_steps=60 --set wandb.enabled=false

Normally launched by lsst/hsc_lsst_train.sbatch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# hipBLASLt is buggy on MI210 for some matrix shapes (HIPBLAS_STATUS_INVALID_VALUE).
# Guarded so the same file runs unchanged on NVIDIA clusters (see CLAUDE.md).
try:
    torch.backends.cuda.preferred_blas_library("hipblas")
except Exception:
    pass

from galaxy_images.galaxy_model.train import main as unified_train_main

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "hsc_lsst_cosmos.json"


def main() -> None:
    unified_train_main(["--config", str(CONFIG), *sys.argv[1:]])


if __name__ == "__main__":
    main()
