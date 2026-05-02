"""
Training entry point for the single-encoder cross-survey ablation.

Usage:
  python galaxy_images/galaxy_model/single_encoder_ablation/train.py

Or with overrides:
  python galaxy_images/galaxy_model/single_encoder_ablation/train.py \\
      --set trainer.num_steps=100000 --set wandb.name=my-run
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
torch.backends.cuda.preferred_blas_library("hipblas")

# Register the single_encoder variant before the unified train.main parses the config.
from galaxy_images.galaxy_model.variants import VARIANTS, VariantSpec
from galaxy_images.galaxy_model.single_encoder_ablation.model import SingleEncoderFlowMatchingModule

VARIANTS["single_encoder"] = VariantSpec(
    name="single_encoder",
    description="Single-encoder cross-survey flow matching (no same-instrument path).",
    model_cls=SingleEncoderFlowMatchingModule,
)

from galaxy_images.galaxy_model.train import main as unified_train_main


def main() -> None:
    default_config = Path(__file__).resolve().parent / "config.json"
    argv = ["--config", str(default_config), *sys.argv[1:]]
    unified_train_main(argv)


if __name__ == "__main__":
    main()
