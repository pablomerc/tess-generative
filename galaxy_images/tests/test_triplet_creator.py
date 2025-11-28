"""
Quick sanity checks and visualization helpers for galaxy triplets.

This mirrors the diagnostic utilities in ``double-encoder-model/triplet_creation.py``
so we can sanity-check the parquet-backed TripletCreator without spinning up the
full training pipeline.
"""

from __future__ import annotations

import argparse
import os
from typing import Tuple

import matplotlib.pyplot as plt
import torch

# Handle imports whether running from parent dir or inside galaxy_images/
try:
    from galaxy_images.galaxy_triplets import TripletCreator
except ImportError:
    # If running from inside galaxy_images/, use relative import
    from galaxy_triplets import TripletCreator


def _to_rgb(image_tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert a 4-band (g,r,i,z) tensor to pseudo-RGB for plotting.
    Uses g,r,i as RGB; clamps to [0,1].
    """

    if image_tensor.shape[0] < 3:
        raise ValueError("Need at least 3 channels to build an RGB preview.")
    rgb = image_tensor[:3]
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    return rgb.permute(1, 2, 0)


def run_triplet_test(dataset_path: str, num_samples: int, anchor_survey: str) -> None:
    creator = TripletCreator(dataset_path=dataset_path)

    print(f"Dataset contains {creator.num_rows} records across {len(creator.files)} shards.")

    fig, axes = plt.subplots(num_samples, 3, figsize=(10, 3 * num_samples))
    if num_samples == 1:
        axes = axes[None, ...]  # normalize indexing

    for row in range(num_samples):
        triplet = creator.create_triplet(anchor_survey=anchor_survey)

        for key in ("ground_truth", "different_galaxy", "cross_instrument"):
            assert key in triplet, f"Missing {key} in triplet output"
            assert isinstance(triplet[key], torch.Tensor), f"{key} is not a tensor"
            assert triplet[key].shape[0] >= 3, f"{key} has unexpected shape {triplet[key].shape}"

        axes[row, 0].imshow(_to_rgb(triplet["ground_truth"]))
        axes[row, 0].set_title(f"Anchor ({anchor_survey})")
        axes[row, 1].imshow(_to_rgb(triplet["different_galaxy"]))
        axes[row, 1].set_title("Different galaxy (same survey)")
        axes[row, 2].imshow(_to_rgb(triplet["cross_instrument"]))
        axes[row, 2].set_title("Same galaxy (cross instrument)")

        for col in range(3):
            axes[row, col].axis("off")

    plt.tight_layout()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Sanity-check TripletCreator outputs.")
    parser.add_argument(
        "--dataset-path",
        default="/mnt/scratch/legacysurvey_hsc_crossmatched/data",
        help="Directory containing parquet shards.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of triplets to visualize.",
    )
    parser.add_argument(
        "--anchor-survey",
        choices=["legacysurvey", "hsc"],
        default="legacysurvey",
        help="Which survey to treat as the anchor image.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset path '{args.dataset_path}' does not exist.")
    run_triplet_test(args.dataset_path, args.num_samples, args.anchor_survey)
