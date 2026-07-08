"""
One-off baseline encoder: random-init ResNet-18 (4-band input, no FC head).

Writes a single H5 with the same layout as prepare_combined.py so
predict_combined.py can probe it without modification. Produces a 512-dim
embedding per image. Used as the "Untrained ResNet-18" reference column
in every per-variant publication plot — compute once, reuse forever.

Usage (from galaxy_model/):
    python downstream_evaluation/engaging/prepare_untrained_resnet.py \
        --index-dir downstream_evaluation/engaging/outputs/index \
        --out       downstream_evaluation/engaging/outputs/embeddings/untrained-resnet.h5
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import torch
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "preferred_blas_library"):
        torch.backends.cuda.preferred_blas_library("hipblas")
except Exception:
    pass

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.models import resnet18

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
from prepare_combined import (
    _BinaryImageDataset,
    _extract_labels,
    _encode_dataset,
    HSC_BANDS, HSC_H, HSC_W, HSC_BYTES,
    LEGACY_BANDS, LEGACY_H, LEGACY_W, LEGACY_BYTES,
)


class _RandomResNet(nn.Module):
    """torchvision ResNet-18 with random weights, 4-band input, FC stripped."""

    def __init__(self, in_channels: int = 4, seed: int = 0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        # Construct with no pretrained weights — random init with default (Kaiming) scheme.
        m = resnet18(weights=None)
        m.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Re-init the new conv1 deterministically so reruns give identical baselines.
        with torch.no_grad():
            nn.init.kaiming_normal_(m.conv1.weight, mode="fan_out", nonlinearity="relu", generator=gen)
        m.fc = nn.Identity()
        self.encoder = m

    def forward(self, x):
        return self.encoder(x)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--hsc-dir", type=Path,
                   default=Path("/home/pablomer/orcd/scratch/hsc_downstream"))
    p.add_argument("--legacy-dir", type=Path,
                   default=Path("/home/pablomer/orcd/scratch/legacy_downstream_full/full_1M"))
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(args.seed)
    model = _RandomResNet(in_channels=4, seed=args.seed).to(device).eval()
    torch.set_grad_enabled(False)
    print(f"Random ResNet-18 (seed={args.seed})  output_dim=512")

    idx_dir = args.index_dir
    overlap_path = idx_dir / "overlap.parquet"
    hsc_only_path = idx_dir / "hsc_only.parquet"
    legacy_only_path = idx_dir / "legacy_only.parquet"

    overlap_df = pd.read_parquet(overlap_path) if overlap_path.exists() else pd.DataFrame()
    hsc_only_df = pd.read_parquet(hsc_only_path) if hsc_only_path.exists() else pd.DataFrame()
    legacy_only_df = pd.read_parquet(legacy_only_path) if legacy_only_path.exists() else pd.DataFrame()

    if args.smoke:
        if len(overlap_df):     overlap_df    = overlap_df.head(512).reset_index(drop=True)
        if len(hsc_only_df):    hsc_only_df   = hsc_only_df.head(512).reset_index(drop=True)
        if len(legacy_only_df): legacy_only_df = legacy_only_df.head(512).reset_index(drop=True)

    print(f"Subsets: overlap={len(overlap_df)}  hsc_only={len(hsc_only_df)}  legacy_only={len(legacy_only_df)}")

    hsc_bin = args.hsc_dir / "hsc_flux.bin"
    leg_bin = args.legacy_dir / "legacy_flux.bin"
    hsc_cat = args.hsc_dir / "catalog.parquet"
    leg_cat = args.legacy_dir / "catalog.parquet"

    n_hsc_bin = os.path.getsize(hsc_bin) // HSC_BYTES
    n_leg_bin = os.path.getsize(leg_bin) // LEGACY_BYTES
    print(f"HSC bin rows: {n_hsc_bin:,}   Legacy bin rows: {n_leg_bin:,}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    has_e2 = False
    with h5py.File(args.out, "w") as fout:
        fout.attrs["checkpoint_path"] = "untrained-resnet18-randominit"
        fout.attrs["checkpoint_name"] = "untrained-resnet"
        fout.attrs["architecture"] = "random_resnet18"
        fout.attrs["has_e2"] = bool(has_e2)
        fout.attrs["embedding_names"] = ["e1"]
        per_subset_n = {}
        per_subset_labels = {}

        if len(overlap_df):
            print("\n[overlap] encoding HSC images")
            hsc_indices = overlap_df["hsc_image_idx"].to_numpy(dtype=np.int64)
            hsc_ds = _BinaryImageDataset(hsc_bin, hsc_indices, HSC_BANDS, HSC_H, HSC_W, "hsc")
            hsc_e1, _ = _encode_dataset(model, hsc_ds, device, args.batch_size, has_e2=False)

            print("[overlap] encoding Legacy images")
            leg_indices = overlap_df["legacy_image_idx"].to_numpy(dtype=np.int64)
            leg_ds = _BinaryImageDataset(leg_bin, leg_indices, LEGACY_BANDS, LEGACY_H, LEGACY_W, "legacy")
            leg_e1, _ = _encode_dataset(model, leg_ds, device, args.batch_size, has_e2=False)

            grp = fout.create_group("overlap")
            grp.create_dataset("hsc_e1", data=hsc_e1, compression="gzip", compression_opts=4)
            grp.create_dataset("legacy_e1", data=leg_e1, compression="gzip", compression_opts=4)
            grp.create_dataset("combined_e1", data=np.concatenate([hsc_e1, leg_e1], axis=1),
                               compression="gzip", compression_opts=4)

            print("[overlap] writing labels")
            hsc_labels = _extract_labels(hsc_cat, hsc_indices, "hsc", "hsc_")
            leg_labels = _extract_labels(leg_cat, leg_indices, "legacy", "legacy_")
            for k, v in {**hsc_labels, **leg_labels}.items():
                grp.create_dataset(f"labels/{k}", data=v, compression="gzip", compression_opts=4)
            per_subset_n["overlap"] = len(overlap_df)
            per_subset_labels["overlap"] = list(hsc_labels.keys()) + list(leg_labels.keys())

        if len(hsc_only_df):
            print("\n[hsc_only] encoding HSC images")
            hsc_indices = hsc_only_df["hsc_image_idx"].to_numpy(dtype=np.int64)
            hsc_ds = _BinaryImageDataset(hsc_bin, hsc_indices, HSC_BANDS, HSC_H, HSC_W, "hsc")
            hsc_e1, _ = _encode_dataset(model, hsc_ds, device, args.batch_size, has_e2=False)
            grp = fout.create_group("hsc_only")
            grp.create_dataset("hsc_e1", data=hsc_e1, compression="gzip", compression_opts=4)
            hsc_labels = _extract_labels(hsc_cat, hsc_indices, "hsc", "hsc_")
            for k, v in hsc_labels.items():
                grp.create_dataset(f"labels/{k}", data=v, compression="gzip", compression_opts=4)
            per_subset_n["hsc_only"] = len(hsc_only_df)
            per_subset_labels["hsc_only"] = list(hsc_labels.keys())

        if len(legacy_only_df):
            print("\n[legacy_only] encoding Legacy images")
            leg_indices = legacy_only_df["legacy_image_idx"].to_numpy(dtype=np.int64)
            leg_ds = _BinaryImageDataset(leg_bin, leg_indices, LEGACY_BANDS, LEGACY_H, LEGACY_W, "legacy")
            leg_e1, _ = _encode_dataset(model, leg_ds, device, args.batch_size, has_e2=False)
            grp = fout.create_group("legacy_only")
            grp.create_dataset("legacy_e1", data=leg_e1, compression="gzip", compression_opts=4)
            leg_labels = _extract_labels(leg_cat, leg_indices, "legacy", "legacy_")
            for k, v in leg_labels.items():
                grp.create_dataset(f"labels/{k}", data=v, compression="gzip", compression_opts=4)
            per_subset_n["legacy_only"] = len(legacy_only_df)
            per_subset_labels["legacy_only"] = list(leg_labels.keys())

        import json
        fout.attrs["num_examples_per_subset"] = json.dumps(per_subset_n)
        fout.attrs["label_columns_per_subset"] = json.dumps(per_subset_labels)

    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
