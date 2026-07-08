"""
Untrained baseline that **matches the base checkpoint's encoder shape** —
i.e. timm resnet18 features_only + 1×1 conv projection to cross_attention_dim,
yielding (B, 4, 16) tokens like `ResNetEncoder` in
double_train_fm_neighbors.py. Random-init (no pretraining), frozen, eval mode.

Usage (from galaxy_model/):
    python downstream_evaluation/engaging/prepare_untrained_resnet_matched.py \
        --index-dir downstream_evaluation/engaging/outputs/index \
        --out       downstream_evaluation/engaging/outputs/embeddings/untrained-resnet-matched.h5

Differences vs. prepare_untrained_resnet.py:
  - Per-image latent is (B, 4, 16) → flatten → (B, 64)  (was 512-dim avgpool).
  - Two encoders (encoder_1, encoder_2) with independent seeds, mirroring the
    physics + instrument encoders in the base config. has_e2=True.
  - Encodes only the `overlap` subset (the only one predict_combined.py
    actually probes).
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

_here = Path(__file__).resolve().parent
_galaxy_model = _here.parents[1]
_repo_root = _galaxy_model.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from galaxy_images.galaxy_model.double_train_fm_neighbors import ResNetEncoder  # noqa: E402

sys.path.insert(0, str(_here))
from prepare_combined import (  # noqa: E402
    _BinaryImageDataset,
    _extract_labels,
    _encode_dataset,
    HSC_BANDS, HSC_H, HSC_W, HSC_BYTES,
    LEGACY_BANDS, LEGACY_H, LEGACY_W, LEGACY_BYTES,
)


class _MatchedRandomEncoder(nn.Module):
    """Wraps two random-init ResNetEncoders with cross_attention_dim=16,
    each producing (B, 4, 16). Matches the base config's encoder_1 / encoder_2."""

    def __init__(self, in_channels: int, cross_attention_dim: int, seed_e1: int, seed_e2: int):
        super().__init__()
        # Two separate random generators so encoder_1 and encoder_2 differ.
        torch.manual_seed(seed_e1)
        self.encoder_1 = ResNetEncoder(
            in_channels=in_channels,
            cross_attention_dim=cross_attention_dim,
            pretrained=False,
            mean_pool=False,
        )
        torch.manual_seed(seed_e2)
        self.encoder_2 = ResNetEncoder(
            in_channels=in_channels,
            cross_attention_dim=cross_attention_dim,
            pretrained=False,
            mean_pool=False,
        )


@torch.no_grad()
def _encode_pair(model: _MatchedRandomEncoder, dataset, device, batch_size):
    """Encode a dataset through both encoders, return (e1_flat, e2_flat) as numpy.
    Each encoder yields (B, T, D) tokens; we flatten to (B, T*D) to match the
    existing H5 layout used by predict_combined.py."""
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        pin_memory=(device.type == "cuda"))
    e1_chunks, e2_chunks = [], []
    for imgs in loader:
        imgs = imgs.to(device, non_blocking=(device.type == "cuda"))
        z1 = model.encoder_1(imgs).flatten(start_dim=1)
        z2 = model.encoder_2(imgs).flatten(start_dim=1)
        e1_chunks.append(z1.float().cpu())
        e2_chunks.append(z2.float().cpu())
    e1 = torch.cat(e1_chunks, dim=0).numpy().astype(np.float32)
    e2 = torch.cat(e2_chunks, dim=0).numpy().astype(np.float32)
    return e1, e2


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--hsc-dir", type=Path,
                   default=Path("/home/pablomer/orcd/scratch/hsc_downstream"))
    p.add_argument("--legacy-dir", type=Path,
                   default=Path("/home/pablomer/orcd/scratch/legacy_downstream_full/full_1M"))
    p.add_argument("--cross-attention-dim", type=int, default=16,
                   help="Token dim — must match the trained-model checkpoint (default 16)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed-e1", type=int, default=0)
    p.add_argument("--seed-e2", type=int, default=1)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = _MatchedRandomEncoder(
        in_channels=4,
        cross_attention_dim=args.cross_attention_dim,
        seed_e1=args.seed_e1,
        seed_e2=args.seed_e2,
    ).to(device).eval()
    torch.set_grad_enabled(False)
    # Probe shapes once.
    dummy = torch.zeros(1, 4, 48, 48, device=device)
    z1 = model.encoder_1(dummy)
    z2 = model.encoder_2(dummy)
    print(f"encoder_1 output {tuple(z1.shape)} -> flat {tuple(z1.flatten(start_dim=1).shape)}")
    print(f"encoder_2 output {tuple(z2.shape)} -> flat {tuple(z2.flatten(start_dim=1).shape)}")

    overlap_path = args.index_dir / "overlap.parquet"
    overlap_df = pd.read_parquet(overlap_path) if overlap_path.exists() else pd.DataFrame()
    print(f"Subsets: overlap={len(overlap_df)}  (hsc_only / legacy_only intentionally skipped)")

    if not len(overlap_df):
        raise SystemExit("overlap.parquet missing or empty — nothing to encode.")

    hsc_bin = args.hsc_dir / "hsc_flux.bin"
    leg_bin = args.legacy_dir / "legacy_flux.bin"
    hsc_cat = args.hsc_dir / "catalog.parquet"
    leg_cat = args.legacy_dir / "catalog.parquet"

    n_hsc_bin = os.path.getsize(hsc_bin) // HSC_BYTES
    n_leg_bin = os.path.getsize(leg_bin) // LEGACY_BYTES
    print(f"HSC bin rows: {n_hsc_bin:,}   Legacy bin rows: {n_leg_bin:,}")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print("\n[overlap] encoding HSC images")
    hsc_indices = overlap_df["hsc_image_idx"].to_numpy(dtype=np.int64)
    hsc_ds = _BinaryImageDataset(hsc_bin, hsc_indices, HSC_BANDS, HSC_H, HSC_W, "hsc")
    hsc_e1, hsc_e2 = _encode_pair(model, hsc_ds, device, args.batch_size)

    print("[overlap] encoding Legacy images")
    leg_indices = overlap_df["legacy_image_idx"].to_numpy(dtype=np.int64)
    leg_ds = _BinaryImageDataset(leg_bin, leg_indices, LEGACY_BANDS, LEGACY_H, LEGACY_W, "legacy")
    leg_e1, leg_e2 = _encode_pair(model, leg_ds, device, args.batch_size)

    with h5py.File(args.out, "w") as fout:
        fout.attrs["checkpoint_path"] = "untrained-resnet-matched-randominit"
        fout.attrs["checkpoint_name"] = "untrained-resnet-matched"
        fout.attrs["architecture"] = "random_resnet18_matched"
        fout.attrs["has_e2"] = True
        fout.attrs["embedding_names"] = ["e1", "e2"]
        fout.attrs["cross_attention_dim"] = int(args.cross_attention_dim)
        fout.attrs["seed_e1"] = int(args.seed_e1)
        fout.attrs["seed_e2"] = int(args.seed_e2)

        grp = fout.create_group("overlap")
        grp.create_dataset("hsc_e1", data=hsc_e1, compression="gzip", compression_opts=4)
        grp.create_dataset("hsc_e2", data=hsc_e2, compression="gzip", compression_opts=4)
        grp.create_dataset("legacy_e1", data=leg_e1, compression="gzip", compression_opts=4)
        grp.create_dataset("legacy_e2", data=leg_e2, compression="gzip", compression_opts=4)
        grp.create_dataset("combined_e1", data=np.concatenate([hsc_e1, leg_e1], axis=1),
                           compression="gzip", compression_opts=4)
        grp.create_dataset("combined_e2", data=np.concatenate([hsc_e2, leg_e2], axis=1),
                           compression="gzip", compression_opts=4)

        print("[overlap] writing labels")
        hsc_labels = _extract_labels(hsc_cat, hsc_indices, "hsc", "hsc_")
        leg_labels = _extract_labels(leg_cat, leg_indices, "legacy", "legacy_")
        for k, v in {**hsc_labels, **leg_labels}.items():
            grp.create_dataset(f"labels/{k}", data=v, compression="gzip", compression_opts=4)

        import json
        fout.attrs["num_examples_per_subset"] = json.dumps({"overlap": len(overlap_df)})
        fout.attrs["label_columns_per_subset"] = json.dumps(
            {"overlap": list(hsc_labels.keys()) + list(leg_labels.keys())}
        )

    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
