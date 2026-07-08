"""
Build a token-mean-pooled copy of an engaging-pipeline H5.

The base-config encoders emit (B, 4, 16) features which prepare_combined.py
flattens to (B, 64) before writing. This script reads such an H5, reshapes
each latent back to (B, T, D//T) where T=4, means across T, and writes a new
H5 with (B, D//T) latents — i.e. 4×16 → 16 per survey, and combined latents
(2×4×16) → (2×16).

CLI:

    python build_meanpool_h5.py \
        --in   outputs/embeddings/base.h5 \
        --out  outputs/embeddings/base-meanpool.h5 \
        [--tokens 4]

All labels and attrs are copied unchanged. The new architecture tag is
suffixed with `-meanpool` so downstream scripts can tell them apart.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


# Latent keys we need to pool. Every other dataset under each subset (the
# `labels/` group) is copied verbatim.
SINGLE_SURVEY_KEYS = ("hsc_e1", "hsc_e2", "legacy_e1", "legacy_e2")
COMBINED_KEYS = ("combined_e1", "combined_e2")


def _pool_single(arr: np.ndarray, tokens: int) -> np.ndarray:
    n, d = arr.shape
    if d % tokens != 0:
        raise ValueError(f"latent dim {d} is not divisible by tokens={tokens}")
    return arr.reshape(n, tokens, d // tokens).mean(axis=1).astype(np.float32)


def _pool_combined(arr: np.ndarray, tokens: int) -> np.ndarray:
    """combined latents are concat([hsc_flat, legacy_flat]) where each flat
    is (B, tokens*D). Pool each half independently and reconcatenate."""
    n, d = arr.shape
    if d % 2 != 0:
        raise ValueError(f"combined dim {d} is not even — cannot split halves")
    half = d // 2
    a = _pool_single(arr[:, :half], tokens)
    b = _pool_single(arr[:, half:], tokens)
    return np.concatenate([a, b], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True, type=Path)
    ap.add_argument("--out", dest="dst", required=True, type=Path)
    ap.add_argument("--tokens", type=int, default=4,
                    help="Number of tokens to mean across (default 4 → 4×16 → 16)")
    args = ap.parse_args()

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"Reading {args.src}")
    with h5py.File(args.src, "r") as fin, h5py.File(args.dst, "w") as fout:
        for k, v in fin.attrs.items():
            fout.attrs[k] = v
        # Tag the architecture so it's obvious downstream.
        arch = fin.attrs.get("architecture", "?")
        if isinstance(arch, bytes):
            arch = arch.decode()
        fout.attrs["architecture"] = f"{arch}-meanpool"
        ck = fin.attrs.get("checkpoint_name", args.src.stem)
        if isinstance(ck, bytes):
            ck = ck.decode()
        fout.attrs["checkpoint_name"] = f"{ck}-meanpool"
        fout.attrs["pool_tokens"] = int(args.tokens)

        for subset in fin.keys():
            grp_in = fin[subset]
            grp_out = fout.create_group(subset)
            for key in grp_in.keys():
                if key == "labels":
                    lbl_out = grp_out.create_group("labels")
                    for col in grp_in["labels"].keys():
                        lbl_out.create_dataset(
                            col,
                            data=grp_in["labels"][col][:],
                            compression="gzip",
                            compression_opts=4,
                        )
                    continue
                arr = np.asarray(grp_in[key][:])
                if key in SINGLE_SURVEY_KEYS:
                    pooled = _pool_single(arr, args.tokens)
                elif key in COMBINED_KEYS:
                    pooled = _pool_combined(arr, args.tokens)
                else:
                    pooled = arr
                grp_out.create_dataset(
                    key, data=pooled, compression="gzip", compression_opts=4
                )
                print(f"  {subset}/{key}: {arr.shape} -> {pooled.shape}")
    print(f"Wrote {args.dst}")


if __name__ == "__main__":
    main()
