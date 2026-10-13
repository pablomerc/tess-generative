#!/usr/bin/env python3
"""Width control: PCA-reduce an embeddings H5 so arms of different latent width are
compared at matched probe input size.

Motivation: the `hsc-vit` arm produces 192-d embeddings while every other arm
(and the contrastive baseline) produces 64-d. The MLP probe's first layer sees 3x
more features, so a raw R² comparison confounds "better representation" with
"wider input". Reducing to 64 components and re-probing separates the two.

This mirrors what the paper already does for AION (`aion/hsc_mean_pca64`), so the
convention is established in this project.

PCA is fit per embedding key on the rows of that key (unsupervised, no labels
touched) and applied in place; the label datasets and group structure of
prepare_combined.py's output are copied through unchanged.

  python pca_reduce_embeddings.py --in jepa-hsc-vit.h5 --out jepa-hsc-vit-pca64.h5 -n 64
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from sklearn.decomposition import PCA

EMB_KEYS = ("hsc_e1", "hsc_e2", "legacy_e1", "legacy_e2", "combined_e1", "combined_e2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True, type=Path)
    ap.add_argument("--out", dest="dst", required=True, type=Path)
    ap.add_argument("-n", "--n-components", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with h5py.File(args.src, "r") as fin, h5py.File(args.dst, "w") as fout:
        for attr, val in fin.attrs.items():
            fout.attrs[attr] = val
        fout.attrs["pca_reduced_from"] = str(args.src.name)
        fout.attrs["pca_n_components"] = args.n_components

        def visit(name, obj):
            if not isinstance(obj, h5py.Dataset):
                return
            arr = obj[()]
            leaf = name.rsplit("/", 1)[-1]
            if leaf in EMB_KEYS and arr.ndim == 2 and arr.shape[1] > args.n_components:
                k = min(args.n_components, arr.shape[0], arr.shape[1])
                p = PCA(n_components=k, random_state=args.seed)
                red = p.fit_transform(arr).astype(np.float32)
                ev = float(p.explained_variance_ratio_.sum())
                print(f"  {name}: {arr.shape} -> {red.shape}  "
                      f"explained_var={ev:.4f}")
                fout.create_dataset(name, data=red)
            else:
                fout.create_dataset(name, data=arr)

        print(f"[pca] {args.src.name} -> {args.dst.name} (n_components={args.n_components})")
        fin.visititems(visit)

    print(f"[pca] wrote {args.dst}")


if __name__ == "__main__":
    main()
