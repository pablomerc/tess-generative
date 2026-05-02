"""Build a gallery latents HDF5 for the hierarchical model in a chosen latent mode.

Mirrors the schema produced by neighbor_search/make_latents_all.py (the "original"
format consumed by search_lens_neighbors.py): keys `idx`, `index_mmu`,
`physics_embedding`, `legacy_physics_embedding`. Only the physics latent is saved
(instrument is unused by search_lens_neighbors.py for hierarchical NN search).

Usage:
  python -m galaxy_images.galaxy_model.neighbor_search_lenses.make_latents_hier \
      --checkpoint galaxy_images/galaxy_model/checkpoints/hier-small/snapshot.ckpt \
      --latent-mode spatial_flat \
      --output galaxy_images/galaxy_model/neighbor_search_lenses/galleries/hier_small/gallery_spatial_flat.h5 \
      --max-examples 110000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader, Subset

_THIS = Path(__file__).resolve()
_PKG_ROOT = _THIS.parents[4]  # tess-generative
_GALAXY_MODEL = _THIS.parents[1]  # galaxy_model
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))
if str(_GALAXY_MODEL) not in sys.path:
    sys.path.insert(0, str(_GALAXY_MODEL))

try:
    torch.backends.cuda.preferred_blas_library("hipblas")
except Exception:
    pass

from galaxy_images.galaxy_model.hierarchical_latent_experiments.common.ckpt import (
    load_trained,
)
from galaxy_images.galaxy_model.hierarchical_latent_experiments.common.latents import (
    extract_physics,
)

DEFAULT_NEIGHBORS_H5 = "/work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5"
DEFAULT_MAX_EXAMPLES = 110_000


def _collate(batch):
    hsc = torch.stack([b[0] for b in batch])
    leg = torch.stack([b[1] for b in batch])
    meta = [b[2] for b in batch]
    return hsc, leg, meta


def main():
    p = argparse.ArgumentParser(description="Build hierarchical-model gallery latents")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--latent-mode", choices=["spatial_flat", "global_vec", "global_concat", "spatial_pooled"],
                   required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--neighbors-h5", default=DEFAULT_NEIGHBORS_H5)
    p.add_argument("--max-examples", type=int, default=DEFAULT_MAX_EXAMPLES)
    p.add_argument("--batch-size", type=int, default=128)
    args = p.parse_args()

    from neighbors import NeighborsSimpleDataset
    full = NeighborsSimpleDataset(hdf5_path=args.neighbors_h5)
    n_total = len(full)
    n_use = min(args.max_examples, n_total)
    dataset = Subset(full, range(n_use))
    print(f"Neighbors dataset: using first {n_use:,} of {n_total:,} examples")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading hierarchical model from {args.checkpoint}")
    model = load_trained(args.checkpoint, device=device)
    print(f"Latent mode: {args.latent_mode}")

    loader = TorchDataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, collate_fn=_collate)

    hdf5_file = None
    row = 0
    with torch.no_grad():
        for hsc_im, leg_im, meta_list in loader:
            h = hsc_im.to(device)
            l = leg_im.to(device)
            hp = extract_physics(model, h, args.latent_mode)
            lp = extract_physics(model, l, args.latent_mode)
            if hp.dim() > 2: hp = hp.flatten(start_dim=1)
            if lp.dim() > 2: lp = lp.flatten(start_dim=1)
            hp = hp.cpu().numpy().astype(np.float32)
            lp = lp.cpu().numpy().astype(np.float32)
            idx_mmu = np.array([m["index_mmu"] for m in meta_list], dtype=np.int64)
            bs = hp.shape[0]

            if hdf5_file is None:
                d = hp.shape[1]
                hdf5_file = h5py.File(args.output, "w")
                hdf5_file.create_dataset("idx",                       shape=(n_use,),    dtype=np.int64)
                hdf5_file.create_dataset("index_mmu",                 shape=(n_use,),    dtype=np.int64)
                hdf5_file.create_dataset("physics_embedding",         shape=(n_use, d),  dtype=np.float32)
                hdf5_file.create_dataset("legacy_physics_embedding",  shape=(n_use, d),  dtype=np.float32)
                hdf5_file.attrs["num_examples"] = n_use
                hdf5_file.attrs["checkpoint"]   = str(args.checkpoint)
                hdf5_file.attrs["latent_mode"]  = args.latent_mode

            end = row + bs
            hdf5_file["idx"][row:end]                        = np.arange(row, end, dtype=np.int64)
            hdf5_file["index_mmu"][row:end]                  = idx_mmu
            hdf5_file["physics_embedding"][row:end]          = hp
            hdf5_file["legacy_physics_embedding"][row:end]   = lp
            row = end
            if row % 10_000 == 0 or row >= n_use:
                print(f"  {row}/{n_use} encoded", flush=True)

    if hdf5_file is not None:
        hdf5_file.close()
    print(f"Saved: {args.output} (latent_mode={args.latent_mode}, dim_recorded_in_attrs)")


if __name__ == "__main__":
    main()
