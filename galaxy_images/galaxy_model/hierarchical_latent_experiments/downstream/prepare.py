"""Prepare downstream H5 for one physics-latent variant of the hierarchical model.

Cluster-only: we run the `neighbors` task; mmu / *_provabgs require MIT-only paths.

Forks galaxy_images/galaxy_model/contrastive_baseline/downstream_evaluation/prepare_all_contrastive.py
— reusing its label aggregation, random-baseline, and H5 writer; replacing only the encoder.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "preferred_blas_library"):
    torch.backends.cuda.preferred_blas_library("hipblas")

_THIS = Path(__file__).resolve()
_PKG_ROOT = _THIS.parents[4]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from galaxy_images.galaxy_model.contrastive_baseline.downstream_evaluation.prepare_all_contrastive import (
    EMBEDDING_NAMES,
    NEIGHBORS_HDF5_DEFAULT,
    NUM_EXAMPLES_NEIGHBORS,
    NEIGHBORS_SEED,
    SHUFFLE_NEIGHBORS,
    _labels_from_metadata_list,
    collate_neighbors,
    make_random_embeddings_like,
    save_downstream_h5_all_variants,
)
from galaxy_images.galaxy_model.neighbors import NeighborsSimpleDataset
from galaxy_images.galaxy_model.hierarchical_latent_experiments.common.ckpt import (
    DEFAULT_CKPT, load_trained, load_untrained,
)
from galaxy_images.galaxy_model.hierarchical_latent_experiments.common.discord_utils import post_text
from galaxy_images.galaxy_model.hierarchical_latent_experiments.common.latents import (
    VARIANTS, extract_both,
)


def _embed_neighbors(model, dataset, variant, batch_size, device, *, shuffle, seed):
    loader_kw = dict(batch_size=batch_size, num_workers=0, collate_fn=collate_neighbors)
    if shuffle:
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)
        loader_kw["shuffle"] = True
        loader_kw["generator"] = g
    else:
        loader_kw["shuffle"] = False
    loader = DataLoader(dataset, **loader_kw)

    hsc_p, hsc_i, leg_p, leg_i = [], [], [], []
    metadata_collected = []
    with torch.no_grad():
        for hsc_im, leg_im, meta in loader:
            metadata_collected.extend(meta)
            h = hsc_im.to(device); l = leg_im.to(device)
            hp, hi = extract_both(model, h, variant)
            lp, li = extract_both(model, l, variant)
            hsc_p.append(hp.float().cpu()); hsc_i.append(hi.float().cpu())
            leg_p.append(lp.float().cpu()); leg_i.append(li.float().cpu())

    hsc_p = torch.cat(hsc_p).numpy(); hsc_i = torch.cat(hsc_i).numpy()
    leg_p = torch.cat(leg_p).numpy(); leg_i = torch.cat(leg_i).numpy()
    hsc_legacy_p = np.concatenate([hsc_p, leg_p], axis=1)
    hsc_legacy_i = np.concatenate([hsc_i, leg_i], axis=1)
    arrs = (hsc_p, hsc_i, leg_p, leg_i, hsc_legacy_p, hsc_legacy_i)
    return arrs, metadata_collected


def prepare_neighbors(variant: str, checkpoint_path: str, output_dir: Path,
                      batch_size: int, seed: int, suffix: str, neighbors_hdf5: str | None):
    hdf5_path = neighbors_hdf5 or NEIGHBORS_HDF5_DEFAULT
    full = NeighborsSimpleDataset(hdf5_path=hdf5_path)
    n_use = min(NUM_EXAMPLES_NEIGHBORS, len(full))
    dataset = Subset(full, range(n_use))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"variant={variant}, n_use={n_use}, device={device}")

    meta_cache = {"labels_n": None, "label_columns": None}

    def _emb_fn(model):
        arrs, metadata_collected = _embed_neighbors(
            model, dataset, variant, batch_size, device,
            shuffle=SHUFFLE_NEIGHBORS, seed=NEIGHBORS_SEED,
        )
        if meta_cache["labels_n"] is None:
            labels_n, label_columns = _labels_from_metadata_list(metadata_collected)
            meta_cache["labels_n"] = labels_n
            meta_cache["label_columns"] = label_columns
        return dict(zip(EMBEDDING_NAMES, arrs))

    print("[real] encoding …")
    model = load_trained(checkpoint_path, device=device)
    real = _emb_fn(model); del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("[untrained] encoding …")
    model_u = load_untrained(checkpoint_path, device=device)
    untrained = _emb_fn(model_u); del model_u
    if device.type == "cuda":
        torch.cuda.empty_cache()

    random_emb = make_random_embeddings_like(real, seed=seed)

    out_path = output_dir / f"downstream_neighbors_{suffix}.h5"
    save_downstream_h5_all_variants(
        out_path, real, untrained, random_emb,
        meta_cache["labels_n"], meta_cache["label_columns"],
    )
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--suffix", required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--neighbors-hdf5", default=None)
    ap.add_argument("--no-discord", action="store_true")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_discord:
        post_text(f"🟢 [downstream/prepare/{args.variant}] starting (suffix={args.suffix})")

    t0 = time.time()
    try:
        path = prepare_neighbors(args.variant, args.ckpt, args.output_dir,
                                 args.batch_size, args.seed, args.suffix,
                                 neighbors_hdf5=args.neighbors_hdf5)
    except Exception as exc:
        if not args.no_discord:
            post_text(f"❌ [downstream/prepare/{args.variant}] FAILED: {type(exc).__name__}: {exc}")
        raise
    elapsed = time.time() - t0
    print(f"saved {path} in {elapsed:.1f}s")
    if not args.no_discord:
        post_text(f"💾 [downstream/prepare/{args.variant}] saved {path.name} in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
