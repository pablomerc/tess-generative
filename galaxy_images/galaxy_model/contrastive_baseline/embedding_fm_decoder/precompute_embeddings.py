"""
Stage 1 — precompute the [embedding, image] dataset for the embedding-conditioned FM decoder.

For each example we take the ANCHOR (== ``target``) image from the efficient neighbors corpus,
run it through the FROZEN contrastive dual-encoder, and store the resulting embeddings alongside
the image the decoder must reconstruct:

    emb_galaxy      (N, 64)          encoder_galaxy(anchor)      — "physics" embedding
    emb_instrument  (N, 64)          encoder_instrument(anchor)  — "instrument/survey" embedding
    targets         (N, 4, 48, 48)   the encoder-ready anchor image (float32)
    survey          (N,)   int8      0 = hsc, 1 = legacy
    row_idx         (N,)   int64     source catalog row of the anchor

The anchor image and its preprocessing are IDENTICAL to what the contrastive model saw at
training time (we reuse ``NeighborsEfficientDataset._preprocess``), so the frozen encoder is
in-distribution.

Anchors alternate HSC/Legacy by index (``idx % 2``), so the first ``--limit`` anchors give a
balanced ~50/50 survey mix, reproducibly.

Usage (see also precompute_embeddings.slurm):

    python -m galaxy_images.galaxy_model.contrastive_baseline.embedding_fm_decoder.precompute_embeddings \
        --data-dir /orcd/pool/007/pablomer/efficient_neighs \
        --ckpt     /orcd/pool/007/pablomer/outputs/contrastive/best-epoch=599-step=240000.ckpt \
        --out-h5   /orcd/pool/007/pablomer/outputs/embedding_fm/anchor_embeddings_100k.h5 \
        --limit 100000 --batch-size 256

Performance note: random-access mmap on a networked (Ceph) filesystem pays a page-fault
round-trip per read. For a big run, stage the ``efficient_neighs`` directory onto local NVMe
scratch first (see efficient_neighs/README.md) and point --data-dir there.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
# ROCm/AMD MI-series: prefer rocBLAS over the buggy hipBLASLt (HIPBLAS_STATUS_INVALID_VALUE on
# certain shapes). Guarded so it stays a harmless no-op on NVIDIA/CUDA (Engaging). Pair with
# TORCH_BLAS_PREFER_HIPBLASLT=0 in the launch env.
try:
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "preferred_blas_library"):
        torch.backends.cuda.preferred_blas_library("hipblas")
except Exception:
    pass
from torch.utils.data import DataLoader, Subset

# Make the repo importable when run as a plain script.
_project_root = Path(__file__).resolve().parents[4]  # .../tess-generative
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from galaxy_images.galaxy_model.neighbors_efficient import NeighborsEfficientDataset
from galaxy_images.galaxy_model.contrastive_baseline.dual_encoder_contrastive import (
    DualEncoderContrastiveModule,
)

DEFAULT_DATA_DIR = "/orcd/pool/007/pablomer/efficient_neighs"
DEFAULT_CKPT = "/orcd/pool/007/pablomer/outputs/contrastive/best-epoch=599-step=240000.ckpt"
DEFAULT_OUT = "/orcd/pool/007/pablomer/outputs/embedding_fm/anchor_embeddings_100k.h5"


class AnchorOnlyEfficientDataset(NeighborsEfficientDataset):
    """Yield ONLY the anchor target image (no samegal, no neighbors).

    Reuses the parent's ``_preprocess`` so the image is identical to training-time inputs,
    but skips all neighbor / same-galaxy I/O — we only need the anchor to embed + reconstruct.
    """

    def __getitem__(self, idx: int):
        row_idx = int(self.anchor_indices[idx])
        anchor_is_hsc = (idx % 2 == 0)
        if anchor_is_hsc:
            target = self._preprocess(row_idx, "hsc")[:4]  # drop HSC y band -> (4, H, W)
            survey = "hsc"
        else:
            target = self._preprocess(row_idx, "legacy")   # (4, H, W)
            survey = "legacy"
        return target, survey, row_idx


def _collate(batch):
    targets = torch.stack([b[0] for b in batch])
    surveys = [b[1] for b in batch]
    row_idxs = [b[2] for b in batch]
    return targets, surveys, row_idxs


def _load_encoder(ckpt_path: str, device: torch.device) -> DualEncoderContrastiveModule:
    # Hyperparameters (embedding_dim etc.) are restored from the checkpoint automatically.
    model = DualEncoderContrastiveModule.load_from_checkpoint(ckpt_path, map_location="cpu")
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def main():
    ap = argparse.ArgumentParser(description="Precompute [embedding, image] dataset from anchors.")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--out-h5", default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=100_000, help="Number of anchors to encode (from the start).")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--crop-size", type=int, default=48)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"[precompute] device={device}")
    print(f"[precompute] ckpt={args.ckpt}")
    model = _load_encoder(args.ckpt, device)
    emb_dim = int(model.hparams.embedding_dim)
    print(f"[precompute] encoder embedding_dim={emb_dim} (galaxy & instrument each)")

    ds = AnchorOnlyEfficientDataset(args.data_dir, crop_size=args.crop_size, max_neighbors=0)
    n_total = len(ds)
    n = min(args.limit, n_total)
    print(f"[precompute] {n_total:,} anchors available; encoding first {n:,}")
    subset = Subset(ds, list(range(n)))

    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate,
        pin_memory=(device.type == "cuda"),
    )

    out_path = Path(args.out_h5)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Running per-survey stats on the stored targets (cheap sanity check).
    stat = {0: [0, 0.0, 0.0], 1: [0, 0.0, 0.0]}  # survey -> [count_pixels, sum, sumsq]

    t0 = time.time()
    chunk = min(args.batch_size, n)
    with h5py.File(out_path, "w") as f:
        d_tgt = f.create_dataset("targets", shape=(n, 4, args.crop_size, args.crop_size),
                                 dtype="float32", compression="lzf",
                                 chunks=(chunk, 4, args.crop_size, args.crop_size))
        d_g = f.create_dataset("emb_galaxy", shape=(n, emb_dim), dtype="float32", compression="lzf")
        d_i = f.create_dataset("emb_instrument", shape=(n, emb_dim), dtype="float32", compression="lzf")
        d_s = f.create_dataset("survey", shape=(n,), dtype="int8")
        d_r = f.create_dataset("row_idx", shape=(n,), dtype="int64")

        ptr = 0
        with torch.no_grad():
            for bi, (targets, surveys, row_idxs) in enumerate(loader):
                x = targets.to(device, dtype=torch.float32, non_blocking=True)
                eg = model.encoder_galaxy(x).float().cpu().numpy()
                ei = model.encoder_instrument(x).float().cpu().numpy()
                b = x.shape[0]

                tnp = targets.numpy().astype(np.float32)
                surv = np.array([0 if s == "hsc" else 1 for s in surveys], dtype=np.int8)

                d_tgt[ptr:ptr + b] = tnp
                d_g[ptr:ptr + b] = eg
                d_i[ptr:ptr + b] = ei
                d_s[ptr:ptr + b] = surv
                d_r[ptr:ptr + b] = np.array(row_idxs, dtype=np.int64)
                ptr += b

                for code in (0, 1):
                    sel = tnp[surv == code]
                    if sel.size:
                        stat[code][0] += sel.size
                        stat[code][1] += float(sel.sum())
                        stat[code][2] += float(np.square(sel).sum())

                if bi % 20 == 0:
                    rate = ptr / max(1e-9, time.time() - t0)
                    print(f"[precompute] {ptr:,}/{n:,}  ({rate:.0f} img/s)", flush=True)

        f.attrs["num_examples"] = n
        f.attrs["embedding_dim_each"] = emb_dim
        f.attrs["cond_dim_concat"] = 2 * emb_dim
        f.attrs["source_data_dir"] = str(args.data_dir)
        f.attrs["contrastive_ckpt"] = str(args.ckpt)
        f.attrs["crop_size"] = args.crop_size

    dt = time.time() - t0
    print(f"[precompute] wrote {n:,} examples to {out_path} in {dt/60:.1f} min")
    for code, name in ((0, "hsc"), (1, "legacy")):
        c, s, ss = stat[code]
        if c:
            mean = s / c
            var = max(0.0, ss / c - mean * mean)
            print(f"[precompute] target[{name}] mean={mean:+.4f} std={var**0.5:.4f} "
                  f"(expected ~0 mean / ~1 std after per-survey normalization)")


if __name__ == "__main__":
    main()
