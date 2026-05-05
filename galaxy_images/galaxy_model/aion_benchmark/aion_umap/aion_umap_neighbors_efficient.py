"""AION UMAP on the same 4096 anchors as the `base_n4096_s314` run.

Reads raw images directly from the `neighbors_efficient` mmap store so the row
identity matches our run exactly. Encodes HSC-only and Legacy-only with the
full AION token budget (no `num_encoder_tokens` cap), mean-pools over tokens,
fits a single 2D UMAP on the concatenated latents, and saves the per-survey
2D coords plus the anchor row indices.

Usage:
    python aion_umap_neighbors_efficient.py --num-anchors 4096 --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import umap

os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")

# --- Make in-repo modules importable when run as a plain script ---
_HERE = Path(__file__).resolve().parent
_AION_BENCH = _HERE.parent
_GM = _AION_BENCH.parent
_REPO = _GM.parent.parent
_VIZ = _GM / "visualization_scripts"
for _p in (str(_REPO), str(_GM), str(_VIZ)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from aion import AION
from aion.codecs import CodecManager
from aion.modalities import LegacySurveyImage, HSCImage
from discord_notify import notify


def _load_mmaps_and_catalog(data_dir: Path):
    meta_path = data_dir / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    hsc_shape = tuple(meta["hsc_image"]["shape"])
    legacy_shape = tuple(meta["legacy_image"]["shape"])
    hsc_mmap = np.memmap(data_dir / "hsc_image.bin", dtype="float16", mode="r", shape=hsc_shape)
    legacy_mmap = np.memmap(data_dir / "legacy_image.bin", dtype="float16", mode="r", shape=legacy_shape)
    catalog = pd.read_parquet(data_dir / "catalog.parquet")
    return hsc_mmap, legacy_mmap, catalog


def encode_anchors(data_dir: Path, num_anchors: int, batch_size: int, device: str):
    hsc_mmap, legacy_mmap, catalog = _load_mmaps_and_catalog(data_dir)

    anchor_indices = catalog.index[catalog["source_type"] == 0].to_numpy().astype(np.int64)
    if num_anchors > len(anchor_indices):
        raise ValueError(f"num_anchors={num_anchors} > available anchors={len(anchor_indices)}")
    anchor_indices = anchor_indices[:num_anchors]

    print(f"[aion-ne] num_anchors={num_anchors}  first 5 row idx={anchor_indices[:5].tolist()}")
    sample_oid = catalog.loc[anchor_indices[:5], ["object_id_hsc", "object_id_legacy"]].to_dict("records")
    print(f"[aion-ne] sample anchor object_ids: {sample_oid}")

    print(f"[aion-ne] Loading AION on device={device} …")
    model = AION.from_pretrained("polymathic-ai/aion-base").to(device)
    model.eval()
    codec_manager = CodecManager(device=device)

    legacy_list, hsc_list = [], []
    t_start = time.perf_counter()
    for start in range(0, num_anchors, batch_size):
        end = min(start + batch_size, num_anchors)
        rows = anchor_indices[start:end]

        # Raw fluxes — AION's codecs handle the rest of preprocessing.
        legacy_arr = np.asarray(legacy_mmap[rows], dtype=np.float32)  # (B, 4, 160, 160)
        hsc_arr = np.asarray(hsc_mmap[rows], dtype=np.float32)        # (B, 5, 160, 160)
        legacy_tensor = torch.from_numpy(legacy_arr).to(device)
        hsc_tensor = torch.from_numpy(hsc_arr).to(device)

        image_leg = LegacySurveyImage(flux=legacy_tensor, bands=["DES-G", "DES-R", "DES-I", "DES-Z"])
        image_hsc = HSCImage(flux=hsc_tensor, bands=["HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y"])

        tokens_leg = codec_manager.encode(image_leg)
        tokens_hsc = codec_manager.encode(image_hsc)

        with torch.no_grad():
            # No num_encoder_tokens cap — let AION emit its full token budget.
            emb_leg = model.encode(tokens_leg)
            emb_hsc = model.encode(tokens_hsc)

        legacy_list.append(emb_leg.mean(dim=1).cpu().numpy())
        hsc_list.append(emb_hsc.mean(dim=1).cpu().numpy())

        elapsed = time.perf_counter() - t_start
        done = end
        rate = done / max(elapsed, 1e-6)
        eta = (num_anchors - done) / max(rate, 1e-6)
        print(f"[aion-ne] {done}/{num_anchors}  ({rate:.2f}/s, eta {eta/60:.1f} min)")

    legacy_latents = np.concatenate(legacy_list, axis=0)
    hsc_latents = np.concatenate(hsc_list, axis=0)
    print(f"[aion-ne] legacy_latents={legacy_latents.shape}  hsc_latents={hsc_latents.shape}")
    return anchor_indices, legacy_latents, hsc_latents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path,
                        default=Path("/work1/jeroenaudenaert/pablomer/data/neighbors_efficient"))
    parser.add_argument("--num-anchors", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=_HERE)
    parser.add_argument("--webhook", type=str, default=os.environ.get("DISCORD_WEBHOOK", ""))
    args = parser.parse_args()

    job_id = os.environ.get("SLURM_JOB_ID", "<no-slurm>")
    host = os.environ.get("SLURMD_NODENAME", socket.gethostname())
    t0 = time.perf_counter()

    notify(args.webhook,
           f"▶️ **AION-UMAP (NE)** start  num_anchors={args.num_anchors}  device={args.device}  "
           f"host=`{host}`  jobid=`{job_id}`")

    try:
        anchor_indices, legacy_latents, hsc_latents = encode_anchors(
            args.data_dir, args.num_anchors, args.batch_size, args.device
        )
        encode_seconds = time.perf_counter() - t0
        notify(args.webhook,
               f"🧠 AION encode done in {encode_seconds/60:.1f} min — fitting UMAP…")

        all_latents = np.concatenate([legacy_latents, hsc_latents], axis=0)
        umap_params = dict(
            n_neighbors=15, min_dist=0.1, n_components=2, metric="euclidean",
            random_state=int(args.seed),
        )
        t = time.perf_counter()
        embedding = umap.UMAP(**umap_params).fit_transform(all_latents)
        umap_seconds = time.perf_counter() - t
        n = len(legacy_latents)
        legacy_umap = embedding[:n]
        hsc_umap = embedding[n:]

        args.out_dir.mkdir(parents=True, exist_ok=True)
        npz_path = args.out_dir / f"aion_umap_data_neighbors_efficient_n{args.num_anchors}.npz"
        meta_path = args.out_dir / f"aion_umap_data_neighbors_efficient_n{args.num_anchors}_metadata.json"
        if npz_path.exists():
            raise FileExistsError(f"Refusing to overwrite {npz_path}")
        np.savez_compressed(
            npz_path,
            legacy_umap=legacy_umap.astype(np.float32),
            hsc_umap=hsc_umap.astype(np.float32),
            anchor_row_idx=anchor_indices.astype(np.int64),
        )
        meta = {
            "num_anchors": int(args.num_anchors),
            "data_dir": str(args.data_dir),
            "seed": int(args.seed),
            "umap_params": umap_params,
            "aion_model_id": "polymathic-ai/aion-base",
            "encoder_token_cap": None,
            "token_pooling": "mean_over_tokens",
            "device": args.device,
            "host": host,
            "slurm_job_id": job_id,
            "encode_seconds": encode_seconds,
            "umap_seconds": umap_seconds,
            "first_5_anchor_row_idx": [int(x) for x in anchor_indices[:5]],
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        total = time.perf_counter() - t0
        notify(args.webhook,
               f"✅ AION-UMAP (NE) done in {total/60:.1f} min\n"
               f"`{npz_path.name}`  legacy {legacy_umap.shape}  hsc {hsc_umap.shape}")
        print(f"[aion-ne] saved {npz_path}")
    except Exception as e:
        tb = traceback.format_exc()
        notify(args.webhook,
               f"❌ AION-UMAP (NE) failed: `{type(e).__name__}: {e}`\n```\n{tb[-1500:]}\n```")
        raise


if __name__ == "__main__":
    main()
