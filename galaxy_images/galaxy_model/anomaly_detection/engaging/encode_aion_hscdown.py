"""Encode HSC downstream dataset with AION (polymathic-ai/aion-base).

Mirrors anomaly_detection/encode_latents_aion.py but reads images from the
HSC downstream memmap dataset instead of neighbours_v2.h5. Saves mean-pooled
HSC embeddings + raw_index. No PCA is applied here — that happens (or doesn't)
at fit time.

Run:
  python anomaly_detection/engaging/encode_aion_hscdown.py \
    --suffix hscdown_474k [--batch-size 32] [--n-max N]
"""
import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ANOMALY_DIR = _HERE.parent
_GALAXY_MODEL = _ANOMALY_DIR.parent
for p in (str(_GALAXY_MODEL), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import h5py
import numpy as np
import torch
from tqdm import tqdm

from aion import AION
from aion.codecs import CodecManager
from aion.modalities import HSCImage

from hsc_downstream_dataset import HSCDownstreamDataset

NUM_ENCODER_TOKENS_HSC = 600
OUTPUT_DIR = _ANOMALY_DIR / "outputs"


def encode(suffix, batch_size, n_max, device_str):
    device = torch.device(device_str)
    print("Loading AION model (polymathic-ai/aion-base)...")
    model = AION.from_pretrained("polymathic-ai/aion-base").to(device)
    codec_manager = CodecManager(device=device_str)
    model.eval()

    full = HSCDownstreamDataset(view="aion")
    n_total = len(full)
    n_use = min(n_max, n_total) if n_max is not None else n_total
    print(f"Encoding {n_use} examples (total: {n_total})  with AION")

    raw_index = np.empty(n_use, dtype=np.int64)
    embeddings_mean_hsc = None  # lazily sized

    for start in tqdm(range(0, n_use, batch_size), desc="Encoding AION"):
        end = min(start + batch_size, n_use)

        # Stack flux (float32) for indices [start:end] from the memmap.
        batch_np = np.stack([full._load_flux(i) for i in range(start, end)])
        batch_imgs = torch.from_numpy(batch_np).to(device)

        image_hsc = HSCImage(
            flux=batch_imgs,
            bands=["HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y"],
        )
        tokens_hsc = codec_manager.encode(image_hsc)

        with torch.no_grad():
            emb_hsc = model.encode(tokens_hsc, num_encoder_tokens=NUM_ENCODER_TOKENS_HSC)

        mean_emb = emb_hsc.mean(dim=1).to(torch.float32).cpu().numpy()
        if embeddings_mean_hsc is None:
            embeddings_mean_hsc = np.empty((n_use, mean_emb.shape[1]), dtype=np.float32)
        embeddings_mean_hsc[start:end] = mean_emb
        raw_index[start:end] = np.arange(start, end, dtype=np.int64)

        del emb_hsc, mean_emb, tokens_hsc, batch_imgs, image_hsc

    print(f"  embeddings_mean_hsc: {embeddings_mean_hsc.shape}")
    print(f"  raw_index: {raw_index.shape}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"anomaly_latents_aion_{suffix}.h5"
    tmp_fd, tmp_path = tempfile.mkstemp(
        suffix=".h5", prefix="anomaly_latents_aion_hscdown_", dir=tempfile.gettempdir()
    )
    os.close(tmp_fd)
    try:
        with h5py.File(tmp_path, "w") as f:
            f.create_dataset("raw_index", data=raw_index, compression="gzip", compression_opts=4)
            f.create_dataset("embeddings_mean_hsc", data=embeddings_mean_hsc, compression="gzip", compression_opts=4)
            f.attrs["n_use"] = n_use
            f.attrs["suffix"] = suffix
            f.attrs["embedding_dim"] = embeddings_mean_hsc.shape[1]
            f.attrs["data_source"] = "hsc_downstream"
        shutil.move(tmp_path, out_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    print(f"Saved: {out_path}")
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suffix", default="hscdown_474k")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-max", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    encode(args.suffix, args.batch_size, args.n_max, args.device)


if __name__ == "__main__":
    main()
