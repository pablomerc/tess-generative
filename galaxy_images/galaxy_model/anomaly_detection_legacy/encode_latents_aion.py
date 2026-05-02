"""
Legacy-modality variant of anomaly_detection/encode_latents_aion.py.

Encodes all source_type=0 (MMU) examples with AION (polymathic-ai/aion-base)
using LegacySurveyImage modality (DES-G/R/I/Z) instead of HSCImage.

The output dataset is named ``embeddings_mean_hsc`` so that fit_and_score.py
works unchanged. ``primary_survey="legacy"`` attribute disambiguates the file.

Run from galaxy_model/:
  python anomaly_detection_legacy/encode_latents_aion.py \
    --suffix aion_legacy_100k \
    [--batch-size 32] [--n-max N] [--device cuda]
"""
import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

_here = Path(__file__).resolve().parent
_src = _here.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import h5py
import numpy as np
import torch
torch.backends.cuda.preferred_blas_library("hipblas")
from tqdm import tqdm

from aion import AION
from aion.codecs import CodecManager
from aion.modalities import LegacySurveyImage

NEIGHBORS_HDF5 = "/work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5"
NUM_ENCODER_TOKENS_LEGACY = 600
OUTPUT_DIR = _here / "outputs"


def encode(suffix, batch_size, n_max, device_str):
    device = torch.device(device_str)
    print(f"Loading AION model (polymathic-ai/aion-base)...")
    model = AION.from_pretrained("polymathic-ai/aion-base").to(device)
    codec_manager = CodecManager(device=device_str)
    model.eval()

    with h5py.File(NEIGHBORS_HDF5, "r") as f:
        indexes_mmu = np.where(f["source_type"][:] == 0)[0]

    n_total = len(indexes_mmu)
    n_use = min(n_max, n_total) if n_max is not None else n_total
    indexes_mmu = indexes_mmu[:n_use]
    print(f"Encoding {n_use} MMU examples (total: {n_total}) — primary survey: LEGACY")

    all_embeddings_mean_leg = []
    all_raw_index = []

    with h5py.File(NEIGHBORS_HDF5, "r") as f:
        for start in tqdm(range(0, n_use, batch_size), desc="Encoding AION (legacy)"):
            end = min(start + batch_size, n_use)
            indices = indexes_mmu[start:end]

            legacy_tensor = torch.from_numpy(f["images_legacy"][indices]).to(device)

            image_leg = LegacySurveyImage(
                flux=legacy_tensor,
                bands=["DES-G", "DES-R", "DES-I", "DES-Z"],
            )
            tokens_leg = codec_manager.encode(image_leg)

            with torch.no_grad():
                emb_leg = model.encode(tokens_leg, num_encoder_tokens=NUM_ENCODER_TOKENS_LEGACY)

            all_embeddings_mean_leg.append(emb_leg.mean(dim=1).cpu().numpy().astype(np.float32))
            all_raw_index.append(indices.astype(np.int64))

    embeddings_mean_leg = np.concatenate(all_embeddings_mean_leg, axis=0)
    raw_index = np.concatenate(all_raw_index, axis=0)

    print(f"  embeddings_mean_legacy: {embeddings_mean_leg.shape}")
    print(f"  raw_index: {raw_index.shape}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"anomaly_latents_aion_{suffix}.h5"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".h5", prefix="anomaly_latents_aion_legacy_", dir=tempfile.gettempdir())
    os.close(tmp_fd)
    try:
        with h5py.File(tmp_path, "w") as f:
            f.create_dataset("raw_index", data=raw_index, compression="gzip", compression_opts=4)
            # Keep the dataset name as embeddings_mean_hsc so fit_and_score.py reads it unchanged.
            f.create_dataset("embeddings_mean_hsc", data=embeddings_mean_leg, compression="gzip", compression_opts=4)
            f.create_dataset("embeddings_mean_legacy", data=embeddings_mean_leg, compression="gzip", compression_opts=4)
            f.attrs["n_use"] = n_use
            f.attrs["suffix"] = suffix
            f.attrs["embedding_dim"] = embeddings_mean_leg.shape[1]
            f.attrs["primary_survey"] = "legacy"
        shutil.move(tmp_path, out_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    print(f"Saved: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", default="aion_legacy_100k")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-max", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    encode(args.suffix, args.batch_size, args.n_max, args.device)


if __name__ == "__main__":
    main()
