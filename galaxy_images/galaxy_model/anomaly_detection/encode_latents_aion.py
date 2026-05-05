"""
Encode all source_type=0 (MMU) examples using AION (polymathic-ai/aion-base).
Saves mean-pooled HSC embeddings + raw_index for anomaly detection.

Run from galaxy_model/ or anomaly_detection/:
  python anomaly_detection/encode_latents_aion.py \
    --suffix best87k \
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
# Force regular hipBLAS instead of hipBLASLt — hipBLASLt is buggy on MI210
# for certain matrix shapes and causes HIPBLAS_STATUS_INVALID_VALUE at runtime.
torch.backends.cuda.preferred_blas_library("hipblas")
from tqdm import tqdm

from aion import AION
from aion.codecs import CodecManager
from aion.modalities import LegacySurveyImage, HSCImage

NEIGHBORS_HDF5 = "/work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5"
NUM_ENCODER_TOKENS_HSC = 600
OUTPUT_DIR = _here / "outputs"


def encode(suffix, batch_size, n_max, device_str, source_types=(0,)):
    device = torch.device(device_str)
    print(f"Loading AION model (polymathic-ai/aion-base)...")
    model = AION.from_pretrained("polymathic-ai/aion-base").to(device)
    codec_manager = CodecManager(device=device_str)
    model.eval()

    source_types_arr = np.asarray(tuple(source_types), dtype=np.int64)
    with h5py.File(NEIGHBORS_HDF5, "r") as f:
        indexes_mmu = np.where(np.isin(f["source_type"][:], source_types_arr))[0]

    n_total = len(indexes_mmu)
    n_use = min(n_max, n_total) if n_max is not None else n_total
    indexes_mmu = indexes_mmu[:n_use]
    print(f"Encoding {n_use} examples  (source_types={tuple(source_types)}, total filtered: {n_total})")

    # Pre-allocate output buffers. Avoids the list-append + np.concatenate path
    # that previously caused unbounded RSS growth and OOM kills on larger runs.
    raw_index = np.empty(n_use, dtype=np.int64)
    embeddings_mean_hsc = None  # lazily sized once we know the embedding dim

    with h5py.File(NEIGHBORS_HDF5, "r") as f:
        for start in tqdm(range(0, n_use, batch_size), desc="Encoding AION"):
            end = min(start + batch_size, n_use)
            indices = indexes_mmu[start:end]

            hsc_tensor = torch.from_numpy(f["images_hsc"][indices]).to(device)

            image_hsc = HSCImage(
                flux=hsc_tensor,
                bands=["HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y"],
            )
            tokens_hsc = codec_manager.encode(image_hsc)

            with torch.no_grad():
                emb_hsc = model.encode(tokens_hsc, num_encoder_tokens=NUM_ENCODER_TOKENS_HSC)

            mean_emb = emb_hsc.mean(dim=1).to(torch.float32).cpu().numpy()
            if embeddings_mean_hsc is None:
                embeddings_mean_hsc = np.empty((n_use, mean_emb.shape[1]), dtype=np.float32)
            embeddings_mean_hsc[start:end] = mean_emb
            raw_index[start:end] = indices.astype(np.int64)

            del emb_hsc, mean_emb, tokens_hsc, hsc_tensor, image_hsc

    print(f"  embeddings_mean_hsc: {embeddings_mean_hsc.shape}")
    print(f"  raw_index: {raw_index.shape}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"anomaly_latents_aion_{suffix}.h5"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".h5", prefix="anomaly_latents_aion_", dir=tempfile.gettempdir())
    os.close(tmp_fd)
    try:
        with h5py.File(tmp_path, "w") as f:
            f.create_dataset("raw_index", data=raw_index, compression="gzip", compression_opts=4)
            f.create_dataset("embeddings_mean_hsc", data=embeddings_mean_hsc, compression="gzip", compression_opts=4)
            f.attrs["n_use"] = n_use
            f.attrs["suffix"] = suffix
            f.attrs["embedding_dim"] = embeddings_mean_hsc.shape[1]
            f.attrs["source_types"] = source_types_arr
        shutil.move(tmp_path, out_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    print(f"Saved: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", default="best87k")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-max", type=int, default=None)
    parser.add_argument(
        "--source-types", type=int, nargs="+", default=[0],
        help="source_type values to keep (default: 0). Use '--source-types 0 1' for the joint pool.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    encode(
        args.suffix, args.batch_size, args.n_max, args.device,
        source_types=tuple(args.source_types),
    )


if __name__ == "__main__":
    main()
