"""
Quick timing test: encode N examples with AION and extrapolate to 10k / 100k.

Run from galaxy_model/:
  python anomaly_detection/time_aion_encoding.py [--n 5] [--batch-size 32]
"""
import argparse
import sys
import time
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

from aion import AION
from aion.codecs import CodecManager
from aion.modalities import LegacySurveyImage, HSCImage

NEIGHBORS_HDF5 = "/work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = args.device
    print(f"Device: {device}")
    print(f"Loading AION model...")
    t0 = time.time()
    model = AION.from_pretrained("polymathic-ai/aion-base").to(device)
    codec_manager = CodecManager(device=device)
    model.eval()
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    with h5py.File(NEIGHBORS_HDF5, "r") as f:
        indexes_mmu = np.where(f["source_type"][:] == 0)[0][:args.n]
        hsc_tensor = torch.from_numpy(f["images_hsc"][indexes_mmu]).to(device)

    image_hsc = HSCImage(flux=hsc_tensor, bands=["HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y"])
    tokens_hsc = codec_manager.encode(image_hsc)

    print(f"Encoding {args.n} examples (batch_size={args.batch_size})...")
    t1 = time.time()
    with torch.no_grad():
        emb = model.encode(tokens_hsc, num_encoder_tokens=600)
    elapsed = time.time() - t1

    print(f"\nResults:")
    print(f"  {args.n} examples in {elapsed:.2f}s")
    print(f"  Per example: {elapsed/args.n*1000:.1f} ms")
    per_example = elapsed / args.n
    for target in [10_000, 100_000]:
        est_s = per_example * target
        est_h = est_s / 3600
        print(f"  Extrapolated {target:,} examples: {est_s/60:.0f} min ({est_h:.1f} h)")
    print(f"  Embedding shape: {emb.shape}  ->  mean-pooled: {emb.mean(dim=1).shape}")


if __name__ == "__main__":
    main()
