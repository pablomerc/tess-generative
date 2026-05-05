"""Fit anomaly detectors directly on the full 768-d AION embeddings (no PCA).

Companion to fit_and_score.py. Only trains the two detectors the user asked for:
  - Normalizing Flow (NSF, zuko) — needs GPU
  - Isolation Forest (sklearn) — CPU

Run from galaxy_model/:
  python anomaly_detection/fit_aion_no_pca.py \
    --aion outputs/anomaly_latents_aion_aion_367k.h5 \
    --suffix aion_nopca_367k \
    [--train-frac 0.8] [--nsf-epochs 50] [--device cuda]
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

# Reuse the exact same scorers as the canonical fit_and_score.py so results
# are directly comparable.
from fit_and_score import _train_test_split, _score_iforest, _score_nsf

OUTPUT_DIR = _here / "outputs"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aion", required=True, help="Path to anomaly_latents_aion_*.h5")
    parser.add_argument("--suffix", required=True, help="Output file suffix")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--nsf-epochs", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    aion_path = Path(args.aion)
    if not aion_path.is_absolute() and not aion_path.exists():
        aion_path = OUTPUT_DIR / args.aion

    print(f"Loading AION latents from {aion_path}")
    with h5py.File(aion_path, "r") as f:
        raw_index = f["raw_index"][:]
        aion_emb = f["embeddings_mean_hsc"][:]
    n, d = aion_emb.shape
    print(f"  N={n}, dim={d}  (full AION embedding, no PCA)")

    train_idx, _ = _train_test_split(n, args.train_frac)
    train_emb = aion_emb[train_idx]
    print(f"  train={len(train_emb)}, test={n - len(train_emb)}")

    scores = {}

    print("\n=== AION raw 768-d: Isolation Forest ===")
    scores["aion/hsc_mean_full/iforest"] = _score_iforest(train_emb, aion_emb)

    print(f"\n=== AION raw 768-d: NSF (epochs={args.nsf_epochs}) ===")
    scores["aion/hsc_mean_full/flow"] = _score_nsf(
        train_emb, aion_emb, args.nsf_epochs, args.device,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"anomaly_scores_{args.suffix}.h5"
    tmp_fd, tmp_path = tempfile.mkstemp(
        suffix=".h5", prefix="anomaly_scores_aion_nopca_", dir=tempfile.gettempdir(),
    )
    os.close(tmp_fd)
    try:
        with h5py.File(tmp_path, "w") as f:
            f.create_dataset("raw_index", data=raw_index, compression="gzip", compression_opts=4)
            for key, arr in scores.items():
                f.create_dataset(key, data=arr, compression="gzip", compression_opts=4)
            f.attrs["suffix"] = args.suffix
            f.attrs["n"] = n
            f.attrs["embedding_dim"] = d
            f.attrs["pca"] = False
            f.attrs["score_keys"] = list(scores.keys())
        shutil.move(tmp_path, out_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    print(f"\nSaved scores: {out_path}")
    print(f"Score keys: {list(scores.keys())}")


if __name__ == "__main__":
    main()
