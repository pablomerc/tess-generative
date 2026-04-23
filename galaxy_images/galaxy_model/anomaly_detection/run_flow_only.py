"""
Compute NSF (normalizing flow) anomaly scores only, patching existing score files.

Run from galaxy_model/:
  python anomaly_detection/run_flow_only.py --suffix ours_10k [--ours ...] [--aion ...]
  python anomaly_detection/run_flow_only.py --suffix aion_10k [--aion ...]
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
import torch.nn as nn
# Force regular hipBLAS instead of hipBLASLt — hipBLASLt is buggy on MI210
# for certain matrix shapes and causes HIPBLAS_STATUS_INVALID_VALUE at runtime.
torch.backends.cuda.preferred_blas_library("hipblas")
from tqdm import tqdm

import zuko

OUTPUT_DIR = _here / "outputs"


def _train_test_split(n, train_frac, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(n * train_frac)
    return idx[:n_train], idx[n_train:]


def _score_nsf(name, train, all_latents, epochs, device_str):
    device = torch.device(device_str)
    dim = train.shape[1]
    flow = zuko.flows.NSF(features=dim, context=0, transforms=6, hidden_features=[64, 64]).to(device)
    optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)

    train_t = torch.tensor(train, dtype=torch.float32, device=device)
    n_train = len(train_t)
    best_loss = float("inf")
    best_state = None

    for epoch in tqdm(range(epochs), desc=f"  NSF [{name}]", leave=False):
        flow.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n_train, 512):
            batch = train_t[perm[i:i + 512]]
            optimizer.zero_grad()
            loss = -flow().log_prob(batch).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg = epoch_loss / max(n_batches, 1)
        if avg < best_loss:
            best_loss = avg
            best_state = {k: v.clone() for k, v in flow.state_dict().items()}

    if best_state:
        flow.load_state_dict(best_state)
    flow.eval()

    all_t = torch.tensor(all_latents, dtype=torch.float32, device=device)
    scores = []
    with torch.no_grad():
        for i in range(0, len(all_t), 512):
            scores.append(-flow().log_prob(all_t[i:i + 512]).cpu().numpy())
    return np.concatenate(scores).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours", default=None)
    parser.add_argument("--aion", default=None)
    parser.add_argument("--suffix", required=True)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--nsf-epochs", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    flow_scores = {}

    if args.ours is not None:
        ours_path = Path(args.ours)
        if not ours_path.is_absolute() and not ours_path.exists():
            ours_path = OUTPUT_DIR / args.ours
        print(f"Loading ours latents from {ours_path}")
        with h5py.File(ours_path, "r") as f:
            raw_index = f["raw_index"][:]
            hsc_flat = f["hsc_flat"][:]
            hsc_mean = f["hsc_mean"][:]
            hsc_legacy_mean = f["hsc_legacy_mean"][:]
        n = len(raw_index)
        train_idx, _ = _train_test_split(n, args.train_frac)
        print(f"  N={n}, train={len(train_idx)}")

        for name, data in [
            ("ours/hsc_flat", hsc_flat),
            ("ours/hsc_mean", hsc_mean),
            ("ours/hsc_legacy_mean", hsc_legacy_mean),
        ]:
            print(f"\n=== NSF: {name} ===")
            flow_scores[f"{name}/flow"] = _score_nsf(name, data[train_idx], data, args.nsf_epochs, args.device)

    if args.aion is not None:
        from sklearn.decomposition import PCA
        aion_path = Path(args.aion)
        if not aion_path.is_absolute() and not aion_path.exists():
            aion_path = OUTPUT_DIR / args.aion
        print(f"Loading AION latents from {aion_path}")
        with h5py.File(aion_path, "r") as f:
            aion_raw_index = f["raw_index"][:]
            aion_emb = f["embeddings_mean_hsc"][:]
        n = len(aion_raw_index)
        train_idx, _ = _train_test_split(n, args.train_frac)
        pca = PCA(n_components=64, random_state=42)
        aion_pca_all = pca.fit_transform(aion_emb)
        aion_pca_train = pca.transform(aion_emb[train_idx])
        print(f"  N={n}, PCA variance explained: {pca.explained_variance_ratio_.sum():.3f}")

        print("\n=== NSF: aion/hsc_mean_pca64 ===")
        flow_scores["aion/hsc_mean_pca64/flow"] = _score_nsf(
            "aion/hsc_mean_pca64", aion_pca_train, aion_pca_all, args.nsf_epochs, args.device
        )

    # Patch existing scores file
    scores_path = OUTPUT_DIR / f"anomaly_scores_{args.suffix}.h5"
    print(f"\nPatching {scores_path} with flow scores...")
    with h5py.File(scores_path, "a") as f:
        for key, arr in flow_scores.items():
            if key in f:
                del f[key]
            f.create_dataset(key, data=arr, compression="gzip", compression_opts=4)
        existing_keys = list(f.attrs.get("score_keys", []))
        new_keys = [k for k in flow_scores if k not in existing_keys]
        f.attrs["score_keys"] = existing_keys + new_keys

    print("Flow scores written:")
    for k, v in flow_scores.items():
        finite = np.isfinite(v).sum()
        print(f"  {k}: {finite}/{len(v)} finite")


if __name__ == "__main__":
    main()
