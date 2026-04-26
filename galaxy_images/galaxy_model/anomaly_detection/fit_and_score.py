"""
Fit anomaly detectors on encoded latents and compute scores for all method × representation combos.

Methods: k-NN (FAISS), Isolation Forest, Mahalanobis distance, Normalizing Flow (NSF via zuko).
Representations (ours): hsc_flat (64D), hsc_mean (16D), hsc_legacy_mean (16D).
Representations (AION): embeddings_mean_hsc -> PCA-64.

Run from galaxy_model/ or anomaly_detection/:
  python anomaly_detection/fit_and_score.py \
    --ours outputs/anomaly_latents_ours_best87k.h5 \
    --aion outputs/anomaly_latents_aion_best87k.h5 \
    --suffix best87k \
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
import torch.nn as nn
# Force regular hipBLAS instead of hipBLASLt — hipBLASLt is buggy on MI210
# for certain matrix shapes and causes HIPBLAS_STATUS_INVALID_VALUE at runtime.
torch.backends.cuda.preferred_blas_library("hipblas")
from tqdm import tqdm

OUTPUT_DIR = _here / "outputs"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _train_test_split(n, train_frac, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(n * train_frac)
    return idx[:n_train], idx[n_train:]


def _score_knn(train, all_latents, k=5):
    import faiss
    dim = train.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.ascontiguousarray(train, dtype=np.float32))
    D, _ = index.search(np.ascontiguousarray(all_latents, dtype=np.float32), k)
    return D.mean(axis=1).astype(np.float32)


def _score_iforest(train, all_latents):
    from sklearn.ensemble import IsolationForest
    clf = IsolationForest(n_estimators=200, contamination="auto", random_state=42, n_jobs=-1)
    clf.fit(train)
    return (-clf.score_samples(all_latents)).astype(np.float32)


def _score_mahalanobis(train, all_latents):
    mu = train.mean(axis=0)
    cov = np.cov(train.T)
    inv_cov = np.linalg.pinv(cov)
    delta = all_latents - mu
    scores = np.einsum("ij,jk,ik->i", delta, inv_cov, delta)
    return scores.astype(np.float32)


def _score_nsf(train, all_latents, epochs, device_str, lr=1e-3, batch_size=512):
    try:
        import zuko
    except ImportError:
        print("  zuko not installed — skipping NSF. Install with: pip install zuko")
        return np.full(len(all_latents), np.nan, dtype=np.float32)

    device = torch.device(device_str)
    dim = train.shape[1]
    flow = zuko.flows.NSF(features=dim, context=0, transforms=6, hidden_features=[64, 64]).to(device)
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)

    train_t = torch.tensor(train, dtype=torch.float32, device=device)
    n_train = len(train_t)
    best_loss = float("inf")
    best_state = None

    for epoch in tqdm(range(epochs), desc="  NSF training", leave=False):
        flow.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n_train, batch_size):
            batch = train_t[perm[i:i + batch_size]]
            optimizer.zero_grad()
            loss = -flow().log_prob(batch).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / max(n_batches, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in flow.state_dict().items()}

    if best_state is not None:
        flow.load_state_dict(best_state)
    flow.eval()

    all_t = torch.tensor(all_latents, dtype=torch.float32, device=device)
    scores = []
    with torch.no_grad():
        for i in range(0, len(all_t), batch_size):
            lp = flow().log_prob(all_t[i:i + batch_size])
            scores.append(-lp.cpu().numpy())
    return np.concatenate(scores, axis=0).astype(np.float32)


def _apply_all_methods(name, train, all_latents, nsf_epochs, device_str, results):
    print(f"  [{name}] k-NN...")
    results[f"{name}/knn"] = _score_knn(train, all_latents)
    print(f"  [{name}] Isolation Forest...")
    results[f"{name}/iforest"] = _score_iforest(train, all_latents)
    print(f"  [{name}] Mahalanobis...")
    results[f"{name}/mahal"] = _score_mahalanobis(train, all_latents)
    print(f"  [{name}] NSF (epochs={nsf_epochs})...")
    results[f"{name}/flow"] = _score_nsf(train, all_latents, nsf_epochs, device_str)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours", default=None, help="Path to anomaly_latents_ours_*.h5")
    parser.add_argument("--aion", default=None, help="Path to anomaly_latents_aion_*.h5")
    parser.add_argument("--suffix", default="best87k")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--nsf-epochs", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--also-instrument-emb", action="store_true",
                        help="Also score instrument_emb from the --ours latents file")
    args = parser.parse_args()

    if args.ours is None and args.aion is None:
        raise ValueError("At least one of --ours or --aion must be provided")

    scores = {}
    raw_index = None

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
        print(f"  N={n}, hsc_flat={hsc_flat.shape}, hsc_mean={hsc_mean.shape}")
        train_idx, _ = _train_test_split(n, args.train_frac)

        print("\n=== Ours: hsc_flat ===")
        _apply_all_methods("ours/hsc_flat", hsc_flat[train_idx], hsc_flat, args.nsf_epochs, args.device, scores)

        print("\n=== Ours: hsc_mean ===")
        _apply_all_methods("ours/hsc_mean", hsc_mean[train_idx], hsc_mean, args.nsf_epochs, args.device, scores)

        print("\n=== Ours: hsc_legacy_mean ===")
        _apply_all_methods("ours/hsc_legacy_mean", hsc_legacy_mean[train_idx], hsc_legacy_mean, args.nsf_epochs, args.device, scores)

        if args.also_instrument_emb:
            with h5py.File(ours_path, "r") as f:
                if "instrument_emb" in f:
                    instrument_emb = f["instrument_emb"][:]
                    print(f"  instrument_emb: {instrument_emb.shape}")
                else:
                    instrument_emb = None
                    print("  WARNING: instrument_emb not found in latents file, skipping")
            if instrument_emb is not None:
                print("\n=== Ours: instrument_emb ===")
                _apply_all_methods("ours/instrument_emb", instrument_emb[train_idx], instrument_emb, args.nsf_epochs, args.device, scores)

    if args.aion is not None:
        aion_path = Path(args.aion)
        if not aion_path.is_absolute() and not aion_path.exists():
            aion_path = OUTPUT_DIR / args.aion
        print(f"\nLoading AION latents from {aion_path}")
        with h5py.File(aion_path, "r") as f:
            aion_raw_index = f["raw_index"][:]
            aion_emb = f["embeddings_mean_hsc"][:]

        if raw_index is None:
            # AION-only run
            raw_index = aion_raw_index
            n = len(raw_index)
            train_idx, _ = _train_test_split(n, args.train_frac)
        elif not np.array_equal(raw_index, aion_raw_index):
            print("  WARNING: raw_index mismatch between ours and AION — aligning by common indices")
            common = np.intersect1d(raw_index, aion_raw_index)
            ours_map = {v: i for i, v in enumerate(raw_index)}
            aion_map = {v: i for i, v in enumerate(aion_raw_index)}
            ours_sel = np.array([ours_map[v] for v in common])
            aion_sel = np.array([aion_map[v] for v in common])
            raw_index = raw_index[ours_sel]
            hsc_flat = hsc_flat[ours_sel]
            hsc_mean = hsc_mean[ours_sel]
            hsc_legacy_mean = hsc_legacy_mean[ours_sel]
            aion_emb = aion_emb[aion_sel]
            n = len(raw_index)
            train_idx, _ = _train_test_split(n, args.train_frac)

        print(f"  AION emb shape before PCA: {aion_emb.shape}")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=64, random_state=42)
        aion_pca = pca.fit_transform(aion_emb[train_idx])
        aion_pca_all = pca.transform(aion_emb)
        print(f"  AION PCA-64 variance explained: {pca.explained_variance_ratio_.sum():.3f}")

        print("\n=== AION: hsc_mean_pca64 ===")
        _apply_all_methods("aion/hsc_mean_pca64", aion_pca_all[train_idx], aion_pca_all, args.nsf_epochs, args.device, scores)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"anomaly_scores_{args.suffix}.h5"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".h5", prefix="anomaly_scores_", dir=tempfile.gettempdir())
    os.close(tmp_fd)
    try:
        with h5py.File(tmp_path, "w") as f:
            f.create_dataset("raw_index", data=raw_index, compression="gzip", compression_opts=4)
            for key, arr in scores.items():
                f.create_dataset(key, data=arr, compression="gzip", compression_opts=4)
            f.attrs["suffix"] = args.suffix
            f.attrs["n"] = n
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
