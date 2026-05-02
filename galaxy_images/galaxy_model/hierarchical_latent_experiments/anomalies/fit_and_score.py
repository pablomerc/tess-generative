"""Fit anomaly detectors on encoded latents (k-NN, IsolationForest, Mahalanobis, NSF).

Forks anomaly_detection/fit_and_score.py — applies the same four scorers to the
hierarchical model's per-variant latents (hsc_flat, hsc_legacy_concat, instrument_hsc).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "preferred_blas_library"):
    torch.backends.cuda.preferred_blas_library("hipblas")

_THIS = Path(__file__).resolve()
_PKG_ROOT = _THIS.parents[4]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from galaxy_images.galaxy_model.hierarchical_latent_experiments.common.discord_utils import post_text


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
    return np.einsum("ij,jk,ik->i", delta, inv_cov, delta).astype(np.float32)


def _score_nsf(train, all_latents, epochs, device_str, lr=1e-3, batch_size=512):
    try:
        import zuko
    except ImportError:
        print("  zuko not installed — skipping NSF.")
        return np.full(len(all_latents), np.nan, dtype=np.float32)

    device = torch.device(device_str)
    dim = train.shape[1]
    flow = zuko.flows.NSF(features=dim, context=0, transforms=6, hidden_features=[64, 64]).to(device)
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)

    train_t = torch.tensor(train, dtype=torch.float32, device=device)
    n_train = len(train_t)
    best_loss = float("inf")
    best_state = None

    for _ in tqdm(range(epochs), desc="  NSF training", leave=False):
        flow.train()
        perm = torch.randperm(n_train, device=device)
        ep_loss, n_batches = 0.0, 0
        for i in range(0, n_train, batch_size):
            batch = train_t[perm[i:i + batch_size]]
            optimizer.zero_grad()
            loss = -flow().log_prob(batch).mean()
            loss.backward()
            optimizer.step()
            ep_loss += loss.item(); n_batches += 1
        avg = ep_loss / max(n_batches, 1)
        if avg < best_loss:
            best_loss = avg
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
    print(f"  [{name}] k-NN..."); results[f"{name}/knn"] = _score_knn(train, all_latents)
    print(f"  [{name}] IsolationForest..."); results[f"{name}/iforest"] = _score_iforest(train, all_latents)
    print(f"  [{name}] Mahalanobis..."); results[f"{name}/mahal"] = _score_mahalanobis(train, all_latents)
    print(f"  [{name}] NSF (epochs={nsf_epochs})..."); results[f"{name}/flow"] = _score_nsf(train, all_latents, nsf_epochs, device_str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", required=True, help="Path to encode_latents output H5")
    ap.add_argument("--out-path", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--nsf-epochs", type=int, default=50)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-discord", action="store_true")
    args = ap.parse_args()

    if not args.no_discord:
        post_text(f"🟢 [anomalies/score/{args.variant}] starting")

    t0 = time.time()
    print(f"Loading latents from {args.latents}")
    with h5py.File(args.latents, "r") as f:
        raw_index = f["raw_index"][:]
        hsc_flat = f["hsc_flat"][:]
        hsc_legacy_concat = f["hsc_legacy_concat"][:]
        instrument_hsc = f["instrument_hsc"][:]

    n = len(raw_index)
    print(f"  N={n}, hsc_flat={hsc_flat.shape}, hsc_legacy_concat={hsc_legacy_concat.shape}, "
          f"instrument_hsc={instrument_hsc.shape}")

    train_idx, _ = _train_test_split(n, args.train_frac)
    scores = {}

    print("\n=== physics: hsc_flat ===")
    _apply_all_methods("phys/hsc_flat", hsc_flat[train_idx], hsc_flat,
                       args.nsf_epochs, args.device, scores)

    print("\n=== physics: hsc_legacy_concat ===")
    _apply_all_methods("phys/hsc_legacy_concat", hsc_legacy_concat[train_idx], hsc_legacy_concat,
                       args.nsf_epochs, args.device, scores)

    print("\n=== instrument: instrument_hsc ===")
    _apply_all_methods("inst/hsc", instrument_hsc[train_idx], instrument_hsc,
                       args.nsf_epochs, args.device, scores)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".h5", prefix="anomaly_scores_", dir=tempfile.gettempdir())
    os.close(fd)
    try:
        with h5py.File(tmp, "w") as f:
            f.create_dataset("raw_index", data=raw_index, compression="gzip", compression_opts=4)
            for k, v in scores.items():
                f.create_dataset(k, data=v, compression="gzip", compression_opts=4)
            f.attrs["variant"] = args.variant
            f.attrs["n"] = n
            f.attrs["score_keys"] = list(scores.keys())
        shutil.move(tmp, out_path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise

    elapsed = time.time() - t0
    print(f"\nSaved {out_path} in {elapsed:.1f}s")
    if not args.no_discord:
        post_text(f"💾 [anomalies/score/{args.variant}] saved {out_path.name} in {elapsed:.0f}s "
                  f"({len(scores)} score columns)")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        try:
            from galaxy_images.galaxy_model.hierarchical_latent_experiments.common.discord_utils import post_text as _pt
            _pt(f"❌ [anomalies/score] FAILED: {type(exc).__name__}: {exc}")
        except Exception:
            pass
        raise
