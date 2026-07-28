"""Stage 2 — fit anomaly detectors on each latent space and score every anchor.

Four detectors x five representations. Detectors are fit on an 80% train split
(seed 42) and then score *all* anchors, exactly as in
``anomaly_detection/fit_and_score.py``:

  knn      mean squared L2 distance to the k=5 nearest training points
  iforest  Isolation Forest, 200 trees
  mahal    squared Mahalanobis distance to the training Gaussian
  flow     negative log-likelihood under a zuko NSF (6 transforms, [64,64])

Two deliberate deviations from the AMD-era script, both recorded in the output
attrs so a later comparison cannot silently mix protocols:

1. **kNN self-exclusion** (``--knn-allow-self`` restores the old behaviour). The
   old script added the train split to the FAISS index and then queried *all*
   points against it, so every training point matched itself at distance 0 and
   got a systematically deflated score. With train_frac=0.8 that biases the
   top-anomaly list ~5x towards the held-out 20%. Here a query that belongs to the
   train split takes neighbours 2..k+1 instead.
2. **Per-dimension standardisation** using train-split statistics, on by default
   (``--no-standardize`` restores the old behaviour). Raw conv1x1 physics latents
   have per-dimension standard deviations spanning ~0.7-3.4, which silently
   weights the wide dimensions in the distance-based detectors (knn) and hurts NSF
   conditioning. Mahalanobis and Isolation Forest are ~invariant to it.

FAISS is not installed in ``torchenv`` on Engaging, so kNN is a chunked
``torch.cdist`` on the GPU — same quantity (squared L2), faster here anyway.

Usage:
  python fit_and_score.py --latents <latents_*.h5> --tag spatial-conv1x1
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

try:
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "preferred_blas_library"):
        torch.backends.cuda.preferred_blas_library("hipblas")
except Exception:
    pass

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

DEFAULT_OUT_DIR = "/orcd/pool/007/pablomer/outputs/contrastive_anomalies"
ALL_REPS = ["physics_hsc", "physics_legacy", "physics_mean", "instrument_hsc", "instrument_legacy"]
ALL_METHODS = ["knn", "iforest", "mahal", "flow"]


def _n_jobs() -> int:
    """Respect the SLURM cpu allocation — joblib's -1 would grab the whole node."""
    env = os.environ.get("SLURM_CPUS_PER_TASK")
    if env:
        return max(1, int(env))
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def train_test_split(n: int, train_frac: float, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(n * train_frac)
    return idx[:n_train], idx[n_train:]


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------

def score_knn(train, all_latents, train_mask, k=5, device_str="cuda", chunk=1024, allow_self=False):
    """Mean squared-L2 distance to the k nearest training points."""
    device = torch.device(device_str)
    ref = torch.as_tensor(train, dtype=torch.float32, device=device)
    # With self-exclusion a train-set query needs one extra neighbour to spare.
    k = min(k, len(ref) - 1) if not allow_self else min(k, len(ref))
    if k < 1:
        raise ValueError(f"train split of {len(ref)} points is too small for kNN")
    take = k if allow_self else k + 1
    out = np.empty(len(all_latents), dtype=np.float32)
    for i in range(0, len(all_latents), chunk):
        q = torch.as_tensor(all_latents[i : i + chunk], dtype=torch.float32, device=device)
        d = torch.cdist(q, ref).pow_(2)                       # squared L2, matches IndexFlatL2
        vals, _ = torch.topk(d, take, dim=1, largest=False)   # ascending
        if allow_self:
            sel = vals[:, :k]
        else:
            # Train members match themselves at distance 0; drop that first column
            # for them only, so a genuine duplicate object is still counted.
            m = torch.as_tensor(train_mask[i : i + chunk], device=device)
            shifted = vals[:, 1 : k + 1]
            kept = vals[:, :k]
            sel = torch.where(m.unsqueeze(1), shifted, kept)
        out[i : i + chunk] = sel.mean(dim=1).cpu().numpy()
        del d
    return out


def score_iforest(train, all_latents):
    from sklearn.ensemble import IsolationForest

    clf = IsolationForest(n_estimators=200, contamination="auto", random_state=42, n_jobs=_n_jobs())
    clf.fit(train)
    return (-clf.score_samples(all_latents)).astype(np.float32)


def score_mahalanobis(train, all_latents):
    mu = train.mean(axis=0)
    inv_cov = np.linalg.pinv(np.cov(train.T))
    delta = all_latents - mu
    return np.einsum("ij,jk,ik->i", delta, inv_cov, delta).astype(np.float32)


def score_nsf(train, all_latents, epochs, device_str, lr=1e-3, batch_size=512, seed=42):
    import zuko

    device = torch.device(device_str)
    torch.manual_seed(seed)
    flow = zuko.flows.NSF(features=train.shape[1], context=0, transforms=6,
                          hidden_features=[64, 64]).to(device)
    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    train_t = torch.as_tensor(train, dtype=torch.float32, device=device)
    n_train = len(train_t)

    best_loss, best_state = float("inf"), None
    with torch.enable_grad():
        for ep in range(epochs):
            flow.train()
            perm = torch.randperm(n_train, device=device)
            tot, nb = 0.0, 0
            for i in range(0, n_train, batch_size):
                batch = train_t[perm[i : i + batch_size]]
                opt.zero_grad()
                loss = -flow().log_prob(batch).mean()
                loss.backward()
                opt.step()
                tot += loss.item()
                nb += 1
            avg = tot / max(nb, 1)
            if avg < best_loss:
                best_loss, best_state = avg, {k: v.detach().clone() for k, v in flow.state_dict().items()}
            if (ep + 1) % 10 == 0:
                print(f"      NSF epoch {ep + 1}/{epochs}  loss={avg:.4f}  best={best_loss:.4f}", flush=True)

    if best_state is not None:
        flow.load_state_dict(best_state)
    flow.eval()

    all_t = torch.as_tensor(all_latents, dtype=torch.float32, device=device)
    parts = []
    with torch.no_grad():
        for i in range(0, len(all_t), batch_size):
            parts.append((-flow().log_prob(all_t[i : i + batch_size])).cpu().numpy())
    return np.concatenate(parts).astype(np.float32)


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latents", required=True)
    p.add_argument("--tag", default=None, help="Output suffix (default: taken from the latents file)")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--reps", nargs="+", default=ALL_REPS, choices=ALL_REPS)
    p.add_argument("--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--knn-k", type=int, default=5)
    p.add_argument("--knn-allow-self", action="store_true",
                   help="Reproduce the legacy FAISS behaviour (train points match themselves at 0)")
    p.add_argument("--no-standardize", dest="standardize", action="store_false")
    p.add_argument("--nsf-epochs", type=int, default=50)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    latents_path = Path(args.latents)
    with h5py.File(latents_path, "r") as f:
        meta = {k: f[k][:] for k in ("row_idx", "ra", "dec", "is_lens")}
        data = {r: f[r][:] for r in args.reps}
        src_attrs = {k: f.attrs[k] for k in f.attrs}
    tag = args.tag or str(src_attrs.get("tag", latents_path.stem.replace("latents_", "")))

    n = len(meta["row_idx"])
    train_idx, test_idx = train_test_split(n, args.train_frac, args.seed)
    train_mask = np.zeros(n, dtype=bool)
    train_mask[train_idx] = True
    print(f"[fit] {latents_path}")
    print(f"[fit] N={n:,}  train={len(train_idx):,}  test={len(test_idx):,}  "
          f"lenses={int(meta['is_lens'].sum())}")
    print(f"[fit] standardize={args.standardize}  knn_self_excluded={not args.knn_allow_self}  "
          f"n_jobs={_n_jobs()}")

    scores: dict[str, np.ndarray] = {}
    for rep in args.reps:
        X = data[rep].astype(np.float32)
        if args.standardize:
            mu = X[train_idx].mean(axis=0)
            sd = X[train_idx].std(axis=0)
            sd[sd < 1e-8] = 1.0
            X = (X - mu) / sd
        Xtr = X[train_idx]
        print(f"\n=== {rep}  dim={X.shape[1]} ===", flush=True)

        for method in args.methods:
            t0 = time.time()
            if method == "knn":
                s = score_knn(Xtr, X, train_mask, k=args.knn_k, device_str=args.device,
                              allow_self=args.knn_allow_self)
            elif method == "iforest":
                s = score_iforest(Xtr, X)
            elif method == "mahal":
                s = score_mahalanobis(Xtr, X)
            elif method == "flow":
                s = score_nsf(Xtr, X, args.nsf_epochs, args.device, seed=args.seed)
            else:
                raise ValueError(method)
            scores[f"{rep}/{method}"] = s
            print(f"  [{method:8s}] {time.time() - t0:6.1f}s  "
                  f"finite={np.isfinite(s).sum():,}/{n:,}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"scores_{tag}.h5"
    fd, tmp = tempfile.mkstemp(suffix=".h5", prefix="scores_", dir=str(out_dir))
    os.close(fd)
    try:
        with h5py.File(tmp, "w") as f:
            for k, v in meta.items():
                f.create_dataset(k, data=v, compression="gzip", compression_opts=4)
            f.create_dataset("in_train", data=train_mask, compression="gzip", compression_opts=4)
            for k, v in scores.items():
                f.create_dataset(k, data=v, compression="gzip", compression_opts=4)
            f.attrs["tag"] = tag
            f.attrs["latents_file"] = str(latents_path)
            f.attrs["checkpoint"] = str(src_attrs.get("checkpoint", ""))
            f.attrs["encoder_pool"] = str(src_attrs.get("encoder_pool", ""))
            f.attrs["n"] = n
            f.attrs["train_frac"] = args.train_frac
            f.attrs["seed"] = args.seed
            f.attrs["standardized"] = args.standardize
            f.attrs["knn_k"] = args.knn_k
            f.attrs["knn_self_excluded"] = not args.knn_allow_self
            f.attrs["nsf_epochs"] = args.nsf_epochs
            f.attrs["score_keys"] = sorted(scores)
        shutil.move(tmp, out_path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise

    print(f"\n[fit] saved: {out_path}")
    print(f"[fit] keys: {sorted(scores)}")


if __name__ == "__main__":
    main()
