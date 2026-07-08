"""Fit Isolation Forest + Normalizing Flow on encoded latents.

Per the HSC-downstream experiment spec, only IForest + NSF are used (no kNN/Mahal).

  --encoder ours-physics   :  reads anomaly_latents_ours_<suffix>.h5  ->  ours/hsc_flat/{iforest,flow}
  --encoder ours-instrument:  reads anomaly_latents_ins_<suffix>.h5   ->  ours/hsc_flat/{iforest,flow}
  --encoder aion           :  reads anomaly_latents_aion_<suffix>.h5  ->  aion/hsc_mean_full/{iforest,flow}
                              NO PCA — raw 768-d AION embeddings.

Output: outputs/anomaly_scores_<encoder_short>_<suffix>.h5
        encoder_short ∈ {ours, ins, aion}.
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
for p in (str(_GALAXY_MODEL), str(_ANOMALY_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import h5py
import numpy as np
import torch
from tqdm import tqdm

# NOTE: the canonical helpers live in anomaly_detection/fit_and_score.py, but
# that module unconditionally calls torch.backends.cuda.preferred_blas_library("hipblas")
# at import time, which errors on NVIDIA GPUs. Inlined here to keep this run
# portable across the engaging (AMD) and orcd (NVIDIA) clusters.

def _train_test_split(n, train_frac, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(n * train_frac)
    return idx[:n_train], idx[n_train:]


def _score_iforest(train, all_latents):
    from sklearn.ensemble import IsolationForest
    clf = IsolationForest(n_estimators=200, contamination="auto",
                          random_state=42, n_jobs=-1)
    clf.fit(train)
    return (-clf.score_samples(all_latents)).astype(np.float32)


def _score_nsf(train, all_latents, epochs, device_str, lr=1e-3, batch_size=512):
    import zuko
    device = torch.device(device_str)
    dim = train.shape[1]
    flow = zuko.flows.NSF(features=dim, context=0, transforms=6,
                          hidden_features=[64, 64]).to(device)
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)

    train_t = torch.tensor(train, dtype=torch.float32, device=device)
    n_train = len(train_t)
    best_loss = float("inf")
    best_state = None

    for _ in tqdm(range(epochs), desc="  NSF training", leave=False):
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


OUTPUT_DIR = _ANOMALY_DIR / "outputs"

ENCODER_CONFIG = {
    "ours-physics": {
        "short": "ours",
        "latents_prefix": "anomaly_latents_ours",
        "h5_key": "hsc_flat",
        "score_namespace": "ours/hsc_flat",
    },
    "ours-instrument": {
        "short": "ins",
        "latents_prefix": "anomaly_latents_ins",
        "h5_key": "hsc_flat",
        "score_namespace": "ours/hsc_flat",
    },
    "aion": {
        "short": "aion",
        "latents_prefix": "anomaly_latents_aion",
        "h5_key": "embeddings_mean_hsc",
        "score_namespace": "aion/hsc_mean_full",
    },
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", required=True, choices=list(ENCODER_CONFIG))
    p.add_argument("--suffix", default="hscdown_474k")
    p.add_argument("--latents", default=None,
                   help="Override the latents file path (default: outputs/<prefix>_<suffix>.h5)")
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--nsf-epochs", type=int, default=50)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    cfg = ENCODER_CONFIG[args.encoder]
    if args.latents is None:
        latents_path = OUTPUT_DIR / f"{cfg['latents_prefix']}_{args.suffix}.h5"
    else:
        latents_path = Path(args.latents)

    print(f"Loading latents from {latents_path}  (key={cfg['h5_key']})")
    with h5py.File(latents_path, "r") as f:
        raw_index = f["raw_index"][:]
        latents = f[cfg["h5_key"]][:]
    n, d = latents.shape
    print(f"  N={n}, dim={d}")

    train_idx, _ = _train_test_split(n, args.train_frac)
    train = latents[train_idx]
    print(f"  train={len(train)}, test={n - len(train)}")

    scores = {}
    ns = cfg["score_namespace"]

    print(f"\n=== {args.encoder} :: Isolation Forest ===")
    scores[f"{ns}/iforest"] = _score_iforest(train, latents)

    print(f"\n=== {args.encoder} :: NSF (epochs={args.nsf_epochs}) ===")
    scores[f"{ns}/flow"] = _score_nsf(train, latents, args.nsf_epochs, args.device)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"anomaly_scores_{cfg['short']}_{args.suffix}.h5"
    tmp_fd, tmp_path = tempfile.mkstemp(
        suffix=".h5", prefix="anomaly_scores_hscdown_", dir=tempfile.gettempdir()
    )
    os.close(tmp_fd)
    try:
        with h5py.File(tmp_path, "w") as f:
            f.create_dataset("raw_index", data=raw_index, compression="gzip", compression_opts=4)
            for k, arr in scores.items():
                f.create_dataset(k, data=arr, compression="gzip", compression_opts=4)
            f.attrs["encoder"] = args.encoder
            f.attrs["suffix"] = args.suffix
            f.attrs["n"] = n
            f.attrs["dim"] = d
            f.attrs["score_keys"] = list(scores.keys())
            f.attrs["data_source"] = "hsc_downstream"
        shutil.move(tmp_path, out_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    print(f"\nSaved scores: {out_path}")
    print(f"Score keys: {list(scores.keys())}")


if __name__ == "__main__":
    main()
