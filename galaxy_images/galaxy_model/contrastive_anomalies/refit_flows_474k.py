"""Refit NSF anomaly flows for every 474k arm, standardized AND unstandardized.

One code path for all arms so the only difference between rows of the comparison is
the encoder (and the standardize flag) — not the flow implementation, split, or seed.

Arms (each is one latents file + one key):
  base-physics     anomaly_latents_ours_hscdown_474k.h5   hsc_flat     (64D)  base/snapshot.ckpt
  base-instrument  anomaly_latents_ins_hscdown_474k.h5    hsc_flat     (64D)  base/snapshot.ckpt
  aion             anomaly_latents_aion_hscdown_474k.h5   embeddings_mean_hsc (768D)
  fmctrl-physics   anomaly_latents_ours_fmctrl_474k.h5    hsc_flat     (64D)  diffusion-ablation FM control
  fmctrl-instrument anomaly_latents_ins_fmctrl_474k.h5    hsc_flat     (64D)  diffusion-ablation FM control
  contra-physics   latents_spatial-conv1x1_hscdown_474k.h5  physics_hsc    (64D)
  contra-instrument latents_spatial-conv1x1_hscdown_474k.h5 instrument_hsc (64D)

Protocol (identical to contrastive_anomalies/fit_and_score.py):
  80/20 split, seed 42 · optional per-dim standardisation fit on train only
  zuko NSF, 6 transforms, hidden [64,64], 50 epochs, Adam 1e-3, batch 512
  score = negative log-likelihood over the full pool (higher = more anomalous)

Pool (``--pool``):
  367k  the paper-figure pool — ``source_type in {0,1}`` of neighbours_v2.h5,
        366,706 galaxies, a strict subset of the 474k store (see
        build_pool367k_index.py). This is the default: it is the pool the
        published anomaly figure was made on.
  474k  the whole hsc_downstream store (108,248 galaxies the paper never scored).

The pool changes the flow's training set and the rank denominator, so scores are
NOT interchangeable between pools — hence the pool tag in every filename.

Output: outputs/contrastive_anomalies/flows474k/scores_<arm>_<std|raw>_<pool>.h5
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import torch

ANOM = Path("/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model/anomaly_detection/outputs")
CONTRA_DIR = Path("/orcd/pool/007/pablomer/outputs/contrastive_anomalies")
OUT_DIR = CONTRA_DIR / "flows474k"
POOL367K = CONTRA_DIR / "pool367k_rows.npy"

ARMS = {
    "base-physics":      (ANOM / "anomaly_latents_ours_hscdown_474k.h5", "hsc_flat"),
    "base-instrument":   (ANOM / "anomaly_latents_ins_hscdown_474k.h5", "hsc_flat"),
    "aion":              (ANOM / "anomaly_latents_aion_hscdown_474k.h5", "embeddings_mean_hsc"),
    "fmctrl-physics":    (ANOM / "anomaly_latents_ours_fmctrl_474k.h5", "hsc_flat"),
    "fmctrl-instrument": (ANOM / "anomaly_latents_ins_fmctrl_474k.h5", "hsc_flat"),
    "contra-physics":    (CONTRA_DIR / "latents_spatial-conv1x1_hscdown_474k.h5", "physics_hsc"),
    "contra-instrument": (CONTRA_DIR / "latents_spatial-conv1x1_hscdown_474k.h5", "instrument_hsc"),
}


def train_test_split(n, train_frac=0.8, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return idx[: int(n * train_frac)]


def score_nsf(train, all_latents, epochs, device_str, lr=1e-3, batch_size=512, seed=42):
    import zuko

    device = torch.device(device_str)
    torch.manual_seed(seed)
    flow = zuko.flows.NSF(features=train.shape[1], context=0, transforms=6,
                          hidden_features=[64, 64]).to(device)
    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    tr = torch.as_tensor(train, dtype=torch.float32, device=device)
    n = len(tr)
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        tot, nb = 0.0, 0
        for i in range(0, n, batch_size):
            loss = -flow().log_prob(tr[perm[i:i + batch_size]]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.item()); nb += 1
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"      epoch {ep+1}/{epochs}  loss={tot/max(nb,1):.4f}", flush=True)
    out = np.empty(len(all_latents), dtype=np.float32)
    allt = torch.as_tensor(all_latents, dtype=torch.float32, device=device)
    with torch.no_grad():
        for i in range(0, len(allt), 4096):
            out[i:i + 4096] = (-flow().log_prob(allt[i:i + 4096])).cpu().numpy()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    p.add_argument("--variants", nargs="+", default=["std", "raw"], choices=["std", "raw"])
    p.add_argument("--pool", default="367k", choices=["367k", "474k"],
                   help="367k = the paper-figure pool (default); 474k = whole store.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pool_rows = np.load(POOL367K) if args.pool == "367k" else None
    if pool_rows is not None:
        print(f"[pool] restricting to the paper pool: {len(pool_rows):,} rows")

    for arm in args.arms:
        lat_path, key = ARMS[arm]
        if not lat_path.exists():
            print(f"[skip] {arm}: missing {lat_path.name}")
            continue
        with h5py.File(lat_path, "r") as f:
            X = f[key][:].astype(np.float32)
            raw_index = f["raw_index"][:] if "raw_index" in f else np.arange(len(X))
            ra = f["ra"][:] if "ra" in f else None
            dec = f["dec"][:] if "dec" in f else None
        if pool_rows is not None:
            sel = np.isin(raw_index, pool_rows)
            if sel.sum() != len(pool_rows):
                print(f"[warn] {arm}: pool covers {int(sel.sum()):,} of {len(pool_rows):,} rows")
            X, raw_index = X[sel], raw_index[sel]
            if ra is not None:
                ra, dec = ra[sel], dec[sel]
        tr = train_test_split(len(X), args.train_frac, args.seed)
        for variant in args.variants:
            out = OUT_DIR / f"scores_{arm}_{variant}_{args.pool}.h5"
            if out.exists() and not args.force:
                print(f"[skip] {out.name} exists")
                continue
            Xf = X
            if variant == "std":
                mu = X[tr].mean(0, keepdims=True)
                sd = X[tr].std(0, keepdims=True)
                sd[sd < 1e-8] = 1.0
                Xf = (X - mu) / sd
            print(f"[fit] {arm} / {variant}  n={len(X):,} dim={X.shape[1]} "
                  f"train={len(tr):,}", flush=True)
            t0 = time.time()
            s = score_nsf(Xf[tr], Xf, args.epochs, args.device, seed=args.seed)
            print(f"[fit] {arm}/{variant} done in {(time.time()-t0)/60:.1f} min "
                  f"(finite {np.isfinite(s).sum():,}/{len(s):,})", flush=True)
            with h5py.File(out, "w") as f:
                f.create_dataset("flow", data=s, compression="gzip", compression_opts=4)
                f.create_dataset("raw_index", data=raw_index, compression="gzip", compression_opts=4)
                if ra is not None:
                    f.create_dataset("ra", data=ra, compression="gzip", compression_opts=4)
                    f.create_dataset("dec", data=dec, compression="gzip", compression_opts=4)
                f.attrs["arm"] = arm
                f.attrs["latents"] = str(lat_path)
                f.attrs["latent_key"] = key
                f.attrs["standardized"] = (variant == "std")
                f.attrs["epochs"] = args.epochs
                f.attrs["seed"] = args.seed
                f.attrs["train_frac"] = args.train_frac
                f.attrs["dim"] = X.shape[1]
                f.attrs["pool"] = args.pool
                f.attrs["n_pool"] = len(X)
            print(f"[fit] wrote {out}")


if __name__ == "__main__":
    main()
