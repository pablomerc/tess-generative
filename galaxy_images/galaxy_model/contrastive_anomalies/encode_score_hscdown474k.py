"""Score a contrastive dual-encoder on the full 474,954-galaxy HSC-downstream corpus.

Why this exists
---------------
``encode_latents.py`` + ``fit_and_score.py`` score the 103,741-anchor ram48 corpus.
The paper's anomaly figure and the FM-control panels are on the 474,954-row
HSC-downstream store, so a top-12 from the anchor corpus is not a like-for-like
comparison (different pool, different rank denominator). This script runs the
contrastive towers over the *same* 474k pool the base checkpoint was scored on.

The preprocessing is identical by construction: the ram48 store the contrastive
model trained on was built with ``preprocess_raw_image(hsc, crop_size=48)[:4]``
(``contrastive_baseline/build_ram48.py``), which is exactly what
``HSCDownstreamDataset(view="ours")`` returns. So the encoder sees in-distribution
inputs, not a rescaled variant.

Protocol matches ``fit_and_score.py``: 80/20 split (seed 42), per-dimension
standardisation fit on train only, zuko NSF (6 transforms, hidden [64,64], 50
epochs, Adam 1e-3), score = negative log-likelihood over the full pool.

Usage
-----
  python contrastive_anomalies/encode_score_hscdown474k.py \
      --ckpt /orcd/pool/007/pablomer/checkpoints_new/contrastive-spatial-conv1x1/best-epoch=53-step=83000.ckpt \
      --tag spatial-conv1x1
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

_HERE = Path(__file__).resolve().parent
_GALAXY_MODEL = _HERE.parent
_PROJECT_ROOT = _GALAXY_MODEL.parents[1]          # .../tess-generative
_ENG = _GALAXY_MODEL / "anomaly_detection" / "engaging"
for _p in (str(_PROJECT_ROOT), str(_GALAXY_MODEL), str(_ENG), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from torch.utils.data import DataLoader  # noqa: E402

from hsc_downstream_dataset import HSCDownstreamDataset  # noqa: E402
from encode_latents import load_contrastive  # noqa: E402

DEFAULT_CKPT = ("/orcd/pool/007/pablomer/checkpoints_new/contrastive-spatial-conv1x1/"
                "best-epoch=53-step=83000.ckpt")
DEFAULT_OUT_DIR = "/orcd/pool/007/pablomer/outputs/contrastive_anomalies"
SUFFIX = "hscdown_474k"


def _collate(batch):
    return torch.stack([b[0] for b in batch]), np.array([b[1] for b in batch], dtype=np.int64)


@torch.no_grad()
def encode_all(model, loader, device, n_total):
    phys, inst, raws = [], [], []
    t0 = time.time()
    seen = 0
    for imgs, raw in loader:
        x = imgs.to(device, non_blocking=True).float()
        phys.append(model.encoder_galaxy(x).flatten(1).float().cpu().numpy())
        inst.append(model.encoder_instrument(x).flatten(1).float().cpu().numpy())
        raws.append(raw)
        seen += len(raw)
        if seen % (256 * 100) < len(raw):
            el = time.time() - t0
            rate = seen / max(el, 1e-6)
            print(f"[encode] {seen:,}/{n_total:,}  ({rate:,.0f} img/s, "
                  f"ETA {(n_total - seen) / max(rate, 1e-6) / 60:.1f} min)", flush=True)
    print(f"[encode] done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return (np.concatenate(phys).astype(np.float32),
            np.concatenate(inst).astype(np.float32),
            np.concatenate(raws))


def train_test_split(n, train_frac=0.8, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    k = int(n * train_frac)
    return idx[:k], idx[k:]


def score_nsf(train, all_latents, epochs, device_str, lr=1e-3, batch_size=512, seed=42):
    """Negative log-likelihood under a zuko NSF fit on `train`. Higher = more anomalous."""
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
            b = tr[perm[i:i + batch_size]]
            loss = -flow().log_prob(b).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.item()); nb += 1
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"    NSF epoch {ep+1}/{epochs}  loss={tot/max(nb,1):.4f}", flush=True)
    out = np.empty(len(all_latents), dtype=np.float32)
    allt = torch.as_tensor(all_latents, dtype=torch.float32, device=device)
    with torch.no_grad():
        for i in range(0, len(allt), 4096):
            out[i:i + 4096] = (-flow().log_prob(allt[i:i + 4096])).cpu().numpy()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--tag", default="spatial-conv1x1")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--nsf-epochs", type=int, default=50)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-standardize", dest="standardize", action="store_false")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n-max", type=int, default=None)
    p.add_argument("--reuse-latents", action="store_true",
                   help="Skip encoding if the latents file already exists.")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lat_path = out_dir / f"latents_{args.tag}_{SUFFIX}.h5"
    sc_path = out_dir / f"scores_{args.tag}_{SUFFIX}.h5"
    device = torch.device(args.device)

    # ── encode ────────────────────────────────────────────────────────────────
    if args.reuse_latents and lat_path.exists():
        print(f"[encode] reusing {lat_path}")
        with h5py.File(lat_path, "r") as f:
            physics = f["physics_hsc"][:]
            instrument = f["instrument_hsc"][:]
            raw_index = f["raw_index"][:]
            ra, dec = f["ra"][:], f["dec"][:]
    else:
        print(f"[encode] ckpt   : {args.ckpt}")
        print(f"[encode] device : {device}")
        model, hp = load_contrastive(args.ckpt, device)
        print(f"[encode] pool={hp.get('encoder_pool','avg')}  class={type(model).__name__}")

        ds = HSCDownstreamDataset(view="ours")
        n_total = len(ds) if args.n_max is None else min(args.n_max, len(ds))
        if args.n_max:
            from torch.utils.data import Subset
            ds_use = Subset(ds, range(n_total))
        else:
            ds_use = ds
        print(f"[encode] corpus: {n_total:,} galaxies (4x48x48, view='ours')")
        loader = DataLoader(ds_use, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=_collate,
                            pin_memory=True, prefetch_factor=4 if args.num_workers else None)
        physics, instrument, raw_index = encode_all(model, loader, device, n_total)
        ra, dec = ds.ra[raw_index], ds.dec[raw_index]

        fd, tmp = tempfile.mkstemp(suffix=".h5", prefix="lat_", dir=str(out_dir))
        os.close(fd)
        try:
            with h5py.File(tmp, "w") as f:
                for k, v in (("physics_hsc", physics), ("instrument_hsc", instrument),
                             ("raw_index", raw_index), ("ra", ra), ("dec", dec)):
                    f.create_dataset(k, data=v, compression="gzip", compression_opts=4)
                f.attrs["checkpoint"] = args.ckpt
                f.attrs["tag"] = args.tag
                f.attrs["suffix"] = SUFFIX
                f.attrs["data_source"] = "hsc_downstream"
                f.attrs["n"] = len(physics)
                f.attrs["self_encoded"] = True
                f.attrs["preprocess"] = "preprocess_raw_image(hsc, crop_size=48)[:4] — same as ram48"
            shutil.move(tmp, lat_path)
        except Exception:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        print(f"[encode] saved {lat_path}")
        for k, v in (("physics_hsc", physics), ("instrument_hsc", instrument)):
            print(f"[encode] {k:16s} {v.shape}  mean={v.mean():+.3f} std={v.std():.3f}")

    # ── fit + score ───────────────────────────────────────────────────────────
    # load_contrastive() calls torch.set_grad_enabled(False) process-wide so the
    # frozen towers can't leak gradients. The NSF fit below needs autograd back.
    torch.set_grad_enabled(True)

    n = len(physics)
    tr_idx, _ = train_test_split(n, args.train_frac, args.seed)
    print(f"[fit] n={n:,}  train={len(tr_idx):,}  standardize={args.standardize}")
    scores = {}
    for space, X in (("physics_hsc", physics), ("instrument_hsc", instrument)):
        Xf = X.astype(np.float32)
        if args.standardize:
            mu = Xf[tr_idx].mean(0, keepdims=True)
            sd = Xf[tr_idx].std(0, keepdims=True)
            sd[sd < 1e-8] = 1.0
            Xf = (Xf - mu) / sd
        print(f"[fit] {space} :: NSF ({args.nsf_epochs} epochs)", flush=True)
        t0 = time.time()
        scores[f"{space}/flow"] = score_nsf(Xf[tr_idx], Xf, args.nsf_epochs,
                                            args.device, seed=args.seed)
        print(f"[fit] {space} done in {(time.time()-t0)/60:.1f} min", flush=True)

    fd, tmp = tempfile.mkstemp(suffix=".h5", prefix="sc_", dir=str(out_dir))
    os.close(fd)
    try:
        with h5py.File(tmp, "w") as f:
            f.create_dataset("raw_index", data=raw_index, compression="gzip", compression_opts=4)
            f.create_dataset("ra", data=ra, compression="gzip", compression_opts=4)
            f.create_dataset("dec", data=dec, compression="gzip", compression_opts=4)
            for k, v in scores.items():
                f.create_dataset(k, data=v, compression="gzip", compression_opts=4)
            f.attrs["checkpoint"] = args.ckpt
            f.attrs["tag"] = args.tag
            f.attrs["suffix"] = SUFFIX
            f.attrs["data_source"] = "hsc_downstream"
            f.attrs["n"] = n
            f.attrs["train_frac"] = args.train_frac
            f.attrs["seed"] = args.seed
            f.attrs["standardized"] = args.standardize
            f.attrs["nsf_epochs"] = args.nsf_epochs
            f.attrs["score_keys"] = list(scores)
        shutil.move(tmp, sc_path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    print(f"[fit] saved {sc_path}")


if __name__ == "__main__":
    main()
