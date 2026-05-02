"""Encode latents for anomaly detection (one variant)."""

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

_THIS = Path(__file__).resolve()
_PKG_ROOT = _THIS.parents[4]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from galaxy_images.galaxy_model.hierarchical_latent_experiments.common.ckpt import (
    DEFAULT_CKPT, load_trained,
)
from galaxy_images.galaxy_model.hierarchical_latent_experiments.common.data_loaders import (
    NEIGHBORS_HDF5_DEFAULT, make_loader, make_pair_dataset,
)
from galaxy_images.galaxy_model.hierarchical_latent_experiments.common.discord_utils import post_text
from galaxy_images.galaxy_model.hierarchical_latent_experiments.common.latents import (
    VARIANTS, extract_both,
)


def encode(variant: str, ckpt: str, hdf5_path: str, out_path: Path,
           batch_size: int, n_max: int | None, device_str: str, post_discord: bool):
    if post_discord:
        post_text(f"🟢 [anomalies/encode/{variant}] starting (n_max={n_max})")

    t0 = time.time()
    device = torch.device(device_str)
    model = load_trained(ckpt, device=device)

    full = make_pair_dataset(hdf5_path=hdf5_path, n=n_max, shuffle=False)
    print(f"variant={variant}, dataset size={len(full)}")
    loader = make_loader(full, batch_size=batch_size, num_workers=4,
                         pin_memory=(device.type == "cuda"))

    raw_idx, hsc_p, leg_p, hsc_i, leg_i = [], [], [], [], []
    with torch.no_grad():
        for hsc, leg, raw, _ in tqdm(loader, desc=f"encode[{variant}]"):
            hsc = hsc.to(device, non_blocking=True)
            leg = leg.to(device, non_blocking=True)
            hp, hi = extract_both(model, hsc, variant)
            lp, li = extract_both(model, leg, variant)
            hsc_p.append(hp.float().cpu().numpy().astype(np.float32))
            leg_p.append(lp.float().cpu().numpy().astype(np.float32))
            hsc_i.append(hi.float().cpu().numpy().astype(np.float32))
            leg_i.append(li.float().cpu().numpy().astype(np.float32))
            raw_idx.append(raw)

    raw_idx = np.concatenate(raw_idx)
    hsc_p = np.concatenate(hsc_p)
    leg_p = np.concatenate(leg_p)
    hsc_i = np.concatenate(hsc_i)
    leg_i = np.concatenate(leg_i)
    hsc_leg_p = np.concatenate([hsc_p, leg_p], axis=1)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp = tempfile.mkstemp(suffix=".h5", prefix="anomaly_latents_", dir=tempfile.gettempdir())
    os.close(fd)
    try:
        with h5py.File(tmp, "w") as f:
            f.create_dataset("raw_index", data=raw_idx, compression="gzip", compression_opts=4)
            # Names mimic the original anomaly script so visualize.py works with minimal change.
            # hsc_flat -> physics latent on HSC; hsc_mean -> kept-name alias for instrument; hsc_legacy_mean -> physics concat.
            f.create_dataset("hsc_flat", data=hsc_p, compression="gzip", compression_opts=4)
            f.create_dataset("legacy_flat", data=leg_p, compression="gzip", compression_opts=4)
            f.create_dataset("hsc_legacy_concat", data=hsc_leg_p, compression="gzip", compression_opts=4)
            f.create_dataset("instrument_hsc", data=hsc_i, compression="gzip", compression_opts=4)
            f.create_dataset("instrument_legacy", data=leg_i, compression="gzip", compression_opts=4)
            f.attrs["variant"] = variant
            f.attrs["n"] = int(raw_idx.shape[0])
            f.attrs["ckpt"] = str(ckpt)
            f.attrs["physics_dim"] = int(hsc_p.shape[1])
            f.attrs["instrument_dim"] = int(hsc_i.shape[1])
        shutil.move(tmp, out_path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise

    elapsed = time.time() - t0
    print(f"saved {out_path} in {elapsed:.1f}s; physics_dim={hsc_p.shape[1]} instrument_dim={hsc_i.shape[1]}")
    if post_discord:
        post_text(f"💾 [anomalies/encode/{variant}] saved {out_path.name} in {elapsed:.0f}s "
                  f"(N={raw_idx.shape[0]}, phys={hsc_p.shape[1]}d, inst={hsc_i.shape[1]}d)")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--hdf5", default=NEIGHBORS_HDF5_DEFAULT)
    ap.add_argument("--out-path", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--n-max", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-discord", action="store_true")
    args = ap.parse_args()

    try:
        encode(args.variant, args.ckpt, args.hdf5, args.out_path,
               args.batch_size, args.n_max, args.device, post_discord=not args.no_discord)
    except Exception as exc:
        if not args.no_discord:
            post_text(f"❌ [anomalies/encode/{args.variant}] FAILED: {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
