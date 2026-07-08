"""Encode HSC downstream dataset with the trained dual-encoder's PHYSICS encoder.

Mirrors anomaly_detection/encode_latents_ours.py but:
  - Uses HSCDownstreamDataset (memmap+parquet, 474,954 examples) instead of
    NeighborsSimpleDataset.
  - No Legacy branch -> no `hsc_legacy_mean` (only `raw_index`, `hsc_flat`,
    `hsc_mean`).

Run from anomaly_detection/engaging/ or galaxy_model/:
  python anomaly_detection/engaging/encode_ours_hscdown.py \
    --suffix hscdown_474k [--batch-size 256] [--n-max N]
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
for p in (str(_GALAXY_MODEL), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from hsc_downstream_dataset import HSCDownstreamDataset

DEFAULT_CHECKPOINT = str(
    _GALAXY_MODEL / "checkpoints/base/snapshot.ckpt"
)
OUTPUT_DIR = _ANOMALY_DIR / "outputs"


def _collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    raw = np.array([b[1] for b in batch], dtype=np.int64)
    return imgs, raw


def _load_model(checkpoint_path, device):
    from double_train_fm_neighbors import ConditionalFlowMatchingModule
    model = ConditionalFlowMatchingModule.load_from_checkpoint(
        checkpoint_path, map_location="cpu"
    )
    model.eval()
    torch.set_grad_enabled(False)
    return model.to(device)


def encode(checkpoint_path, suffix, batch_size, n_max, device_str):
    device = torch.device(device_str)
    print(f"Loading model from {checkpoint_path}")
    model = _load_model(checkpoint_path, device)
    cross_dim = model.hparams.cross_attention_dim
    print(f"  cross_attention_dim={cross_dim}")

    full = HSCDownstreamDataset(view="ours")
    n_total = len(full)
    n_use = min(n_max, n_total) if n_max is not None else n_total
    dataset = Subset(full, list(range(n_use))) if n_use < n_total else full
    print(f"  Encoding {n_use} examples (total: {n_total})")

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, collate_fn=_collate,
        pin_memory=(device.type == "cuda"),
    )

    raw_index = np.empty(n_use, dtype=np.int64)
    hsc_flat = None
    hsc_mean = None

    written = 0
    with torch.no_grad():
        for imgs, raw in tqdm(loader, desc="Encoding (ours-physics)"):
            imgs = imgs.to(device, non_blocking=True)
            enc = model.encoder_1(imgs)  # (B, n_tokens, cross_dim)
            flat = enc.flatten(start_dim=1).to(torch.float32).cpu().numpy()
            mean = enc.mean(dim=1).to(torch.float32).cpu().numpy()
            b = flat.shape[0]
            if hsc_flat is None:
                hsc_flat = np.empty((n_use, flat.shape[1]), dtype=np.float32)
                hsc_mean = np.empty((n_use, mean.shape[1]), dtype=np.float32)
                n_tokens = enc.shape[1]
            hsc_flat[written:written + b] = flat
            hsc_mean[written:written + b] = mean
            raw_index[written:written + b] = raw
            written += b

    print(f"  raw_index: {raw_index.shape}")
    print(f"  hsc_flat:  {hsc_flat.shape}  (n_tokens={n_tokens}, cross_dim={cross_dim})")
    print(f"  hsc_mean:  {hsc_mean.shape}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"anomaly_latents_ours_{suffix}.h5"
    tmp_fd, tmp_path = tempfile.mkstemp(
        suffix=".h5", prefix="anomaly_latents_ours_hscdown_", dir=tempfile.gettempdir()
    )
    os.close(tmp_fd)
    try:
        with h5py.File(tmp_path, "w") as f:
            f.create_dataset("raw_index", data=raw_index, compression="gzip", compression_opts=4)
            f.create_dataset("hsc_flat", data=hsc_flat, compression="gzip", compression_opts=4)
            f.create_dataset("hsc_mean", data=hsc_mean, compression="gzip", compression_opts=4)
            f.attrs["n_use"] = n_use
            f.attrs["checkpoint"] = str(checkpoint_path)
            f.attrs["suffix"] = suffix
            f.attrs["cross_attention_dim"] = cross_dim
            f.attrs["n_tokens"] = int(n_tokens)
            f.attrs["data_source"] = "hsc_downstream"
        shutil.move(tmp_path, out_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    print(f"Saved: {out_path}")
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--suffix", default="hscdown_474k")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-max", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    encode(args.checkpoint, args.suffix, args.batch_size, args.n_max, args.device)


if __name__ == "__main__":
    main()
