"""
Legacy-primary variant of anomaly_detection/encode_latents_instrument.py.

Encodes all source_type=0 (MMU) examples with encoder_2 (instrument latent),
using Legacy images as the primary input. Dataset names match the HSC pipeline
so fit_and_score.py works unchanged; primary_survey="legacy" disambiguates.

Run from galaxy_model/:
  python anomaly_detection_legacy/encode_latents_instrument.py \
    --checkpoint outputs/neighbors_all_attn/2026-04-05/checkpoints/best-epoch=228-step=87000.ckpt \
    --suffix ins_legacy_100k \
    [--batch-size 256] [--n-max N] [--device cuda]
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
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from neighbors import NeighborsSimpleDataset

NEIGHBORS_HDF5 = "/work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5"
DEFAULT_CHECKPOINT = str(
    _src / "outputs/neighbors_all_attn/2026-04-05/checkpoints/best-epoch=228-step=87000.ckpt"
)
OUTPUT_DIR = _here / "outputs"


def _collate_fn(batch):
    hsc_imgs = torch.stack([b[0] for b in batch])
    leg_imgs = torch.stack([b[1] for b in batch])
    raw_indices = np.array([b[2]["index_mmu"] for b in batch], dtype=np.int64)
    return hsc_imgs, leg_imgs, raw_indices


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
    ins_dim = model.instrument_zdim
    print(f"  instrument_zdim={ins_dim}")

    print(f"Loading dataset from {NEIGHBORS_HDF5}")
    full_dataset = NeighborsSimpleDataset(hdf5_path=NEIGHBORS_HDF5)
    n_total = len(full_dataset)
    n_use = min(n_max, n_total) if n_max is not None else n_total
    dataset = Subset(full_dataset, list(range(n_use)))
    print(f"  Encoding {n_use} examples (total MMU: {n_total}) — primary survey: LEGACY")

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, collate_fn=_collate_fn, pin_memory=(device.type == "cuda"),
    )

    all_raw_index = []
    all_hsc_flat = []
    all_hsc_mean = []
    all_hsc_legacy_mean = []

    with torch.no_grad():
        for hsc_imgs, leg_imgs, raw_indices in tqdm(loader, desc="Encoding"):
            hsc_imgs = hsc_imgs.to(device)
            leg_imgs = leg_imgs.to(device)

            enc2_leg = model.encoder_2(leg_imgs)   # primary
            enc2_hsc = model.encoder_2(hsc_imgs)   # auxiliary

            hsc_flat = enc2_leg.flatten(start_dim=1).cpu().numpy().astype(np.float32)
            hsc_mean = enc2_leg.mean(dim=1).cpu().numpy().astype(np.float32)
            hsc_legacy_mean = torch.cat([enc2_leg, enc2_hsc], dim=1).mean(dim=1).cpu().numpy().astype(np.float32)

            all_raw_index.append(raw_indices)
            all_hsc_flat.append(hsc_flat)
            all_hsc_mean.append(hsc_mean)
            all_hsc_legacy_mean.append(hsc_legacy_mean)

    raw_index = np.concatenate(all_raw_index, axis=0)
    hsc_flat = np.concatenate(all_hsc_flat, axis=0)
    hsc_mean = np.concatenate(all_hsc_mean, axis=0)
    hsc_legacy_mean = np.concatenate(all_hsc_legacy_mean, axis=0)

    print(f"  raw_index: {raw_index.shape}")
    print(f"  hsc_flat:  {hsc_flat.shape}  (n_tokens={enc2_leg.shape[1]}, ins_dim={ins_dim})")
    print(f"  hsc_mean:  {hsc_mean.shape}")
    print(f"  hsc_legacy_mean: {hsc_legacy_mean.shape}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"anomaly_latents_ours_{suffix}.h5"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".h5", prefix="anomaly_latents_ins_legacy_", dir=tempfile.gettempdir())
    os.close(tmp_fd)
    try:
        with h5py.File(tmp_path, "w") as f:
            f.create_dataset("raw_index", data=raw_index, compression="gzip", compression_opts=4)
            f.create_dataset("hsc_flat", data=hsc_flat, compression="gzip", compression_opts=4)
            f.create_dataset("hsc_mean", data=hsc_mean, compression="gzip", compression_opts=4)
            f.create_dataset("hsc_legacy_mean", data=hsc_legacy_mean, compression="gzip", compression_opts=4)
            f.attrs["n_use"] = n_use
            f.attrs["checkpoint"] = str(checkpoint_path)
            f.attrs["suffix"] = suffix
            f.attrs["encoder"] = "encoder_2 (instrument)"
            f.attrs["instrument_zdim"] = int(ins_dim)
            f.attrs["n_tokens"] = int(enc2_leg.shape[1])
            f.attrs["primary_survey"] = "legacy"
        shutil.move(tmp_path, out_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    print(f"Saved: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--suffix", default="ins_legacy_100k")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-max", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    encode(args.checkpoint, args.suffix, args.batch_size, args.n_max, args.device)


if __name__ == "__main__":
    main()
