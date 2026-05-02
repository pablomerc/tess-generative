"""
Legacy-primary variant of anomaly_detection/encode_latents_contrastive.py.

Saves three latent representations per example using the dual-encoder contrastive
model, but with Legacy images as primary input. Dataset names match the HSC
pipeline so fit_and_score.py works unchanged. Semantics:
  hsc_flat        = encoder_galaxy(leg_imgs)        (single-vector, same as hsc_mean)
  hsc_mean        = encoder_galaxy(leg_imgs)
  hsc_legacy_mean = mean(encoder_galaxy(leg), encoder_galaxy(hsc))
  instrument_emb  = encoder_instrument(leg_imgs)

Run from galaxy_model/:
  python anomaly_detection_legacy/encode_latents_contrastive.py \
    --checkpoint /path/to/best-epoch=XX-step=YY.ckpt \
    --suffix contrastive_legacy_100k \
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
OUTPUT_DIR = _here / "outputs"


def _collate_fn(batch):
    hsc_imgs = torch.stack([b[0] for b in batch])
    leg_imgs = torch.stack([b[1] for b in batch])
    raw_indices = np.array([b[2]["index_mmu"] for b in batch], dtype=np.int64)
    return hsc_imgs, leg_imgs, raw_indices


def _load_model(checkpoint_path, device):
    _contrastive_dir = _src / "contrastive_baseline"
    if str(_contrastive_dir) not in sys.path:
        sys.path.insert(0, str(_contrastive_dir))
    from dual_encoder_contrastive import DualEncoderContrastiveModule
    model = DualEncoderContrastiveModule.load_from_checkpoint(
        checkpoint_path, map_location="cpu"
    )
    model.eval()
    torch.set_grad_enabled(False)
    return model.to(device)


def encode(checkpoint_path, suffix, batch_size, n_max, device_str):
    device = torch.device(device_str)
    print(f"Loading model from {checkpoint_path}")
    model = _load_model(checkpoint_path, device)
    embedding_dim = model.hparams.embedding_dim
    print(f"  embedding_dim={embedding_dim}")

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
    all_hsc_mean = []
    all_hsc_legacy_mean = []
    all_instrument_emb = []

    with torch.no_grad():
        for hsc_imgs, leg_imgs, raw_indices in tqdm(loader, desc="Encoding"):
            hsc_imgs = hsc_imgs.to(device)
            leg_imgs = leg_imgs.to(device)

            z_galaxy_leg = model.encoder_galaxy(leg_imgs)      # primary
            z_galaxy_hsc = model.encoder_galaxy(hsc_imgs)
            z_instrument = model.encoder_instrument(leg_imgs)  # primary

            hsc_mean = z_galaxy_leg.cpu().numpy().astype(np.float32)
            hsc_legacy_mean = ((z_galaxy_leg + z_galaxy_hsc) / 2).cpu().numpy().astype(np.float32)
            instrument_emb = z_instrument.cpu().numpy().astype(np.float32)

            all_raw_index.append(raw_indices)
            all_hsc_mean.append(hsc_mean)
            all_hsc_legacy_mean.append(hsc_legacy_mean)
            all_instrument_emb.append(instrument_emb)

    raw_index = np.concatenate(all_raw_index, axis=0)
    hsc_mean = np.concatenate(all_hsc_mean, axis=0)
    hsc_legacy_mean = np.concatenate(all_hsc_legacy_mean, axis=0)
    instrument_emb = np.concatenate(all_instrument_emb, axis=0)

    print(f"  raw_index:       {raw_index.shape}")
    print(f"  hsc_mean:        {hsc_mean.shape}")
    print(f"  hsc_legacy_mean: {hsc_legacy_mean.shape}")
    print(f"  instrument_emb:  {instrument_emb.shape}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"anomaly_latents_ours_{suffix}.h5"
    tmp_fd, tmp_path = tempfile.mkstemp(
        suffix=".h5", prefix="anomaly_latents_contrastive_legacy_", dir=tempfile.gettempdir()
    )
    os.close(tmp_fd)
    try:
        with h5py.File(tmp_path, "w") as f:
            f.create_dataset("raw_index",        data=raw_index,        compression="gzip", compression_opts=4)
            f.create_dataset("hsc_flat",         data=hsc_mean,         compression="gzip", compression_opts=4)
            f.create_dataset("hsc_mean",         data=hsc_mean,         compression="gzip", compression_opts=4)
            f.create_dataset("hsc_legacy_mean",  data=hsc_legacy_mean,  compression="gzip", compression_opts=4)
            f.create_dataset("instrument_emb",   data=instrument_emb,   compression="gzip", compression_opts=4)
            f.attrs["n_use"] = n_use
            f.attrs["checkpoint"] = str(checkpoint_path)
            f.attrs["suffix"] = suffix
            f.attrs["embedding_dim"] = embedding_dim
            f.attrs["model_type"] = "DualEncoderContrastiveModule"
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
    parser.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    parser.add_argument("--suffix", required=True, help="Output file suffix")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-max", type=int, default=None, help="Cap number of examples (default: all)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    encode(args.checkpoint, args.suffix, args.batch_size, args.n_max, args.device)


if __name__ == "__main__":
    main()
