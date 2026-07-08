"""
Step 0 (shared): encode the redshift-labeled HSC subset into the Ours-Physics latent.

Selects every row of hsc_downstream/catalog.parquet with a finite desi_z > 0 (~9,210),
loads each HSC image from hsc_flux.bin with the exact training preprocessing, encodes
the physics latent (encoder_1 mean-pooled), and writes a single HDF5 consumed by both
the unconditional (Job A) and conditional (Job B) scoring jobs.

Run from galaxy_model/ (or anywhere):
  python anomaly_detection/redshift_experiment/encode_redshift_subset.py \
    [--checkpoint .../checkpoints/base/snapshot.ckpt] \
    [--batch-size 256] [--limit N] [--device cuda]
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import common  # noqa: E402  (sets sys.path, applies hipblas)

_HERE = Path(__file__).resolve().parent
OUTPUT_DIR = _HERE / "outputs"


def main():
    p = argparse.ArgumentParser(description="Encode redshift-labeled HSC subset -> physics/instrument latent.")
    p.add_argument("--checkpoint", default=common.DEFAULT_CHECKPOINT)
    p.add_argument("--catalog", type=Path, default=common.DEFAULT_CATALOG)
    p.add_argument("--images-bin", type=Path, default=common.DEFAULT_IMAGES_BIN)
    p.add_argument("--encoder", choices=list(common.ENCODE_FNS), default="physics",
                   help="Which latent space to encode: encoder_1 (physics) or encoder_2 (instrument).")
    p.add_argument("--out", type=Path, default=None,
                   help="Output H5. Default: outputs/latents_redshift[_instrument].h5 keyed to --encoder.")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--limit", type=int, default=None, help="Smoke test: only first N selected rows.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if args.out is None:
        suffix = "" if args.encoder == "physics" else "_instrument"
        args.out = OUTPUT_DIR / f"latents_redshift{suffix}.h5"
    encode_fn = common.ENCODE_FNS[args.encoder]
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # --- select redshift-labeled rows (positional row == .bin record) ---
    df = pd.read_parquet(args.catalog).reset_index(drop=True)
    z = pd.to_numeric(df["desi_z"], errors="coerce").to_numpy(dtype=np.float64)
    sel = np.where(np.isfinite(z) & (z > 0))[0].astype(np.int64)  # positional indices
    print(f"Catalog rows: {len(df):,} | finite desi_z>0: {len(sel):,}")
    if args.limit is not None:
        sel = sel[: args.limit]
        print(f"  [smoke] limited to {len(sel)} rows")

    desi_z = z[sel].astype(np.float32)
    ra = pd.to_numeric(df["ra"], errors="coerce").to_numpy(dtype=np.float64)[sel] if "ra" in df else np.full(len(sel), np.nan)
    dec = pd.to_numeric(df["dec"], errors="coerce").to_numpy(dtype=np.float64)[sel] if "dec" in df else np.full(len(sel), np.nan)
    image_idx_col = df["image_idx"].to_numpy()[sel] if "image_idx" in df else sel.copy()

    # PROVABGS physics properties used by Fix 3's multi-property distribution figure.
    # All 5 confirmed 100% finite-coverage on the desi_z>0 subset (see plan: Fix 3).
    provabgs_cols = ["provabgs_logmstar", "provabgs_tage_mw", "provabgs_z_mw", "provabgs_avg_sfr"]
    props = {}
    for col in provabgs_cols:
        if col in df.columns:
            arr = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)[sel]
            props[col] = arr.astype(np.float32)
            n_fin = int(np.isfinite(arr).sum())
            print(f"  {col}: {n_fin}/{len(sel)} finite, range [{np.nanmin(arr):.3g}, {np.nanmax(arr):.3g}]")
        else:
            print(f"  {col}: NOT in catalog -- skipping")

    # --- encode ---
    device = torch.device(args.device)
    print(f"Device: {device} | encoder: {args.encoder} | checkpoint: {args.checkpoint}")
    model = common.load_model(args.checkpoint, device=str(device))

    ds = common.HSCBinDataset(args.images_bin, sel)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
                        pin_memory=(device.type == "cuda"))
    parts, total = [], 0
    with torch.no_grad():
        for imgs in loader:
            imgs = imgs.to(device)
            parts.append(encode_fn(model, imgs).float().cpu().numpy())
            total += imgs.size(0)
            if total % 2048 < args.batch_size:
                print(f"  encoded {total}/{len(ds)}")
    hsc_mean = np.concatenate(parts, axis=0).astype(np.float32)
    print(f"hsc_mean shape: {hsc_mean.shape}")

    # --- write ---
    with h5py.File(args.out, "w") as f:
        f.create_dataset("hsc_mean", data=hsc_mean, compression="gzip", compression_opts=4)
        f.create_dataset("record_idx", data=sel, compression="gzip", compression_opts=4)
        f.create_dataset("image_idx", data=np.asarray(image_idx_col, dtype=np.int64), compression="gzip", compression_opts=4)
        f.create_dataset("desi_z", data=desi_z, compression="gzip", compression_opts=4)
        f.create_dataset("ra", data=ra.astype(np.float64), compression="gzip", compression_opts=4)
        f.create_dataset("dec", data=dec.astype(np.float64), compression="gzip", compression_opts=4)
        g = f.create_group("props")
        for col, arr in props.items():
            g.create_dataset(col, data=arr, compression="gzip", compression_opts=4)
        f.attrs["checkpoint"] = str(args.checkpoint)
        f.attrs["encoder"] = args.encoder
        f.attrs["images_bin"] = str(args.images_bin)
        f.attrs["n"] = len(sel)
        f.attrs["latent_dim"] = hsc_mean.shape[1]
    print(f"Saved {args.out}  (N={len(sel)}, D={hsc_mean.shape[1]}, props={list(props.keys())})")


if __name__ == "__main__":
    main()
