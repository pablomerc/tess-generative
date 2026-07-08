"""
Package the HSC ∩ Legacy overlap subset into a single self-contained HDF5 file
for transfer to another cluster.

Selection: rows of outputs/index/overlap.parquet that also have a DESI redshift
(desi_z) — the "~5k examples with both images and physics properties" subset.

Outputs one HDF5 file with:
  hsc_flux       (N, 5, 160, 160) float16   raw HSC g/r/i/z/y flux from .bin
  legacy_flux    (N, 4, 160, 160) float16   raw Legacy g/r/i/z flux from .bin
  redshift       (N,) float32               desi_z
  redshift_err   (N,) float32               desi_zerr
  physics/<col>  (N,) float32               provabgs columns (logmstar, sfr, ...)
  meta/<col>     metadata: image_idx, object_id, ra, dec, sep_arcsec
"""

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

HSC_BANDS, HSC_H, HSC_W = 5, 160, 160
LEGACY_BANDS, LEGACY_H, LEGACY_W = 4, 160, 160
HSC_BYTES = HSC_BANDS * HSC_H * HSC_W * 2  # float16
LEGACY_BYTES = LEGACY_BANDS * LEGACY_H * LEGACY_W * 2

PHYSICS_COLS = [
    "provabgs_logmstar",
    "provabgs_tage_mw",
    "provabgs_avg_sfr",
    "provabgs_z_mw",
    "provabgs_w1_mag",
]


def read_images(bin_path: Path, indices: np.ndarray, bands: int, h: int, w: int) -> np.ndarray:
    n = len(indices)
    out = np.empty((n, bands, h, w), dtype=np.float16)
    rec_bytes = bands * h * w * 2
    with open(bin_path, "rb") as fh:
        for i, idx in enumerate(indices):
            fh.seek(int(idx) * rec_bytes)
            buf = fh.read(rec_bytes)
            out[i] = np.frombuffer(buf, dtype=np.float16).reshape(bands, h, w)
            if (i + 1) % 1000 == 0:
                print(f"    {i+1}/{n}", flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", type=Path,
                   default=Path("outputs/index/overlap.parquet"))
    p.add_argument("--hsc-dir", type=Path,
                   default=Path("/home/pablomer/orcd/scratch/hsc_downstream"))
    p.add_argument("--legacy-dir", type=Path,
                   default=Path("/home/pablomer/orcd/scratch/legacy_downstream_full/full_1M"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--require-redshift", action="store_true", default=True)
    args = p.parse_args()

    here = Path(__file__).resolve().parent
    if not args.index.is_absolute():
        args.index = here / args.index

    print(f"Reading index: {args.index}")
    overlap = pd.read_parquet(args.index)
    print(f"  rows: {len(overlap):,}")

    physics_cols = ["desi_z", "desi_zerr"] + PHYSICS_COLS

    print(f"Reading HSC catalog physics cols")
    hsc_cat = pd.read_parquet(
        args.hsc_dir / "catalog.parquet",
        columns=["image_idx", "object_id"] + physics_cols,
    ).set_index("image_idx")
    print(f"Reading Legacy catalog physics cols")
    leg_cat = pd.read_parquet(
        args.legacy_dir / "catalog.parquet",
        columns=["image_idx", "object_id"] + physics_cols,
    ).set_index("image_idx")

    hsc_rows = hsc_cat.loc[overlap["hsc_image_idx"].values].reset_index(drop=True)
    leg_rows = leg_cat.loc[overlap["legacy_image_idx"].values].reset_index(drop=True)

    # The two surveys' DESI columns are populated on the same DESI targets, so
    # use either side. Prefer HSC-side; fall back to Legacy-side per-row.
    z = hsc_rows["desi_z"].where(hsc_rows["desi_z"].notna(), leg_rows["desi_z"]).values
    zerr = hsc_rows["desi_zerr"].where(hsc_rows["desi_zerr"].notna(), leg_rows["desi_zerr"]).values

    keep = ~pd.isna(z) if args.require_redshift else np.ones(len(z), dtype=bool)
    overlap = overlap.iloc[keep].reset_index(drop=True)
    z = z[keep].astype(np.float32)
    zerr = zerr[keep].astype(np.float32)
    hsc_rows = hsc_rows.iloc[keep].reset_index(drop=True)
    leg_rows = leg_rows.iloc[keep].reset_index(drop=True)
    print(f"  rows with redshift: {len(overlap):,}")

    hsc_indices = overlap["hsc_image_idx"].to_numpy(dtype=np.int64)
    leg_indices = overlap["legacy_image_idx"].to_numpy(dtype=np.int64)

    n_hsc_bin = os.path.getsize(args.hsc_dir / "hsc_flux.bin") // HSC_BYTES
    n_leg_bin = os.path.getsize(args.legacy_dir / "legacy_flux.bin") // LEGACY_BYTES
    assert (hsc_indices < n_hsc_bin).all()
    assert (leg_indices < n_leg_bin).all()

    print(f"Reading HSC images ({len(hsc_indices):,} × {HSC_BANDS}×{HSC_H}×{HSC_W} float16)")
    hsc_flux = read_images(args.hsc_dir / "hsc_flux.bin",
                           hsc_indices, HSC_BANDS, HSC_H, HSC_W)
    print(f"Reading Legacy images ({len(leg_indices):,} × {LEGACY_BANDS}×{LEGACY_H}×{LEGACY_W} float16)")
    legacy_flux = read_images(args.legacy_dir / "legacy_flux.bin",
                              leg_indices, LEGACY_BANDS, LEGACY_H, LEGACY_W)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {args.out}")
    with h5py.File(args.out, "w") as f:
        f.attrs["description"] = (
            "HSC ∩ Legacy overlap subset with DESI redshifts. HSC bands=g,r,i,z,y; "
            "Legacy bands=g,r,i,z. Images are raw float16 nanomaggy fluxes, "
            "160x160 pixels at native survey scale (HSC pixscale ~0.168\"/pix, "
            "Legacy pixscale ~0.262\"/pix)."
        )
        f.attrs["hsc_bands"] = ["g", "r", "i", "z", "y"]
        f.attrs["legacy_bands"] = ["g", "r", "i", "z"]
        f.attrs["n"] = len(overlap)

        f.create_dataset("hsc_flux", data=hsc_flux,
                         compression="gzip", compression_opts=4, shuffle=True)
        f.create_dataset("legacy_flux", data=legacy_flux,
                         compression="gzip", compression_opts=4, shuffle=True)
        f.create_dataset("redshift", data=z,
                         compression="gzip", compression_opts=4)
        f.create_dataset("redshift_err", data=zerr,
                         compression="gzip", compression_opts=4)

        # Physics properties — one per column. Take HSC side first, fall back to Legacy.
        for col in PHYSICS_COLS:
            v = hsc_rows[col].where(hsc_rows[col].notna(), leg_rows[col]).values
            v = v.astype(np.float32)
            f.create_dataset(f"physics/{col}", data=v,
                             compression="gzip", compression_opts=4)

        # Meta — coordinates and source indices/IDs
        f.create_dataset("meta/ra", data=overlap["ra"].to_numpy(np.float64))
        f.create_dataset("meta/dec", data=overlap["dec"].to_numpy(np.float64))
        f.create_dataset("meta/sep_arcsec", data=overlap["sep_arcsec"].to_numpy(np.float32))
        f.create_dataset("meta/hsc_image_idx", data=hsc_indices)
        f.create_dataset("meta/legacy_image_idx", data=leg_indices)
        f.create_dataset("meta/hsc_object_id",
                         data=overlap["hsc_object_id"].to_numpy(np.int64))
        f.create_dataset("meta/legacy_object_id",
                         data=overlap["legacy_object_id"].astype(str).values.astype("S"))

    sz = os.path.getsize(args.out)
    print(f"Done. {args.out}  ({sz/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
