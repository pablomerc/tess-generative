"""
Extract HSC ivar + PSF stats + center-cropped images from raw parquet files.

For each parquet row, extracts:
  - hsc_mean_ivar       — mean ivar across 5 bands × pixels (scalar)
  - hsc_ivar_per_band   — per-band mean ivar (5,)
  - hsc_psf_fwhm        — PSF FWHM per band in arcsec (5,), from hsc_image['psf_fwhm']
  - hsc_psf_fwhm_avg    — mean PSF FWHM across bands (scalar)
  - hsc_cmodel_mag      — HSC cmodel magnitudes g/r/i/z/y (5,)
  - hsc_images          — center-cropped flux image, 4 bands g/r/i/z, (4, 48, 48)

Output HDF5 is compact (only matched rows) and includes 'hdf5_row_idx' for
joining back to neighbours_v2.h5 images (images_hsc, images_legacy, etc.).

Usage:
  python extract_hsc_stats.py                          # all 396 files
  python extract_hsc_stats.py --n-files 1              # test: 1 file only
  python extract_hsc_stats.py --n-files 1 --output /work1/.../hsc_stats_test.h5
"""

import argparse
import io
import multiprocessing as mp
import os
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

# ============= CONFIGURATION =============

PARQUET_DIR = Path("/work1/jeroenaudenaert/pablomer/data/raw_mmu/data")
NEIGHBORS_HDF5 = "/work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5"
OUTPUT_DEFAULT = Path("/work1/jeroenaudenaert/pablomer/data/hsc_ivar_psf_stats.h5")
OUTPUT_DIR = Path(__file__).resolve().parent

DISCORD_WEBHOOK = (
    "https://discord.com/api/webhooks/1497979386144493680/"
    "VA-xWhfTWzc-oeC5EvPzyqEk_MW52wZsK2RyLS0egfhHHHhBxrmb9NGawy0rIpfvn3Zo"
)

PARQUET_COLS = [
    "hsc_object_id", "hsc_image",
    "g_cmodel_mag", "r_cmodel_mag", "i_cmodel_mag", "z_cmodel_mag", "y_cmodel_mag",
]

BANDS = ["g", "r", "i", "z", "y"]
CROP_SIZE = 48


# ============= JOIN MAP =============

def build_join_map(hdf5_path: str) -> dict:
    """Build {hsc_object_id_str -> hdf5_row_idx} from neighbours_v2.h5."""
    with h5py.File(hdf5_path, "r") as f:
        oids = f["object_id_hsc"][:]
    join_map = {oid.decode(): i for i, oid in enumerate(oids)}
    print(f"Join map built: {len(join_map):,} HDF5 rows")
    return join_map


# ============= PER-FILE PROCESSING =============

def _band_to_2d(band_data) -> np.ndarray:
    """Convert a parquet band array (possibly ragged object array) to (H, W) float32."""
    raw = np.asarray(band_data)
    if raw.dtype == object:
        return np.stack([np.asarray(r, dtype=np.float32) for r in raw])
    return raw.astype(np.float32)


def _band_mean_f64(band_data) -> float:
    """Mean of a parquet band ivar array; handles ragged object arrays."""
    raw = np.asarray(band_data)
    if raw.dtype == object:
        vals = np.concatenate([np.asarray(r, dtype=np.float64).ravel() for r in raw])
    else:
        vals = raw.astype(np.float64).ravel()
    return float(vals.mean())


def process_file(args):
    """Process one parquet file. Returns list of record tuples."""
    parquet_path, join_map = args
    try:
        df = pd.read_parquet(parquet_path, columns=PARQUET_COLS, engine="pyarrow")
    except Exception as e:
        print(f"  ERROR reading {parquet_path}: {e}")
        return []

    records = []
    for row in df.itertuples(index=False):
        hdf5_row = join_map.get(row.hsc_object_id)
        if hdf5_row is None:
            continue

        img = row.hsc_image
        psf = np.asarray(img["psf_fwhm"], dtype=np.float32)                          # (5,)
        ivar_per_band = np.array(
            [_band_mean_f64(img["ivar"][b]) for b in range(5)],
            dtype=np.float32,
        )                                                                              # (5,)
        mags = np.array(
            [row.g_cmodel_mag, row.r_cmodel_mag, row.i_cmodel_mag,
             row.z_cmodel_mag, row.y_cmodel_mag],
            dtype=np.float32,
        )                                                                              # (5,)

        # Center-crop flux image, 4 bands g/r/i/z (band index 0-3)
        flux = np.stack(
            [_band_to_2d(img["flux"][b]) for b in range(4)]
        )                                                                              # (4, H, W)
        H, W = flux.shape[1], flux.shape[2]
        y0, x0 = (H - CROP_SIZE) // 2, (W - CROP_SIZE) // 2
        crop = flux[:, y0:y0 + CROP_SIZE, x0:x0 + CROP_SIZE]                         # (4, 48, 48)

        records.append((
            hdf5_row,
            float(ivar_per_band.mean()),   # hsc_mean_ivar
            ivar_per_band,                 # hsc_ivar_per_band
            psf,                           # hsc_psf_fwhm
            float(np.nanmean(psf)),        # hsc_psf_fwhm_avg
            mags,                          # hsc_cmodel_mag
            crop,                          # hsc_images (4, 48, 48)
        ))
    return records


# ============= HDF5 OUTPUT =============

def save_hdf5(records: list, output_path: Path):
    records.sort(key=lambda r: r[0])  # sort by hdf5_row_idx
    n = len(records)

    hdf5_row_idx   = np.array([r[0] for r in records], dtype=np.int32)
    mean_ivar      = np.array([r[1] for r in records], dtype=np.float32)
    ivar_per_band  = np.stack([r[2] for r in records], axis=0).astype(np.float32)
    psf_fwhm       = np.stack([r[3] for r in records], axis=0).astype(np.float32)
    psf_fwhm_avg   = np.array([r[4] for r in records], dtype=np.float32)
    cmodel_mag     = np.stack([r[5] for r in records], axis=0).astype(np.float32)
    images         = np.stack([r[6] for r in records], axis=0).astype(np.float32)    # (N, 4, 48, 48)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.create_dataset("hdf5_row_idx",       data=hdf5_row_idx)
        f.create_dataset("hsc_mean_ivar",      data=mean_ivar)
        f.create_dataset("hsc_ivar_per_band",  data=ivar_per_band)
        f.create_dataset("hsc_psf_fwhm",       data=psf_fwhm)
        f.create_dataset("hsc_psf_fwhm_avg",   data=psf_fwhm_avg)
        f.create_dataset("hsc_cmodel_mag",     data=cmodel_mag)
        f.create_dataset("hsc_images",         data=images,
                         chunks=(min(n, 256), 4, CROP_SIZE, CROP_SIZE))
        f.attrs["bands"] = BANDS
        f.attrs["n_examples"] = n

    print(f"\nSaved {output_path}  ({n:,} rows)")
    for key in ["hsc_mean_ivar", "hsc_psf_fwhm_avg"]:
        arr = mean_ivar if key == "hsc_mean_ivar" else psf_fwhm_avg
        finite = arr[np.isfinite(arr)]
        print(f"  {key}: min={finite.min():.4f}  max={finite.max():.4f}  "
              f"mean={finite.mean():.4f}  zeros={(arr == 0).sum()}")


# ============= 2D DENSITY PLOT =============

def percentile_filter_normalize(arr, p_lo=1.0, p_hi=99.0):
    """Drop values outside [p_lo, p_hi] percentile range; normalize remainder to [0,1]."""
    lo = np.nanpercentile(arr, p_lo)
    hi = np.nanpercentile(arr, p_hi)
    mask = (arr >= lo) & (arr <= hi)
    normed = (arr[mask] - lo) / (hi - lo + 1e-10)
    return normed, mask, lo, hi


def plot_2d_density(mean_ivar: np.ndarray, psf_fwhm_avg: np.ndarray,
                    output_path: Path, title_suffix: str = "", bins: int = 100):
    valid = np.isfinite(mean_ivar) & np.isfinite(psf_fwhm_avg)
    iv  = mean_ivar[valid]
    psf = psf_fwhm_avg[valid]

    _, iv_mask,  iv_lo,  iv_hi  = percentile_filter_normalize(iv)
    _, psf_mask, psf_lo, psf_hi = percentile_filter_normalize(psf)
    keep = iv_mask & psf_mask
    n_dropped = (~keep).sum()
    iv_f  = (iv[keep]  - iv_lo)  / (iv_hi  - iv_lo  + 1e-10)
    psf_f = (psf[keep] - psf_lo) / (psf_hi - psf_lo + 1e-10)

    ivar_label = f"HSC mean ivar  [0={iv_lo:.0f}, 1={iv_hi:.0f}]"
    psf_label  = f"HSC PSF FWHM avg  [0={psf_lo:.2f}\", 1={psf_hi:.2f}\"]"
    configs = [
        (iv_f, psf_f, f"p1–p99 drop-and-normalize (linear)\n{n_dropped:,} dropped", False),
        (iv_f, psf_f, f"p1–p99 drop-and-normalize (log count)\n{n_dropped:,} dropped", True),
    ]

    n_panels = len(configs)
    fig = plt.figure(figsize=(n_panels * 9, 8))
    outer = fig.add_gridspec(1, n_panels, wspace=0.38)

    for col_idx, (ix, px, title, log_scale) in enumerate(configs):
        inner = outer[col_idx].subgridspec(
            2, 2, height_ratios=[1, 4], width_ratios=[4, 1], hspace=0.05, wspace=0.05
        )
        ax_top   = fig.add_subplot(inner[0, 0])
        ax_main  = fig.add_subplot(inner[1, 0])
        ax_right = fig.add_subplot(inner[1, 1])
        ax_cb    = fig.add_subplot(inner[0, 1])
        ax_cb.set_visible(False)

        norm = plt.matplotlib.colors.LogNorm() if log_scale else None
        _, _, _, im = ax_main.hist2d(ix, px, bins=bins, cmap="viridis", norm=norm)
        fig.colorbar(im, ax=ax_cb, fraction=0.8, pad=0.05).set_label("Count", fontsize=8)
        ax_main.set_xlabel(ivar_label, fontsize=9)
        ax_main.set_ylabel(psf_label, fontsize=9)

        ax_top.hist(ix, bins=bins, color="steelblue", edgecolor="none")
        ax_top.set_xlim(ax_main.get_xlim())
        ax_top.set_xticks([])
        ax_top.set_ylabel("Count", fontsize=8)
        ax_top.tick_params(axis="y", labelsize=7)
        ax_top.set_title(title, fontsize=10, fontweight="bold", pad=4)

        ax_right.hist(px, bins=bins, color="tomato", edgecolor="none",
                      orientation="horizontal")
        ax_right.set_ylim(ax_main.get_ylim())
        ax_right.set_yticks([])
        ax_right.set_xlabel("Count", fontsize=8)
        ax_right.tick_params(axis="x", labelsize=7)

    fig.suptitle(
        f"2D density: HSC mean ivar vs HSC PSF FWHM  {title_suffix}",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


# ============= DISCORD =============

def send_to_discord(webhook_url: str, file_path: Path, message: str = ""):
    with open(file_path, "rb") as f:
        data = f.read()
    resp = requests.post(
        webhook_url,
        data={"content": message} if message else {},
        files={"file": (file_path.name, io.BytesIO(data), "image/png")},
    )
    if resp.status_code in (200, 204):
        print(f"  Sent {file_path.name} to Discord.")
    else:
        print(f"  WARNING: Discord {resp.status_code}: {resp.text[:200]}")


# ============= MAIN =============

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-dir", type=Path, default=PARQUET_DIR)
    parser.add_argument("--hdf5-path", default=NEIGHBORS_HDF5)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--n-files", type=int, default=None,
                        help="Number of parquet files to process (default: all)")
    parser.add_argument("--n-workers", type=int, default=8)
    args = parser.parse_args()

    files = sorted(args.parquet_dir.glob("train-*.parquet"))
    if args.n_files is not None:
        files = files[:args.n_files]
    print(f"Processing {len(files)} parquet file(s) with {args.n_workers} workers")

    # Build join map
    join_map = build_join_map(args.hdf5_path)

    # Process files in parallel
    pool_args = [(str(f), join_map) for f in files]
    with mp.Pool(args.n_workers) as pool:
        results = pool.map(process_file, pool_args)

    all_records = [rec for batch in results for rec in batch]
    n_total_rows = sum(len(r) for r in results)  # same as len(all_records)
    print(f"\nMatched {len(all_records):,} / ~{len(files) * 257:,} rows")

    if not all_records:
        print("ERROR: no records extracted — check join key format")
        return

    # Save HDF5
    save_hdf5(all_records, args.output)

    # 2D density plot
    with h5py.File(args.output, "r") as f:
        mean_ivar    = np.array(f["hsc_mean_ivar"])
        psf_fwhm_avg = np.array(f["hsc_psf_fwhm_avg"])

    suffix = f"(n={len(all_records):,}, {len(files)} file{'s' if len(files) > 1 else ''})"
    plot_path = OUTPUT_DIR / f"hsc_ivar_psf_density{'_test' if args.n_files == 1 else ''}.png"
    plot_2d_density(mean_ivar, psf_fwhm_avg, plot_path, title_suffix=suffix)

    send_to_discord(
        DISCORD_WEBHOOK, plot_path,
        f"**HSC ivar × PSF 2D density** {suffix}\noutput: `{args.output.name}`",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
