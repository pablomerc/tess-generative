"""
Post a gallery of 20 candidate HSC targets at SNR percentiles to Discord, each
labeled with stats_idx and hdf5_row so the user can pick which to traverse.
"""

import argparse
import io
import sys
import time
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from discord_notify import notify as _notify

STATS_HDF5   = Path("/work1/jeroenaudenaert/pablomer/data/hsc_ivar_psf_stats.h5")
METRICS_HDF5 = Path("/work1/jeroenaudenaert/pablomer/data/hsc_noise_metrics.h5")
OUTPUT_DIR   = Path(__file__).resolve().parent / "snr_traversal_results"
DISCORD_WEBHOOK = (
    "https://discord.com/api/webhooks/1497979386144493680/"
    "VA-xWhfTWzc-oeC5EvPzyqEk_MW52wZsK2RyLS0egfhHHHhBxrmb9NGawy0rIpfvn3Zo"
)


def row_scale_rgb(img_chw):
    x = img_chw[:3].astype(np.float32)
    vmin = x.reshape(3, -1).min(axis=1).reshape(3, 1, 1)
    vmax = x.reshape(3, -1).max(axis=1).reshape(3, 1, 1)
    y = (x - vmin) / (vmax - vmin + 1e-8)
    return np.transpose(np.clip(y, 0, 1), (1, 2, 0))


def pick_indices(values, percentiles):
    finite = np.isfinite(values)
    finite_idx = np.where(finite)[0]
    sub = values[finite_idx]
    order = np.argsort(sub)
    n = len(order)
    ranks = np.clip(np.round(percentiles / 100.0 * (n - 1)).astype(int), 0, n - 1)
    seen, out_ranks = set(), []
    for r in ranks:
        rr = r
        while rr in seen and rr < n - 1:
            rr += 1
        seen.add(rr)
        out_ranks.append(rr)
    return finite_idx[order[np.array(out_ranks)]]


def send_to_discord(file_path, message):
    with open(file_path, "rb") as f:
        data = f.read()
    resp = requests.post(
        DISCORD_WEBHOOK,
        data={"content": message},
        files={"file": (file_path.name, io.BytesIO(data), "image/png")},
        timeout=60,
    )
    if resp.status_code in (200, 204):
        print(f"  Sent {file_path.name}")
    else:
        print(f"  WARNING: Discord {resp.status_code}: {resp.text[:200]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-images", type=int, default=20)
    parser.add_argument("--p-lo", type=float, default=1.0)
    parser.add_argument("--p-hi", type=float, default=99.0)
    parser.add_argument("--n-cols", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Loading metrics ===")
    with h5py.File(METRICS_HDF5, "r") as mf:
        rows_metrics = np.array(mf["hdf5_row_idx"], dtype=np.int64)
        snr_neg_4band = np.array(mf["snr_neg_4band"], dtype=np.float64)
    snr_pos = -snr_neg_4band

    with h5py.File(STATS_HDF5, "r") as sf:
        rows_stats = np.array(sf["hdf5_row_idx"], dtype=np.int64)
        ivar_all = np.array(sf["hsc_mean_ivar"], dtype=np.float32)
        psf_all  = np.array(sf["hsc_psf_fwhm_avg"], dtype=np.float32)
        if not np.array_equal(rows_metrics, rows_stats):
            raise RuntimeError("metrics/stats row mismatch")

        # Pick by snr_neg_4band so the percentile ramp goes clean → noisy
        percentiles = np.linspace(args.p_lo, args.p_hi, args.n_images)
        idx = pick_indices(snr_neg_4band, percentiles)
        print(f"  picked stats_idx range: {idx.min()}..{idx.max()}")

        # Sorted-unique read for h5py fancy indexing
        order = np.argsort(idx)
        sorted_idx = idx[order]
        unique_sorted, inverse = np.unique(sorted_idx, return_inverse=True)
        imgs_unique = sf["hsc_images"][unique_sorted]
        imgs_sorted = imgs_unique[inverse]
        # Restore original order
        inv = np.argsort(order)
        images = imgs_sorted[inv]

    chosen_snr  = snr_pos[idx]
    chosen_ivar = ivar_all[idx]
    chosen_psf  = psf_all[idx]
    chosen_row  = rows_metrics[idx]

    n_cols = args.n_cols
    n_rows = int(np.ceil(args.n_images / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 3.0 * n_rows), squeeze=False)
    for k in range(n_rows * n_cols):
        ax = axes[k // n_cols, k % n_cols]
        if k >= args.n_images:
            ax.axis("off"); continue
        ax.imshow(row_scale_rgb(images[k]))
        ax.set_title(
            f"p{percentiles[k]:.0f}  stats_idx={int(idx[k])}  row={int(chosen_row[k])}\n"
            f"SNR={chosen_snr[k]:.1f}  ivar={chosen_ivar[k]:.0f}  PSF={chosen_psf[k]:.2f}\"",
            fontsize=8,
        )
        ax.axis("off")

    fig.suptitle(
        f"SNR-traversal target candidates  —  {args.n_images} HSC images at "
        f"SNR_neg percentiles {args.p_lo:.0f}–{args.p_hi:.0f}  (low pct = clean, high pct = noisy)\n"
        "Reply with the `stats_idx` values you want for the traversal.",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()

    out_path = args.output_dir / "snr_target_candidates.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

    msg = (
        "**SNR traversal — pick your targets**\n"
        "20 candidate HSC images shown at evenly-spaced percentiles of `-SNR` "
        "(left/top = cleanest, right/bottom = noisiest). Reply with the "
        "`stats_idx` values you want me to traverse (e.g. \"5824, 50000, 98112\")."
    )
    send_to_discord(out_path, msg)
    _notify("✅ SNR target gallery posted to Discord")


if __name__ == "__main__":
    main()
