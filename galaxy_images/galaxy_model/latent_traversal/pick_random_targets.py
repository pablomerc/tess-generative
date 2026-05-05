"""
Post a gallery of 64 randomly-chosen HSC targets near the SNR p60 band to
Discord, each labeled with stats_idx so the user can pick favorites for the
SNR traversal. Uses the same per-image 2-98 percentile (vis_indep) style as
snr_traversal_full.py.
"""

import argparse
import io
import sys
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


def vis_indep(img_chw):
    rgb = img_chw[:3].astype(np.float32).copy()
    lo, hi = np.percentile(rgb, [2, 98])
    rgb = (rgb - lo) / (hi - lo + 1e-8)
    return np.transpose(np.clip(rgb, 0, 1), (1, 2, 0))


def send_to_discord(file_path, message):
    with open(file_path, "rb") as f:
        data = f.read()
    resp = requests.post(
        DISCORD_WEBHOOK,
        data={"content": message},
        files={"file": (file_path.name, io.BytesIO(data), "image/png")},
        timeout=120,
    )
    if resp.status_code in (200, 204):
        print(f"  Sent {file_path.name}")
    else:
        print(f"  WARNING: Discord {resp.status_code}: {resp.text[:200]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-images", type=int, default=64)
    parser.add_argument("--n-cols", type=int, default=8)
    parser.add_argument("--snr-percentile", type=float, default=60.0,
                        help="Center percentile of snr_pos for the band")
    parser.add_argument("--snr-half-window", type=float, default=2.0,
                        help="Half-width (in percentile points) around the center")
    parser.add_argument("--seed", type=int, default=0)
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

        valid = np.isfinite(snr_pos)
        valid_snr = snr_pos[valid]
        p_lo = args.snr_percentile - args.snr_half_window
        p_hi = args.snr_percentile + args.snr_half_window
        snr_lo = np.percentile(valid_snr, p_lo)
        snr_hi = np.percentile(valid_snr, p_hi)
        band_mask = valid & (snr_pos >= snr_lo) & (snr_pos <= snr_hi)
        candidate_idx = np.where(band_mask)[0]
        print(f"  SNR band p{p_lo:.1f}-p{p_hi:.1f} -> "
              f"[{snr_lo:.2f}, {snr_hi:.2f}], candidates={len(candidate_idx):,}")

        if len(candidate_idx) < args.n_images:
            raise RuntimeError(
                f"Not enough candidates ({len(candidate_idx)}) for n_images={args.n_images}; "
                "widen --snr-half-window")

        rng = np.random.default_rng(args.seed)
        idx = rng.choice(candidate_idx, size=args.n_images, replace=False)
        idx = np.sort(idx)
        print(f"  picked {len(idx)} stats_idx (seed={args.seed})")

        order = np.argsort(idx)
        sorted_idx = idx[order]
        unique_sorted, inverse = np.unique(sorted_idx, return_inverse=True)
        imgs_unique = sf["hsc_images"][unique_sorted]
        imgs_sorted = imgs_unique[inverse]
        inv = np.argsort(order)
        images = imgs_sorted[inv]

    chosen_snr  = snr_pos[idx]
    chosen_ivar = ivar_all[idx]
    chosen_psf  = psf_all[idx]
    chosen_row  = rows_metrics[idx]

    n_cols = args.n_cols
    n_rows = int(np.ceil(args.n_images / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.4 * n_cols, 2.7 * n_rows),
        squeeze=False,
    )
    for k in range(n_rows * n_cols):
        ax = axes[k // n_cols, k % n_cols]
        if k >= args.n_images:
            ax.axis("off"); continue
        ax.imshow(vis_indep(images[k]))
        ax.set_title(
            f"stats_idx={int(idx[k])}  row={int(chosen_row[k])}\n"
            f"SNR={chosen_snr[k]:.1f}  ivar={chosen_ivar[k]:.0f}  "
            f"PSF={chosen_psf[k]:.2f}\"",
            fontsize=7,
        )
        ax.axis("off")

    fig.suptitle(
        f"Random HSC targets near SNR p{args.snr_percentile:.0f}  "
        f"(band p{p_lo:.1f}-p{p_hi:.1f}, SNR in [{snr_lo:.2f}, {snr_hi:.2f}])  "
        f"n={args.n_images} | seed={args.seed} | vis=indep (per-image 2-98 pct)\n"
        "Reply with the `stats_idx` values you want me to traverse.",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()

    out_path = args.output_dir / f"random_targets_p{int(args.snr_percentile)}_seed{args.seed}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

    msg = (
        f"**Random HSC target candidates — SNR p{args.snr_percentile:.0f} band**\n"
        f"{args.n_images} HSC images, randomly drawn from stats_idxs whose SNR sits "
        f"in p{p_lo:.1f}-p{p_hi:.1f} (SNR in [{snr_lo:.2f}, {snr_hi:.2f}]). "
        f"seed={args.seed}, visualization = per-image 2-98 percentile (matches indep style).\n"
        "Reply with the `stats_idx` values you want me to run the full SNR traversal on."
    )
    send_to_discord(out_path, msg)
    _notify(f"✅ Random p{args.snr_percentile:.0f} target gallery posted to Discord")


if __name__ == "__main__":
    main()
