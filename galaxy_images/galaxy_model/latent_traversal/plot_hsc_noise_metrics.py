"""
For each of 6 noise metrics × {iband, 4band}, pick 20 HSC images at percentiles
linspace(1, 99, 20) and post a row-scaled grid PNG to Discord.
12 grids total.
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

STATS_HDF5      = Path("/work1/jeroenaudenaert/pablomer/data/hsc_ivar_psf_stats.h5")
METRICS_HDF5    = Path("/work1/jeroenaudenaert/pablomer/data/hsc_noise_metrics.h5")
OUTPUT_DIR      = Path(__file__).resolve().parent / "noise_grids"
DISCORD_WEBHOOK = (
    "https://discord.com/api/webhooks/1497979386144493680/"
    "VA-xWhfTWzc-oeC5EvPzyqEk_MW52wZsK2RyLS0egfhHHHhBxrmb9NGawy0rIpfvn3Zo"
)

METRIC_NAMES = [
    "corner_sky_rms",
    "outer_annulus_sigma",
    "adjacent_diff_sigma",
    "masked_mad_sigma",
    "high_freq_power",
    "snr_neg",
]

METRIC_LABEL = {
    "corner_sky_rms":      "Corner sky RMS",
    "outer_annulus_sigma": "Median 1/√ivar (outer annulus)",
    "adjacent_diff_sigma": "Adjacent-pixel diff σ",
    "masked_mad_sigma":    "MAD (source-masked)",
    "high_freq_power":     "High-k FFT power",
    "snr_neg":             "−(cmodel flux/fluxerr)",
}


# ============= UTILS (mirrored from plot_hsc_ivar_percentiles.py) =============

def row_scale_rgb(img_chw: np.ndarray) -> np.ndarray:
    x = img_chw[:3].astype(np.float32)
    vmin = x.reshape(3, -1).min(axis=1).reshape(3, 1, 1)
    vmax = x.reshape(3, -1).max(axis=1).reshape(3, 1, 1)
    y = (x - vmin) / (vmax - vmin + 1e-8)
    return np.transpose(np.clip(y, 0, 1), (1, 2, 0))


def send_to_discord(webhook_url: str, file_path: Path, message: str = "", retries: int = 3):
    for attempt in range(retries):
        try:
            with open(file_path, "rb") as f:
                data = f.read()
            resp = requests.post(
                webhook_url,
                data={"content": message} if message else {},
                files={"file": (file_path.name, io.BytesIO(data), "image/png")},
                timeout=60,
            )
            if resp.status_code in (200, 204):
                print(f"  Sent {file_path.name}")
                return
            if resp.status_code == 429:
                wait = float(resp.json().get("retry_after", 2.0))
                print(f"  Rate-limited; sleeping {wait:.1f}s")
                time.sleep(wait + 0.5)
                continue
            print(f"  WARNING: Discord {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            print(f"  WARNING: Discord post failed (attempt {attempt+1}): {e}")
            time.sleep(2.0)
    print(f"  GIVING UP on {file_path.name}")


def pick_percentile_indices(values: np.ndarray, percentiles: np.ndarray) -> np.ndarray:
    """Pick indices closest to each percentile. Walks the sorted array to avoid duplicates."""
    finite = np.isfinite(values)
    finite_idx = np.where(finite)[0]
    if len(finite_idx) < len(percentiles):
        raise RuntimeError(f"Only {len(finite_idx)} finite values; need {len(percentiles)}")

    sub_values = values[finite_idx]
    order = np.argsort(sub_values)
    n = len(order)

    ranks = np.clip(np.round(percentiles / 100.0 * (n - 1)).astype(int), 0, n - 1)
    seen, out_ranks = set(), []
    for r in ranks:
        rr = r
        while rr in seen and rr < n - 1:
            rr += 1
        seen.add(rr)
        out_ranks.append(rr)
    sub_idx = order[np.array(out_ranks)]
    return finite_idx[sub_idx]


# ============= PLOT =============

def plot_grid(
    images: np.ndarray, percentiles: np.ndarray, values: np.ndarray, rows: np.ndarray,
    metric_label: str, band_mode: str, n_cols: int, output_path: Path,
):
    n_imgs = len(images)
    n_rows = int(np.ceil(n_imgs / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.0 * n_cols, 2.4 * n_rows),
        squeeze=False,
    )
    for k in range(n_rows * n_cols):
        ax = axes[k // n_cols, k % n_cols]
        if k >= n_imgs:
            ax.axis("off"); continue
        ax.imshow(row_scale_rgb(images[k]))
        ax.set_title(
            f"p{percentiles[k]:.0f}  row={int(rows[k])}\n"
            f"{metric_label.split()[0]}={values[k]:.3g}",
            fontsize=8,
        )
        ax.axis("off")

    fig.suptitle(
        f"{metric_label}  ({band_mode})  —  20 HSC images, percentiles 1–99, row-scaled",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============= MAIN =============

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-path",   type=Path, default=STATS_HDF5)
    parser.add_argument("--metrics-path", type=Path, default=METRICS_HDF5)
    parser.add_argument("--output-dir",   type=Path, default=OUTPUT_DIR)
    parser.add_argument("--n-images",     type=int, default=20)
    parser.add_argument("--p-lo",         type=float, default=1.0)
    parser.add_argument("--p-hi",         type=float, default=99.0)
    parser.add_argument("--n-cols",       type=int, default=5)
    parser.add_argument("--sleep-between", type=float, default=1.0,
                        help="Seconds to sleep between Discord posts (rate-limit cushion)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Loading metrics from {args.metrics_path} ===")
    with h5py.File(args.metrics_path, "r") as mf:
        rows_metrics = np.array(mf["hdf5_row_idx"], dtype=np.int64)
        metrics = {
            (name, mode): np.array(mf[f"{name}_{mode}"], dtype=np.float64)
            for name in METRIC_NAMES for mode in ("iband", "4band")
        }

    print(f"=== Loading stats from {args.stats_path} ===")
    with h5py.File(args.stats_path, "r") as sf:
        rows_stats = np.array(sf["hdf5_row_idx"], dtype=np.int64)
        if not np.array_equal(rows_metrics, rows_stats):
            raise RuntimeError("Row alignment mismatch between metrics and stats files")

        percentiles = np.linspace(args.p_lo, args.p_hi, args.n_images)
        plan = []
        for name in METRIC_NAMES:
            for mode in ("iband", "4band"):
                vals = metrics[(name, mode)]
                idx = pick_percentile_indices(vals, percentiles)
                plan.append((name, mode, idx, vals[idx]))

        # Read all needed images at once (sorted unique indices for fancy indexing)
        all_idx = np.unique(np.concatenate([p[2] for p in plan]))
        print(f"  Loading {len(all_idx)} unique images from hsc_images")
        imgs_all = sf["hsc_images"][all_idx]   # (M,4,48,48)

    # Map plan-idx -> position in imgs_all
    idx_to_pos = {int(i): p for p, i in enumerate(all_idx)}

    print("=== Building & posting grids ===")
    _notify(f"🖼️ plot grids: building {len(plan)} percentile grids")
    for name, mode, idx, vals in plan:
        positions = np.array([idx_to_pos[int(i)] for i in idx])
        images = imgs_all[positions]
        rows_for_plot = rows_metrics[idx]
        out_path = args.output_dir / f"{name}_{mode}.png"
        plot_grid(
            images, percentiles, vals, rows_for_plot,
            metric_label=METRIC_LABEL[name],
            band_mode=mode,
            n_cols=args.n_cols,
            output_path=out_path,
        )
        msg = (
            f"**{METRIC_LABEL[name]} ({mode})** — 20 HSC images at percentiles 1–99 "
            f"(row-scaled, per training)"
        )
        send_to_discord(DISCORD_WEBHOOK, out_path, msg)
        time.sleep(args.sleep_between)

    _notify(f"✅ plot grids: all {len(plan)} grids posted")
    print("\nDone.")


if __name__ == "__main__":
    main()
