"""
Pick 20 HSC images ranked by HSC mean ivar at percentiles linspace(1, 99, 20)
and post a single grid figure to Discord, using the training-time "row scaled"
visualization (per-image per-channel min/max → RGB in [0, 1], first 3 bands).
"""

import argparse
import io
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests

STATS_DEFAULT = Path("/work1/jeroenaudenaert/pablomer/data/hsc_ivar_psf_stats.h5")
OUTPUT_DIR = Path(__file__).resolve().parent
DISCORD_WEBHOOK = (
    "https://discord.com/api/webhooks/1497979386144493680/"
    "VA-xWhfTWzc-oeC5EvPzyqEk_MW52wZsK2RyLS0egfhHHHhBxrmb9NGawy0rIpfvn3Zo"
)


def row_scale_rgb(img_chw: np.ndarray) -> np.ndarray:
    """(C,H,W) → (H,W,3) in [0,1] via per-channel min/max on first 3 bands.
    Mirrors _row_scale_rgb from train_fm.py with vmin/vmax = per-channel amin/amax."""
    x = img_chw[:3].astype(np.float32)
    vmin = x.reshape(3, -1).min(axis=1).reshape(3, 1, 1)
    vmax = x.reshape(3, -1).max(axis=1).reshape(3, 1, 1)
    y = (x - vmin) / (vmax - vmin + 1e-8)
    y = np.clip(y, 0.0, 1.0)
    return np.transpose(y, (1, 2, 0))


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


def pick_percentile_indices(values: np.ndarray, percentiles: np.ndarray) -> np.ndarray:
    """For each target percentile, return the index of the example whose value is
    closest to the percentile threshold. Picks unique indices by walking the sorted
    array — avoids duplicates when the distribution has ties."""
    order = np.argsort(values)
    sorted_vals = values[order]
    n = len(sorted_vals)
    # rank position within sorted array for each percentile
    ranks = np.clip(np.round(percentiles / 100.0 * (n - 1)).astype(int), 0, n - 1)
    # de-duplicate while keeping monotonic order
    seen = set()
    out_ranks = []
    for r in ranks:
        rr = r
        while rr in seen and rr < n - 1:
            rr += 1
        seen.add(rr)
        out_ranks.append(rr)
    return order[np.array(out_ranks)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-path", type=Path, default=STATS_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--n-images", type=int, default=20)
    parser.add_argument("--p-lo", type=float, default=1.0)
    parser.add_argument("--p-hi", type=float, default=99.0)
    parser.add_argument("--n-cols", type=int, default=5)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Loading {args.stats_path} ===")
    with h5py.File(args.stats_path, "r") as f:
        mean_ivar = np.array(f["hsc_mean_ivar"], dtype=np.float32)
        hdf5_row_idx = np.array(f["hdf5_row_idx"], dtype=np.int64)
        n = len(mean_ivar)
        print(f"  n={n:,}  ivar: min={mean_ivar.min():.2f}  max={mean_ivar.max():.2f}")

        percentiles = np.linspace(args.p_lo, args.p_hi, args.n_images)
        print(f"  percentiles: {percentiles}")
        chosen = pick_percentile_indices(mean_ivar, percentiles)

        # Load just the chosen images. h5py needs sorted unique indices for fancy indexing.
        sort_order = np.argsort(chosen)
        sorted_chosen = chosen[sort_order]
        imgs_sorted = f["hsc_images"][sorted_chosen]  # (n_images, 4, 48, 48)
        # Restore original (percentile-ordered) order
        inv = np.argsort(sort_order)
        images = imgs_sorted[inv]

    chosen_ivar = mean_ivar[chosen]
    chosen_rows = hdf5_row_idx[chosen]
    print("  chosen ivar values:", np.array2string(chosen_ivar, precision=1))

    # === Figure ===
    n_cols = args.n_cols
    n_rows = int(np.ceil(args.n_images / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.0 * n_cols, 2.4 * n_rows),
        squeeze=False,
    )
    for k in range(n_rows * n_cols):
        ax = axes[k // n_cols, k % n_cols]
        if k >= args.n_images:
            ax.axis("off")
            continue
        rgb = row_scale_rgb(images[k])
        ax.imshow(rgb)
        ax.set_title(
            f"p{percentiles[k]:.0f}  row={int(chosen_rows[k])}\n"
            f"ivar={chosen_ivar[k]:.0f}",
            fontsize=8,
        )
        ax.axis("off")

    fig.suptitle(
        f"HSC images at {args.n_images} HSC mean-ivar percentiles "
        f"(linspace {args.p_lo:.0f}–{args.p_hi:.0f}) — row-scaled (per-image per-channel min/max)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()

    out_path = args.output_dir / "hsc_ivar_percentile_grid.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

    print("\n=== Sending to Discord ===")
    msg = (
        f"**HSC images at {args.n_images} ivar percentiles** "
        f"(linspace {args.p_lo:.0f}–{args.p_hi:.0f}, row-scaled per training)"
    )
    send_to_discord(DISCORD_WEBHOOK, out_path, msg)
    print("\nDone.")


if __name__ == "__main__":
    main()
