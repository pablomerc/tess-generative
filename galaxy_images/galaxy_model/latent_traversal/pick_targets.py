"""
Show 20 randomly selected HSC targets as a labeled grid and send to Discord.
Prints the stats_idx for each label so the chosen one can be passed to
quality_conditioning_traversal.py --target-idx.

Usage:
  python pick_targets.py [--seed 7] [--stats-path ...]
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

STATS_DEFAULT   = Path("/work1/jeroenaudenaert/pablomer/data/hsc_ivar_psf_stats.h5")
CACHE_DEFAULT   = Path(__file__).resolve().parent / "ivar_psf_cache.npz"
OUTPUT_DIR      = Path(__file__).resolve().parent
DISCORD_WEBHOOK = (
    "https://discord.com/api/webhooks/1497979386144493680/"
    "VA-xWhfTWzc-oeC5EvPzyqEk_MW52wZsK2RyLS0egfhHHHhBxrmb9NGawy0rIpfvn3Zo"
)


def tensor_to_rgb(img: np.ndarray, p: float = 99.5) -> np.ndarray:
    """(C, H, W) float → (H, W, 3) RGB in [0,1], per-channel percentile stretch."""
    rgb = img[:3].copy().astype(np.float32)
    rgb = np.transpose(rgb, (1, 2, 0))
    for i in range(3):
        lo = np.percentile(rgb[:, :, i], 100 - p)
        hi = np.percentile(rgb[:, :, i], p)
        rgb[:, :, i] = np.clip(rgb[:, :, i], lo, hi)
        ch = rgb[:, :, i]
        lo2, hi2 = ch.min(), ch.max()
        rgb[:, :, i] = (ch - lo2) / (hi2 - lo2 + 1e-8)
    return rgb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",       type=int,  default=7)
    parser.add_argument("--n",          type=int,  default=20)
    parser.add_argument("--stats-path", type=Path, default=STATS_DEFAULT)
    parser.add_argument("--cache-path", type=Path, default=CACHE_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load quality scores from cache
    c = np.load(args.cache_path)
    score_sum = c["score_sum"].astype(np.float32)
    score_pca = c["score_pca"].astype(np.float32)
    ivar_all  = c["ivar"].astype(np.float32)
    psf_all   = c["psf"].astype(np.float32)
    N = len(score_sum)

    # Random sample
    rng = np.random.default_rng(args.seed)
    chosen_idxs = rng.choice(N, size=args.n, replace=False)
    chosen_idxs = sorted(chosen_idxs)   # sort for reproducibility in printout

    # Load HSC images from stats HDF5
    with h5py.File(args.stats_path, "r") as f:
        images = np.array(f["hsc_images"][chosen_idxs], dtype=np.float32)  # (n, 4, 48, 48)

    # Print mapping for reference
    print(f"\nSeed={args.seed}  |  {args.n} randomly selected targets:\n")
    header = f"{'#':>3}  {'stats_idx':>10}  {'sum_score':>10}  {'pca_score':>10}  {'ivar':>8}  {'PSF':>6}"
    print(header)
    print("-" * 58)
    for i, idx in enumerate(chosen_idxs):
        print(f"{i+1:>3}  {idx:>10}  {score_sum[idx]:>10.3f}  {score_pca[idx]:>10.3f}  "
              f"{ivar_all[idx]:>8.0f}  {psf_all[idx]:>6.3f}")

    # Plot 4×5 grid
    rows, cols = 4, 5
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.6))
    for i, (idx, img) in enumerate(zip(chosen_idxs, images)):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        ax.imshow(tensor_to_rgb(img))
        ax.set_title(
            f"#{i+1}  idx={idx}\nsum={score_sum[idx]:.2f}  pca={score_pca[idx]:.2f}\n"
            f"ivar={ivar_all[idx]:.0f}  PSF={psf_all[idx]:.2f}\"",
            fontsize=7,
        )
        ax.axis("off")

    fig.suptitle(
        f"Pick a target (1–{args.n}) — seed={args.seed}\n"
        "Reply with the number to get the full quality traversal",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()

    out = args.output_dir / f"pick_targets_seed{args.seed}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}")

    # Send to Discord
    with open(out, "rb") as fh:
        data = fh.read()
    resp = requests.post(
        DISCORD_WEBHOOK,
        data={"content": f"**Pick a target (1–{args.n})** — reply with the number for a full traversal plot  (seed={args.seed})"},
        files={"file": (out.name, io.BytesIO(data), "image/png")},
    )
    if resp.status_code in (200, 204):
        print("Sent to Discord.")
    else:
        print(f"WARNING: Discord {resp.status_code}: {resp.text[:200]}")


if __name__ == "__main__":
    main()
