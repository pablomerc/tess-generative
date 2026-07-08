"""
Sanity-check the HSC × Legacy overlap matches by plotting paired images
side-by-side, in the row-scaled style used by training validation grids.

For N random rows from outputs/index/overlap.parquet:
  - Read the raw HSC image (5,160,160 float16) by file-seek into hsc_flux.bin
  - Read the raw Legacy image (4,160,160 float16) by file-seek into legacy_flux.bin
  - Apply the same preprocessing the model sees (preprocess_raw_image, drop y for HSC)
  - For each pair, take the first 3 channels (g/r/i) as RGB, share per-channel
    vmin/vmax across the pair (taken from HSC) so the two panels are directly
    comparable, then plot HSC | Legacy.

Saves a single PNG and (optionally) posts it to a Discord webhook.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from galaxy_images.galaxy_model.neighbors import preprocess_raw_image  # noqa: E402

HSC_BANDS, HSC_H, HSC_W = 5, 160, 160
LEGACY_BANDS, LEGACY_H, LEGACY_W = 4, 160, 160
HSC_BYTES = HSC_BANDS * HSC_H * HSC_W * 2
LEGACY_BYTES = LEGACY_BANDS * LEGACY_H * LEGACY_W * 2


def _read_raw(bin_path: Path, idx: int, bands: int, h: int, w: int, nbytes: int) -> torch.Tensor:
    with open(bin_path, "rb") as fh:
        fh.seek(int(idx) * nbytes)
        buf = fh.read(nbytes)
    raw = np.frombuffer(buf, dtype=np.float16).copy().reshape(bands, h, w).astype(np.float32)
    return torch.from_numpy(raw)


def _row_scale_rgb(x_chw: torch.Tensor, vmin: torch.Tensor, vmax: torch.Tensor) -> np.ndarray:
    """Per-channel min-max into [0,1], match training row-scaled style."""
    x = x_chw[:3]
    vmin_t = vmin.view(3, 1, 1).to(x.dtype)
    vmax_t = vmax.view(3, 1, 1).to(x.dtype)
    y = (x - vmin_t) / (vmax_t - vmin_t + 1e-8)
    y = y.clamp(0, 1)
    return y.permute(1, 2, 0).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path,
                    default=_HERE / "outputs" / "index" / "overlap.parquet",
                    help="overlap.parquet from build_overlap_dataset.py")
    ap.add_argument("--hsc-dir", type=Path,
                    default=Path("/home/pablomer/orcd/scratch/hsc_downstream"))
    ap.add_argument("--legacy-dir", type=Path,
                    default=Path("/home/pablomer/orcd/scratch/legacy_downstream_full/full_1M"))
    ap.add_argument("--n", type=int, default=16, help="number of pairs to plot")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=_HERE / "outputs" / "plots" / "overlap_check.png")
    ap.add_argument("--webhook", type=str,
                    default=os.environ.get("DISCORD_WEBHOOK", ""),
                    help="Discord webhook URL. If empty, skip posting.")
    ap.add_argument("--message", type=str, default="overlap visual check (HSC | Legacy)")
    args = ap.parse_args()

    df = pd.read_parquet(args.index)
    print(f"loaded {len(df)} overlap rows from {args.index}")
    if len(df) == 0:
        raise SystemExit("overlap.parquet is empty — no pairs to plot")

    rng = np.random.default_rng(args.seed)
    n = min(args.n, len(df))
    sel = rng.choice(len(df), size=n, replace=False)
    sel.sort()
    rows = df.iloc[sel].reset_index(drop=True)

    hsc_bin = args.hsc_dir / "hsc_flux.bin"
    leg_bin = args.legacy_dir / "legacy_flux.bin"
    for p in (hsc_bin, leg_bin):
        if not p.is_file():
            raise FileNotFoundError(p)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(n, 2, figsize=(5, 2.4 * n), squeeze=False)
    axes[0, 0].set_title("HSC", fontsize=11)
    axes[0, 1].set_title("Legacy", fontsize=11)

    for i, row in rows.iterrows():
        hsc_raw = _read_raw(hsc_bin, row["hsc_image_idx"], HSC_BANDS, HSC_H, HSC_W, HSC_BYTES)
        leg_raw = _read_raw(leg_bin, row["legacy_image_idx"], LEGACY_BANDS, LEGACY_H, LEGACY_W, LEGACY_BYTES)

        hsc_img = preprocess_raw_image(hsc_raw, survey="hsc", crop_size=48)[:4]
        leg_img = preprocess_raw_image(leg_raw, survey="legacy", crop_size=48)

        ref = hsc_img[:3]
        vmin = ref.amin(dim=(1, 2))
        vmax = ref.amax(dim=(1, 2))

        hsc_vis = _row_scale_rgb(hsc_img, vmin, vmax)
        leg_vis = _row_scale_rgb(leg_img, vmin, vmax)

        axes[i, 0].imshow(hsc_vis)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(leg_vis)
        axes[i, 1].axis("off")
        axes[i, 0].text(
            -0.05, 0.5,
            f"ra={row['ra']:.4f}\ndec={row['dec']:.4f}\nsep={row['sep_arcsec']:.2f}\"",
            transform=axes[i, 0].transAxes,
            ha="right", va="center", fontsize=7,
        )

    plt.tight_layout()
    plt.savefig(args.out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")

    if args.webhook:
        msg = f"{args.message} — n={n}, seed={args.seed}"
        cmd = [
            "curl", "-s", "-S", "-f",
            "-F", f"payload_json={{\"content\":\"{msg}\"}}",
            "-F", f"file=@{args.out}",
            args.webhook,
        ]
        try:
            subprocess.run(cmd, check=True)
            print("posted to Discord")
        except subprocess.CalledProcessError as e:
            print(f"discord post failed: {e}", file=sys.stderr)
    else:
        print("no --webhook / DISCORD_WEBHOOK set; skipping post")


if __name__ == "__main__":
    main()
