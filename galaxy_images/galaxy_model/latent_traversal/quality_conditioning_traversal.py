"""
Quality-conditioned latent traversal.

Like psf_conditioning_traversal_v2.py but uses the combined quality metric
(ivar × PSF) instead of Legacy PSF alone. Runs for both metrics:
  - sum score: z_ivar - z_psf  (high = more signal + sharper)
  - PCA score: PC1 projection  (positive = better quality)

For each metric, 5 quality buckets (Q p5 … Q p95) are selected from the
stats cache. The model is conditioned on each bucket of 5 Legacy neighbors
and generates an image, showing how quality conditioning changes the output.

Layout per figure (2 rows × 7 cols):
  Row 0 — conditioning:  [Target Legacy] [HSC source] [Qp5 cond] … [Qp95 cond]
  Row 1 — generated:     [     —       ] [    —     ] [Gen Qp5 ] … [Gen Qp95 ]

Outputs (2 metrics × 3 vis methods per target):
  quality_traversal_sum_{idx:05d}_kb{K}_{vis}.png
  quality_traversal_pca_{idx:05d}_kb{K}_{vis}.png

Usage:
  python quality_conditioning_traversal.py --target-idx 50000
  python quality_conditioning_traversal.py --target-quality-pct 50   # target at quality median
"""

import argparse
import io
import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parents[2]   # latent_traversal → galaxy_model → galaxy_images → tess-generative
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import os
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
torch.backends.cuda.preferred_blas_library("hipblas")

from galaxy_images.galaxy_model.double_train_fm_neighbors import ConditionalFlowMatchingModule
from galaxy_images.galaxy_model.neighbors import preprocess_raw_image

# ============= CONFIGURATION =============

MODEL_CHECKPOINT = (
    "/work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model/"
    "outputs/neighbors_all_attn/checkpoints/best-epoch=228-step=87000.ckpt"
)
NEIGHBORS_HDF5  = "/work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5"
CACHE_DEFAULT   = Path(__file__).resolve().parent / "ivar_psf_cache.npz"
OUTPUT_DIR      = Path(__file__).resolve().parent / "quality_traversal_results"
DISCORD_WEBHOOK = (
    "https://discord.com/api/webhooks/1497979386144493680/"
    "VA-xWhfTWzc-oeC5EvPzyqEk_MW52wZsK2RyLS0egfhHHHhBxrmb9NGawy0rIpfvn3Zo"
)

CROP_SIZE    = 48
BUCKET_SIZE  = 5

QUALITY_LEVELS = [
    ("Q p5",   5),
    ("Q p25", 25),
    ("Q p50", 50),
    ("Q p75", 75),
    ("Q p95", 95),
]

METRICS = [
    ("sum", "sum score (z_ivar − z_psf)"),
    ("pca", "PCA score (PC1)"),
]


# ============= HELPERS =============

def determine_device():
    device = torch.device("cpu")
    if torch.cuda.is_available():
        for gpu_id in range(torch.cuda.device_count()):
            try:
                torch.tensor([1.0], device=f"cuda:{gpu_id}")
                device = torch.device(f"cuda:{gpu_id}")
                print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
                break
            except RuntimeError:
                continue
    if device.type == "cpu":
        print("No GPU available, using CPU (will be slow)")
    return device


def vis_indep(img: np.ndarray) -> np.ndarray:
    rgb = img[:3].copy()
    lo, hi = np.percentile(rgb, [2, 98])
    rgb = (rgb - lo) / (hi - lo + 1e-8)
    return np.transpose(np.clip(rgb, 0, 1), (1, 2, 0))


def vis_rowscale(img: np.ndarray, vmin: np.ndarray, vmax: np.ndarray) -> np.ndarray:
    rgb = img[:3].copy()
    vmin3 = vmin[:, None, None]
    vmax3 = vmax[:, None, None]
    rgb = (rgb - vmin3) / (vmax3 - vmin3 + 1e-8)
    return np.transpose(np.clip(rgb, 0, 1), (1, 2, 0))


def vis_zscale(img: np.ndarray) -> np.ndarray:
    from astropy.visualization import ZScaleInterval
    interval = ZScaleInterval()
    rgb = np.zeros((3, img.shape[1], img.shape[2]), dtype=np.float32)
    for c in range(3):
        ch = img[c]
        vmin, vmax = interval.get_limits(ch)
        rgb[c] = np.clip((ch - vmin) / (vmax - vmin + 1e-8), 0, 1)
    return np.transpose(rgb, (1, 2, 0))


def discord_text(webhook_url: str, message: str):
    try:
        resp = requests.post(webhook_url, json={"content": message}, timeout=10)
        if resp.status_code not in (200, 204):
            print(f"  WARNING: Discord {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"  WARNING: Discord send failed: {e}")


def send_to_discord(webhook_url: str, file_path: Path, message: str = ""):
    with open(file_path, "rb") as fh:
        data = fh.read()
    resp = requests.post(
        webhook_url,
        data={"content": message} if message else {},
        files={"file": (file_path.name, io.BytesIO(data), "image/png")},
    )
    if resp.status_code in (200, 204):
        print(f"  Sent {file_path.name} to Discord.")
    else:
        print(f"  WARNING: Discord {resp.status_code}: {resp.text[:200]}")


# ============= DATA LOADING =============

def load_cache(cache_path: Path):
    """Load quality scores and HDF5 row indices from the analysis cache."""
    c = np.load(cache_path)
    scores = {
        "sum": c["score_sum"].astype(np.float32),
        "pca": c["score_pca"].astype(np.float32),
    }
    hdf5_row_idx = c["hdf5_row_idx"].astype(np.int32)
    ivar = c["ivar"].astype(np.float32)
    psf  = c["psf"].astype(np.float32)
    N = len(hdf5_row_idx)
    print(f"Loaded cache: {N:,} examples from {cache_path.name}")
    for name, sc in scores.items():
        print(f"  {name}: [{sc.min():.3f}, {sc.max():.3f}]")
    return scores, hdf5_row_idx, ivar, psf


def find_bucket_idxs(score: np.ndarray, percentile: float, n: int = BUCKET_SIZE):
    """Return n stats-array indices whose score is closest to the given percentile."""
    target_val = np.nanpercentile(score, percentile)
    distances  = np.abs(score - target_val)
    nearest    = np.argsort(distances)[:n]
    return nearest, float(score[nearest].mean())


def load_legacy(hdf5_path: str, raw_row: int) -> torch.Tensor:
    with h5py.File(hdf5_path, "r") as f:
        img_raw = np.asarray(f["images_legacy"][raw_row], dtype=np.float32)
    return preprocess_raw_image(img_raw, survey="legacy", crop_size=CROP_SIZE)


def load_hsc(hdf5_path: str, raw_row: int) -> torch.Tensor:
    with h5py.File(hdf5_path, "r") as f:
        img_raw = np.asarray(f["images_hsc"][raw_row], dtype=np.float32)
    return preprocess_raw_image(img_raw, survey="hsc", crop_size=CROP_SIZE)[:4]


# ============= PLOTTING =============

def make_figure(
    target_legacy_np: np.ndarray,
    target_hsc_np: np.ndarray,
    target_score: float,
    target_ivar: float,
    target_psf: float,
    bucket_cond_images: list,    # list of (4,48,48) np arrays, one per level (closest)
    bucket_labels: list,         # list of str
    bucket_score_avgs: list,     # list of float
    bucket_ivar_avgs: list,
    bucket_psf_avgs: list,
    generated_images: list,      # list of (4,48,48) np arrays, one per level
    target_idx: int,
    metric_name: str,
    metric_label: str,
    bucket_size: int,
    vis_method: str,
    rowscale_vmin: np.ndarray,
    rowscale_vmax: np.ndarray,
) -> plt.Figure:

    def apply_vis(img_chw):
        if vis_method == "indep":
            return vis_indep(img_chw)
        elif vis_method == "rowscale":
            return vis_rowscale(img_chw, rowscale_vmin, rowscale_vmax)
        elif vis_method == "zscale":
            return vis_zscale(img_chw)
        raise ValueError(vis_method)

    n_levels = len(QUALITY_LEVELS)
    n_cols = 2 + n_levels
    fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 2.5, 2 * 2.8))

    def show(ax, img, title=None):
        ax.imshow(apply_vis(img))
        ax.set_axis_off()
        if title:
            ax.set_title(title, fontsize=8)

    def blank(ax, label=None):
        ax.set_axis_off()
        if label:
            ax.text(0.5, 0.5, label, ha="center", va="center",
                    fontsize=9, transform=ax.transAxes, color="gray")

    # Row 0: conditioning
    show(axes[0, 0], target_legacy_np,
         f"Target HSC\nscore={target_score:.2f}\nivar={target_ivar:.0f} PSF={target_psf:.2f}\"")
    show(axes[0, 1], target_hsc_np, "Legacy source\n(same galaxy)")
    for col, (lbl, cond_img, sc_avg, iv_avg, ps_avg) in enumerate(
        zip(bucket_labels, bucket_cond_images, bucket_score_avgs, bucket_ivar_avgs, bucket_psf_avgs)
    ):
        show(axes[0, 2 + col], cond_img,
             f"{lbl}\nscore={sc_avg:.2f}\nivar={iv_avg:.0f} PSF={ps_avg:.2f}\"")

    # Row 1: generated
    blank(axes[1, 0])
    blank(axes[1, 1])
    for col, (gen_img, lbl) in enumerate(zip(generated_images, bucket_labels)):
        show(axes[1, 2 + col], gen_img, f"Generated\n({lbl})")

    # Row labels
    VIS_LABELS = {"indep": "Per-image min-max", "rowscale": "Row-scaled to target", "zscale": "ZScale"}
    for row_i, row_lbl in enumerate(["Conditioning", "Generated"]):
        ax0 = axes[row_i, 0]
        ax0.set_axis_on()
        ax0.set_xticks([]); ax0.set_yticks([])
        for sp in ax0.spines.values():
            sp.set_visible(False)
        ax0.set_ylabel(row_lbl, fontsize=11, fontweight="bold", labelpad=6)

    fig.suptitle(
        f"Quality traversal — {metric_label}  (target stats_idx={target_idx}, K={bucket_size} bucket)\n"
        f"{VIS_LABELS.get(vis_method, vis_method)}",
        fontsize=11, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    return fig


# ============= MAIN =============

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-idx", type=int, default=None,
                        help="Stats-array index of the target galaxy (0..N-1)")
    parser.add_argument("--target-quality-pct", type=float, default=None,
                        help="Pick target at this percentile of sum score (alternative to --target-idx)")
    parser.add_argument("--cache-path", type=Path, default=CACHE_DEFAULT)
    parser.add_argument("--hdf5-path", type=str, default=NEIGHBORS_HDF5)
    parser.add_argument("--checkpoint", type=str, default=MODEL_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--bucket-size", type=int, default=BUCKET_SIZE)
    parser.add_argument("--no-discord", action="store_true")
    parser.add_argument("--gallery-discord", action="store_true",
                        help="In batch mode: only send pca+rowscale to Discord (not all 6 images)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = determine_device()

    job_id   = os.environ.get("SLURM_JOB_ID", "local")
    hostname = os.uname().nodename

    def notify(msg: str):
        if not args.no_discord:
            discord_text(DISCORD_WEBHOOK, msg)

    notify(f"**Quality traversal started** | job={job_id} | host={hostname}")

    # ---- 1. Load cache ----
    scores, hdf5_row_idx, ivar_all, psf_all = load_cache(args.cache_path)
    N = len(hdf5_row_idx)

    # ---- 2. Resolve target index ----
    if args.target_idx is not None:
        target_stats_idx = args.target_idx
    elif args.target_quality_pct is not None:
        # Use sum score as reference for picking the target
        ref_score = scores["sum"]
        target_val = np.nanpercentile(ref_score, args.target_quality_pct)
        target_stats_idx = int(np.argmin(np.abs(ref_score - target_val)))
        print(f"--target-quality-pct {args.target_quality_pct} → stats_idx={target_stats_idx} "
              f"(sum_score={ref_score[target_stats_idx]:.3f})")
    else:
        target_stats_idx = N // 2
        print(f"No target specified, defaulting to stats_idx={target_stats_idx}")

    if target_stats_idx < 0 or target_stats_idx >= N:
        raise ValueError(f"--target-idx {target_stats_idx} out of range (0..{N-1})")

    target_raw_row = int(hdf5_row_idx[target_stats_idx])
    target_ivar    = float(ivar_all[target_stats_idx])
    target_psf     = float(psf_all[target_stats_idx])
    print(f"\nTarget: stats_idx={target_stats_idx}, hdf5_row={target_raw_row}, "
          f"ivar={target_ivar:.0f}, PSF={target_psf:.3f}\"")
    for mname, sc in scores.items():
        print(f"  {mname} score: {sc[target_stats_idx]:.3f}")

    score_summary = "  ".join(f"{m}={scores[m][target_stats_idx]:.2f}" for m, _ in METRICS)
    notify(
        f"**Target selected** | stats_idx={target_stats_idx} | "
        f"ivar={target_ivar:.0f}  PSF={target_psf:.2f}\" | {score_summary}"
    )

    # Per-target subfolder + shared gallery folder
    target_dir  = args.output_dir / f"target_{target_stats_idx:05d}"
    gallery_dir = args.output_dir / "gallery_pca_rowscale"
    target_dir.mkdir(parents=True, exist_ok=True)
    gallery_dir.mkdir(parents=True, exist_ok=True)

    # ---- 3. Load target images ----
    # HSC is the target domain (what we generate); Legacy is the same-galaxy condition.
    print("\n--- Loading target images ---")
    target_hsc    = load_hsc(args.hdf5_path, target_raw_row)     # (4, 48, 48) — target to reconstruct
    target_legacy = load_legacy(args.hdf5_path, target_raw_row)  # (4, 48, 48) — same-galaxy condition

    rowscale_ref = target_hsc.numpy()[:3]
    rowscale_vmin = rowscale_ref.min(axis=(1, 2))
    rowscale_vmax = rowscale_ref.max(axis=(1, 2))

    # ---- 4. Find buckets + load HSC conditioning images (per metric) ----
    # Instrument neighbors are HSC images at different quality levels.
    all_bucket_data = {}   # metric_name → list of (label, cond_imgs, score_avg, ivar_avg, psf_avg)
    for mname, _ in METRICS:
        sc = scores[mname]
        print(f"\n--- {mname}: finding quality buckets (n={args.bucket_size}) ---")
        buckets = []
        for label, pct in QUALITY_LEVELS:
            bucket_idxs, sc_avg = find_bucket_idxs(sc, pct, n=args.bucket_size)
            iv_avg  = float(ivar_all[bucket_idxs].mean())
            ps_avg  = float(psf_all[bucket_idxs].mean())
            sc_vals = [f"{sc[i]:.2f}" for i in bucket_idxs]
            print(f"  {label}: avg_score={sc_avg:.3f}  ivar={iv_avg:.0f}  PSF={ps_avg:.3f}\"  scores={sc_vals}")

            cond_imgs = []
            for bi in bucket_idxs:
                raw_row = int(hdf5_row_idx[bi])
                cond_imgs.append(load_hsc(args.hdf5_path, raw_row))  # HSC neighbors
            buckets.append((label, cond_imgs, sc_avg, iv_avg, ps_avg))
        all_bucket_data[mname] = buckets

    # ---- 5. Load model ----
    print(f"\n--- Loading model ---")
    model = ConditionalFlowMatchingModule.load_from_checkpoint(
        args.checkpoint, map_location="cpu"
    )
    model.eval()
    torch.set_grad_enabled(False)
    model = model.to(device)
    print("Model loaded.")

    samegal  = target_legacy.unsqueeze(0).to(device)  # (1, 4, 48, 48) — Legacy as same-galaxy cond.
    masks_kb = torch.ones(1, args.bucket_size, dtype=torch.bool, device=device)

    # ---- 6. Generate (per metric) ----
    generated_per_metric = {}
    for mname, mlabel in METRICS:
        print(f"\n--- Generating: metric={mname} ---")
        notify(f"**Generating [{mname}]** target={target_stats_idx} — 5 quality levels ...")
        buckets = all_bucket_data[mname]
        gen_images = []
        for label, cond_imgs, sc_avg, _, _ in buckets:
            sameins = torch.stack(cond_imgs).unsqueeze(0).to(device)   # (1, K, 4, 48, 48)
            print(f"  {label} (avg score={sc_avg:.3f}) ...")
            gen = model.sample(samegal, sameins, masks=masks_kb)
            gen_images.append(gen.squeeze(0).cpu().numpy())             # (4, 48, 48)
        generated_per_metric[mname] = gen_images
        notify(f"**Done generating [{mname}]** target={target_stats_idx}")

    # ---- 7. Plot ----
    print("\n--- Plotting ---")
    VIS_METHODS = ["indep", "rowscale", "zscale"]
    saved_paths = []

    for mname, mlabel in METRICS:
        buckets   = all_bucket_data[mname]
        gen_imgs  = generated_per_metric[mname]
        target_sc = float(scores[mname][target_stats_idx])

        bucket_labels     = [b[0] for b in buckets]
        bucket_cond_imgs  = [b[1][0].numpy() for b in buckets]  # closest HSC example per bucket
        bucket_sc_avgs    = [b[2] for b in buckets]
        bucket_ivar_avgs  = [b[3] for b in buckets]
        bucket_psf_avgs   = [b[4] for b in buckets]

        for vis in VIS_METHODS:
            fig = make_figure(
                target_legacy_np=target_hsc.numpy(),     # col 0: target HSC (what we reconstruct)
                target_hsc_np=target_legacy.numpy(),     # col 1: Legacy (same-galaxy condition)
                target_score=target_sc,
                target_ivar=target_ivar,
                target_psf=target_psf,
                bucket_cond_images=bucket_cond_imgs,
                bucket_labels=bucket_labels,
                bucket_score_avgs=bucket_sc_avgs,
                bucket_ivar_avgs=bucket_ivar_avgs,
                bucket_psf_avgs=bucket_psf_avgs,
                generated_images=gen_imgs,
                target_idx=target_stats_idx,
                metric_name=mname,
                metric_label=mlabel,
                bucket_size=args.bucket_size,
                vis_method=vis,
                rowscale_vmin=rowscale_vmin,
                rowscale_vmax=rowscale_vmax,
            )
            fname = f"quality_traversal_{mname}_{target_stats_idx:05d}_kb{args.bucket_size}_{vis}.png"
            out   = target_dir / fname
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {out}")
            saved_paths.append((out, mname, vis))

            # Copy pca+rowscale to gallery
            if mname == "pca" and vis == "rowscale":
                import shutil
                shutil.copy2(out, gallery_dir / fname)

    # ---- 8. Discord ----
    if not args.no_discord:
        print("\n--- Sending to Discord ---")
        if args.gallery_discord:
            # Batch mode: only send pca+rowscale to Discord
            for path, mname, vis in saved_paths:
                if mname == "pca" and vis == "rowscale":
                    send_to_discord(
                        DISCORD_WEBHOOK, path,
                        f"**Quality traversal [pca/rowscale]** target={target_stats_idx} K={args.bucket_size}",
                    )
        else:
            for path, mname, vis in saved_paths:
                send_to_discord(
                    DISCORD_WEBHOOK, path,
                    f"**Quality traversal [{mname}]** target={target_stats_idx} K={args.bucket_size} vis={vis}",
                )
        notify(f"**Target {target_stats_idx} complete** — {len(saved_paths)} figures saved")

    print("\nDone.")


if __name__ == "__main__":
    main()
