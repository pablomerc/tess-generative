"""
SNR-conditioned latent traversal.

Uses the 4-band cmodel-flux-SNR axis (= mean over g/r/i/z of cmodel_flux/cmodel_fluxerr,
stored as `snr_neg_4band` = -SNR in /work1/.../hsc_noise_metrics.h5) as the noise/quality
axis for the instrument-side conditioning.

For a chosen target galaxy, picks 5 SNR buckets (p5..p95 of snr_neg_4band):
  - p5  = lowest snr_neg = HIGHEST SNR  (cleanest)
  - p95 = highest snr_neg = LOWEST SNR  (noisiest)

For each bucket, finds K nearest HSC examples to the percentile, feeds them as
sameins to the dual-encoder flow-matching model (Legacy-of-target as samegal),
and generates one HSC image. Plots a 2×7 grid (target + 5 buckets) with three
visualizations (indep, rowscale, zscale).

Usage:
  python snr_conditioning_traversal.py --target-snr-pct 50
  python snr_conditioning_traversal.py --target-idx 12345
"""

import argparse
import io
import os
import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_script_dir))

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
from discord_notify import notify as _notify

# ============= CONFIGURATION =============

MODEL_CHECKPOINT = (
    "/work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model/"
    "outputs/neighbors_all_attn/checkpoints/best-epoch=228-step=87000.ckpt"
)
NEIGHBORS_HDF5  = "/work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5"
METRICS_HDF5    = Path("/work1/jeroenaudenaert/pablomer/data/hsc_noise_metrics.h5")
STATS_HDF5      = Path("/work1/jeroenaudenaert/pablomer/data/hsc_ivar_psf_stats.h5")
OUTPUT_DIR      = Path(__file__).resolve().parent / "snr_traversal_results"
DISCORD_WEBHOOK = (
    "https://discord.com/api/webhooks/1497979386144493680/"
    "VA-xWhfTWzc-oeC5EvPzyqEk_MW52wZsK2RyLS0egfhHHHhBxrmb9NGawy0rIpfvn3Zo"
)

CROP_SIZE   = 48
BUCKET_SIZE = 5

# Ordered noisy→clean (left-to-right reads as "gradual denoising").
# Percentile is on snr_neg_4band: p95 = lowest SNR = noisiest neighbors;
#                                  p5  = highest SNR = cleanest neighbors.
SNR_LEVELS = [
    ("Noisy neighbors (p95)", 95),
    ("p75",                   75),
    ("p50",                   50),
    ("p25",                   25),
    ("Clean neighbors (p5)",   5),
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
    rgb = (rgb - vmin[:, None, None]) / (vmax[:, None, None] - vmin[:, None, None] + 1e-8)
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


def send_to_discord(webhook_url: str, file_path: Path, message: str = ""):
    with open(file_path, "rb") as fh:
        data = fh.read()
    resp = requests.post(
        webhook_url,
        data={"content": message} if message else {},
        files={"file": (file_path.name, io.BytesIO(data), "image/png")},
    )
    if resp.status_code in (200, 204):
        print(f"  Sent {file_path.name}")
    else:
        print(f"  WARNING: Discord {resp.status_code}: {resp.text[:200]}")


def load_legacy(hdf5_path: str, raw_row: int) -> torch.Tensor:
    with h5py.File(hdf5_path, "r") as f:
        img_raw = np.asarray(f["images_legacy"][raw_row], dtype=np.float32)
    return preprocess_raw_image(img_raw, survey="legacy", crop_size=CROP_SIZE)


def load_hsc(hdf5_path: str, raw_row: int) -> torch.Tensor:
    with h5py.File(hdf5_path, "r") as f:
        img_raw = np.asarray(f["images_hsc"][raw_row], dtype=np.float32)
    return preprocess_raw_image(img_raw, survey="hsc", crop_size=CROP_SIZE)[:4]


def find_bucket_idxs(score: np.ndarray, valid_mask: np.ndarray, percentile: float, n: int):
    """n stats-array indices with finite score closest to the given percentile of valid scores."""
    valid_score = score[valid_mask]
    target_val = np.nanpercentile(valid_score, percentile)
    distances = np.where(valid_mask, np.abs(score - target_val), np.inf)
    nearest = np.argsort(distances)[:n]
    return nearest, float(target_val), float(score[nearest].mean())


# ============= PLOTTING =============

def make_figure(
    target_hsc_np, target_legacy_np,
    target_snr_actual, target_ivar, target_psf,
    bucket_cond_images, bucket_labels, bucket_snr_avgs, bucket_ivar_avgs, bucket_psf_avgs,
    generated_images, target_idx, bucket_size, mode,
    vis_method, rowscale_vmin, rowscale_vmax,
):
    def apply_vis(img):
        if vis_method == "indep":   return vis_indep(img)
        if vis_method == "rowscale": return vis_rowscale(img, rowscale_vmin, rowscale_vmax)
        if vis_method == "zscale":   return vis_zscale(img)
        raise ValueError(vis_method)

    n_levels = len(SNR_LEVELS)
    n_cols = 2 + n_levels
    fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 2.5, 2 * 2.8))

    def show(ax, img, title=None):
        ax.imshow(apply_vis(img)); ax.set_axis_off()
        if title: ax.set_title(title, fontsize=8)

    def blank(ax):
        ax.set_axis_off()

    # Row 0: conditioning
    show(axes[0, 0], target_hsc_np,
         f"Target HSC\nSNR={target_snr_actual:.1f}\nivar={target_ivar:.0f} PSF={target_psf:.2f}\"")
    show(axes[0, 1], target_legacy_np, "Legacy source\n(same galaxy)")
    for col, (lbl, cond_img, snr_avg, iv_avg, ps_avg) in enumerate(
        zip(bucket_labels, bucket_cond_images, bucket_snr_avgs, bucket_ivar_avgs, bucket_psf_avgs)
    ):
        show(axes[0, 2 + col], cond_img,
             f"{lbl}\nSNR={snr_avg:.1f}\nivar={iv_avg:.0f} PSF={ps_avg:.2f}\"")

    # Row 1: generated
    blank(axes[1, 0]); blank(axes[1, 1])
    for col, (gen_img, lbl) in enumerate(zip(generated_images, bucket_labels)):
        show(axes[1, 2 + col], gen_img, f"Generated\n({lbl})")

    for row_i, row_lbl in enumerate(["Conditioning", "Generated"]):
        ax0 = axes[row_i, 0]
        ax0.set_axis_on(); ax0.set_xticks([]); ax0.set_yticks([])
        for sp in ax0.spines.values(): sp.set_visible(False)
        ax0.set_ylabel(row_lbl, fontsize=11, fontweight="bold", labelpad=6)

    VIS_LABELS = {"indep": "Per-image min-max", "rowscale": "Row-scaled to target", "zscale": "ZScale"}
    fig.suptitle(
        f"SNR traversal — cmodel flux/fluxerr (4-band avg)  "
        f"(target stats_idx={target_idx}, mode={mode})\n{VIS_LABELS.get(vis_method, vis_method)}",
        fontsize=11, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    return fig


# ============= MAIN =============

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-idx", type=int, default=None)
    parser.add_argument("--target-snr-pct", type=float, default=None,
                        help="Pick target at this percentile of SNR (0=worst, 100=best). "
                             "Internally translates to (100-pct) of snr_neg_4band.")
    parser.add_argument("--metrics-path", type=Path, default=METRICS_HDF5)
    parser.add_argument("--stats-path",   type=Path, default=STATS_HDF5)
    parser.add_argument("--hdf5-path",    type=str, default=NEIGHBORS_HDF5)
    parser.add_argument("--checkpoint",   type=str, default=MODEL_CHECKPOINT)
    parser.add_argument("--output-dir",   type=Path, default=OUTPUT_DIR)
    parser.add_argument("--bucket-size",  type=int, default=BUCKET_SIZE)
    parser.add_argument("--mode", choices=["k5", "k10", "k1x5"], default="k5",
                        help="k5: 5 distinct neighbors. k10: 10 distinct neighbors. "
                             "k1x5: 1 nearest neighbor repeated 5x.")
    parser.add_argument("--no-discord",   action="store_true")
    parser.add_argument("--gallery-discord", action="store_true",
                        help="Only send rowscale figure to Discord (saves all locally)")
    args = parser.parse_args()

    # Resolve mode → (K_select, K_pass, repeat)
    if args.mode == "k5":
        n_select, n_pass, repeat_one = 5, 5, False
    elif args.mode == "k10":
        n_select, n_pass, repeat_one = 10, 10, False
    elif args.mode == "k1x5":
        n_select, n_pass, repeat_one = 1, 5, True
    else:
        raise ValueError(args.mode)
    print(f"Mode={args.mode}: select {n_select} neighbor(s) per bucket, "
          f"pass {n_pass} to model{' (repeated)' if repeat_one else ''}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = determine_device()

    job_id = os.environ.get("SLURM_JOB_ID", "local")

    def notify(msg):
        if not args.no_discord:
            _notify(msg)

    # ---- 1. Load score + context ----
    print(f"=== Loading metrics from {args.metrics_path} ===")
    with h5py.File(args.metrics_path, "r") as mf:
        rows_metrics = np.array(mf["hdf5_row_idx"], dtype=np.int64)
        snr_neg_4band = np.array(mf["snr_neg_4band"], dtype=np.float64)  # higher = noisier
    with h5py.File(args.stats_path, "r") as sf:
        rows_stats = np.array(sf["hdf5_row_idx"], dtype=np.int64)
        ivar_all   = np.array(sf["hsc_mean_ivar"], dtype=np.float32)
        psf_all    = np.array(sf["hsc_psf_fwhm_avg"], dtype=np.float32)
    if not np.array_equal(rows_metrics, rows_stats):
        raise RuntimeError("noise_metrics rows misaligned with stats rows")
    hdf5_row_idx = rows_metrics
    N = len(hdf5_row_idx)

    snr_pos = -snr_neg_4band  # actual SNR (cmodel flux/fluxerr 4-band avg)
    valid = np.isfinite(snr_neg_4band)
    print(f"  N={N:,}  finite SNR={valid.sum():,}  "
          f"SNR range=[{np.nanmin(snr_pos):.2f}, {np.nanmax(snr_pos):.2f}]")

    # ---- 2. Resolve target ----
    if args.target_idx is not None:
        target_stats_idx = args.target_idx
    elif args.target_snr_pct is not None:
        # user passes pct of SNR (high=clean). snr_neg pct is 100-pct.
        target_neg_pct = 100.0 - args.target_snr_pct
        target_val = np.nanpercentile(snr_neg_4band[valid], target_neg_pct)
        d = np.where(valid, np.abs(snr_neg_4band - target_val), np.inf)
        target_stats_idx = int(np.argmin(d))
        print(f"--target-snr-pct {args.target_snr_pct} → snr_neg pct {target_neg_pct} → "
              f"stats_idx={target_stats_idx} (SNR={snr_pos[target_stats_idx]:.2f})")
    else:
        target_stats_idx = N // 2

    if not valid[target_stats_idx]:
        raise RuntimeError(f"Target idx {target_stats_idx} has invalid SNR")

    target_raw_row = int(hdf5_row_idx[target_stats_idx])
    target_snr     = float(snr_pos[target_stats_idx])
    target_ivar    = float(ivar_all[target_stats_idx])
    target_psf     = float(psf_all[target_stats_idx])
    print(f"\nTarget: stats_idx={target_stats_idx}  hdf5_row={target_raw_row}  "
          f"SNR={target_snr:.2f}  ivar={target_ivar:.0f}  PSF={target_psf:.3f}\"")

    notify(
        f"**SNR traversal target** | job={job_id} | stats_idx={target_stats_idx} | "
        f"SNR={target_snr:.2f}  ivar={target_ivar:.0f}  PSF={target_psf:.2f}\""
    )

    target_dir  = args.output_dir / f"target_{target_stats_idx:05d}"
    gallery_dir = args.output_dir / "gallery_rowscale"
    target_dir.mkdir(parents=True, exist_ok=True)
    gallery_dir.mkdir(parents=True, exist_ok=True)

    # ---- 3. Load target images ----
    print("\n--- Loading target images ---")
    target_hsc    = load_hsc(args.hdf5_path, target_raw_row)
    target_legacy = load_legacy(args.hdf5_path, target_raw_row)
    rowscale_ref  = target_hsc.numpy()[:3]
    rowscale_vmin = rowscale_ref.min(axis=(1, 2))
    rowscale_vmax = rowscale_ref.max(axis=(1, 2))

    # ---- 4. Build SNR buckets ----
    print(f"\n--- Building 5 SNR buckets (mode={args.mode}, n_select={n_select}, n_pass={n_pass}) ---")
    bucket_data = []
    for label, neg_pct in SNR_LEVELS:
        idxs, target_neg, neg_avg = find_bucket_idxs(snr_neg_4band, valid, neg_pct, n_select)
        snr_avg = -neg_avg
        iv_avg  = float(ivar_all[idxs].mean())
        ps_avg  = float(psf_all[idxs].mean())
        snr_vals = [f"{snr_pos[i]:.1f}" for i in idxs]
        print(f"  {label:24s}  SNR_avg={snr_avg:7.2f}  ivar={iv_avg:6.0f}  PSF={ps_avg:.3f}\"  members={snr_vals}")
        cond_imgs = [load_hsc(args.hdf5_path, int(hdf5_row_idx[bi])) for bi in idxs]
        if repeat_one:
            cond_imgs = cond_imgs * n_pass  # repeat the single nearest neighbor n_pass times
        bucket_data.append((label, cond_imgs, snr_avg, iv_avg, ps_avg))

    # ---- 5. Load model ----
    print(f"\n--- Loading model ---")
    model = ConditionalFlowMatchingModule.load_from_checkpoint(args.checkpoint, map_location="cpu")
    model.eval()
    torch.set_grad_enabled(False)
    model = model.to(device)
    print("Model loaded.")

    samegal  = target_legacy.unsqueeze(0).to(device)
    masks_kb = torch.ones(1, n_pass, dtype=torch.bool, device=device)

    # ---- 6. Generate per bucket ----
    print("\n--- Generating ---")
    notify(f"**Generating** target={target_stats_idx} — 5 SNR buckets ...")
    gen_images = []
    for label, cond_imgs, snr_avg, _, _ in bucket_data:
        sameins = torch.stack(cond_imgs).unsqueeze(0).to(device)
        print(f"  {label} (SNR_avg={snr_avg:.2f}) ...")
        gen = model.sample(samegal, sameins, masks=masks_kb)
        gen_images.append(gen.squeeze(0).cpu().numpy())

    # ---- 7. Plot 3 vis methods ----
    print("\n--- Plotting ---")
    saved_paths = []
    for vis in ["indep", "rowscale", "zscale"]:
        fig = make_figure(
            target_hsc_np=target_hsc.numpy(),
            target_legacy_np=target_legacy.numpy(),
            target_snr_actual=target_snr,
            target_ivar=target_ivar,
            target_psf=target_psf,
            bucket_cond_images=[b[1][0].numpy() for b in bucket_data],
            bucket_labels=[b[0] for b in bucket_data],
            bucket_snr_avgs=[b[2] for b in bucket_data],
            bucket_ivar_avgs=[b[3] for b in bucket_data],
            bucket_psf_avgs=[b[4] for b in bucket_data],
            generated_images=gen_images,
            target_idx=target_stats_idx,
            bucket_size=n_pass,
            mode=args.mode,
            vis_method=vis,
            rowscale_vmin=rowscale_vmin,
            rowscale_vmax=rowscale_vmax,
        )
        fname = f"snr_traversal_{target_stats_idx:05d}_{args.mode}_{vis}.png"
        out = target_dir / fname
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append((out, vis))
        print(f"  Saved {out}")
        if vis == "rowscale":
            import shutil
            shutil.copy2(out, gallery_dir / fname)

    # ---- 8. Discord ----
    if not args.no_discord:
        for path, vis in saved_paths:
            if args.gallery_discord and vis != "rowscale":
                continue
            send_to_discord(
                DISCORD_WEBHOOK, path,
                f"**SNR traversal [{args.mode}]** target={target_stats_idx} "
                f"vis={vis} | target SNR={target_snr:.1f}"
            )
        notify(f"**Target {target_stats_idx} complete** — {len(saved_paths)} figures saved")

    print("\nDone.")


if __name__ == "__main__":
    main()
