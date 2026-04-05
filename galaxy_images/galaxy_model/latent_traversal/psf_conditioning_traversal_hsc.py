"""
PSF conditioning traversal for HSC targets using HSC instrument buckets.

This mirrors psf_conditioning_traversal_v2.py, but flips the conditioning
direction to the HSC-anchor case used by the neighbors dataset:
  - target image: HSC
  - same-galaxy conditioning: Legacy
  - same-instrument conditioning buckets: HSC

HSC PSF FWHM is derived from shape moments using the same formula as the
downstream evaluation code, then averaged over g/i/r/z to define bucketed PSF
levels.

Outputs (3 vis methods, bucket size embedded in filename):
  psf_conditioning_traversal_hsc_<idx>_kb<N>_indep.png
  psf_conditioning_traversal_hsc_<idx>_kb<N>_rowscale.png
  psf_conditioning_traversal_hsc_<idx>_kb<N>_zscale.png

Run:
  python psf_conditioning_traversal_hsc.py [--target-idx 0] [--n-examples 100000] [--bucket-size 5]
  python psf_conditioning_traversal_hsc.py [--target-idx 0] [--n-examples 100000] [--bucket-size 15]
"""

import argparse
import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import os
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
torch.backends.cuda.preferred_blas_library("hipblas")

from galaxy_images.galaxy_model.double_train_fm_neighbors import ConditionalFlowMatchingModule
from galaxy_images.galaxy_model.neighbors import preprocess_raw_image


MODEL_CHECKPOINT = (
    "/work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model/"
    "outputs/neighbors_all_attn/checkpoints/best-epoch=228-step=87000.ckpt"
)
NEIGHBORS_HDF5 = "/work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5"
OUTPUT_DIR = Path(__file__).resolve().parent / "latent_traversal_hsc"

HSC_PSF_BANDS = ("g", "i", "r", "z")
DEFAULT_BUCKET_SIZE = 5
CROP_SIZE = 48
DEFAULT_NUM_EXAMPLES = 100_000
DEFAULT_TARGET_IDX = 0

PSF_LEVELS = [
    ("PSF max", 100),
    ("PSF p75", 75),
    ("PSF p50", 50),
    ("PSF p25", 25),
    ("PSF min", 0),
]


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


def compute_hsc_psf_seeing(shape11, shape22):
    """HSC PSF FWHM in arcsec from shape moments (pixel scale 0.168)."""
    pixel_scale_hsc = 0.168
    return 2.355 * np.sqrt((shape11 + shape22) / 2) * pixel_scale_hsc


def vis_indep(img: np.ndarray) -> np.ndarray:
    """Per-image percentile normalization for more robust visualization."""
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


def load_psf_and_indexes(hdf5_path: str, n_examples: int):
    """Load avg HSC PSF FWHM (across g/i/r/z) for the first n_examples MMU rows."""
    with h5py.File(hdf5_path, "r") as f:
        sources = f["source_type"][:]
        indexes_mmu = np.where(sources == 0)[0]
        n_use = min(n_examples, len(indexes_mmu))
        indexes_mmu = indexes_mmu[:n_use]

        bands = []
        for band in HSC_PSF_BANDS:
            key11 = f"hsc_{band}_sdssshape_psf_shape11"
            key22 = f"hsc_{band}_sdssshape_psf_shape22"
            if key11 not in f or key22 not in f:
                raise KeyError(f"Missing HSC PSF shape keys: {key11}, {key22}")
            shape11 = np.asarray(f[key11][indexes_mmu], dtype=np.float32)
            shape22 = np.asarray(f[key22][indexes_mmu], dtype=np.float32)
            bands.append(compute_hsc_psf_seeing(shape11, shape22).astype(np.float32))

    psf_avg = np.nanmean(np.stack(bands, axis=0), axis=0)
    print(
        f"Loaded {n_use} MMU rows. HSC PSF avg range: "
        f"[{np.nanmin(psf_avg):.3f}, {np.nanmax(psf_avg):.3f}] arcsec"
    )
    return indexes_mmu, psf_avg


def find_bucket_idxs(psf_avg: np.ndarray, percentile: float, n: int = DEFAULT_BUCKET_SIZE):
    """Return (dataset_idxs, avg_psf) for the n examples closest to `percentile`."""
    valid = np.where(np.isfinite(psf_avg))[0]
    target_val = np.nanpercentile(psf_avg[valid], percentile)
    distances = np.abs(psf_avg[valid] - target_val)
    nearest = np.argsort(distances)[:n]
    dataset_idxs = valid[nearest]
    avg_psf = float(psf_avg[dataset_idxs].mean())
    return dataset_idxs, avg_psf


def load_and_preprocess(hdf5_path: str, raw_row_idx: int, survey: str) -> torch.Tensor:
    key = "images_hsc" if survey == "hsc" else "images_legacy"
    with h5py.File(hdf5_path, "r") as f:
        img_raw = np.asarray(f[key][raw_row_idx], dtype=np.float32)
    img = preprocess_raw_image(img_raw, survey=survey, crop_size=CROP_SIZE)
    return img[:4] if survey == "hsc" else img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-idx", type=int, default=DEFAULT_TARGET_IDX,
                        help="Dataset index (into MMU rows) for the target example")
    parser.add_argument("--n-examples", type=int, default=DEFAULT_NUM_EXAMPLES,
                        help="Number of MMU examples to scan for PSF statistics")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--hdf5-path", type=str, default=NEIGHBORS_HDF5)
    parser.add_argument("--checkpoint", type=str, default=MODEL_CHECKPOINT)
    parser.add_argument("--bucket-size", type=int, default=DEFAULT_BUCKET_SIZE,
                        help="Number of distinct HSC neighbors per PSF bucket (e.g. 5 or 15)")
    args = parser.parse_args()
    bucket_size = args.bucket_size

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = determine_device()

    print("\n--- Loading HSC PSF metadata ---")
    indexes_mmu, psf_avg = load_psf_and_indexes(args.hdf5_path, args.n_examples)

    print(f"\n--- Finding HSC PSF buckets (n={bucket_size} per level) ---")
    psf_buckets = []
    for label, pct in PSF_LEVELS:
        ds_idxs, avg_psf = find_bucket_idxs(psf_avg, pct, n=bucket_size)
        psf_buckets.append((label, ds_idxs, avg_psf))
        psf_vals = [f"{psf_avg[di]:.3f}" for di in ds_idxs]
        print(f"  {label:10s} (p{pct:3.0f}) -> avg PSF={avg_psf:.3f}, bucket PSFs={psf_vals}")

    target_ds_idx = args.target_idx
    if target_ds_idx < 0 or target_ds_idx >= len(indexes_mmu):
        raise ValueError(f"--target-idx {target_ds_idx} out of range (0 to {len(indexes_mmu) - 1})")
    target_raw_row = int(indexes_mmu[target_ds_idx])
    print(
        f"\n--- Target: dataset_idx={target_ds_idx}, raw_row={target_raw_row}, "
        f"HSC PSF={psf_avg[target_ds_idx]:.3f} arcsec ---"
    )

    target_hsc = load_and_preprocess(args.hdf5_path, target_raw_row, "hsc")
    target_legacy = load_and_preprocess(args.hdf5_path, target_raw_row, "legacy")

    print("\n--- Loading HSC PSF bucket images ---")
    bucket_images = []
    for label, ds_idxs, avg_psf in psf_buckets:
        imgs = []
        for di in ds_idxs:
            raw_row = int(indexes_mmu[di])
            imgs.append(load_and_preprocess(args.hdf5_path, raw_row, "hsc"))
        bucket_images.append(imgs)
        print(f"  {label}: loaded {len(imgs)} images, avg HSC PSF={avg_psf:.3f}")

    print(f"\n--- Loading model from {args.checkpoint} ---")
    model = ConditionalFlowMatchingModule.load_from_checkpoint(args.checkpoint, map_location="cpu")
    model.eval()
    torch.set_grad_enabled(False)
    model = model.to(device)
    print("Model loaded.")

    print(f"\n--- Running HSC generation for each PSF bucket (K={bucket_size}) ---")
    samegal = target_legacy.unsqueeze(0).to(device)
    masks_kb = torch.ones(1, bucket_size, dtype=torch.bool, device=device)

    generated_images_kb = []
    for i, (label, ds_idxs, avg_psf) in enumerate(psf_buckets):
        imgs = bucket_images[i]
        sameins_kb = torch.stack(imgs).unsqueeze(0).to(device)
        print(f"  K={bucket_size} - {label} (avg HSC PSF={avg_psf:.3f})...")
        generated_images_kb.append(
            model.sample(samegal, sameins_kb, masks=masks_kb).squeeze(0).cpu().numpy()
        )

    tgt_np = target_hsc.numpy()[:3]
    rowscale_vmin = tgt_np.min(axis=(1, 2))
    rowscale_vmax = tgt_np.max(axis=(1, 2))

    print("\n--- Plotting ---")
    n_cols = 2 + len(PSF_LEVELS)
    vis_methods = [
        ("indep", "Per-image min-max (independent)"),
        ("rowscale", "Row-scaled to target HSC"),
        ("zscale", "Astropy ZScaleInterval"),
    ]

    def apply_vis(method: str, img_chw: np.ndarray) -> np.ndarray:
        if method == "indep":
            return vis_indep(img_chw)
        if method == "rowscale":
            return vis_rowscale(img_chw, rowscale_vmin, rowscale_vmax)
        if method == "zscale":
            return vis_zscale(img_chw)
        raise ValueError(method)

    def show(ax, img_chw, method, title=None):
        ax.imshow(apply_vis(method, img_chw))
        ax.set_axis_off()
        if title:
            ax.set_title(title, fontsize=9)

    def blank(ax, label=None):
        ax.set_axis_off()
        if label:
            ax.text(0.5, 0.5, label, ha="center", va="center",
                    fontsize=9, transform=ax.transAxes, color="gray")

    for method, method_label in vis_methods:
        fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 2.5, 2 * 2.8))

        show(axes[0, 0], target_hsc.numpy(), method, f"Target HSC\nPSF={psf_avg[target_ds_idx]:.2f}\"")
        show(axes[0, 1], target_legacy.numpy(), method, "Legacy source\n(same galaxy)")
        for col, (label, ds_idxs, avg_psf) in enumerate(psf_buckets):
            show(
                axes[0, 2 + col],
                bucket_images[col][0].numpy(),
                method,
                f"{label}\n(avg PSF={avg_psf:.2f}\", n={bucket_size})",
            )

        blank(axes[1, 0])
        blank(axes[1, 1])
        for col, gen_np in enumerate(generated_images_kb):
            label = psf_buckets[col][0]
            show(axes[1, 2 + col], gen_np, method, f"Generated\n({label})")

        for row_idx, row_label in enumerate(["Conditioning", "Generated"]):
            ax0 = axes[row_idx, 0]
            ax0.set_axis_on()
            ax0.set_xticks([])
            ax0.set_yticks([])
            for sp in ax0.spines.values():
                sp.set_visible(False)
            ax0.set_ylabel(row_label, fontsize=11, fontweight="bold", labelpad=6)

        fig.suptitle(
            f"HSC PSF conditioning traversal (target dataset_idx={target_ds_idx}, K={bucket_size} bucket) - {method_label}\n"
            f"Same Legacy source, same noise - HSC PSF conditioning uses {bucket_size} real neighbors per bucket",
            fontsize=12,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()

        out_path = args.output_dir / (
            f"psf_conditioning_traversal_hsc_{args.target_idx:05d}_kb{bucket_size}_{method}.png"
        )
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
