"""
PSF conditioning traversal v2 — bucket neighbors.

Like v1 but instead of repeating a single representative image K=15 times,
we pick --bucket-size distinct Legacy images closest to each PSF percentile
and pass them as real distinct neighbors.

Layout (2 rows × 7 cols):
  Row 0 — conditioning images:
    [Target Legacy] [HSC source] [PSF_max cond] [PSF_p75 cond] [PSF_p50 cond] [PSF_p25 cond] [PSF_min cond]
    (conditioning column shows the single closest example; title reports avg PSF of the bucket)
  Row 1 — generated images:
    [     —       ] [    —     ] [Gen_max      ] [Gen_p75     ] [Gen_p50     ] [Gen_p25     ] [Gen_min     ]

Outputs (3 vis methods, bucket size embedded in filename):
  psf_conditioning_traversal_v2_<idx>_kb<N>_indep.png     — K=N bucket, per-image min-max
  psf_conditioning_traversal_v2_<idx>_kb<N>_rowscale.png  — K=N bucket, row-scaled to target Legacy
  psf_conditioning_traversal_v2_<idx>_kb<N>_zscale.png    — K=N bucket, astropy ZScaleInterval

Run:
  python psf_conditioning_traversal_v2.py [--target-idx 0] [--n-examples 95000] [--bucket-size 5]
  python psf_conditioning_traversal_v2.py [--target-idx 0] [--n-examples 95000] [--bucket-size 15]
"""

import argparse
import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parents[3]   # .../tess-generative
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

# ============= CONFIGURATION =============

MODEL_CHECKPOINT = (
    '/work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model/'
    'outputs/neighbors_all_attn/checkpoints/best-epoch=228-step=87000.ckpt'
)
NEIGHBORS_HDF5 = "/work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5"
OUTPUT_DIR = Path(__file__).resolve().parent

LEGACY_PSF_BANDS = ["legacy_PSFSIZE_G", "legacy_PSFSIZE_I", "legacy_PSFSIZE_R", "legacy_PSFSIZE_Z"]
DEFAULT_BUCKET_SIZE = 5    # overridable via --bucket-size
CROP_SIZE = 48
DEFAULT_NUM_EXAMPLES = 102_000
DEFAULT_TARGET_IDX = 0

# 5 PSF levels: (label, percentile)
PSF_LEVELS = [
    ("PSF max",  100),
    ("PSF p75",   75),
    ("PSF p50",   50),
    ("PSF p25",   25),
    ("PSF min",    0),
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
    """Per-image percentile normalization for more robust visualization.
    img: (C, H, W) numpy float → (H, W, 3) RGB in [0, 1]."""
    rgb = img[:3].copy()
    lo, hi = np.percentile(rgb, [2, 98])
    rgb = (rgb - lo) / (hi - lo + 1e-8)
    return np.transpose(np.clip(rgb, 0, 1), (1, 2, 0))


def vis_rowscale(img: np.ndarray, vmin: np.ndarray, vmax: np.ndarray) -> np.ndarray:
    """Row-scaled visualization: fixed per-channel vmin/vmax from the target Legacy image.
    img: (C, H, W) numpy float; vmin/vmax: (3,) → (H, W, 3) RGB in [0, 1]."""
    rgb = img[:3].copy()  # (3, H, W)
    vmin3 = vmin[:, None, None]
    vmax3 = vmax[:, None, None]
    rgb = (rgb - vmin3) / (vmax3 - vmin3 + 1e-8)
    return np.transpose(np.clip(rgb, 0, 1), (1, 2, 0))


def vis_zscale(img: np.ndarray) -> np.ndarray:
    """Astropy ZScaleInterval stretch, applied per channel.
    img: (C, H, W) numpy float → (H, W, 3) RGB in [0, 1]."""
    from astropy.visualization import ZScaleInterval
    interval = ZScaleInterval()
    rgb = np.zeros((3, img.shape[1], img.shape[2]), dtype=np.float32)
    for c in range(3):
        ch = img[c]
        vmin, vmax = interval.get_limits(ch)
        rgb[c] = np.clip((ch - vmin) / (vmax - vmin + 1e-8), 0, 1)
    return np.transpose(rgb, (1, 2, 0))


# ============= DATA LOADING =============

def load_psf_and_indexes(hdf5_path: str, n_examples: int):
    """Load avg Legacy PSF size (across g/i/r/z) for the first n_examples MMU rows."""
    with h5py.File(hdf5_path, "r") as f:
        sources = f["source_type"][:]
        indexes_mmu = np.where(sources == 0)[0]
        n_use = min(n_examples, len(indexes_mmu))
        indexes_mmu = indexes_mmu[:n_use]

        bands = []
        for key in LEGACY_PSF_BANDS:
            if key not in f:
                raise KeyError(f"Key '{key}' not found in HDF5.")
            bands.append(np.asarray(f[key][indexes_mmu], dtype=np.float32))

    psf_avg = np.nanmean(np.stack(bands, axis=0), axis=0)  # (N,)
    print(f"Loaded {n_use} MMU rows. PSF avg range: [{np.nanmin(psf_avg):.3f}, {np.nanmax(psf_avg):.3f}] arcsec")
    return indexes_mmu, psf_avg


def find_bucket_idxs(psf_avg: np.ndarray, percentile: float, n: int = DEFAULT_BUCKET_SIZE):
    """Return (dataset_idxs, avg_psf) for the n examples closest to `percentile`.
    dataset_idxs are sorted closest-first."""
    valid = np.where(np.isfinite(psf_avg))[0]
    target_val = np.nanpercentile(psf_avg[valid], percentile)
    distances = np.abs(psf_avg[valid] - target_val)
    nearest = np.argsort(distances)[:n]
    dataset_idxs = valid[nearest]
    avg_psf = float(psf_avg[dataset_idxs].mean())
    return dataset_idxs, avg_psf


def load_and_preprocess(hdf5_path: str, raw_row_idx: int, survey: str) -> torch.Tensor:
    """Load one raw image from HDF5 and preprocess it. Returns (C, H, W) tensor."""
    key = "images_hsc" if survey == "hsc" else "images_legacy"
    with h5py.File(hdf5_path, "r") as f:
        img_raw = np.asarray(f[key][raw_row_idx], dtype=np.float32)
    return preprocess_raw_image(img_raw, survey=survey, crop_size=CROP_SIZE)


# ============= MAIN =============

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
                        help="Number of distinct Legacy neighbors per PSF bucket (e.g. 5 or 15)")
    args = parser.parse_args()
    bucket_size = args.bucket_size

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = determine_device()

    # ---- 1. PSF statistics ----
    print("\n--- Loading PSF metadata ---")
    indexes_mmu, psf_avg = load_psf_and_indexes(args.hdf5_path, args.n_examples)

    # ---- 2. Find PSF buckets (bucket_size nearest examples per percentile) ----
    print(f"\n--- Finding PSF buckets (n={bucket_size} per level) ---")
    psf_buckets = []   # list of (label, dataset_idxs, avg_psf)
    for label, pct in PSF_LEVELS:
        ds_idxs, avg_psf = find_bucket_idxs(psf_avg, pct, n=bucket_size)
        psf_buckets.append((label, ds_idxs, avg_psf))
        psf_vals = [f"{psf_avg[di]:.3f}" for di in ds_idxs]
        print(f"  {label:10s} (p{pct:3.0f}) → avg PSF={avg_psf:.3f}, bucket PSFs={psf_vals}")

    # ---- 3. Load target images ----
    target_ds_idx = args.target_idx
    if target_ds_idx < 0 or target_ds_idx >= len(indexes_mmu):
        raise ValueError(f"--target-idx {target_ds_idx} out of range (0 to {len(indexes_mmu)-1})")
    target_raw_row = int(indexes_mmu[target_ds_idx])
    print(f"\n--- Target: dataset_idx={target_ds_idx}, raw_row={target_raw_row}, PSF={psf_avg[target_ds_idx]:.3f} arcsec ---")

    # Legacy (target domain) — 4 channels after preprocessing
    target_legacy = load_and_preprocess(args.hdf5_path, target_raw_row, "legacy")  # (4, 48, 48)
    # HSC (samegal / physics conditioning) — 5 channels → slice to 4
    target_hsc = load_and_preprocess(args.hdf5_path, target_raw_row, "hsc")[:4]    # (4, 48, 48)

    # ---- 4. Load PSF bucket images ----
    print("\n--- Loading PSF bucket images ---")
    # bucket_images[i] = list of BUCKET_SIZE (4, 48, 48) tensors, closest example first
    bucket_images = []
    for label, ds_idxs, avg_psf in psf_buckets:
        imgs = []
        for di in ds_idxs:
            raw_row = int(indexes_mmu[di])
            imgs.append(load_and_preprocess(args.hdf5_path, raw_row, "legacy"))
        bucket_images.append(imgs)
        print(f"  {label}: loaded {len(imgs)} images, avg PSF={avg_psf:.3f}")

    # ---- 5. Load model ----
    print(f"\n--- Loading model from {args.checkpoint} ---")
    model = ConditionalFlowMatchingModule.load_from_checkpoint(
        args.checkpoint, map_location="cpu"
    )
    model.eval()
    torch.set_grad_enabled(False)
    model = model.to(device)
    print("Model loaded.")

    # ---- 6. Generate for each PSF bucket (K=bucket_size) ----
    print(f"\n--- Running generation for each PSF bucket (K={bucket_size}) ---")
    samegal = target_hsc.unsqueeze(0).to(device)   # (1, 4, 48, 48)
    masks_kb = torch.ones(1, bucket_size, dtype=torch.bool, device=device)

    generated_images_kb = []
    for i, (label, ds_idxs, avg_psf) in enumerate(psf_buckets):
        imgs = bucket_images[i]  # list of bucket_size tensors (4, 48, 48)

        # K=bucket_size: all bucket images as distinct neighbors → (1, K, 4, 48, 48)
        sameins_kb = torch.stack(imgs).unsqueeze(0).to(device)
        print(f"  K={bucket_size} — {label} (avg PSF={avg_psf:.3f})...")
        generated_images_kb.append(
            model.sample(samegal, sameins_kb, masks=masks_kb).squeeze(0).cpu().numpy()
        )

    # ---- 7. Pre-compute row-scale reference (target Legacy per-channel min/max) ----
    tgt_np = target_legacy.numpy()[:3]          # (3, H, W)
    rowscale_vmin = tgt_np.min(axis=(1, 2))     # (3,)
    rowscale_vmax = tgt_np.max(axis=(1, 2))     # (3,)

    # ---- 8. Plot — one figure per (vis method × K setting) ----
    print("\n--- Plotting ---")
    n_psf = len(PSF_LEVELS)
    n_cols = 2 + n_psf   # Target Legacy, HSC, then one per PSF bucket

    VIS_METHODS = [
        ("indep",    "Per-image min-max (independent)"),
        ("rowscale", "Row-scaled to target Legacy"),
        ("zscale",   "Astropy ZScaleInterval"),
    ]

    def apply_vis(method: str, img_chw: np.ndarray) -> np.ndarray:
        if method == "indep":
            return vis_indep(img_chw)
        elif method == "rowscale":
            return vis_rowscale(img_chw, rowscale_vmin, rowscale_vmax)
        elif method == "zscale":
            return vis_zscale(img_chw)
        raise ValueError(method)

    def show(ax, img_chw, method, title=None):
        rgb = apply_vis(method, img_chw)
        ax.imshow(rgb)
        ax.set_axis_off()
        if title:
            ax.set_title(title, fontsize=9)

    def blank(ax, label=None):
        ax.set_axis_off()
        if label:
            ax.text(0.5, 0.5, label, ha="center", va="center",
                    fontsize=9, transform=ax.transAxes, color="gray")

    for method, method_label in VIS_METHODS:
        fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 2.5, 2 * 2.8))

        # Row 0: conditioning images (show closest bucket example)
        show(axes[0, 0], target_legacy.numpy(), method, f"Target Legacy\nPSF={psf_avg[target_ds_idx]:.2f}\"")
        show(axes[0, 1], target_hsc.numpy(), method, "HSC source\n(same galaxy)")
        for col, (label, ds_idxs, avg_psf) in enumerate(psf_buckets):
            show(axes[0, 2 + col], bucket_images[col][0].numpy(), method,
                 f"{label}\n(avg PSF={avg_psf:.2f}\", n={bucket_size})")

        # Row 1: generated images
        blank(axes[1, 0])
        blank(axes[1, 1])
        for col, gen_np in enumerate(generated_images_kb):
            label = psf_buckets[col][0]
            show(axes[1, 2 + col], gen_np, method, f"Generated\n({label})")

        # Row labels
        for row_idx, row_label in enumerate(["Conditioning", "Generated"]):
            ax0 = axes[row_idx, 0]
            ax0.set_axis_on()
            ax0.set_xticks([]); ax0.set_yticks([])
            for sp in ax0.spines.values():
                sp.set_visible(False)
            ax0.set_ylabel(row_label, fontsize=11, fontweight="bold", labelpad=6)

        fig.suptitle(
            f"PSF conditioning traversal v2 (target dataset_idx={target_ds_idx}, K={bucket_size} bucket) — {method_label}\n"
            f"Same HSC source, same noise — PSF conditioning uses {bucket_size} real neighbors per bucket",
            fontsize=12, fontweight="bold", y=1.02,
        )
        plt.tight_layout()

        out_path = args.output_dir / f"psf_conditioning_traversal_v2_{args.target_idx:05d}_kb{bucket_size}_{method}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
