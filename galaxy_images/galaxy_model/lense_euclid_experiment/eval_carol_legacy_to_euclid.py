"""
Carol Legacy -> predicted HSC -> real Euclid follow-up figure.

For each strong lens in q1_lenses_in_legacy_footprint.csv, plot three panels:
    1) Legacy (the input the model sees)
    2) Predicted HSC (mean of 5 model samples)
    3) Real Euclid VIS+NIR follow-up (the rgb_0.png composite from inside lens.zip,
       same as strong_lenses_carol/replot.py uses; center-cropped to match the
       predicted-HSC field of view via --euclid-fov)

Conditioning HSC neighbors are sampled at random from the source_type==0 pool
in neighbours_v2.h5 (no pre-computed neighbor indices exist for Carol lenses).

Usage:
    python eval_carol_legacy_to_euclid.py --num-lenses 2 --steps 50  # smoke test
    python eval_carol_legacy_to_euclid.py                            # full run (20 lenses)
    python eval_carol_legacy_to_euclid.py --replot                   # replot from cache
"""
from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from astropy.io import fits
from PIL import Image

torch.backends.cuda.preferred_blas_library("hipblas")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

_this = Path(__file__).resolve()
_root = _this.parents[4]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from galaxy_images.galaxy_model.neighbors import preprocess_raw_image
from galaxy_images.galaxy_model.hierarchical_attention.double_train_fm_neighbors_hier_global_ins import (
    HierarchicalGlobalInstrumentFlowMatchingModule,
)

CAROL_DIR = Path("/work1/jeroenaudenaert/pablomer/data/strong_lenses_carol")
CSV_PATH = CAROL_DIR / "q1_lenses_in_legacy_footprint.csv"
LEGACY_FITS_DIR = CAROL_DIR / "legacy_fits"
LENS_ZIP = CAROL_DIR / "lens.zip"
NEIGHBOURS_H5 = Path("/work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5")
DEFAULT_CKPT = (
    "/work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model"
    "/hierarchical_attention/outputs/neighbors_hier_global_ins"
    "/2026-04-22_2/checkpoints/latest-step=step=201000.ckpt"
)

MAX_NEIGHBORS = 5
NUM_SAMPLES_PER_COND = 5
NUM_INTEGRATION_STEPS = 100
CROP_SIZE = 48

LEGACY_NATIVE_PIX = 0.262   # arcsec/px (DESI Legacy DR10)
HSC_PIX = 0.168             # arcsec/px
EUCLID_VIS_PIX = 0.1        # arcsec/px

# After zoom_legacy_image (0.64x crop + resample to original size) and crop_size=48
# both Legacy-input and Predicted-HSC live at HSC pixel scale -> 48 * 0.168 = 8.06"
LEGACY_INPUT_FOV = CROP_SIZE * HSC_PIX
HSC_FOV = CROP_SIZE * HSC_PIX
EUCLID_FOV = 300 * EUCLID_VIS_PIX  # 30"


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- data loading ----------------------------------------------------------

def _load_hsc_pool_indices() -> np.ndarray:
    print("Loading HSC pool indices (source_type==0)...")
    with h5py.File(NEIGHBOURS_H5, "r") as f:
        source_type = f["source_type"][:]
    pool = np.where(source_type == 0)[0]
    print(f"  HSC pool size: {len(pool):,}")
    return pool


def _load_euclid_rgb_from_zip(zf: zipfile.ZipFile, id_str: str) -> np.ndarray | None:
    """Return the pre-rendered Euclid RGB PNG from the zip as (H, W, 3) uint8.

    This is the same composite that strong_lenses_carol/replot.py loads via
    `euclid_rgb_0`, but read directly from inside lens.zip (the upstream paths
    in the CSV point to a different machine).
    """
    member = f"lens/{id_str}/rgb_0.png"
    try:
        with zf.open(member) as f:
            buf = io.BytesIO(f.read())
        with Image.open(buf) as im:
            return np.asarray(im.convert("RGB"))
    except (KeyError, OSError):
        return None


def _load_legacy_fits(id_str: str) -> np.ndarray | None:
    path = LEGACY_FITS_DIR / f"{id_str}.fits"
    if not path.exists():
        return None
    with fits.open(path) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float32)  # (4, 96, 96)
    return data


def load_lenses(num_lenses: int | None, seed: int = 42):
    """Load N lenses sorted by expert_score (descending). Return tensors + metadata."""
    df = pd.read_csv(CSV_PATH)
    if "expert_score" in df.columns:
        df = df.sort_values("expert_score", ascending=False).reset_index(drop=True)
    print(f"Loaded {len(df)} rows from {CSV_PATH.name}")

    hsc_pool = _load_hsc_pool_indices()
    rng = np.random.default_rng(seed)

    legacy_inputs = []   # preprocessed (4, 48, 48)
    legacy_raws = []     # raw (4, 96, 96) for asinh visualization
    euclid_rgbs = []     # pre-rendered RGB PNG (H, W, 3) uint8 from lens.zip
    sameins = []         # (MAX_NEIGHBORS, 4, 48, 48)
    masks = []
    metadata = []

    with zipfile.ZipFile(LENS_ZIP) as zf, h5py.File(NEIGHBOURS_H5, "r") as nf:
        all_hsc_images = nf["images_hsc"]

        for _, row in df.iterrows():
            id_str = str(row["id_str"])
            raw_legacy = _load_legacy_fits(id_str)
            if raw_legacy is None:
                print(f"  skip {id_str}: missing Legacy FITS")
                continue
            euclid_rgb = _load_euclid_rgb_from_zip(zf, id_str)
            if euclid_rgb is None:
                print(f"  skip {id_str}: missing Euclid rgb_0.png in zip")
                continue

            legacy_input = preprocess_raw_image(
                raw_legacy, survey="legacy", crop_size=CROP_SIZE
            )

            # Random HSC neighbors from the pool
            slot_ids = rng.choice(hsc_pool, size=MAX_NEIGHBORS, replace=False)
            sort_order = np.argsort(slot_ids)
            sorted_ids = np.array(slot_ids, dtype=np.int64)[sort_order]
            loaded = all_hsc_images[sorted_ids]  # (K, 5, 160, 160)
            unsort_order = np.argsort(sort_order)
            loaded = loaded[unsort_order]
            neigh_tensors = [
                preprocess_raw_image(loaded[k], survey="hsc", crop_size=CROP_SIZE)[:4]
                for k in range(MAX_NEIGHBORS)
            ]

            legacy_inputs.append(legacy_input)
            legacy_raws.append(torch.from_numpy(raw_legacy))
            euclid_rgbs.append(euclid_rgb)
            sameins.append(torch.stack(neigh_tensors))
            masks.append(torch.ones(MAX_NEIGHBORS, dtype=torch.bool))
            metadata.append({
                "id_str": id_str,
                "ra": float(row.get("right_ascension", float("nan"))),
                "dec": float(row.get("declination", float("nan"))),
                "grade": str(row.get("grade", "?")),
                "expert_score": float(row.get("expert_score", float("nan"))),
                "theta_E": float(row.get("einstein_radius_eff_median", float("nan"))),
                "field": str(row.get("euclid_field", "")) or "outside-EDF",
            })

            if num_lenses is not None and len(metadata) >= num_lenses:
                break

    print(f"Loaded {len(metadata)} lenses with both Legacy and Euclid available")
    return (
        torch.stack(legacy_inputs),
        torch.stack(legacy_raws),
        euclid_rgbs,
        torch.stack(sameins),
        torch.stack(masks),
        metadata,
    )


# --- visualization ---------------------------------------------------------

def _normalize_for_vis(img_chw: torch.Tensor) -> np.ndarray:
    img = img_chw.detach().cpu().clone().float()
    img = img - img.amin()
    if img.amax() > 0:
        img = img / img.amax()
    return img.permute(1, 2, 0).numpy()


def _row_scale_rgb(x_chw: torch.Tensor, vmin: torch.Tensor, vmax: torch.Tensor) -> np.ndarray:
    """Same per-channel scaling as the training-time row-scaled grid."""
    x = x_chw[:3].detach().cpu().float()
    vmin_t = torch.as_tensor(vmin, dtype=x.dtype).view(3, 1, 1)
    vmax_t = torch.as_tensor(vmax, dtype=x.dtype).view(3, 1, 1)
    y = (x - vmin_t) / (vmax_t - vmin_t + 1e-8)
    return y.clamp(0, 1).permute(1, 2, 0).numpy()


def _asinh_stretch(img: np.ndarray, q: float = 99.5, a: float = 0.1) -> np.ndarray:
    img = np.nan_to_num(np.asarray(img, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    vmax = np.percentile(img, q)
    if vmax <= 0:
        vmax = max(float(img.max()), 1e-6)
    x = np.clip(img / vmax, 0.0, None)
    x = np.arcsinh(x / a) / np.arcsinh(1.0 / a)
    return np.clip(x, 0.0, 1.0)


def _legacy_rgb_asinh(raw_legacy: np.ndarray) -> np.ndarray:
    """raw_legacy: (4, H, W) float32 in g, r, i, z order. Returns (H, W, 3) RGB."""
    g, r, _, z = raw_legacy[0], raw_legacy[1], raw_legacy[2], raw_legacy[3]
    return np.stack([_asinh_stretch(z), _asinh_stretch(r), _asinh_stretch(g)], axis=-1)


def _center_crop_euclid(rgb: np.ndarray, fov_arcsec: float) -> np.ndarray:
    """Center-crop a (H, W, 3) Euclid PNG to fov_arcsec on a side.

    Carol's rgb_0.png is rendered at the Euclid VIS pixel scale (0.1″/px),
    300×300 → 30″ FoV. We crop to the requested arcsec FoV centered on the lens.
    """
    H, W = rgb.shape[:2]
    px = int(round(fov_arcsec / EUCLID_VIS_PIX))
    px = max(8, min(px, H, W))
    y0 = (H - px) // 2
    x0 = (W - px) // 2
    return rgb[y0:y0 + px, x0:x0 + px]


def _add_scale_bar(ax, fov_arcsec: float, bar_arcsec: float, color: str = "white"):
    """Draw a scale bar in the lower-right of an axes showing image of side fov_arcsec."""
    bar_frac = bar_arcsec / fov_arcsec
    pad = 0.05
    x0 = 1.0 - pad - bar_frac
    y0 = pad
    rect = Rectangle((x0, y0), bar_frac, 0.02,
                     transform=ax.transAxes, facecolor=color,
                     edgecolor="black", linewidth=0.5)
    ax.add_patch(rect)
    ax.text(x0 + bar_frac / 2, y0 + 0.04, f"{bar_arcsec:g}″",
            transform=ax.transAxes, color=color, ha="center", va="bottom",
            fontsize=7,
            path_effects=None)


def save_plot(
    legacy_inputs, legacy_raws, euclid_rgbs, hsc_panels_chw, metadata,
    out_path: Path, mode: str = "normalized",
    euclid_fov: float = HSC_FOV,
):
    """Render figure: rows = lenses, cols = [Legacy | Predicted HSC | Euclid].

    `hsc_panels_chw` is a (N, C, H, W) tensor of the HSC images to show
    (typically the mean across model draws).
    `euclid_fov` is the Euclid FoV in arcseconds (center-cropped from the
    native 30″ rgb_0.png).
    """
    n = len(metadata)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n), squeeze=False)

    col_titles = [
        f"Legacy (input, ~{LEGACY_INPUT_FOV:.1f}″)",
        f"Predicted HSC (~{HSC_FOV:.1f}″)",
        f"Euclid VIS+NIR (~{euclid_fov:.1f}″)",
    ]
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=10)

    for i in range(n):
        meta = metadata[i]

        if mode == "asinh":
            legacy_panel = _legacy_rgb_asinh(legacy_raws[i].numpy())
            legacy_fov = 96 * LEGACY_NATIVE_PIX
            hsc_panel = _normalize_for_vis(hsc_panels_chw[i][:3])
        elif mode == "row_scaled":
            # Use the chosen HSC sample's per-channel min/max as vmin/vmax
            # (matches the training-time `val/sample_grid_row_scaled` style,
            # but with the model output as the reference instead of the GT target).
            ref = hsc_panels_chw[i][:3]
            vmin = ref.amin(dim=(1, 2))
            vmax = ref.amax(dim=(1, 2))
            legacy_panel = _row_scale_rgb(legacy_inputs[i][:3], vmin, vmax)
            hsc_panel = _row_scale_rgb(ref, vmin, vmax)
            legacy_fov = LEGACY_INPUT_FOV
        else:
            legacy_panel = _normalize_for_vis(legacy_inputs[i][:3])
            hsc_panel = _normalize_for_vis(hsc_panels_chw[i][:3])
            legacy_fov = LEGACY_INPUT_FOV

        # Real Euclid: pre-rendered RGB PNG from inside lens.zip, center-cropped
        # to euclid_fov arcsec. Imshow with origin='upper' to match the source.
        euclid_panel = _center_crop_euclid(euclid_rgbs[i], euclid_fov)

        axes[i, 0].imshow(legacy_panel, origin="lower")
        axes[i, 0].set_xticks([]); axes[i, 0].set_yticks([])
        axes[i, 1].imshow(hsc_panel, origin="lower")
        axes[i, 1].set_xticks([]); axes[i, 1].set_yticks([])
        axes[i, 2].imshow(euclid_panel, origin="upper")
        axes[i, 2].set_xticks([]); axes[i, 2].set_yticks([])

        euclid_bar = 1.0 if euclid_fov <= 10 else (2.0 if euclid_fov <= 20 else 5.0)
        _add_scale_bar(axes[i, 0], legacy_fov, 2.0)
        _add_scale_bar(axes[i, 1], HSC_FOV, 2.0)
        _add_scale_bar(axes[i, 2], euclid_fov, euclid_bar)

        # Row label on the leftmost panel
        theta_e = meta["theta_E"]
        theta_str = f"{theta_e:.2f}″" if not np.isnan(theta_e) else "--"
        axes[i, 0].set_ylabel(
            f"{meta['id_str'][:18]}...\nGrade {meta['grade']}  "
            r"$\theta_E$=" + theta_str,
            fontsize=8, rotation=0, labelpad=60, va="center", ha="right",
        )

    suptitle_mode = {
        "asinh": "asinh stretch on raw",
        "row_scaled": "row-scaled to predicted-HSC vmin/vmax",
    }.get(mode, "per-image min-max norm")
    fig.suptitle(
        f"Strong-lens follow-up scenario — {n} Carol lenses ({suptitle_mode})\n"
        "Legacy detection → model-predicted HSC → real Euclid follow-up",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path.resolve()}")
    return out_path


def save_posterior_plot(
    legacy_inputs, legacy_raws, euclid_rgbs, all_samples, metadata,
    out_path: Path, num_samples: int = 4,
    euclid_fov: float = HSC_FOV,
):
    """6-column figure: Legacy | Sample 1..N | Euclid (row-scaled).

    Each row uses per-channel vmin/vmax taken from the across-sample mean for
    that lens, so all sample panels in the row share the same color scale.
    """
    n = len(metadata)
    ncols = 2 + num_samples
    fig, axes = plt.subplots(n, ncols, figsize=(2.4 * ncols, 2.4 * n), squeeze=False)

    col_titles = (
        [f"Legacy (~{LEGACY_INPUT_FOV:.1f}″)"]
        + [f"Sample {j + 1}" for j in range(num_samples)]
        + [f"Euclid (~{euclid_fov:.1f}″)"]
    )
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=10)

    for i in range(n):
        meta = metadata[i]
        samples = all_samples[i][:num_samples, :3]      # (num_samples, 3, H, W)
        ref = samples.mean(dim=0)                       # (3, H, W)
        vmin = ref.amin(dim=(1, 2))
        vmax = ref.amax(dim=(1, 2))

        legacy_panel = _row_scale_rgb(legacy_inputs[i][:3], vmin, vmax)
        axes[i, 0].imshow(legacy_panel, origin="lower")
        axes[i, 0].set_xticks([]); axes[i, 0].set_yticks([])
        _add_scale_bar(axes[i, 0], LEGACY_INPUT_FOV, 2.0)

        for j in range(num_samples):
            panel = _row_scale_rgb(samples[j], vmin, vmax)
            ax = axes[i, 1 + j]
            ax.imshow(panel, origin="lower")
            ax.set_xticks([]); ax.set_yticks([])
            _add_scale_bar(ax, HSC_FOV, 2.0)

        euclid_panel = _center_crop_euclid(euclid_rgbs[i], euclid_fov)
        ax_e = axes[i, -1]
        ax_e.imshow(euclid_panel, origin="upper")
        ax_e.set_xticks([]); ax_e.set_yticks([])
        euclid_bar = 1.0 if euclid_fov <= 10 else (2.0 if euclid_fov <= 20 else 5.0)
        _add_scale_bar(ax_e, euclid_fov, euclid_bar)

        theta_e = meta["theta_E"]
        theta_str = f"{theta_e:.2f}″" if not np.isnan(theta_e) else "--"
        axes[i, 0].set_ylabel(
            f"{meta['id_str'][:18]}...\nGrade {meta['grade']}  "
            r"$\theta_E$=" + theta_str,
            fontsize=8, rotation=0, labelpad=60, va="center", ha="right",
        )

    fig.suptitle(
        f"Posterior samples — {n} Carol lenses (row-scaled to per-lens sample mean)\n"
        "Legacy detection → 4 model draws → real Euclid follow-up",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path.resolve()}")
    return out_path


# --- inference -------------------------------------------------------------

def generate_samples(
    model, legacy_inputs, sameins, masks, device, num_steps: int,
):
    n = legacy_inputs.shape[0]
    all_samples = []
    for i in range(n):
        samegal = legacy_inputs[i:i + 1].to(device).repeat(NUM_SAMPLES_PER_COND, 1, 1, 1)
        si = sameins[i:i + 1].to(device).repeat(NUM_SAMPLES_PER_COND, 1, 1, 1, 1)
        m = masks[i:i + 1].to(device).repeat(NUM_SAMPLES_PER_COND, 1)
        with torch.no_grad():
            samples = model.sample(samegal, si, masks=m, num_steps=num_steps)
        all_samples.append(samples.cpu())
        print(f"  sampled lens {i + 1}/{n}")
    return all_samples


# --- entry -----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--num-lenses", type=int, default=None,
                        help="Limit to N lenses (default: all 20)")
    parser.add_argument("--out-dir",
                        default="outputs/carol_legacy_to_euclid")
    parser.add_argument("--steps", type=int, default=NUM_INTEGRATION_STEPS)
    parser.add_argument("--replot", action="store_true")
    parser.add_argument("--euclid-fov", type=float, default=HSC_FOV,
                        help="Center-crop the Euclid panel to this FoV in arcsec "
                             f"(default: match predicted-HSC FoV ≈ {HSC_FOV:.1f}″; "
                             f"native rgb_0.png FoV is ~{EUCLID_FOV:.1f}″)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "samples_cache.pt"

    if args.replot:
        print(f"Loading cache from {cache_path}")
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        legacy_inputs = cache["legacy_inputs"]
        legacy_raws = cache["legacy_raws"]
        euclid_rgbs = cache.get("euclid_rgbs")
        if euclid_rgbs is None:
            # Older cache: re-fetch RGB PNGs from the zip using metadata id_str.
            print("  cache predates rgb_0.png loader — re-reading Euclid PNGs from zip")
            metadata = cache["metadata"]
            euclid_rgbs = []
            with zipfile.ZipFile(LENS_ZIP) as zf:
                for m in metadata:
                    rgb = _load_euclid_rgb_from_zip(zf, m["id_str"])
                    euclid_rgbs.append(rgb)
        all_samples = cache["all_samples"]
        metadata = cache["metadata"]
    else:
        device = get_device()
        print(f"Device: {device}")

        legacy_inputs, legacy_raws, euclid_rgbs, sameins, masks, metadata = load_lenses(
            args.num_lenses,
        )
        if len(metadata) == 0:
            sys.exit("No lenses with both Legacy and Euclid found.")

        print(f"\nLoading checkpoint {args.checkpoint}")
        model = HierarchicalGlobalInstrumentFlowMatchingModule.load_from_checkpoint(
            args.checkpoint, map_location=device,
        )
        model.to(device).eval()

        print(f"\nGenerating samples (steps={args.steps})...")
        all_samples = generate_samples(
            model, legacy_inputs, sameins, masks, device, num_steps=args.steps,
        )

        torch.save({
            "legacy_inputs": legacy_inputs,
            "legacy_raws": legacy_raws,
            "euclid_rgbs": euclid_rgbs,
            "all_samples": all_samples,
            "metadata": metadata,
        }, cache_path)
        print(f"Saved cache to {cache_path.resolve()}")

    # Mean across the NUM_SAMPLES_PER_COND draws per lens.
    mean_samples = torch.stack([s.mean(dim=0) for s in all_samples])

    save_plot(legacy_inputs, legacy_raws, euclid_rgbs, mean_samples, metadata,
              out_dir / "lens_carol_legacy_to_euclid_normalized.png",
              mode="normalized", euclid_fov=args.euclid_fov)
    save_plot(legacy_inputs, legacy_raws, euclid_rgbs, mean_samples, metadata,
              out_dir / "lens_carol_legacy_to_euclid_asinh.png",
              mode="asinh", euclid_fov=args.euclid_fov)
    save_plot(legacy_inputs, legacy_raws, euclid_rgbs, mean_samples, metadata,
              out_dir / "lens_carol_legacy_to_euclid_row_scaled.png",
              mode="row_scaled", euclid_fov=args.euclid_fov)
    save_posterior_plot(legacy_inputs, legacy_raws, euclid_rgbs, all_samples, metadata,
                        out_dir / "lens_carol_legacy_to_euclid_posterior.png",
                        num_samples=4, euclid_fov=args.euclid_fov)


if __name__ == "__main__":
    main()
