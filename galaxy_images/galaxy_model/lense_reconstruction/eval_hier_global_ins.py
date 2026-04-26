"""
Evaluate HierarchicalGlobalInstrumentFlowMatchingModule on gravitational lens candidates.

Loads lenses from lens_reconstruction_dataset.h5, preprocesses them the same way
as NeighborsDataset (crop, normalize, zoom for legacy), then generates samples and
saves validation-style plots (per-image normalized + row-scaled from target).

Usage:
    # Default: top-5 neighbors (ranks 1-5)
    python eval_hier_global_ins.py --checkpoint <path> --num-lenses 8 --out-dir <dir>

    # Offset: ranks 6-10 (random fallback for missing slots, marked with red border)
    python eval_hier_global_ins.py --neighbor-start 5 --out-dir <dir>

    # Replot from cache without re-running inference:
    python eval_hier_global_ins.py --replot --out-dir <dir>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

# Prefer hipBLAS over hipBLASLt — avoids HIPBLAS_STATUS_INVALID_VALUE on AMD GPUs
torch.backends.cuda.preferred_blas_library("hipblas")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Path setup
_this = Path(__file__).resolve()
_root = _this.parents[4]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from galaxy_images.galaxy_model.neighbors import preprocess_raw_image
from galaxy_images.galaxy_model.hierarchical_attention.double_train_fm_neighbors_hier_global_ins import (
    HierarchicalGlobalInstrumentFlowMatchingModule,
)

LENS_H5 = _this.parent / "lens_reconstruction_dataset.h5"
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
RED_BORDER_PX = 4


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_hsc_pool_indices() -> np.ndarray:
    """Return all h5 indices where source_type == 0 (HSC-survey entries)."""
    print("Loading HSC pool indices (source_type==0)...")
    with h5py.File(NEIGHBOURS_H5, "r") as f:
        source_type = f["source_type"][:]
    pool = np.where(source_type == 0)[0]
    print(f"  HSC pool size: {len(pool):,}")
    return pool


def load_lenses(num_lenses: int, neighbor_start: int = 0, seed: int = 42):
    """Load and preprocess lens images + same-instrument neighbors.

    neighbor_start: first rank to use (0-indexed). E.g. 0 → ranks 1-5, 5 → ranks 6-10.
    For each of the MAX_NEIGHBORS slots, if the neighbor at rank (neighbor_start + k) is
    unavailable (index == -1), a random HSC image (source_type==0) is substituted and
    flagged in random_flags.

    Returns:
        targets:      (N, 4, 48, 48)
        samegals:     (N, 4, 48, 48)
        sameins:      (N, MAX_NEIGHBORS, 4, 48, 48)
        masks:        (N, MAX_NEIGHBORS) bool  — True for all slots (real or random)
        random_flags: (N, MAX_NEIGHBORS) bool  — True where a random image was substituted
    """
    hsc_pool = _load_hsc_pool_indices()
    rng = np.random.default_rng(seed)

    with h5py.File(LENS_H5, "r") as lf:
        n_total = lf["images_hsc"].shape[0]
        n = min(num_lenses, n_total)
        print(f"Loading {n}/{n_total} lenses from {LENS_H5}")
        raw_hsc = lf["images_hsc"][:n]           # (n, 5, 160, 160)
        raw_legacy = lf["images_legacy"][:n]      # (n, 4, 160, 160)
        neighbor_idx = lf["neighbor_idx_hsc"][:n] # (n, 100)

    targets_list, samegals_list, sameins_list, masks_list, random_flags_list = [], [], [], [], []

    with h5py.File(NEIGHBOURS_H5, "r") as nf:
        all_hsc_images = nf["images_hsc"]

        for i in range(n):
            target = preprocess_raw_image(raw_hsc[i], survey="hsc", crop_size=CROP_SIZE)[:4]
            samegal = preprocess_raw_image(raw_legacy[i], survey="legacy", crop_size=CROP_SIZE)

            # Collect neighbor h5 indices for the requested rank window
            all_neigh_ids = neighbor_idx[i]           # up to 100 entries, -1 = missing
            valid_ids = all_neigh_ids[all_neigh_ids != -1]

            slot_ids = []     # h5 index to load for each slot
            is_random = []    # whether slot was filled with a random image

            for k in range(MAX_NEIGHBORS):
                rank = neighbor_start + k
                if rank < len(valid_ids):
                    slot_ids.append(valid_ids[rank])
                    is_random.append(False)
                else:
                    # Fall back: random HSC image from source_type==0 pool
                    slot_ids.append(int(rng.choice(hsc_pool)))
                    is_random.append(True)

            # Batch-load in sorted order then re-order
            sort_order = np.argsort(slot_ids)
            sorted_h5_ids = np.array(slot_ids, dtype=np.int64)[sort_order]
            loaded_imgs = all_hsc_images[sorted_h5_ids]   # (MAX_NEIGHBORS, 5, 160, 160)

            # Unsort
            unsort_order = np.argsort(sort_order)
            loaded_imgs = loaded_imgs[unsort_order]

            neigh_tensors = [
                preprocess_raw_image(loaded_imgs[k], survey="hsc", crop_size=CROP_SIZE)[:4]
                for k in range(MAX_NEIGHBORS)
            ]

            padded = torch.stack(neigh_tensors)           # (MAX_NEIGHBORS, 4, 48, 48)
            mask = torch.ones(MAX_NEIGHBORS, dtype=torch.bool)  # all slots filled
            random_flag = torch.tensor(is_random, dtype=torch.bool)

            targets_list.append(target)
            samegals_list.append(samegal)
            sameins_list.append(padded)
            masks_list.append(mask)
            random_flags_list.append(random_flag)

            if (i + 1) % 10 == 0 or i == n - 1:
                n_rand = sum(is_random)
                print(f"  Preprocessed {i+1}/{n} lenses  ({n_rand} random slots)")

    return (
        torch.stack(targets_list),       # (n, 4, 48, 48)
        torch.stack(samegals_list),      # (n, 4, 48, 48)
        torch.stack(sameins_list),       # (n, MAX_NEIGHBORS, 4, 48, 48)
        torch.stack(masks_list),         # (n, MAX_NEIGHBORS)
        torch.stack(random_flags_list),  # (n, MAX_NEIGHBORS)
    )


# ── Visualization helpers ────────────────────────────────────────────────────

def _normalize_for_vis(img: torch.Tensor) -> np.ndarray:
    img = img.clone()
    img = img - img.min()
    if img.max() > 0:
        img = img / img.max()
    return img.cpu().permute(1, 2, 0).numpy()


def _row_scale_rgb(x_chw: torch.Tensor, vmin: torch.Tensor, vmax: torch.Tensor) -> np.ndarray:
    x = x_chw[:3]
    vmin_t = torch.as_tensor(vmin, device=x.device, dtype=x.dtype).view(3, 1, 1)
    vmax_t = torch.as_tensor(vmax, device=x.device, dtype=x.dtype).view(3, 1, 1)
    y = (x - vmin_t) / (vmax_t - vmin_t + 1e-8)
    return y.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()


def _red_border(tile: np.ndarray, px: int = RED_BORDER_PX) -> np.ndarray:
    tile = tile.copy()
    tile[:px, :] = [1.0, 0.0, 0.0]
    tile[-px:, :] = [1.0, 0.0, 0.0]
    tile[:, :px] = [1.0, 0.0, 0.0]
    tile[:, -px:] = [1.0, 0.0, 0.0]
    return tile


def _make_2x2_mosaic(
    si_neighbors: torch.Tensor,
    mask: torch.Tensor,
    random_flags: torch.Tensor,
    vis_fn,
) -> np.ndarray:
    """Stitch first 4 neighbors into a 2×2 grid.
    Black tile if mask[k] is False; red border if random_flags[k] is True.
    """
    H, W = si_neighbors.shape[-2], si_neighbors.shape[-1]
    tiles = []
    for k in range(4):
        if k < si_neighbors.shape[0] and mask[k]:
            tile = vis_fn(si_neighbors[k, :3])
            if random_flags[k]:
                tile = _red_border(tile)
        else:
            tile = np.zeros((H, W, 3), dtype=np.float32)
        tiles.append(tile)
    top = np.concatenate([tiles[0], tiles[1]], axis=1)
    bot = np.concatenate([tiles[2], tiles[3]], axis=1)
    return np.concatenate([top, bot], axis=0)


# ── Plot rendering ───────────────────────────────────────────────────────────

def save_plots(
    targets, samegals, sameins, masks, random_flags, all_samples,
    out_dir: Path, n: int, neighbor_start: int = 0,
):
    """Render and save both figure variants from (cached) tensors."""
    rank_lo = neighbor_start + 1
    rank_hi = neighbor_start + MAX_NEIGHBORS
    neigh_label = f"SameIns (ranks {rank_lo}-{rank_hi})"

    num_cols = 3 + NUM_SAMPLES_PER_COND + 1
    col_titles = (
        ["SameGal", neigh_label, "Target"]
        + [f"Sample {j+1}" for j in range(NUM_SAMPLES_PER_COND)]
        + ["Mean"]
    )

    fig_orig, axes_orig = plt.subplots(n, num_cols, figsize=(2 * num_cols, 2 * n), squeeze=False)
    fig_row, axes_row = plt.subplots(n, num_cols, figsize=(2 * num_cols, 2 * n), squeeze=False)
    for j, title in enumerate(col_titles):
        axes_orig[0, j].set_title(title, fontsize=10)
        axes_row[0, j].set_title(title, fontsize=10)

    for i in range(n):
        target = targets[i]
        samegal = samegals[i]
        si = sameins[i]
        mask_i = masks[i]
        rf_i = random_flags[i]
        samples = all_samples[i]
        mean_sample = samples.mean(dim=0)

        target_chw = target[:3]
        vmin = target_chw.amin(dim=(1, 2))
        vmax = target_chw.amax(dim=(1, 2))

        mosaic_orig = _make_2x2_mosaic(si, mask_i, rf_i, lambda x: _normalize_for_vis(x))
        mosaic_row  = _make_2x2_mosaic(si, mask_i, rf_i, lambda x: _row_scale_rgb(x, vmin, vmax))

        for axes, mosaic, vis in [
            (axes_orig, mosaic_orig, lambda x: _normalize_for_vis(x)),
            (axes_row,  mosaic_row,  lambda x: _row_scale_rgb(x, vmin, vmax)),
        ]:
            axes[i, 0].imshow(vis(samegal[:3]))
            axes[i, 0].axis("off")
            axes[i, 1].imshow(mosaic)
            axes[i, 1].axis("off")
            axes[i, 2].imshow(vis(target[:3]))
            axes[i, 2].axis("off")
            for j in range(NUM_SAMPLES_PER_COND):
                axes[i, 3 + j].imshow(vis(samples[j, :3]))
                axes[i, 3 + j].axis("off")
            axes[i, -1].imshow(vis(mean_sample[:3]))
            axes[i, -1].axis("off")

    saved = []
    rand_note = "  (red border = random HSC fallback)" if random_flags.any() else ""
    for fig, name in [(fig_orig, "lens_eval_normalized.png"), (fig_row, "lens_eval_row_scaled.png")]:
        fig.suptitle(
            f"Lens evaluation — hier-global-ins (step=201k) — {n} lenses — "
            f"neighbor ranks {rank_lo}-{rank_hi}{rand_note}\n"
            "Cols: SameGal | SameIns (2×2) | Target | Sample 1-5 | Mean",
            fontsize=10,
        )
        fig.tight_layout()
        out_path = out_dir / name
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")
        saved.append(out_path)
    return saved


# ── Inference ────────────────────────────────────────────────────────────────

def generate_and_plot(
    model: HierarchicalGlobalInstrumentFlowMatchingModule,
    targets, samegals, sameins, masks, random_flags,
    out_dir: Path, device: torch.device,
    num_steps: int = NUM_INTEGRATION_STEPS,
    neighbor_start: int = 0,
):
    n = targets.shape[0]
    all_samples = []

    for i in range(n):
        samegal_i = samegals[i : i + 1].to(device)
        si_i      = sameins[i : i + 1].to(device)
        mask_i    = masks[i : i + 1].to(device)

        samegal_rep = samegal_i.repeat(NUM_SAMPLES_PER_COND, 1, 1, 1)
        si_rep      = si_i.repeat(NUM_SAMPLES_PER_COND, 1, 1, 1, 1)
        mask_rep    = mask_i.repeat(NUM_SAMPLES_PER_COND, 1)

        with torch.no_grad():
            samples = model.sample(
                samegal_rep, si_rep, masks=mask_rep, num_steps=num_steps,
            )  # (NUM_SAMPLES_PER_COND, 4, 48, 48)

        all_samples.append(samples.cpu())
        print(f"  Generated samples for lens {i+1}/{n}")

    cache_path = out_dir / "samples_cache.pt"
    torch.save(
        {
            "targets": targets, "samegals": samegals,
            "sameins": sameins, "masks": masks,
            "random_flags": random_flags,
            "all_samples": all_samples,
            "neighbor_start": neighbor_start,
        },
        cache_path,
    )
    print(f"Saved sample cache to {cache_path}")

    return save_plots(
        targets, samegals, sameins, masks, random_flags,
        all_samples, out_dir, n, neighbor_start,
    )


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate hier-global-ins on gravitational lenses")
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--num-lenses", type=int, default=8)
    parser.add_argument("--out-dir", default="lense_reconstruction/outputs/lens_eval_hier_global_ins")
    parser.add_argument("--steps", type=int, default=NUM_INTEGRATION_STEPS)
    parser.add_argument("--neighbor-start", type=int, default=0,
                        help="First neighbor rank to use (0-indexed). 0 = ranks 1-5, 5 = ranks 6-10.")
    parser.add_argument("--replot", action="store_true",
                        help="Skip inference, replot from cached samples_cache.pt")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    if args.replot:
        cache_path = out_dir / "samples_cache.pt"
        print(f"Loading cached samples from {cache_path}")
        cache = torch.load(cache_path, map_location="cpu")
        saved_paths = save_plots(
            cache["targets"], cache["samegals"], cache["sameins"],
            cache["masks"], cache.get("random_flags", torch.zeros(len(cache["all_samples"]), MAX_NEIGHBORS, dtype=torch.bool)),
            cache["all_samples"], out_dir,
            len(cache["all_samples"]),
            cache.get("neighbor_start", 0),
        )
    else:
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Num lenses: {args.num_lenses}")
        print(f"Integration steps: {args.steps}")
        print(f"Neighbor ranks: {args.neighbor_start + 1} – {args.neighbor_start + MAX_NEIGHBORS}")

        print("\nLoading and preprocessing lenses...")
        targets, samegals, sameins, masks, random_flags = load_lenses(
            args.num_lenses, neighbor_start=args.neighbor_start,
        )
        print(f"  targets: {targets.shape}, samegals: {samegals.shape}")
        print(f"  sameins: {sameins.shape}, masks: {masks.shape}")
        print(f"  random slots total: {random_flags.sum().item()}")

        print("\nLoading model checkpoint...")
        model = HierarchicalGlobalInstrumentFlowMatchingModule.load_from_checkpoint(
            args.checkpoint, map_location=device,
        )
        model.to(device)
        model.eval()
        print("  Model loaded.")

        print("\nGenerating samples and saving plots...")
        saved_paths = generate_and_plot(
            model, targets, samegals, sameins, masks, random_flags,
            out_dir, device, num_steps=args.steps, neighbor_start=args.neighbor_start,
        )

    print("\nDone. Saved files:")
    for p in saved_paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
