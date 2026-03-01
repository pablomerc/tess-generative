"""
Counterfactual visualization for "weird" HSC indices from the neighbors model.

- Loads the double encoder flow matching model (trained with neighbours_train.py / double_train_fm_neighbors.py).
- Loads NeighborsPrecomputedDataset and takes items at the given indices (target=HSC, samegal=legacy, HSC neighbors).
- Plots each weird example with the same layout as during training (SameGal | SameIns 1st | Target | Samples | Mean) and saves.
- Plots neighbors of each target in a separate figure.
- Counterfactual: one legacy image (from idx+1), two HSC instrument pairs (idx, idx+2), two reconstructions; one plot.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

# Project root: galaxy_model
_SCRIPT_DIR = Path(__file__).resolve().parent
_GALAXY_MODEL_DIR = _SCRIPT_DIR.parent
if str(_GALAXY_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_GALAXY_MODEL_DIR))

from neighbors import NeighborsPrecomputedDataset, simple_collate
from double_train_fm_neighbors import ConditionalFlowMatchingModule

# Indices into the neighbors dataset that correspond to "weird" HSC examples (from find_weird_examples.ipynb)
INTERESTING_INDICES_HSC = [
    1670,
    1784,
    5001,
    5029,
    5457,
    6270,
    # 29319,
    # 96762,
    # 146356,
    # 149370,
    # 149375,
    # 149502,
]

# Default paths (same as neighbours_train.py)
PRECOMPUTED_H5 = "/data/vision/billf/scratch/pablomer/data/neighbor_batches/neighbours_vds.h5"
DEFAULT_CHECKPOINT = "/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy-flow-matching-neighbours/g2g9kvr4/checkpoints/latest-step=step=75000.ckpt"
OUTPUT_DIR = _SCRIPT_DIR / "counterfactual_outputs"
NUM_SAMPLES_PER_ROW = 3  # number of stochastic samples in training-style grid
MAX_NEIGHBORS_SHOW = 5   # max neighbor columns in neighbor-only plot


def _normalize_for_vis(img: torch.Tensor) -> torch.Tensor:
    """Normalize image to [0, 1] for visualization (first 3 channels as RGB)."""
    img = img.clone()
    if img.dim() == 4:
        img = img[0]
    x = img[:3]
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return x


def plot_training_style_grid(
    model: ConditionalFlowMatchingModule,
    targets: torch.Tensor,
    samegals: torch.Tensor,
    sameins: torch.Tensor,
    masks: torch.Tensor | None,
    device: torch.device,
    save_path: Path,
    num_samples_per_cond: int = NUM_SAMPLES_PER_ROW,
) -> None:
    """
    Same layout as validation during training: SameGal | SameIns (1st) | Target | Sample1 | ... | Mean.
    Uses row-scaled RGB from target for consistent comparison.
    """
    n = targets.shape[0]
    num_cols = 3 + num_samples_per_cond + 1  # samegal + sameins_first + target + samples + mean

    def _row_scale_rgb(x_chw: torch.Tensor, vmin: torch.Tensor, vmax: torch.Tensor) -> torch.Tensor:
        x = x_chw[:3]
        vmin_t = vmin.to(x.device, x.dtype).view(3, 1, 1)
        vmax_t = vmax.to(x.device, x.dtype).view(3, 1, 1)
        y = (x - vmin_t) / (vmax_t - vmin_t + 1e-8)
        y = y.clamp(0, 1)
        return y.permute(1, 2, 0)

    fig, axes = plt.subplots(n, num_cols, figsize=(2 * num_cols, 2 * n), squeeze=False)
    col_titles = ["SameGal (Legacy)", "SameIns (1st)", "Target (HSC)"] + [f"Sample {j+1}" for j in range(num_samples_per_cond)] + ["Mean"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=10)

    for i in range(n):
        samegal = samegals[i : i + 1].to(device)
        target = targets[i : i + 1].to(device)
        sameins_i = sameins[i : i + 1].to(device)
        sameins_first = sameins_i[:, 0:1]
        samegal_rep = samegal.repeat(num_samples_per_cond, 1, 1, 1)
        sameins_rep = sameins_i.repeat(num_samples_per_cond, 1, 1, 1, 1)
        masks_i = None
        if masks is not None:
            m = masks[i : i + 1].to(device)
            masks_i = m.repeat(num_samples_per_cond, 1)

        with torch.no_grad():
            samples = model.sample(samegal_rep, sameins_rep, masks=masks_i)
        mean_sample = samples.mean(dim=0, keepdim=True)

        target_chw = target[0, :3]
        vmin = target_chw.amin(dim=(1, 2))
        vmax = target_chw.amax(dim=(1, 2))

        # SameGal
        vis = _row_scale_rgb(samegal[0, :3], vmin, vmax).detach().cpu().numpy()
        axes[i, 0].imshow(vis)
        axes[i, 0].axis("off")
        # SameIns first
        vis = _row_scale_rgb(sameins_first[0, 0, :3], vmin, vmax).detach().cpu().numpy()
        axes[i, 1].imshow(vis)
        axes[i, 1].axis("off")
        # Target
        vis = _row_scale_rgb(target[0, :3], vmin, vmax).detach().cpu().numpy()
        axes[i, 2].imshow(vis)
        axes[i, 2].axis("off")
        # Samples
        for j in range(num_samples_per_cond):
            vis = _row_scale_rgb(samples[j, :3], vmin, vmax).detach().cpu().numpy()
            axes[i, 3 + j].imshow(vis)
            axes[i, 3 + j].axis("off")
        # Mean
        vis = _row_scale_rgb(mean_sample[0, :3], vmin, vmax).detach().cpu().numpy()
        axes[i, -1].imshow(vis)
        axes[i, -1].axis("off")

    plt.suptitle("Weird HSC examples: Target=HSC, SameGal=Legacy, SameIns=HSC neighbors", fontsize=10)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved training-style grid to {save_path}")


def plot_neighbors_only(
    targets: torch.Tensor,
    samegals: torch.Tensor,
    sameins: torch.Tensor,
    masks: torch.Tensor,
    save_path: Path,
    max_neighbors_show: int = MAX_NEIGHBORS_SHOW,
) -> None:
    """One row per example: Target | SameGal | Neighbor 1 | ... | Neighbor k."""
    n = targets.shape[0]
    n_cols = 2 + max_neighbors_show

    def _to_rgb(img: torch.Tensor) -> torch.Tensor:
        x = img.detach().cpu().float()
        if x.dim() == 4:
            x = x[0]
        x = x[:3].permute(1, 2, 0).numpy()
        lo, hi = x.min(), x.max()
        if hi > lo:
            x = (x - lo) / (hi - lo)
        return x

    fig, axes = plt.subplots(n, n_cols, figsize=(2 * n_cols, 2 * n), squeeze=False)
    for i in range(n):
        axes[i, 0].imshow(_to_rgb(targets[i]))
        axes[i, 0].set_title("Target (HSC)" if i == 0 else None)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(_to_rgb(samegals[i]))
        axes[i, 1].set_title("SameGal (Legacy)" if i == 0 else None)
        axes[i, 1].axis("off")
        n_valid = int(masks[i].sum().item())
        for j in range(max_neighbors_show):
            ax = axes[i, 2 + j]
            if j < n_valid:
                ax.imshow(_to_rgb(sameins[i, j]))
                ax.set_title(f"Neighbor {j+1}" if i == 0 else None)
            else:
                ax.set_facecolor("0.9")
            ax.axis("off")
    plt.suptitle("Neighbors of each weird HSC target", fontsize=10)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved neighbors grid to {save_path}")


def plot_counterfactual_legacy_two_hsc(
    model: ConditionalFlowMatchingModule,
    legacy_image: torch.Tensor,
    hsc_idx: torch.Tensor,
    hsc_idx2: torch.Tensor,
    sameins_legacy: torch.Tensor,
    masks_legacy: torch.Tensor | None,
    device: torch.device,
    save_path: Path,
) -> None:
    """
    One legacy image, two HSC instrument pairs (from row idx and idx+2), two reconstructions of the legacy.
    We reconstruct the legacy image conditioned on (samegal=HSC from idx, sameins=Legacy neighbors) and
    (samegal=HSC from idx+2, sameins=Legacy neighbors). Layout: Legacy | HSC (idx) | HSC (idx+2) | Recon (cond idx) | Recon (cond idx+2).
    """
    samegal_1 = hsc_idx.unsqueeze(0).to(device)
    samegal_2 = hsc_idx2.unsqueeze(0).to(device)
    sameins_leg = sameins_legacy.unsqueeze(0).to(device)
    m_leg = masks_legacy.unsqueeze(0).to(device) if masks_legacy is not None else None

    with torch.no_grad():
        recon_1 = model.sample(samegal_1, sameins_leg, masks=m_leg)
        recon_2 = model.sample(samegal_2, sameins_leg, masks=m_leg)

    def _to_rgb(img: torch.Tensor) -> torch.Tensor:
        x = img.detach().cpu().float()
        if x.dim() == 4:
            x = x[0]
        x = x[:3].permute(1, 2, 0).numpy()
        lo, hi = x.min(), x.max()
        if hi > lo:
            x = (x - lo) / (hi - lo)
        return x

    fig, axes = plt.subplots(1, 5, figsize=(10, 2.5))
    titles = ["Legacy (other galaxy)", "HSC pair (idx)", "HSC pair (idx+2)", "Recon (cond idx)", "Recon (cond idx+2)"]
    imgs = [
        legacy_image,
        hsc_idx,
        hsc_idx2,
        recon_1[0],
        recon_2[0],
    ]
    for ax, title, img in zip(axes, titles, imgs):
        ax.imshow(_to_rgb(img))
        ax.set_title(title)
        ax.axis("off")
    plt.suptitle("Counterfactual: one legacy, two HSC conditionings, two reconstructions", fontsize=10)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved counterfactual plot to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Counterfactual plots for weird HSC indices")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="Path to model checkpoint")
    parser.add_argument("--data", type=str, default=PRECOMPUTED_H5, help="Path to neighbours precomputed H5")
    parser.add_argument("--indices", type=int, nargs="+", default=None, help="Dataset indices (default: INTERESTING_INDICES_HSC)")
    parser.add_argument("--out-dir", type=str, default=str(OUTPUT_DIR), help="Output directory for plots")
    parser.add_argument("--no-training-grid", action="store_true", help="Skip training-style grid plot")
    parser.add_argument("--no-neighbors-grid", action="store_true", help="Skip neighbors-only grid plot")
    parser.add_argument("--no-counterfactual", action="store_true", help="Skip counterfactual (legacy + two HSC) plot")
    parser.add_argument("--counterfactual-base-idx", type=int, default=None, help="Base index for counterfactual (legacy from base+1, HSC from base and base+2). Default: first interesting index.")
    args = parser.parse_args()

    indices = args.indices if args.indices is not None else INTERESTING_INDICES_HSC
    out_dir = Path(args.out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"Loading checkpoint {args.checkpoint} ...")
    model = ConditionalFlowMatchingModule.load_from_checkpoint(args.checkpoint, map_location="cpu")
    model.eval()
    torch.set_grad_enabled(False)
    model = model.to(device)

    # Load dataset and take only HSC rows at the given indices
    print(f"Loading dataset {args.data} ...")
    dataset = NeighborsPrecomputedDataset(args.data)
    n_total = len(dataset)

    # Filter to indices that exist and are HSC (target = HSC, samegal = legacy, sameins = HSC neighbors)
    hsc_indices = []
    for idx in indices:
        if idx < 0 or idx >= n_total:
            print(f"  Skip index {idx} (out of range [0, {n_total}))")
            continue
        meta = dataset.meta_survey[idx]
        if meta != "hsc":
            print(f"  Skip index {idx} (anchor_survey={meta}, need 'hsc')")
            continue
        hsc_indices.append(idx)

    if not hsc_indices:
        print("No valid HSC indices; nothing to plot.")
        return

    # Batch of weird examples
    batch = simple_collate([dataset[i] for i in hsc_indices])
    targets, samegals, sameins, masks, metadata = batch
    targets = targets.to(device)
    samegals = samegals.to(device)
    sameins = sameins.to(device)
    masks = masks.to(device) if masks is not None else None

    # 1) Training-style grid (Target=HSC, SameGal=Legacy, SameIns=HSC neighbors)
    if not args.no_training_grid:
        plot_training_style_grid(
            model, targets, samegals, sameins, masks, device,
            save_path=out_dir / "weird_hsc_training_style_grid.png",
            num_samples_per_cond=NUM_SAMPLES_PER_ROW,
        )

    # 2) Neighbors of each target
    if not args.no_neighbors_grid:
        plot_neighbors_only(
            targets.cpu(), samegals.cpu(), sameins.cpu(), masks.cpu() if masks is not None else torch.ones_like(sameins[:, :, 0, 0, 0]),
            save_path=out_dir / "weird_hsc_neighbors_grid.png",
            max_neighbors_show=MAX_NEIGHBORS_SHOW,
        )

    # 3) Counterfactual: one legacy image (from base_idx+1), two HSC instrument pairs (base_idx, base_idx+2), two reconstructions of the legacy
    if not args.no_counterfactual:
        base_idx = args.counterfactual_base_idx if args.counterfactual_base_idx is not None else hsc_indices[0]
        idx_legacy = base_idx + 1
        idx_hsc2 = base_idx + 2
        if idx_legacy >= n_total or idx_hsc2 >= n_total:
            print(f"  Counterfactual: base_idx={base_idx} would need idx+1 and idx+2 in range; skipping.")
        else:
            # Legacy row at idx_legacy: target = legacy, samegal = HSC, sameins = Legacy neighbors
            t_leg, s_leg, si_leg, mask_leg, meta_leg = dataset[idx_legacy]
            if meta_leg["anchor_survey"] != "legacy":
                print(f"  Counterfactual: row idx+1={idx_legacy} has anchor_survey={meta_leg['anchor_survey']}; need Legacy row. Using legacy = samegal of that row.")
                legacy_image = s_leg
                sameins_legacy = si_leg
                masks_legacy = mask_leg
            else:
                legacy_image = t_leg
                sameins_legacy = si_leg
                masks_legacy = mask_leg

            # HSC images from base_idx and idx+2 (target when row is HSC)
            t1, s1, si1, mask1, _ = dataset[base_idx]
            t2, s2, si2, mask2, _ = dataset[idx_hsc2]
            hsc_from_idx = t1 if dataset.meta_survey[base_idx] == "hsc" else s1
            hsc_from_idx2 = t2 if dataset.meta_survey[idx_hsc2] == "hsc" else s2

            plot_counterfactual_legacy_two_hsc(
                model,
                legacy_image=legacy_image,
                hsc_idx=hsc_from_idx,
                hsc_idx2=hsc_from_idx2,
                sameins_legacy=sameins_legacy,
                masks_legacy=masks_legacy,
                device=device,
                save_path=out_dir / "counterfactual_legacy_two_hsc.png",
            )


if __name__ == "__main__":
    main()
