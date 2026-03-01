"""
Re-visualize precomputed neighbors from neighbors_summary.csv using raw images
from neighbours_v2.h5 and the same RGB helper as tests/triplet_images.py.

Unlike search_neighbors.py, this script does NOT recompute kNN in latent space.
Instead, it:
  - reads a row from neighbor_search/query_results/neighbors_summary.csv
  - uses the recorded neighbor indices + sources ('hsc' / 'legacy')
  - loads images from /data/vision/billf/scratch/pablomer/data/neighbours_v2.h5
    directly (full 160x160 flux images, before any preprocessing)
  - plots query + neighbors in a 5-row grid, but with triplet_images-style RGB

Run from galaxy_model/:

  python neighbor_search/neighbors_nicer.py --query-idx 5

You can also override paths with --summary-csv and --neighbors-h5.
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

_here = Path(__file__).resolve().parent
_src = _here.parent  # galaxy_model (contains neighbors.py)
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

# Also add project root so we can import galaxy_images.*
_root_dir = _here.parents[2]  # .../projects/tess-generative
if str(_root_dir) not in sys.path:
    sys.path.insert(0, str(_root_dir))

from galaxy_images.image_preprocessing import CenterCrop  # noqa: E402


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_SUMMARY_CSV = _here / "query_results" / "neighbors_summary.csv"
DEFAULT_NEIGHBORS_HDF5 = Path("/data/vision/billf/scratch/pablomer/data/neighbours_v2.h5")
K_NEIGHBORS = 10  # neighbors_summary.csv was written with K=10
_CROP_SIZE = 48  # center-crop to 64x64 from original 160x160

# Instantiate a single cropper to reuse
_CROPPER = CenterCrop(crop_size=_CROP_SIZE)


# ---------------------------------------------------------------------------
# Visualization helper (copied from tests/triplet_images.tensor_to_rgb)
# ---------------------------------------------------------------------------

def tensor_to_rgb(tensor: torch.Tensor, channels: Sequence[int] = (0, 1, 2), percentile_clip: float = 99.5) -> np.ndarray:
    """
    Convert a (C,H,W) tensor into an RGB image (H,W,3) using percentile clipping.

    This mirrors tests/triplet_images.tensor_to_rgb and operates per-image
    (no shared vmin/vmax across the grid).
    """
    if tensor.ndim != 3:
        raise ValueError(f"Expected tensor with shape (C,H,W), got {tuple(tensor.shape)}")

    # Extract the specified channels (e.g., first 3)
    c_indices = list(channels)
    if max(c_indices) >= tensor.shape[0]:
        raise ValueError(
            f"Requested channels {c_indices} but tensor has only {tensor.shape[0]} channels"
        )
    rgb = tensor[c_indices].cpu().numpy()  # (3, H, W)
    rgb = np.transpose(rgb, (1, 2, 0))     # (H, W, 3)

    # Percentile-based clipping per channel
    for i in range(3):
        p_low = np.percentile(rgb[:, :, i], 100 - percentile_clip)
        p_high = np.percentile(rgb[:, :, i], percentile_clip)
        rgb[:, :, i] = np.clip(rgb[:, :, i], p_low, p_high)

    # Normalize each channel to [0,1]
    for i in range(3):
        ch = rgb[:, :, i]
        ch_min, ch_max = ch.min(), ch.max()
        if ch_max > ch_min:
            rgb[:, :, i] = (ch - ch_min) / (ch_max - ch_min)
        else:
            rgb[:, :, i] = 0.0

    return rgb


# ---------------------------------------------------------------------------
# CSV parsing helpers
# ---------------------------------------------------------------------------

def _parse_semicolon_ints(s: str) -> List[int]:
    if not s:
        return []
    return [int(x) for x in s.split(";") if x.strip() != ""]


def _parse_semicolon_strs(s: str) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(";") if x.strip() != ""]


def load_summary_row(summary_csv: Path, query_idx: int) -> dict:
    """
    Load a single row from neighbors_summary.csv matching query_idx.
    """
    with summary_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["query_idx"]) == query_idx:
                return row
    raise ValueError(f"query_idx={query_idx} not found in {summary_csv}")


def build_neighbor_lists(row: dict) -> Tuple[
    List[Tuple[int, str]],
    List[Tuple[int, str]],
    List[Tuple[int, str]],
    List[Tuple[int, str]],
]:
    """
    From a CSV row, build 4 neighbor lists:
      - hsc_inst_neighbors, hsc_phys_neighbors, leg_inst_neighbors, leg_phys_neighbors
    Each is a list of (dataset_idx, source_str) where source_str is 'hsc' or 'legacy'.
    """
    def _pairs(prefix: str) -> List[Tuple[int, str]]:
        idxs = _parse_semicolon_ints(row[f"{prefix}_indices"])
        srcs = _parse_semicolon_strs(row[f"{prefix}_sources"])
        if len(idxs) != len(srcs):
            raise ValueError(
                f"Length mismatch for {prefix}: {len(idxs)} indices vs {len(srcs)} sources"
            )
        return list(zip(idxs, srcs))

    hsc_inst = _pairs("hsc_inst")
    hsc_phys = _pairs("hsc_phys")
    leg_inst = _pairs("leg_inst")
    leg_phys = _pairs("leg_phys")
    return hsc_inst, hsc_phys, leg_inst, leg_phys


# ---------------------------------------------------------------------------
# Image loading helpers (raw 160x160 from neighbours_v2.h5)
# ---------------------------------------------------------------------------


def _center_crop_tensor(image: torch.Tensor) -> torch.Tensor:
    """
    Apply CenterCrop from image_preprocessing to a single (C,H,W) tensor.
    Crops from 160x160 down to 64x64 (configurable via _CROP_SIZE).
    """
    if image.ndim != 3:
        raise ValueError(f"Expected (C,H,W) tensor, got shape {tuple(image.shape)}")
    # CenterCrop expects [B, C, H, W]
    img_batched = image.unsqueeze(0)
    cropped = _CROPPER(img_batched)
    return cropped.squeeze(0)


def get_indexes_mmu(h5_file: h5py.File) -> np.ndarray:
    """
    Reproduce NeighborsSimpleDataset's filtering:
      indexes_mmu = np.where(source_type == 0)[0]
    so that dataset indices (0..N-1) used in latents/CSV map to raw rows.
    """
    sources = h5_file["source_type"][:]
    return np.where(sources == 0)[0]


def load_query_images_h5(
    h5_file: h5py.File,
    indexes_mmu: np.ndarray,
    query_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return (query_hsc, query_legacy) tensors for dataset index query_idx,
    using full-resolution images from neighbours_v2.h5.
    """
    if query_idx < 0 or query_idx >= len(indexes_mmu):
        raise IndexError(f"query_idx={query_idx} out of range for indexes_mmu (len={len(indexes_mmu)})")
    raw_idx = int(indexes_mmu[query_idx])
    img_hsc = torch.from_numpy(h5_file["images_hsc"][raw_idx]).float()
    img_legacy = torch.from_numpy(h5_file["images_legacy"][raw_idx]).float()

    # Center-crop to 64x64 using the shared cropper
    img_hsc = _center_crop_tensor(img_hsc)
    img_legacy = _center_crop_tensor(img_legacy)
    return img_hsc, img_legacy


def load_images_for_neighbors_h5(
    h5_file: h5py.File,
    indexes_mmu: np.ndarray,
    neighbor_list: List[Tuple[int, str]],
):
    """
    neighbor_list: list of (dataset_idx, source) where source is 'hsc' or 'legacy'.
    Returns (images, sources, indices) lists of same length, using full-resolution
    160x160 images from neighbours_v2.h5.
    """
    if not neighbor_list:
        return [], [], []

    unique_indices = sorted({idx for idx, _ in neighbor_list})
    idx_to_rank = {idx: i for i, idx in enumerate(unique_indices)}

    # Load each unique dataset index once from HDF5
    hsc_list: List[torch.Tensor] = []
    leg_list: List[torch.Tensor] = []
    for dataset_idx in unique_indices:
        if dataset_idx < 0 or dataset_idx >= len(indexes_mmu):
            raise IndexError(
                f"Neighbor dataset_idx={dataset_idx} out of range for indexes_mmu (len={len(indexes_mmu)})"
            )
        raw_idx = int(indexes_mmu[dataset_idx])
        img_hsc = torch.from_numpy(h5_file["images_hsc"][raw_idx]).float()
        img_legacy = torch.from_numpy(h5_file["images_legacy"][raw_idx]).float()

        # Center-crop to 64x64
        img_hsc = _center_crop_tensor(img_hsc)
        img_legacy = _center_crop_tensor(img_legacy)

        hsc_list.append(img_hsc)
        leg_list.append(img_legacy)

    images: List[torch.Tensor] = []
    sources: List[str] = []
    indices: List[int] = []
    for dataset_idx, source in neighbor_list:
        r = idx_to_rank[dataset_idx]
        img = hsc_list[r] if source == "hsc" else leg_list[r]
        images.append(img)
        sources.append(source)
        indices.append(dataset_idx)

    return images, sources, indices


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_query_and_neighbors_triplet_style(
    query_hsc: torch.Tensor,
    query_legacy: torch.Tensor,
    hsc_inst_images: List[torch.Tensor],
    hsc_inst_sources: List[str],
    hsc_inst_indices: List[int],
    hsc_phys_images: List[torch.Tensor],
    hsc_phys_sources: List[str],
    hsc_phys_indices: List[int],
    leg_inst_images: List[torch.Tensor],
    leg_inst_sources: List[str],
    leg_inst_indices: List[int],
    leg_phys_images: List[torch.Tensor],
    leg_phys_sources: List[str],
    leg_phys_indices: List[int],
    query_idx: int,
    out_path: Path,
):
    """
    Layout similar to search_neighbors.plot_query_and_neighbors, but each tile
    is rendered via tensor_to_rgb (triplet_images-style).
    """
    n_cols = 12  # match search_neighbors grid (neighbors at cols 1..10)
    n_rows = 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.3, n_rows * 1.2))
    for ax in axes.flat:
        ax.set_axis_off()

    def show_img(
        ax,
        t: torch.Tensor,
        title: str | None = None,
        title_color: str = "black",
    ):
        """
        Render a single tile using triplet_images-style normalization:
        percentile clipping + per-image, per-channel min-max.
        """
        try:
            rgb = tensor_to_rgb(t)
            ax.imshow(rgb)
        except Exception:
            # Fallback to grayscale with per-image normalization
            x = t.detach().cpu().float().numpy()
            if x.ndim == 3:
                x = x[0]
            x = (x - x.min()) / (x.max() - x.min() + 1e-8)
            ax.imshow(x, cmap="gray")
        if title:
            ax.set_title(title, fontsize=8, color=title_color)
        ax.set_axis_off()

    def src_label(s: str) -> str:
        return "(HSC)" if s == "hsc" else "(Leg)"

    # Row 0, center: Query HSC and Query Legacy (cols 2, 3)
    show_img(axes[0, 2], query_hsc, "Query HSC")
    show_img(axes[0, 3], query_legacy, "Query Legacy")

    # Row 1: HSC query → instrument kNN
    for j, img in enumerate(hsc_inst_images):
        if j >= K_NEIGHBORS or 1 + j >= n_cols:
            break
        is_counterpart = hsc_inst_indices[j] == query_idx and hsc_inst_sources[j] == "legacy"
        is_cross_survey = hsc_inst_sources[j] == "legacy"
        if is_counterpart:
            color = "red"
        elif is_cross_survey:
            color = "gold"
        else:
            color = "black"
        ax = axes[1, 1 + j]
        show_img(
            ax,
            img,
            f"HSC inst kNN {j+1} {src_label(hsc_inst_sources[j])}",
            title_color=color,
        )
        ax.set_xlabel(f"idx {hsc_inst_indices[j]}", fontsize=7)

    # Row 2: HSC query → physics kNN
    for j, img in enumerate(hsc_phys_images):
        if j >= K_NEIGHBORS or 1 + j >= n_cols:
            break
        is_counterpart = hsc_phys_indices[j] == query_idx and hsc_phys_sources[j] == "legacy"
        is_cross_survey = hsc_phys_sources[j] == "legacy"
        if is_counterpart:
            color = "red"
        elif is_cross_survey:
            color = "gold"
        else:
            color = "black"
        ax = axes[2, 1 + j]
        show_img(
            ax,
            img,
            f"HSC phys kNN {j+1} {src_label(hsc_phys_sources[j])}",
            title_color=color,
        )
        ax.set_xlabel(f"idx {hsc_phys_indices[j]}", fontsize=7)

    # Row 3: Legacy query → instrument kNN
    for j, img in enumerate(leg_inst_images):
        if j >= K_NEIGHBORS or 1 + j >= n_cols:
            break
        is_counterpart = leg_inst_indices[j] == query_idx and leg_inst_sources[j] == "hsc"
        is_cross_survey = leg_inst_sources[j] == "hsc"
        if is_counterpart:
            color = "red"
        elif is_cross_survey:
            color = "gold"
        else:
            color = "black"
        ax = axes[3, 1 + j]
        show_img(
            ax,
            img,
            f"Leg inst kNN {j+1} {src_label(leg_inst_sources[j])}",
            title_color=color,
        )
        ax.set_xlabel(f"idx {leg_inst_indices[j]}", fontsize=7)

    # Row 4: Legacy query → physics kNN
    for j, img in enumerate(leg_phys_images):
        if j >= K_NEIGHBORS or 1 + j >= n_cols:
            break
        is_counterpart = leg_phys_indices[j] == query_idx and leg_phys_sources[j] == "hsc"
        is_cross_survey = leg_phys_sources[j] == "hsc"
        if is_counterpart:
            color = "red"
        elif is_cross_survey:
            color = "gold"
        else:
            color = "black"
        ax = axes[4, 1 + j]
        show_img(
            ax,
            img,
            f"Leg phys kNN {j+1} {src_label(leg_phys_sources[j])}",
            title_color=color,
        )
        ax.set_xlabel(f"idx {leg_phys_indices[j]}", fontsize=7)

    fig.suptitle(
        f"Neighbors for query_idx={query_idx} (triplet_images-style RGB)\n"
        "Colors: gold = cross-survey neighbor, red = direct counterpart",
        fontsize=9,
    )
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Re-plot neighbors for a given query_idx using neighbors_summary.csv "
            "and neighbours_v2.h5, with triplet_images-style RGB visualization."
        )
    )
    p.add_argument(
        "--query-idx",
        type=int,
        required=True,
        help="Dataset index used when building neighbors_summary.csv (same as in search_neighbors.py).",
    )
    p.add_argument(
        "--summary-csv",
        type=Path,
        default=DEFAULT_SUMMARY_CSV,
        help=f"Path to neighbors_summary.csv (default: {DEFAULT_SUMMARY_CSV})",
    )
    p.add_argument(
        "--neighbors-h5",
        type=Path,
        default=DEFAULT_NEIGHBORS_HDF5,
        help=f"Path to neighbours_v2.h5 (default: {DEFAULT_NEIGHBORS_HDF5})",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output figure path (default: neighbor_search/query_results/query_<idx>_nice.png)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    summary_csv = args.summary_csv
    neighbors_h5 = args.neighbors_h5
    if not summary_csv.is_file():
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")
    if not neighbors_h5.is_file():
        raise FileNotFoundError(f"Neighbors HDF5 not found: {neighbors_h5}")

    print(f"Loading summary row for query_idx={args.query_idx} from {summary_csv}")
    row = load_summary_row(summary_csv, args.query_idx)
    (
        hsc_inst_neighbors,
        hsc_phys_neighbors,
        leg_inst_neighbors,
        leg_phys_neighbors,
    ) = build_neighbor_lists(row)

    print(f"Opening neighbors HDF5 from {neighbors_h5}")
    with h5py.File(neighbors_h5, "r") as f:
        indexes_mmu = get_indexes_mmu(f)

        print("Loading query images (full 160x160)...")
        query_hsc, query_legacy = load_query_images_h5(f, indexes_mmu, args.query_idx)

        print("Loading neighbor images for all four categories (full 160x160)...")
        hsc_inst_images, hsc_inst_sources, hsc_inst_indices = load_images_for_neighbors_h5(
            f, indexes_mmu, hsc_inst_neighbors
        )
        hsc_phys_images, hsc_phys_sources, hsc_phys_indices = load_images_for_neighbors_h5(
            f, indexes_mmu, hsc_phys_neighbors
        )
        leg_inst_images, leg_inst_sources, leg_inst_indices = load_images_for_neighbors_h5(
            f, indexes_mmu, leg_inst_neighbors
        )
        leg_phys_images, leg_phys_sources, leg_phys_indices = load_images_for_neighbors_h5(
            f, indexes_mmu, leg_phys_neighbors
        )

        if args.out is None:
            out_path = _here / "query_results" / f"query_{args.query_idx}_nice.png"
        else:
            out_path = args.out

        print(f"Plotting neighbors to {out_path} ...")
        plot_query_and_neighbors_triplet_style(
            query_hsc,
            query_legacy,
            hsc_inst_images,
            hsc_inst_sources,
            hsc_inst_indices,
            hsc_phys_images,
            hsc_phys_sources,
            hsc_phys_indices,
            leg_inst_images,
            leg_inst_sources,
            leg_inst_indices,
            leg_phys_images,
            leg_phys_sources,
            leg_phys_indices,
            args.query_idx,
            out_path,
        )


if __name__ == "__main__":
    main()
