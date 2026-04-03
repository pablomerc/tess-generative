"""
UMAP of AION latents for Legacy and HSC galaxy images.

Three encoding modes (all with num_encoder_tokens=256):
  1. Legacy-only:  encode Legacy image alone
  2. HSC-only:     encode HSC image alone
  3. Joint:        encode both images together

For each mode, Legacy and HSC latents are concatenated, a single UMAP is fit,
and points are colored by survey with 20 highlighted pairs.

Run from galaxy_model/ or aion_benchmark/aion_umap/:
  python aion_benchmark/aion_umap/aion_umap.py
"""
import os
import sys
from pathlib import Path

os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "0"

_here = Path(__file__).resolve().parent
_aion_benchmark = _here.parent
_src = _aion_benchmark.parent
for _p in [str(_src), str(_aion_benchmark)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import h5py
import numpy as np
import torch
import umap
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase
from tqdm import tqdm

from aion import AION
from aion.codecs import CodecManager
from aion.modalities import LegacySurveyImage, HSCImage

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NEIGHBORS_HDF5 = "/work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5"
OUTPUT_DIR = _here
# NUM_EXAMPLES = 4096
NUM_EXAMPLES=4096
BATCH_SIZE = 32
DEVICE = "cpu"
SEED = 42
NUM_ENCODER_TOKENS = 512

# UMAP
UMAP_PARAMS = {
    "n_neighbors": 15,
    "min_dist": 0.1,
    "n_components": 2,
    "metric": "euclidean",
    "random_state": SEED,
}

# Visual style (match visualization_scripts/plot_umap_from_file.py)
COLOR_HSC = "#e8c4a0"
COLOR_LEGACY = "#8eb8e8"
COLOR_JOINT = "#b8d4b8"
PAIR_COLORS = ["#70a845", "#b460bd", "#4aac8d", "#c85979", "#b49041"]
PAIR_MARKERS = ["x", "s", "o", "^"]
NUM_PAIRS = 20
POINT_SIZE = 20
PAIR_MARKER_SIZE = 200
PAIR_LINEWIDTHS = 3
ALPHA = 1.0
DPI = 150
TITLE_FONTSIZE = 23
AXIS_FONTSIZE = 23
LEGEND_FONTSIZE = 17
TICK_FONTSIZE = 17
LEGEND_MARKER_SIZE = 8


# ---------------------------------------------------------------------------
# Custom legend handler for the 'Pairs' row
# ---------------------------------------------------------------------------
class _PairsHandle:
    pass


class _HandlerPairs(HandlerBase):
    def __init__(self, markers, **kwargs):
        super().__init__(**kwargs)
        self.markers = markers

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        artists = []
        y_center = ydescent + height / 2
        n = len(self.markers)
        for i, mk in enumerate(self.markers):
            x = xdescent + width * (i + 0.5) / n
            line = Line2D(
                [x], [y_center],
                marker=mk, color="white",
                markerfacecolor="white", markeredgecolor="black",
                markeredgewidth=1.2, markersize=fontsize * 0.8,
                linestyle="None", transform=trans,
            )
            artists.append(line)
        return artists


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------
def encode_pairs(hdf5_path: str, n_use: int, batch_size: int, device: str):
    """
    Encode Legacy and HSC images with AION in three modes:
      - Legacy-only, HSC-only, Joint (both together)
    All use the same num_encoder_tokens for comparable latent shapes.
    Returns (legacy_latents, hsc_latents, joint_latents) as numpy arrays of shape [N, D].
    """
    model = AION.from_pretrained("polymathic-ai/aion-base").to(device)
    model.eval()
    codec_manager = CodecManager(device=device)

    legacy_list, hsc_list, joint_list = [], [], []

    with h5py.File(hdf5_path, "r") as f:
        indices = np.where(f["source_type"][:] == 0)[0][:n_use]
        n_actual = len(indices)
        print(f"Encoding {n_actual} pairs...")

        for start in tqdm(range(0, n_actual, batch_size), desc="Encoding"):
            end = min(start + batch_size, n_actual)
            batch_idx = indices[start:end]

            legacy_tensor = torch.from_numpy(f["images_legacy"][batch_idx])
            hsc_tensor = torch.from_numpy(f["images_hsc"][batch_idx])

            image_leg = LegacySurveyImage(
                flux=legacy_tensor,
                bands=["DES-G", "DES-R", "DES-I", "DES-Z"],
            )
            image_hsc = HSCImage(
                flux=hsc_tensor,
                bands=["HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y"],
            )

            tokens_leg = codec_manager.encode(image_leg)
            tokens_hsc = codec_manager.encode(image_hsc)
            tokens_joint = codec_manager.encode(image_hsc, image_leg)

            with torch.no_grad():
                emb_leg = model.encode(tokens_leg, num_encoder_tokens=NUM_ENCODER_TOKENS)
                emb_hsc = model.encode(tokens_hsc, num_encoder_tokens=NUM_ENCODER_TOKENS)
                emb_joint = model.encode(tokens_joint, num_encoder_tokens=NUM_ENCODER_TOKENS)

            # Mean pool over token dimension: [B, T, D] -> [B, D]
            legacy_list.append(emb_leg.mean(dim=1).cpu().numpy())
            hsc_list.append(emb_hsc.mean(dim=1).cpu().numpy())
            joint_list.append(emb_joint.mean(dim=1).cpu().numpy())

    legacy_latents = np.concatenate(legacy_list, axis=0)
    hsc_latents = np.concatenate(hsc_list, axis=0)
    joint_latents = np.concatenate(joint_list, axis=0)
    print(f"Legacy latents: {legacy_latents.shape}")
    print(f"HSC latents:    {hsc_latents.shape}")
    print(f"Joint latents:  {joint_latents.shape}")
    return legacy_latents, hsc_latents, joint_latents


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Device: {DEVICE}")
    print(f"Neighbors HDF5: {NEIGHBORS_HDF5}")
    print(f"Num examples: {NUM_EXAMPLES}")
    print(f"Num encoder tokens: {NUM_ENCODER_TOKENS}")

    legacy_latents, hsc_latents, joint_latents = encode_pairs(
        NEIGHBORS_HDF5, NUM_EXAMPLES, BATCH_SIZE, DEVICE
    )

    print("\n--- UMAP: all three encoding types together ---")
    all_latents = np.concatenate([legacy_latents, hsc_latents, joint_latents], axis=0)
    n = len(legacy_latents)
    print(f"Fitting UMAP on {all_latents.shape[0]} points (dim={all_latents.shape[1]})...")
    reducer = umap.UMAP(**UMAP_PARAMS)
    embedding = reducer.fit_transform(all_latents)

    legacy_umap = embedding[:n]
    hsc_umap = embedding[n:2*n]
    joint_umap = embedding[2*n:]

    output_path = OUTPUT_DIR / "aion_umap3.png"
    plot_umap_three(legacy_umap, hsc_umap, joint_umap, output_path)

    np_path = OUTPUT_DIR / "aion_umap3_data.npz"
    np.savez_compressed(
        np_path,
        legacy_umap=legacy_umap,
        hsc_umap=hsc_umap,
        joint_umap=joint_umap,
    )
    print(f"Raw UMAP coordinates saved to: {np_path}")


def plot_umap_three(legacy_umap, hsc_umap, joint_umap, output_path):
    n = len(legacy_umap)
    np.random.seed(SEED)
    selected_indices = np.random.choice(n, size=NUM_PAIRS, replace=False)
    print(f"Highlighted pairs: {selected_indices}")

    fig, ax = plt.subplots(1, 1, figsize=(12, 9))

    ax.scatter(
        legacy_umap[:, 0], legacy_umap[:, 1],
        s=POINT_SIZE, c=COLOR_LEGACY, alpha=ALPHA, zorder=2,
    )
    ax.scatter(
        hsc_umap[:, 0], hsc_umap[:, 1],
        s=POINT_SIZE, c=COLOR_HSC, alpha=ALPHA, zorder=2,
    )
    ax.scatter(
        joint_umap[:, 0], joint_umap[:, 1],
        s=POINT_SIZE, c=COLOR_JOINT, alpha=ALPHA, zorder=2,
    )

    # Highlighted triplets: same marker/color for all 3 versions of one galaxy
    for i, idx in enumerate(selected_indices):
        color = PAIR_COLORS[i % len(PAIR_COLORS)]
        marker = PAIR_MARKERS[i % len(PAIR_MARKERS)]
        lw_outline = PAIR_LINEWIDTHS + 2 if marker == "x" else PAIR_LINEWIDTHS

        for umap_arr in (legacy_umap, hsc_umap, joint_umap):
            if marker == "x":
                ax.scatter(
                    umap_arr[idx, 0], umap_arr[idx, 1],
                    marker=marker, s=PAIR_MARKER_SIZE, c=["black"],
                    linewidths=lw_outline, zorder=4, alpha=1.0,
                )
            ax.scatter(
                umap_arr[idx, 0], umap_arr[idx, 1],
                marker=marker, s=PAIR_MARKER_SIZE, c=[color],
                linewidths=PAIR_LINEWIDTHS, zorder=5, edgecolors="black", alpha=1.0,
            )

    ax.set_title("AION Latent Space (UMAP)", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax.set_xlabel("UMAP Component 1", fontsize=AXIS_FONTSIZE)
    ax.set_ylabel("UMAP Component 2", fontsize=AXIS_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_LEGACY,
               markeredgecolor="black", markersize=LEGEND_MARKER_SIZE, label="Legacy-only"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_HSC,
               markeredgecolor="black", markersize=LEGEND_MARKER_SIZE, label="HSC-only"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_JOINT,
               markeredgecolor="black", markersize=LEGEND_MARKER_SIZE, label="Joint"),
        _PairsHandle(),
    ]
    legend_labels = ["Legacy-only", "HSC-only", "Joint", "Pairs"]
    ax.legend(
        handles=legend_handles,
        labels=legend_labels,
        fontsize=LEGEND_FONTSIZE,
        handlelength=4,
        handler_map={_PairsHandle: _HandlerPairs(PAIR_MARKERS)},
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DPI)
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
