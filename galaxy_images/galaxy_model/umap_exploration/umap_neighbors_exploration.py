"""
UMAP exploration of the neighbors model's latent spaces.

Uses the same pretrained model as power_autocorrelation_analysis.py.
Encodes N examples (both HSC and Legacy images) through encoder_1 (physics)
and encoder_2 (instrument), then produces UMAP visualizations:

  1) Colored by Survey (HSC vs Legacy)
  2) Colored by properties available in the neighbors HDF5 dataset
  3) [TODO] Colored by properties from the MMU dataset
"""

import argparse
import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import os
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")

import re
import matplotlib
matplotlib.use("Agg")

import torch
torch.backends.cuda.preferred_blas_library("hipblas")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import umap
from torch.utils.data import DataLoader, Subset

from galaxy_images.galaxy_model.double_train_fm_neighbors import ConditionalFlowMatchingModule
from galaxy_images.galaxy_model.neighbors import NeighborsSimpleDataset

# ============= CONFIGURATION =============

MODEL_CHECKPOINT = (
    '/work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model/'
    'outputs/neighbors_all_attn/checkpoints/best-epoch=228-step=87000.ckpt'
)
HDF5_PATH = '/work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5'

DEFAULT_NUM_EXAMPLES = 2000
BATCH_SIZE = 256

COLOR_HSC = '#e8c4a0'
COLOR_LEGACY = '#8eb8e8'

UMAP_PARAMS = {
    'n_neighbors': 15,
    'min_dist': 0.1,
    'n_components': 2,
    'metric': 'euclidean',
    'random_state': 42,
}

SKIP_META_KEYS = frozenset({'idx', 'index_mmu'})

# Averaging patterns: regex -> averaged name.
# Same convention as downstream_evaluation/final/makeplot_v2.py and aion_vs_ours_all.py
AVERAGE_PATTERNS = {
    r"^legacy_GALDEPTH_": "legacy_GALDEPTH",
    r"^legacy_NOBS_": "legacy_NOBS",
    r"^legacy_PSFSIZE_": "legacy_PSFSIZE",
    r"^legacy_PSFDEPTH_": "legacy_PSFDEPTH",
    r"^hsc_.*_variance_value$": "hsc_variance_value",
    r"^hsc_.*_psf_fwhm$": "hsc_psf_fwhm",
}

# Readable labels for averaged properties (matches aion_vs_ours_all.py)
DISPLAY_LABELS = {
    "hsc_variance_value": "HSC Variance",
    "hsc_psf_fwhm": "HSC PSF Size",
    "legacy_GALDEPTH": "Legacy Galaxy Depth",
    "legacy_NOBS": "Legacy # Observations",
    "legacy_PSFSIZE": "Legacy PSF Size",
    "legacy_PSFDEPTH": "Legacy PSF Depth",
    "EBV": "E(B-V)",
}


# ============= HELPERS =============

def collate_simple(batch):
    hsc = torch.stack([b[0] for b in batch])
    leg = torch.stack([b[1] for b in batch])
    meta = [b[2] for b in batch]
    return hsc, leg, meta


def compute_hsc_psf_fwhm(shape11, shape22):
    """Derive PSF FWHM (arcsec) from HSC sdssshape moments."""
    pixel_scale_hsc = 0.168
    return 2.355 * np.sqrt((shape11 + shape22) / 2) * pixel_scale_hsc


def extract_numeric_metadata(metadata_collected):
    """Extract all numeric columns from metadata dicts, plus derived PSF FWHM."""
    meta_keys = list(metadata_collected[0].keys())
    valid_params = []
    param_arrays = {}

    for key in meta_keys:
        if key in SKIP_META_KEYS:
            continue
        try:
            vals = np.array([m[key] for m in metadata_collected], dtype=np.float64)
            if vals.ndim != 1:
                continue
            finite_frac = np.isfinite(vals).mean()
            if finite_frac < 0.5:
                continue
            valid_params.append(key)
            param_arrays[key] = vals.astype(np.float32)
        except (TypeError, ValueError):
            continue

    for band in ("g", "i", "r", "z"):
        k11 = f"hsc_{band}_sdssshape_psf_shape11"
        k22 = f"hsc_{band}_sdssshape_psf_shape22"
        if k11 in param_arrays and k22 in param_arrays:
            fwhm = compute_hsc_psf_fwhm(param_arrays[k11], param_arrays[k22])
            name = f"hsc_{band}_psf_fwhm"
            param_arrays[name] = fwhm.astype(np.float32)
            valid_params.append(name)

    return valid_params, param_arrays


def average_multiband_properties(valid_params, param_arrays):
    """Average multi-band properties across channels, matching downstream_evaluation logic.

    Returns a reduced (name_list, arrays_dict) where per-band entries are replaced
    by a single channel-averaged entry.
    """
    consumed = set()
    averaged_params = []
    averaged_arrays = {}

    for pattern, avg_name in AVERAGE_PATTERNS.items():
        matching = [p for p in valid_params if re.search(pattern, p)]
        if not matching:
            continue
        stacked = np.stack([param_arrays[p] for p in matching], axis=0)
        averaged_arrays[avg_name] = np.nanmean(stacked, axis=0).astype(np.float32)
        averaged_params.append(avg_name)
        consumed.update(matching)

    for p in valid_params:
        if p not in consumed:
            averaged_params.append(p)
            averaged_arrays[p] = param_arrays[p]

    return averaged_params, averaged_arrays


# ============= PLOTTING =============

def plot_survey_umap(hsc_umap_1, leg_umap_1, hsc_umap_2, leg_umap_2,
                     num_pairs, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    ax1.scatter(hsc_umap_1[:, 0], hsc_umap_1[:, 1],
                s=5, alpha=0.5, c=COLOR_HSC, label='HSC')
    ax1.scatter(leg_umap_1[:, 0], leg_umap_1[:, 1],
                s=5, alpha=0.5, c=COLOR_LEGACY, label='Legacy')
    ax1.set_title('Encoder 1 — Physics Space', fontsize=13)
    ax1.set_xlabel('UMAP 1')
    ax1.set_ylabel('UMAP 2')
    ax1.legend(markerscale=4)
    ax1.grid(True, alpha=0.3)

    ax2.scatter(hsc_umap_2[:, 0], hsc_umap_2[:, 1],
                s=5, alpha=0.5, c=COLOR_HSC, label='HSC')
    ax2.scatter(leg_umap_2[:, 0], leg_umap_2[:, 1],
                s=5, alpha=0.5, c=COLOR_LEGACY, label='Legacy')
    ax2.set_title('Encoder 2 — Instrument Space', fontsize=13)
    ax2.set_xlabel('UMAP 1')
    ax2.set_ylabel('UMAP 2')
    ax2.legend(markerscale=4)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f'UMAP Colored by Survey (N={num_pairs} pairs)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = output_dir / 'umap_by_survey.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path.name}")


def plot_neighbors_properties(hsc_umap_1, leg_umap_1, hsc_umap_2, leg_umap_2,
                              valid_params, param_arrays, num_pairs, output_dir,
                              filename='umap_neighbors_properties.png'):
    """Big grid: one row per property, 4 columns (Phys HSC, Phys Leg, Inst HSC, Inst Leg)."""
    num_params = len(valid_params)
    if num_params == 0:
        print("  No valid numeric metadata — skipping neighbors properties plot.")
        return

    fig, axes = plt.subplots(num_params, 4, figsize=(24, 5.5 * num_params))
    if num_params == 1:
        axes = axes.reshape(1, -1)

    col_configs = [
        (hsc_umap_1, 'Physics — HSC'),
        (leg_umap_1, 'Physics — Legacy'),
        (hsc_umap_2, 'Instrument — HSC'),
        (leg_umap_2, 'Instrument — Legacy'),
    ]

    for row_idx, param_name in enumerate(valid_params):
        display_name = DISPLAY_LABELS.get(param_name, param_name)
        values = param_arrays[param_name]
        finite_mask = np.isfinite(values)
        vmin = np.nanpercentile(values[finite_mask], 5) if finite_mask.any() else 0
        vmax = np.nanpercentile(values[finite_mask], 95) if finite_mask.any() else 1
        if vmax <= vmin:
            vmax = vmin + 1e-6
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        for col_idx, (emb, space_label) in enumerate(col_configs):
            ax = axes[row_idx, col_idx]
            sc = ax.scatter(
                emb[:, 0], emb[:, 1],
                s=8, c=values, cmap='viridis', alpha=0.5,
                edgecolors='none', norm=norm,
            )
            ax.set_title(f'{display_name}\n{space_label}', fontsize=10)
            ax.set_xlabel('UMAP 1', fontsize=8)
            ax.set_ylabel('UMAP 2', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
            plt.colorbar(sc, ax=ax, label=display_name, shrink=0.8)

        if (row_idx + 1) % 5 == 0:
            print(f"    Plotted {row_idx + 1}/{num_params} parameters...")

    fig.suptitle(
        f'UMAP Colored by Neighbors Properties (N={num_pairs} pairs)',
        fontsize=16, y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    path = output_dir / filename
    fig.savefig(path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path.name}")


# ============= MAIN =============

def parse_args():
    parser = argparse.ArgumentParser(description="UMAP exploration of neighbors model latent spaces")
    parser.add_argument("--n-examples", type=int, default=DEFAULT_NUM_EXAMPLES,
                        help=f"Number of examples to use (default: {DEFAULT_NUM_EXAMPLES})")
    return parser.parse_args()


def main():
    args = parse_args()
    num_examples = args.n_examples

    output_dir = _script_dir / f"figures_{num_examples}"
    output_dir.mkdir(exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Num examples: {num_examples}")
    print(f"Output directory: {output_dir}")

    # ---- Load model ----
    print("Loading model from checkpoint...")
    model = ConditionalFlowMatchingModule.load_from_checkpoint(
        MODEL_CHECKPOINT, map_location='cpu'
    )
    model.eval()
    model.to(device)
    print("Model loaded.")

    # ---- Load dataset ----
    print(f"Opening NeighborsSimpleDataset from {HDF5_PATH}...")
    full_dataset = NeighborsSimpleDataset(hdf5_path=HDF5_PATH)
    n_use = min(num_examples, len(full_dataset))
    dataset = Subset(full_dataset, list(range(n_use)))
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_simple,
    )
    print(f"Will process {n_use} examples (dataset total: {len(full_dataset)})")

    # ---- Encode all images ----
    print("\nEncoding images through both encoders...")
    hsc_emb1_list, hsc_emb2_list = [], []
    leg_emb1_list, leg_emb2_list = [], []
    metadata_collected = []

    with torch.no_grad():
        for batch_idx, (hsc_im, leg_im, meta_list) in enumerate(loader):
            hsc_im = hsc_im.to(device)
            leg_im = leg_im.to(device)

            hsc_emb1_list.append(model.encoder_1(hsc_im).cpu())
            hsc_emb2_list.append(model.encoder_2(hsc_im).cpu())
            leg_emb1_list.append(model.encoder_1(leg_im).cpu())
            leg_emb2_list.append(model.encoder_2(leg_im).cpu())
            metadata_collected.extend(meta_list)

            processed = min((batch_idx + 1) * BATCH_SIZE, n_use)
            print(f"  Encoded {processed}/{n_use}")

    hsc_emb1 = torch.cat(hsc_emb1_list).flatten(start_dim=1).numpy()
    hsc_emb2 = torch.cat(hsc_emb2_list).flatten(start_dim=1).numpy()
    leg_emb1 = torch.cat(leg_emb1_list).flatten(start_dim=1).numpy()
    leg_emb2 = torch.cat(leg_emb2_list).flatten(start_dim=1).numpy()

    all_emb1 = np.concatenate([hsc_emb1, leg_emb1], axis=0)
    all_emb2 = np.concatenate([hsc_emb2, leg_emb2], axis=0)
    num_hsc = hsc_emb1.shape[0]

    print(f"Embeddings: encoder_1 {all_emb1.shape}, encoder_2 {all_emb2.shape}")

    # ---- Compute UMAPs ----
    print("\nComputing UMAP for Encoder 1 (Physics)...")
    reducer_1 = umap.UMAP(**UMAP_PARAMS)
    umap_1 = reducer_1.fit_transform(all_emb1)
    hsc_umap_1, leg_umap_1 = umap_1[:num_hsc], umap_1[num_hsc:]

    print("Computing UMAP for Encoder 2 (Instrument)...")
    reducer_2 = umap.UMAP(**UMAP_PARAMS)
    umap_2 = reducer_2.fit_transform(all_emb2)
    hsc_umap_2, leg_umap_2 = umap_2[:num_hsc], umap_2[num_hsc:]

    # ---- Save embeddings for later reuse ----
    np.savez_compressed(
        output_dir / 'umap_embeddings.npz',
        hsc_umap_1=hsc_umap_1, leg_umap_1=leg_umap_1,
        hsc_umap_2=hsc_umap_2, leg_umap_2=leg_umap_2,
    )
    print(f"Saved UMAP coordinates to {output_dir / 'umap_embeddings.npz'}")

    # ===============================================================
    # PLOT 1: Colored by Survey
    # ===============================================================
    print("\n--- Plot 1: Colored by Survey ---")
    plot_survey_umap(hsc_umap_1, leg_umap_1, hsc_umap_2, leg_umap_2,
                     num_hsc, output_dir)

    # ===============================================================
    # PLOT 2: Colored by neighbors dataset properties (channel-averaged)
    # ===============================================================
    print("\n--- Plot 2: Colored by Neighbors Properties ---")
    all_params, all_arrays = extract_numeric_metadata(metadata_collected)
    print(f"  Extracted {len(all_params)} raw numeric columns")

    avg_params, avg_arrays = average_multiband_properties(all_params, all_arrays)
    print(f"  After channel averaging: {len(avg_params)} properties:")
    for p in avg_params:
        label = DISPLAY_LABELS.get(p, p)
        arr = avg_arrays[p]
        print(f"    {label:40s} ({p})  range=[{np.nanmin(arr):.4g}, {np.nanmax(arr):.4g}]")

    plot_neighbors_properties(
        hsc_umap_1, leg_umap_1, hsc_umap_2, leg_umap_2,
        avg_params, avg_arrays, num_hsc, output_dir,
    )

    # ===============================================================
    # PLOT 3: [TODO] Colored by MMU dataset properties
    # ===============================================================
    # The MMU dataset lives in a separate HDF5 file with matched
    # HSC/Legacy pairs plus photometric/structural metadata.
    # Uncomment and adapt when the MMU data is available on this cluster.
    #
    # MMU_HDF5_PATH = '/path/to/preprocessed_hsc_legacy_48x48_all.h5'
    # MMU_METADATA_PATH = '/path/to/preprocessed_hsc_legacy_metadata_8192.h5'
    # MMU_PARAMS = [
    #     "EBV", "FLUX_G", "FLUX_I", "FLUX_R", "FLUX_W1", "FLUX_W2",
    #     "FLUX_W3", "FLUX_W4", "FLUX_Z", "SHAPE_E1", "SHAPE_E2", "SHAPE_R",
    #     "a_g", "a_i", "a_r", "a_y", "a_z",
    #     "g_cmodel_mag", "g_cmodel_magerr",
    #     "g_sdssshape_psf_shape11", "g_sdssshape_psf_shape12", "g_sdssshape_psf_shape22",
    #     "i_cmodel_mag", "i_cmodel_magerr", "i_extendedness_value",
    #     "i_sdssshape_psf_shape11", "i_sdssshape_psf_shape12", "i_sdssshape_psf_shape22",
    #     "r_cmodel_mag", "r_sdssshape_psf_shape11", "r_sdssshape_psf_shape12",
    #     "r_sdssshape_psf_shape22",
    #     "y_cmodel_mag", "y_cmodel_magerr", "y_extendedness_value",
    #     "z_cmodel_mag", "z_sdssshape_psf_shape11", "z_sdssshape_psf_shape12",
    #     "z_sdssshape_psf_shape22",
    # ]
    #
    # To use the MMU dataset:
    #   1. Load the HSCLegacyDatasetZoom with an idx_list of NUM_EXAMPLES indices
    #   2. Encode through model.encoder_1 / model.encoder_2
    #   3. Load metadata from MMU_METADATA_PATH (h5py, keyed by MMU_PARAMS)
    #   4. Compute UMAP on the MMU embeddings
    #   5. Call plot_neighbors_properties(...) with the MMU param arrays
    #
    # import h5py
    # from galaxy_images.galaxy_model.data import HSCLegacyDatasetZoom
    #
    # mmu_dataset = HSCLegacyDatasetZoom(
    #     hdf5_path=MMU_HDF5_PATH,
    #     idx_list=list(range(NUM_EXAMPLES)),
    # )
    # mmu_loader = DataLoader(mmu_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    #
    # mmu_hsc_e1, mmu_hsc_e2, mmu_leg_e1, mmu_leg_e2 = [], [], [], []
    # with torch.no_grad():
    #     for hsc_im, leg_im in mmu_loader:
    #         hsc_im, leg_im = hsc_im.to(device), leg_im.to(device)
    #         mmu_hsc_e1.append(model.encoder_1(hsc_im).cpu())
    #         mmu_hsc_e2.append(model.encoder_2(hsc_im).cpu())
    #         mmu_leg_e1.append(model.encoder_1(leg_im).cpu())
    #         mmu_leg_e2.append(model.encoder_2(leg_im).cpu())
    #
    # mmu_all_e1 = np.concatenate([
    #     torch.cat(mmu_hsc_e1).flatten(1).numpy(),
    #     torch.cat(mmu_leg_e1).flatten(1).numpy(),
    # ])
    # mmu_all_e2 = np.concatenate([
    #     torch.cat(mmu_hsc_e2).flatten(1).numpy(),
    #     torch.cat(mmu_leg_e2).flatten(1).numpy(),
    # ])
    #
    # mmu_reducer_1 = umap.UMAP(**UMAP_PARAMS)
    # mmu_umap_1 = mmu_reducer_1.fit_transform(mmu_all_e1)
    # mmu_hsc_umap_1 = mmu_umap_1[:NUM_EXAMPLES]
    # mmu_leg_umap_1 = mmu_umap_1[NUM_EXAMPLES:]
    #
    # mmu_reducer_2 = umap.UMAP(**UMAP_PARAMS)
    # mmu_umap_2 = mmu_reducer_2.fit_transform(mmu_all_e2)
    # mmu_hsc_umap_2 = mmu_umap_2[:NUM_EXAMPLES]
    # mmu_leg_umap_2 = mmu_umap_2[NUM_EXAMPLES:]
    #
    # with h5py.File(MMU_METADATA_PATH, 'r') as f:
    #     mmu_param_arrays = {}
    #     mmu_valid_params = []
    #     for param in MMU_PARAMS:
    #         if param in f:
    #             vals = f[param][:NUM_EXAMPLES].astype(np.float32)
    #             mmu_param_arrays[param] = vals
    #             mmu_valid_params.append(param)
    #
    # plot_survey_umap(
    #     mmu_hsc_umap_1, mmu_leg_umap_1,
    #     mmu_hsc_umap_2, mmu_leg_umap_2,
    #     NUM_EXAMPLES, output_dir,
    # )
    # plot_neighbors_properties(
    #     mmu_hsc_umap_1, mmu_leg_umap_1,
    #     mmu_hsc_umap_2, mmu_leg_umap_2,
    #     mmu_valid_params, mmu_param_arrays,
    #     NUM_EXAMPLES, output_dir,
    # )

    print(f"\nDone! Figures saved to {output_dir}")


if __name__ == '__main__':
    main()
