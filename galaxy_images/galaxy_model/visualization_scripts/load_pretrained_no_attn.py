"""
Load a pretrained no-attn double-encoder model and generate UMAP, PCA, and t-SNE
latent space visualizations with overlapping HSC and Legacy points.

This expects checkpoints produced by `double_train_fm_no-attn.py`.
"""

import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import importlib.util
from torch.utils.data import DataLoader
from data import HSCLegacyDataset
import time

import umap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# Path to the training script that defines the no-attn ConditionalFlowMatchingModule
NO_ATTN_TRAIN_SCRIPT = (
    "/data/vision/billf/scratch/pablomer/projects/tess-generative/"
    "galaxy_images/galaxy_model/double_train_fm_no-attn.py"
)


def _load_no_attn_class():
    """Dynamically load ConditionalFlowMatchingModule from the no-attn training script."""
    spec = importlib.util.spec_from_file_location(
        "double_train_fm_no_attn_module", NO_ATTN_TRAIN_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec from {NO_ATTN_TRAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    if not hasattr(module, "ConditionalFlowMatchingModule"):
        raise AttributeError(
            f"'ConditionalFlowMatchingModule' not found in {NO_ATTN_TRAIN_SCRIPT}"
        )
    return module.ConditionalFlowMatchingModule


# Single no-attn checkpoint (update here if you want to try other runs)
checkpoint_path = (
    "/data/vision/billf/scratch/pablomer/projects/tess-generative/"
    # "galaxy-flow-matching/p9tj82az/checkpoints/latest-step=step=75000.ckpt"
    "galaxy-flow-matching/p9tj82az/checkpoints/best-epoch=161-step=60000-val_loss=0.2111.ckpt"
)

zoom = True

# Tag used in filenames
mode_tag = "_noattn"

# Control flags
GENERATE_UMAP = True   # Set to False to skip UMAP generation and plotting
GENERATE_PCA = True    # Set to False to skip PCA generation and plotting
GENERATE_TSNE = True   # Set to False to skip t-SNE generation and plotting
SHOW_PAIRS = True      # Set to False to skip marking pairs on the plots


def main():
    # Load the no-attn class definition from the training script
    ConditionalFlowMatchingModule = _load_no_attn_class()

    # Determine device: try to find a working GPU, fallback to CPU
    device = torch.device("cpu")
    if torch.cuda.is_available():
        for gpu_id in range(torch.cuda.device_count()):
            try:
                test_tensor = torch.tensor([1.0], device=f"cuda:{gpu_id}")
                del test_tensor
                torch.cuda.empty_cache()
                device = torch.device(f"cuda:{gpu_id}")
                print(f"Using GPU {gpu_id}")
                break
            except RuntimeError:
                print(f"GPU {gpu_id} is not available, trying next...")
                continue
        if device.type == "cpu":
            print("No working GPU found, using CPU")

    print(f"\nLoading no-attn model from checkpoint: {checkpoint_path}")
    model = ConditionalFlowMatchingModule.load_from_checkpoint(
        checkpoint_path, map_location="cpu"
    )

    model.eval()
    torch.set_grad_enabled(False)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")

    # Time dataset initialization
    dataset_start = time.perf_counter()
    dataset = HSCLegacyDataset(
        hdf5_path="/data/vision/billf/scratch/pablomer/legacysurvey_hsc/"
        "preprocessed_hsc_legacy_48x48_all.h5",
        idx_list=list(range(95_000, 97_048)),
    )
    dataset_time = time.perf_counter() - dataset_start

    # DataLoader
    loader_start = time.perf_counter()
    train_loader = DataLoader(dataset, batch_size=2048, shuffle=True, num_workers=4)
    loader_time = time.perf_counter() - loader_start

    # First batch timing
    batch_start = time.perf_counter()
    batch = next(iter(train_loader))
    batch_time = time.perf_counter() - batch_start

    actual_batch_size = batch[0].shape[0]
    total_time = dataset_time + loader_time + batch_time

    print(f"\nTiming breakdown for {actual_batch_size} examples:")
    print(f"  Dataset initialization (HDF5 → memory): {dataset_time:.4f} s")
    print(f"  DataLoader creation:                    {loader_time:.4f} s")
    print(f"  First batch retrieval:                  {batch_time:.4f} s")
    print(f"  Total time:                             {total_time:.4f} s")

    # Move images to device
    hsc_images = batch[0].to(device)
    legacy_images = batch[1].to(device)

    # Encode images with both encoders
    print("\nEncoding images with no-attn encoders...")
    with torch.no_grad():
        hsc_embeddings_1 = model.encoder_1(hsc_images)
        legacy_embeddings_1 = model.encoder_1(legacy_images)
        hsc_embeddings_2 = model.encoder_2(hsc_images)
        legacy_embeddings_2 = model.encoder_2(legacy_images)

    print(f"\nEncoding results (no-attn):")
    print(f"  HSC images shape:         {hsc_images.shape}")
    print(f"  HSC embeddings 1 shape:   {hsc_embeddings_1.shape}")
    print(f"  HSC embeddings 2 shape:   {hsc_embeddings_2.shape}")
    print(f"  Legacy images shape:      {legacy_images.shape}")
    print(f"  Legacy embeddings 1 shape:{legacy_embeddings_1.shape}")
    print(f"  Legacy embeddings 2 shape:{legacy_embeddings_2.shape}")

    # Embeddings are already (B, embed_dim); no flattening needed
    all_embeddings_1 = torch.concat([hsc_embeddings_1, legacy_embeddings_1], dim=0)
    all_embeddings_2 = torch.concat([hsc_embeddings_2, legacy_embeddings_2], dim=0)

    num_hsc = hsc_embeddings_1.shape[0]
    dim = hsc_embeddings_1.shape[1]

    # Output directory for this script
    figures_dir = (
        Path(
            "/data/vision/billf/scratch/pablomer/projects/tess-generative/"
            "galaxy_images/galaxy_model/figures/latentspace-noattn"
        )
    )
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ===== UMAP =====
    if GENERATE_UMAP:
        umap_params = {
            "n_neighbors": 15,
            "min_dist": 0.1,
            "n_components": 2,
            "metric": "euclidean",
            "random_state": 42,
        }

        print("\nStarting UMAP calculation for Encoder 1 (no-attn)...")
        reducer_1 = umap.UMAP(**umap_params)
        embedding_1 = reducer_1.fit_transform(all_embeddings_1.cpu().numpy())
        hsc_embedding_1 = embedding_1[:num_hsc]
        legacy_embedding_1 = embedding_1[num_hsc:]

        print("Starting UMAP calculation for Encoder 2 (no-attn)...")
        reducer_2 = umap.UMAP(**umap_params)
        embedding_2 = reducer_2.fit_transform(all_embeddings_2.cpu().numpy())
        hsc_embedding_2 = embedding_2[:num_hsc]
        legacy_embedding_2 = embedding_2[num_hsc:]

        # Optionally highlight random HSC–Legacy pairs
        selected_indices = None
        pair_colors = None
        if SHOW_PAIRS:
            np.random.seed(42)
            num_pairs_to_highlight = 5
            selected_indices = np.random.choice(
                num_hsc, size=num_pairs_to_highlight, replace=False
            )
            print(
                f"\nSelected {num_pairs_to_highlight} random pairs to highlight: "
                f"indices {selected_indices}"
            )
            pair_colors = plt.cm.tab10(
                np.linspace(0, 1, num_pairs_to_highlight)
            )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Encoder 1 plot
        ax1.scatter(
            hsc_embedding_1[:, 0],
            hsc_embedding_1[:, 1],
            s=5,
            label="HSC",
            alpha=0.6,
            c="blue",
        )
        ax1.scatter(
            legacy_embedding_1[:, 0],
            legacy_embedding_1[:, 1],
            s=5,
            label="Legacy",
            alpha=0.6,
            c="orange",
        )
        if SHOW_PAIRS and selected_indices is not None:
            for i, idx in enumerate(selected_indices):
                color = pair_colors[i]
                ax1.scatter(
                    hsc_embedding_1[idx, 0],
                    hsc_embedding_1[idx, 1],
                    marker="x",
                    s=200,
                    c=[color],
                    linewidths=3,
                    zorder=5,
                )
                ax1.scatter(
                    legacy_embedding_1[idx, 0],
                    legacy_embedding_1[idx, 1],
                    marker="x",
                    s=200,
                    c=[color],
                    linewidths=3,
                    zorder=5,
                )
        ax1.set_title("Encoder 1 (Same Galaxy) - No-Attn UMAP")
        ax1.set_xlabel("UMAP Component 1")
        ax1.set_ylabel("UMAP Component 2")
        ax1.legend()
        ax1.grid(True)

        # Encoder 2 plot
        ax2.scatter(
            hsc_embedding_2[:, 0],
            hsc_embedding_2[:, 1],
            s=5,
            label="HSC",
            alpha=0.6,
            c="blue",
        )
        ax2.scatter(
            legacy_embedding_2[:, 0],
            legacy_embedding_2[:, 1],
            s=5,
            label="Legacy",
            alpha=0.6,
            c="orange",
        )
        if SHOW_PAIRS and selected_indices is not None:
            for i, idx in enumerate(selected_indices):
                color = pair_colors[i]
                ax2.scatter(
                    hsc_embedding_2[idx, 0],
                    hsc_embedding_2[idx, 1],
                    marker="x",
                    s=200,
                    c=[color],
                    linewidths=3,
                    zorder=5,
                )
                ax2.scatter(
                    legacy_embedding_2[idx, 0],
                    legacy_embedding_2[idx, 1],
                    marker="x",
                    s=200,
                    c=[color],
                    linewidths=3,
                    zorder=5,
                )
        ax2.set_title("Encoder 2 (Same Instrument) - No-Attn UMAP")
        ax2.set_xlabel("UMAP Component 1")
        ax2.set_ylabel("UMAP Component 2")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        zoom_suffix = "_zoom" if zoom else ""
        umap_path = (
            figures_dir
            / f"umap_both_encoders_zdim{dim}{mode_tag}{zoom_suffix}.png"
        )
        plt.savefig(umap_path, dpi=150)
        plt.close()

        print(f"\n[no-attn] Combined UMAP plot saved to '{umap_path}'")
        print(f"  HSC points: {num_hsc}")
        print(f"  Legacy points: {len(legacy_embedding_1)}")

    # ===== PCA =====
    if GENERATE_PCA:
        pca_params = {"n_components": 2, "random_state": 42}

        print("\nStarting PCA calculation for Encoder 1 (no-attn)...")
        pca_1 = PCA(**pca_params)
        embedding_1_pca = pca_1.fit_transform(all_embeddings_1.cpu().numpy())
        explained_variance_1 = pca_1.explained_variance_ratio_

        hsc_embedding_1_pca = embedding_1_pca[:num_hsc]
        legacy_embedding_1_pca = embedding_1_pca[num_hsc:]

        print("Starting PCA calculation for Encoder 2 (no-attn)...")
        pca_2 = PCA(**pca_params)
        embedding_2_pca = pca_2.fit_transform(all_embeddings_2.cpu().numpy())
        explained_variance_2 = pca_2.explained_variance_ratio_

        hsc_embedding_2_pca = embedding_2_pca[:num_hsc]
        legacy_embedding_2_pca = embedding_2_pca[num_hsc:]

        selected_indices = None
        pair_colors = None
        if SHOW_PAIRS:
            np.random.seed(42)
            num_pairs_to_highlight = 5
            selected_indices = np.random.choice(
                num_hsc, size=num_pairs_to_highlight, replace=False
            )
            print(
                f"\n[PCA no-attn] Selected {num_pairs_to_highlight} random pairs to "
                f"highlight: indices {selected_indices}"
            )
            pair_colors = plt.cm.tab10(
                np.linspace(0, 1, num_pairs_to_highlight)
            )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Encoder 1 PCA
        ax1.scatter(
            hsc_embedding_1_pca[:, 0],
            hsc_embedding_1_pca[:, 1],
            s=5,
            label="HSC",
            alpha=0.6,
            c="blue",
        )
        ax1.scatter(
            legacy_embedding_1_pca[:, 0],
            legacy_embedding_1_pca[:, 1],
            s=5,
            label="Legacy",
            alpha=0.6,
            c="orange",
        )
        if SHOW_PAIRS and selected_indices is not None:
            for i, idx in enumerate(selected_indices):
                color = pair_colors[i]
                ax1.scatter(
                    hsc_embedding_1_pca[idx, 0],
                    hsc_embedding_1_pca[idx, 1],
                    marker="x",
                    s=200,
                    c=[color],
                    linewidths=3,
                    zorder=5,
                )
                ax1.scatter(
                    legacy_embedding_1_pca[idx, 0],
                    legacy_embedding_1_pca[idx, 1],
                    marker="x",
                    s=200,
                    c=[color],
                    linewidths=3,
                    zorder=5,
                )
        ax1.set_title(
            "Encoder 1 (Same Galaxy) - No-Attn PCA\n"
            f"Explained variance: {explained_variance_1.sum():.2%}"
        )
        ax1.set_xlabel(f"PC1 ({explained_variance_1[0]:.2%} var)")
        ax1.set_ylabel(f"PC2 ({explained_variance_1[1]:.2%} var)")
        ax1.legend()
        ax1.grid(True)

        # Encoder 2 PCA
        ax2.scatter(
            hsc_embedding_2_pca[:, 0],
            hsc_embedding_2_pca[:, 1],
            s=5,
            label="HSC",
            alpha=0.6,
            c="blue",
        )
        ax2.scatter(
            legacy_embedding_2_pca[:, 0],
            legacy_embedding_2_pca[:, 1],
            s=5,
            label="Legacy",
            alpha=0.6,
            c="orange",
        )
        if SHOW_PAIRS and selected_indices is not None:
            for i, idx in enumerate(selected_indices):
                color = pair_colors[i]
                ax2.scatter(
                    hsc_embedding_2_pca[idx, 0],
                    hsc_embedding_2_pca[idx, 1],
                    marker="x",
                    s=200,
                    c=[color],
                    linewidths=3,
                    zorder=5,
                )
                ax2.scatter(
                    legacy_embedding_2_pca[idx, 0],
                    legacy_embedding_2_pca[idx, 1],
                    marker="x",
                    s=200,
                    c=[color],
                    linewidths=3,
                    zorder=5,
                )
        ax2.set_title(
            "Encoder 2 (Same Instrument) - No-Attn PCA\n"
            f"Explained variance: {explained_variance_2.sum():.2%}"
        )
        ax2.set_xlabel(f"PC1 ({explained_variance_2[0]:.2%} var)")
        ax2.set_ylabel(f"PC2 ({explained_variance_2[1]:.2%} var)")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        zoom_suffix = "_zoom" if zoom else ""
        pca_path = (
            figures_dir
            / f"pca_both_encoders_zdim{dim}{mode_tag}{zoom_suffix}.png"
        )
        plt.savefig(pca_path, dpi=150)
        plt.close()

        print(f"\n[no-attn] Combined PCA plot saved to '{pca_path}'")
        print(f"  HSC points: {num_hsc}")
        print(f"  Legacy points: {len(legacy_embedding_1_pca)}")

    # ===== t-SNE =====
    if GENERATE_TSNE:
        tsne_params = {
            "n_components": 2,
            "perplexity": 30,
            "random_state": 42,
            "max_iter": 1000,
        }

        print("\nStarting t-SNE calculation for Encoder 1 (no-attn)...")
        print("  (This may take a while for large datasets)")
        tsne_1 = TSNE(**tsne_params)
        embedding_1_tsne = tsne_1.fit_transform(all_embeddings_1.cpu().numpy())
        hsc_embedding_1_tsne = embedding_1_tsne[:num_hsc]
        legacy_embedding_1_tsne = embedding_1_tsne[num_hsc:]

        print("\nStarting t-SNE calculation for Encoder 2 (no-attn)...")
        print("  (This may take a while for large datasets)")
        tsne_2 = TSNE(**tsne_params)
        embedding_2_tsne = tsne_2.fit_transform(all_embeddings_2.cpu().numpy())
        hsc_embedding_2_tsne = embedding_2_tsne[:num_hsc]
        legacy_embedding_2_tsne = embedding_2_tsne[num_hsc:]

        selected_indices = None
        pair_colors = None
        if SHOW_PAIRS:
            np.random.seed(42)
            num_pairs_to_highlight = 5
            selected_indices = np.random.choice(
                num_hsc, size=num_pairs_to_highlight, replace=False
            )
            print(
                f"\n[t-SNE no-attn] Selected {num_pairs_to_highlight} random pairs "
                f"to highlight: indices {selected_indices}"
            )
            pair_colors = plt.cm.tab10(
                np.linspace(0, 1, num_pairs_to_highlight)
            )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Encoder 1 t-SNE
        ax1.scatter(
            hsc_embedding_1_tsne[:, 0],
            hsc_embedding_1_tsne[:, 1],
            s=5,
            label="HSC",
            alpha=0.6,
            c="blue",
        )
        ax1.scatter(
            legacy_embedding_1_tsne[:, 0],
            legacy_embedding_1_tsne[:, 1],
            s=5,
            label="Legacy",
            alpha=0.6,
            c="orange",
        )
        if SHOW_PAIRS and selected_indices is not None:
            for i, idx in enumerate(selected_indices):
                color = pair_colors[i]
                ax1.scatter(
                    hsc_embedding_1_tsne[idx, 0],
                    hsc_embedding_1_tsne[idx, 1],
                    marker="x",
                    s=200,
                    c=[color],
                    linewidths=3,
                    zorder=5,
                )
                ax1.scatter(
                    legacy_embedding_1_tsne[idx, 0],
                    legacy_embedding_1_tsne[idx, 1],
                    marker="x",
                    s=200,
                    c=[color],
                    linewidths=3,
                    zorder=5,
                )
        ax1.set_title("Encoder 1 (Same Galaxy) - No-Attn t-SNE")
        ax1.set_xlabel("t-SNE Component 1")
        ax1.set_ylabel("t-SNE Component 2")
        ax1.legend()
        ax1.grid(True)

        # Encoder 2 t-SNE
        ax2.scatter(
            hsc_embedding_2_tsne[:, 0],
            hsc_embedding_2_tsne[:, 1],
            s=5,
            label="HSC",
            alpha=0.6,
            c="blue",
        )
        ax2.scatter(
            legacy_embedding_2_tsne[:, 0],
            legacy_embedding_2_tsne[:, 1],
            s=5,
            label="Legacy",
            alpha=0.6,
            c="orange",
        )
        if SHOW_PAIRS and selected_indices is not None:
            for i, idx in enumerate(selected_indices):
                color = pair_colors[i]
                ax2.scatter(
                    hsc_embedding_2_tsne[idx, 0],
                    hsc_embedding_2_tsne[idx, 1],
                    marker="x",
                    s=200,
                    c=[color],
                    linewidths=3,
                    zorder=5,
                )
                ax2.scatter(
                    legacy_embedding_2_tsne[idx, 0],
                    legacy_embedding_2_tsne[idx, 1],
                    marker="x",
                    s=200,
                    c=[color],
                    linewidths=3,
                    zorder=5,
                )
        ax2.set_title("Encoder 2 (Same Instrument) - No-Attn t-SNE")
        ax2.set_xlabel("t-SNE Component 1")
        ax2.set_ylabel("t-SNE Component 2")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        zoom_suffix = "_zoom" if zoom else ""
        tsne_path = (
            figures_dir
            / f"tsne_both_encoders_zdim{dim}{mode_tag}{zoom_suffix}.png"
        )
        plt.savefig(tsne_path, dpi=150)
        plt.close()

        print(f"\n[no-attn] Combined t-SNE plot saved to '{tsne_path}'")
        print(f"  HSC points: {num_hsc}")
        print(f"  Legacy points: {len(legacy_embedding_1_tsne)}")


if __name__ == "__main__":
    main()
