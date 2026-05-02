"""
Nearest-neighbor search in physics latent space for interesting gravitational lenses.

Encodes a set of lenses from lens_reconstruction_dataset.h5 with the trained model
and queries the combined HSC+Legacy gallery in physics space for each lens.
Produces one figure per lens in the style of neighbor_search/search_neighbors.py.

Lens indices are 1-based (as listed on the lens file). The script subtracts 1
internally to convert to 0-based dataset indices.

Run from galaxy_model/:
  python neighbor_search_lenses/search_lens_neighbors.py \
    --latents neighbor_search/neighbor_latents_2026-04-05.h5
"""
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

_here = Path(__file__).resolve().parent
_src = _here.parent  # galaxy_model/
_root = _src.parents[1]  # tess-generative/
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import argparse

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from sklearn.neighbors import NearestNeighbors

from galaxy_images.image_preprocessing import CenterCrop

# Force hipBLAS on MI210 (same as anomaly_detection scripts)
try:
    torch.backends.cuda.preferred_blas_library("hipblas")
except Exception:
    pass

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

NEIGHBORS_HDF5 = "/work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5"
DEFAULT_CHECKPOINT = str(
    _src / "outputs/neighbors_all_attn/2026-04-05/checkpoints/best-epoch=228-step=87000.ckpt"
)
DEFAULT_LENS_H5 = str(_src / "lense_reconstruction/lens_reconstruction_dataset.h5")
DEFAULT_LENS_INDICES = "5,8,12,18,20,29,32,33,40,41,44,46,48,49,53,63,64,66,67,68,70"
K_NEIGHBORS = 10
_CROP_SIZE = 64
_CROPPER = CenterCrop(crop_size=_CROP_SIZE)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_model(checkpoint_path: str, device: torch.device):
    """Load either base ConditionalFlowMatchingModule or hierarchical model.

    Tries the hierarchical class first (it's what hier-small / hier ckpts use);
    falls back to the baseline class for `base` checkpoints.
    """
    last_exc = None
    try:
        from galaxy_images.galaxy_model.hierarchical_attention.double_train_fm_neighbors_hier_global_ins import (
            HierarchicalGlobalInstrumentFlowMatchingModule,
        )
        model = HierarchicalGlobalInstrumentFlowMatchingModule.load_from_checkpoint(
            checkpoint_path, map_location="cpu"
        )
    except Exception as exc:
        last_exc = exc
        try:
            from double_train_fm_neighbors import ConditionalFlowMatchingModule
            model = ConditionalFlowMatchingModule.load_from_checkpoint(
                checkpoint_path, map_location="cpu"
            )
        except Exception as exc2:
            raise RuntimeError(
                f"Could not load checkpoint as hierarchical "
                f"({type(last_exc).__name__}: {last_exc}) or baseline "
                f"({type(exc2).__name__}: {exc2})."
            )
    model.eval()
    torch.set_grad_enabled(False)
    return model.to(device)


def _is_hierarchical(model) -> bool:
    return hasattr(model, "encode_image")


# ---------------------------------------------------------------------------
# Lens encoding
# ---------------------------------------------------------------------------


def _encode_lenses(
    model,
    device: torch.device,
    raw_hsc: np.ndarray,
    raw_legacy: np.ndarray,
    batch_size: int = 64,
    latent_mode: str = "combined",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode raw lens images (shape (n, C, H, W)) with model.encoder_1 (physics).
    Returns (hsc_physics, leg_physics) each of shape (n, D).

    For hierarchical models, `latent_mode` selects which slice of the physics
    latent to use:
      - "spatial_flat" -> physics["spatial_flat"]      (multi-level spatial tokens)
      - "global_vec"   -> physics["global_vec"]        (global conditioning vector)
      - "combined"     -> concat(global_vec, spatial_flat)  (default)

    For the baseline (non-hierarchical) model, `latent_mode` is ignored and the
    legacy `encoder_1(x).flatten(...)` path is used.
    """
    from neighbors import preprocess_raw_image

    use_hier = _is_hierarchical(model)
    if use_hier:
        from galaxy_images.galaxy_model.hierarchical_latent_experiments.common.latents import (
            extract_physics,
        )
        hier_variant = {
            "spatial_flat": "spatial_flat",
            "global_vec": "global_vec",
            "combined": "global_concat",
        }[latent_mode]

    n = raw_hsc.shape[0]
    hsc_tensors = torch.stack([
        preprocess_raw_image(raw_hsc[i], "hsc", 48)[:4]
        for i in range(n)
    ])  # (n, 4, 48, 48)
    leg_tensors = torch.stack([
        preprocess_raw_image(raw_legacy[i], "legacy", 48)[:4]
        for i in range(n)
    ])  # (n, 4, 48, 48)

    hsc_phys_list, leg_phys_list = [], []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            h_batch = hsc_tensors[start:end].to(device)
            l_batch = leg_tensors[start:end].to(device)
            if use_hier:
                hp = extract_physics(model, h_batch, hier_variant).cpu()
                lp = extract_physics(model, l_batch, hier_variant).cpu()
                if hp.dim() > 2:
                    hp = hp.flatten(start_dim=1); lp = lp.flatten(start_dim=1)
            else:
                hp = model.encoder_1(h_batch).cpu().flatten(start_dim=1)
                lp = model.encoder_1(l_batch).cpu().flatten(start_dim=1)
            hsc_phys_list.append(hp); leg_phys_list.append(lp)

    hsc_phys = torch.cat(hsc_phys_list, dim=0).numpy().astype(np.float32)
    leg_phys = torch.cat(leg_phys_list, dim=0).numpy().astype(np.float32)
    return hsc_phys, leg_phys


# ---------------------------------------------------------------------------
# Image loading for display
# ---------------------------------------------------------------------------


def _center_crop_tensor(t: torch.Tensor) -> torch.Tensor:
    return _CROPPER(t.unsqueeze(0)).squeeze(0)


def _load_lens_display(lens_h5_file: h5py.File, lens_0idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load and crop lens display images (64×64) from lens_reconstruction_dataset.h5."""
    img_hsc = torch.from_numpy(lens_h5_file["images_hsc"][lens_0idx]).float()
    img_leg = torch.from_numpy(lens_h5_file["images_legacy"][lens_0idx]).float()
    return _center_crop_tensor(img_hsc), _center_crop_tensor(img_leg)


def _load_nn_display(
    nb_h5_file: h5py.File,
    raw_h5_row: int,
    source: str,
) -> torch.Tensor:
    """Load and crop a gallery image (64×64) by raw h5 row and survey."""
    key = "images_hsc" if source == "hsc" else "images_legacy"
    img = torch.from_numpy(nb_h5_file[key][raw_h5_row]).float()
    return _center_crop_tensor(img)


# ---------------------------------------------------------------------------
# Visualization helpers (matching neighbors_plot.py)
# ---------------------------------------------------------------------------


def tensor_to_rgb(tensor: torch.Tensor, channels: Sequence[int] = (0, 1, 2), percentile_clip: float = 99.5) -> np.ndarray:
    c_indices = list(channels)
    rgb = tensor[c_indices].cpu().numpy()   # (3, H, W)
    rgb = np.transpose(rgb, (1, 2, 0))      # (H, W, 3)
    for i in range(3):
        p_low = np.percentile(rgb[:, :, i], 100 - percentile_clip)
        p_high = np.percentile(rgb[:, :, i], percentile_clip)
        rgb[:, :, i] = np.clip(rgb[:, :, i], p_low, p_high)
    for i in range(3):
        ch = rgb[:, :, i]
        ch_min, ch_max = ch.min(), ch.max()
        rgb[:, :, i] = (ch - ch_min) / (ch_max - ch_min) if ch_max > ch_min else 0.0
    return rgb


def _show_img(ax, t: torch.Tensor, title: Optional[str] = None, title_color: str = "black",
              is_lens: bool = False):
    try:
        n_ch = t.shape[0]
        channels = (0, 1, 2) if n_ch >= 3 else (0, 0, 0)
        ax.imshow(tensor_to_rgb(t, channels))
    except Exception:
        x = t[0].numpy()
        ax.imshow((x - x.min()) / (x.max() - x.min() + 1e-8), cmap="gray")
    if title:
        ax.set_title(title, fontsize=8, color=title_color, fontweight="bold")
    ax.set_axis_off()
    if is_lens:
        rect = Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                         fill=False, edgecolor="royalblue", linewidth=4, clip_on=False)
        ax.add_patch(rect)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

N_COLS = 12   # col 2 = Query HSC, col 3 = Query Legacy; cols 1..10 = NNs
N_ROWS = 3    # row 0 = query; row 1 = HSC-phys NNs; row 2 = Leg-phys NNs


def plot_lens_neighbors(
    query_hsc: torch.Tensor,
    query_legacy: torch.Tensor,
    hsc_nns: List[Tuple[int, str, int]],   # (combined_pos, survey, raw_h5_row) × k
    leg_nns: List[Tuple[int, str, int]],
    hsc_nn_images: List[torch.Tensor],
    leg_nn_images: List[torch.Tensor],
    user_idx: int,
    lens_object_id: str,
    lens_gallery_pos: Optional[int],        # HSC gallery position of this lens (counterpart detection)
    gallery_index_mmu: np.ndarray,          # full combined index_mmu (hsc then legacy)
    all_lens_h5_indices: set,               # set of raw h5 rows that are known lenses
    out_path: Path,
):
    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(N_COLS * 1.8, N_ROWS * 1.8))
    for ax in axes.flat:
        ax.set_axis_off()

    # Row 0: query pair (matching search_neighbors.py placement at cols 2–3)
    _show_img(axes[0, 2], query_hsc, "Query HSC")
    _show_img(axes[0, 3], query_legacy, "Query Legacy")

    def src_label(s: str) -> str:
        return "(HSC)" if s == "hsc" else "(Leg)"

    def nn_color(combined_pos: int, source: str, query_source: str) -> str:
        if lens_gallery_pos is not None:
            if combined_pos == lens_gallery_pos and source != query_source:
                return "red"   # direct counterpart (same galaxy, other survey)
        if source != query_source:
            return "gold"      # cross-survey
        return "black"

    # Row 1: physics NNs from HSC query
    for j, (cpos, nn_src, raw_row) in enumerate(hsc_nns):
        col = 1 + j
        if col >= N_COLS:
            break
        color = nn_color(cpos, nn_src, "hsc")
        is_lens = raw_row in all_lens_h5_indices
        _show_img(
            axes[1, col],
            hsc_nn_images[j],
            f"HSC phys kNN {j+1} {src_label(nn_src)}",
            title_color=color,
            is_lens=is_lens,
        )
        axes[1, col].set_xlabel(f"h5row {raw_row}", fontsize=7)

    # Row 2: physics NNs from Legacy query
    for j, (cpos, nn_src, raw_row) in enumerate(leg_nns):
        col = 1 + j
        if col >= N_COLS:
            break
        color = nn_color(cpos, nn_src, "legacy")
        is_lens = raw_row in all_lens_h5_indices
        _show_img(
            axes[2, col],
            leg_nn_images[j],
            f"Leg phys kNN {j+1} {src_label(nn_src)}",
            title_color=color,
            is_lens=is_lens,
        )
        axes[2, col].set_xlabel(f"h5row {raw_row}", fontsize=7)

    # Separator line between query row and NN rows
    fig.canvas.draw()
    left = min(ax.get_position().x0 for ax in axes[0, :])
    right = max(ax.get_position().x1 for ax in axes[0, :])
    sep_y = (axes[0, 0].get_position().y0 + axes[1, 0].get_position().y1) / 2
    fig.add_artist(Line2D([left, right], [sep_y, sep_y],
                          transform=fig.transFigure, linewidth=1.5, color="black", linestyle="--"))

    fig.suptitle(
        f"Lens {user_idx} ({lens_object_id})  —  physics kNN top-{K_NEIGHBORS} in combined HSC+Legacy space\n"
        "gold = cross-survey  |  red = direct counterpart  |  black = same survey  |  blue border = known lens",
        fontsize=9,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Physics-space kNN search for interesting gravitational lenses."
    )
    p.add_argument(
        "--latents",
        type=Path,
        default=None,
        help="Path to neighbor_latents_*.h5 (auto-detected from neighbor_search/ if omitted)",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Model checkpoint for encoding lenses",
    )
    p.add_argument(
        "--lens-h5",
        type=Path,
        default=DEFAULT_LENS_H5,
        help="Path to lens_reconstruction_dataset.h5",
    )
    p.add_argument(
        "--neighbors-h5",
        type=Path,
        default=NEIGHBORS_HDF5,
        help="Path to neighbours_v2.h5 (gallery images)",
    )
    p.add_argument(
        "--lens-indices",
        type=str,
        default=DEFAULT_LENS_INDICES,
        help="Comma-separated 1-based lens indices",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_here / "outputs",
        help="Output directory for figures",
    )
    p.add_argument("--k", type=int, default=K_NEIGHBORS, help="Number of NNs per lens")
    p.add_argument("--device", type=str, default=None, help="Device (default: cuda if available)")
    p.add_argument("--batch-size", type=int, default=64, help="Encoding batch size")
    p.add_argument("--latent-mode", choices=["spatial_flat", "global_vec", "combined"],
                   default="combined",
                   help="Hierarchical physics-latent slice. Ignored for non-hierarchical models.")
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve latents path
    latents_path = args.latents
    if latents_path is None:
        candidates = list((_src / "neighbor_search").glob("neighbor_latents_*.h5"))
        if len(candidates) == 1:
            latents_path = candidates[0]
            print(f"Using gallery latents: {latents_path}")
        elif not candidates:
            raise FileNotFoundError(
                "No neighbor_latents_*.h5 in neighbor_search/. "
                "Run neighbor_search/make_latents_all.py first (see run_lens_neighbors.sh)."
            )
        else:
            raise FileNotFoundError(
                f"Multiple latent files found: {candidates}. Pass --latents explicitly."
            )
    if not latents_path.is_file():
        raise FileNotFoundError(f"Latents file not found: {latents_path}")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    # Parse 1-based lens indices
    user_indices = [int(x) for x in args.lens_indices.split(",") if x.strip()]
    lens_0indices = [u - 1 for u in user_indices]
    print(f"Processing {len(user_indices)} lenses (1-based): {user_indices}")

    # --- Load gallery latents (supports both original and extended format) ---
    print(f"Loading gallery latents from {latents_path} ...")
    with h5py.File(latents_path, "r") as f:
        if "hsc_index_mmu" in f:
            # Extended format: asymmetric HSC / Legacy arrays
            hsc_index_mmu = f["hsc_index_mmu"][:]       # (N_hsc,)
            leg_index_mmu = f["legacy_index_mmu"][:]    # (N_leg,)
            hsc_phys_gallery = f["hsc_physics"][:]      # (N_hsc, D)
            leg_phys_gallery  = f["legacy_physics"][:]  # (N_leg, D)
            fmt = "extended"
        else:
            # Original format: symmetric, N_hsc == N_leg
            hsc_index_mmu = f["index_mmu"][:]
            leg_index_mmu = hsc_index_mmu           # same rows for both surveys
            hsc_phys_gallery = f["physics_embedding"][:]
            if "legacy_physics_embedding" not in f:
                raise KeyError("legacy_physics_embedding not found; re-run make_latents_all.py")
            leg_phys_gallery = f["legacy_physics_embedding"][:]
            fmt = "original"

    N_hsc = len(hsc_index_mmu)
    N_leg = len(leg_index_mmu)
    print(f"Gallery: {N_hsc:,} HSC + {N_leg:,} Legacy = {N_hsc+N_leg:,} combined  [{fmt} format]")
    if hsc_phys_gallery.shape[1] != leg_phys_gallery.shape[1]:
        raise ValueError(
            f"Gallery dim mismatch: HSC={hsc_phys_gallery.shape[1]}, Legacy={leg_phys_gallery.shape[1]}"
        )

    combined_physics = np.concatenate([hsc_phys_gallery, leg_phys_gallery], axis=0)
    gallery_index_mmu = np.concatenate([hsc_index_mmu, leg_index_mmu], axis=0)  # (N_hsc+N_leg,)

    def pos_to_survey_and_raw_row(pos: int):
        """Map combined-space position to (survey, raw_h5_row)."""
        if pos < N_hsc:
            return "hsc", int(hsc_index_mmu[pos])
        return "legacy", int(leg_index_mmu[pos - N_hsc])

    k_query = args.k + 2  # extra slots for potential self-matches
    print(f"Fitting NearestNeighbors (k={k_query}) on combined physics space ...")
    nn_phys = NearestNeighbors(
        n_neighbors=min(k_query, N_hsc + N_leg), metric="euclidean"
    ).fit(combined_physics)

    # Lookup: raw h5 row → HSC gallery position (for counterpart detection)
    h5idx_to_hsc_pos = {int(h): i for i, h in enumerate(hsc_index_mmu)}

    # --- Load lens dataset ---
    print(f"Loading lens dataset from {args.lens_h5} ...")
    with h5py.File(args.lens_h5, "r") as f:
        n_lenses_total = f["h5_index"].shape[0]
        for idx0 in lens_0indices:
            if idx0 < 0 or idx0 >= n_lenses_total:
                raise IndexError(
                    f"Lens 0-index {idx0} (user index {idx0+1}) out of range "
                    f"[0, {n_lenses_total-1}] (lens dataset has {n_lenses_total} entries)."
                )
        lens_h5_indices = f["h5_index"][:]        # (n_lenses,) raw h5 rows
        lens_object_ids = f["object_id_hsc"][:]   # (n_lenses,) bytes
        raw_hsc_all = f["images_hsc"][:]          # (n_lenses, 5, H, W)
        raw_leg_all = f["images_legacy"][:]       # (n_lenses, 4, H, W)

    # Set of all known-lens raw h5 rows (for blue-border cross-matching)
    all_lens_h5_indices = set(int(x) for x in lens_h5_indices)

    # --- Encode all lenses with model ---
    print(f"Loading model from {args.checkpoint} ...")
    model = _load_model(args.checkpoint, device)
    print(f"Encoding lenses (physics space, latent_mode={args.latent_mode}) ...")
    lens_hsc_phys, lens_leg_phys = _encode_lenses(
        model, device, raw_hsc_all, raw_leg_all, batch_size=args.batch_size,
        latent_mode=args.latent_mode,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"Encoded {len(lens_hsc_phys)} lenses; embedding dim = {lens_hsc_phys.shape[1]}")
    if lens_hsc_phys.shape[1] != combined_physics.shape[1]:
        raise ValueError(
            f"Lens embedding dim {lens_hsc_phys.shape[1]} does not match gallery dim "
            f"{combined_physics.shape[1]}. The gallery was built with a different "
            f"checkpoint or latent_mode. Rebuild the gallery with make_latents_hier.py "
            f"using --latent-mode {args.latent_mode}."
        )

    # --- Process each lens ---
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSearching NNs and plotting to {args.out_dir} ...")

    with h5py.File(args.neighbors_h5, "r") as nb_h5, \
         h5py.File(args.lens_h5, "r") as lens_h5:

        for user_idx, lens_0idx in zip(user_indices, lens_0indices):
            print(f"  Lens {user_idx} (0-idx={lens_0idx}) ...")

            obj_id = lens_object_ids[lens_0idx]
            if isinstance(obj_id, bytes):
                obj_id = obj_id.decode("utf-8")

            lens_gallery_pos = h5idx_to_hsc_pos.get(int(lens_h5_indices[lens_0idx]), None)

            # Query from HSC embedding
            q_hsc = lens_hsc_phys[lens_0idx:lens_0idx+1]
            dists_hsc, inds_hsc = nn_phys.kneighbors(q_hsc, n_neighbors=k_query)
            hsc_nns = _filter_nns(inds_hsc[0], dists_hsc[0], pos_to_survey_and_raw_row, args.k)

            # Query from Legacy embedding
            q_leg = lens_leg_phys[lens_0idx:lens_0idx+1]
            dists_leg, inds_leg = nn_phys.kneighbors(q_leg, n_neighbors=k_query)
            leg_nns = _filter_nns(inds_leg[0], dists_leg[0], pos_to_survey_and_raw_row, args.k)

            # Load query display images
            query_hsc_img, query_leg_img = _load_lens_display(lens_h5, lens_0idx)

            # Load NN display images
            hsc_nn_images = [_load_nn_display(nb_h5, raw_row, src) for _, src, raw_row in hsc_nns]
            leg_nn_images = [_load_nn_display(nb_h5, raw_row, src) for _, src, raw_row in leg_nns]

            out_path = args.out_dir / f"lens_{user_idx:03d}_neighbors.png"
            plot_lens_neighbors(
                query_hsc_img,
                query_leg_img,
                hsc_nns,
                leg_nns,
                hsc_nn_images,
                leg_nn_images,
                user_idx=user_idx,
                lens_object_id=obj_id,
                lens_gallery_pos=lens_gallery_pos,
                gallery_index_mmu=gallery_index_mmu,
                all_lens_h5_indices=all_lens_h5_indices,
                out_path=out_path,
            )

    print(f"\nDone. Figures saved to {args.out_dir}")


def _filter_nns(
    positions: np.ndarray,
    distances: np.ndarray,
    pos_to_survey_and_raw_row,
    k: int,
) -> List[Tuple[int, str, int]]:
    """Return list of (combined_pos, survey, raw_h5_row), excluding self-matches (dist≈0)."""
    result: List[Tuple[int, str, int]] = []
    for pos, dist in zip(positions, distances):
        if dist < 1e-6:
            continue
        survey, raw_row = pos_to_survey_and_raw_row(int(pos))
        result.append((int(pos), survey, raw_row))
        if len(result) == k:
            break
    return result


if __name__ == "__main__":
    main()
