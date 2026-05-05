"""
HSC-only and HSC+Legacy lens kNN search comparing our model vs AION (Lens 48 default).

Two modes:
  hsc_only  — gallery = all HSC images (source_type∈{0,1}, ~366k)
  combined  — gallery = HSC + Legacy images (~571k)

Each mode produces one figure with 4 result rows:
  Ours · L2  |  Ours · Cosine  |  AION · L2  |  AION · Cosine

Run from galaxy_model/:
  python neighbor_search_lenses/search_lens_hsc_aion.py --mode hsc_only --lens-index 48
  python neighbor_search_lenses/search_lens_hsc_aion.py --mode combined --lens-index 48
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple

_here = Path(__file__).resolve().parent
_src = _here.parent
_root = _src.parents[1]
for _p in [str(_src), str(_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from sklearn.neighbors import NearestNeighbors

from galaxy_images.image_preprocessing import CenterCrop

try:
    torch.backends.cuda.preferred_blas_library("hipblas")
except Exception:
    pass

NEIGHBORS_HDF5 = "/work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5"
DEFAULT_CHECKPOINT = str(
    _src / "outputs/neighbors_all_attn/2026-04-05/checkpoints/best-epoch=228-step=87000.ckpt"
)
DEFAULT_LENS_H5 = str(_src / "lense_reconstruction/lens_reconstruction_dataset.h5")
DEFAULT_OURS_LATENTS = str(_src / "neighbor_search/neighbor_latents_extended.h5")
DEFAULT_AION_HSC = str(_src / "anomaly_detection/outputs/anomaly_latents_aion_hsc_extended.h5")
DEFAULT_AION_LEGACY = str(_src / "anomaly_detection/outputs/anomaly_latents_aion_legacy_extended.h5")
# overlap-only files for combined mode (source_type=0 only, ~103k each)
DEFAULT_AION_OVERLAP_HSC = str(_src / "anomaly_detection/outputs/anomaly_latents_aion_overlap_hsc.h5")
DEFAULT_AION_OVERLAP_LEGACY = str(_src / "anomaly_detection/outputs/anomaly_latents_aion_overlap_legacy.h5")
K_NEIGHBORS = 10
_CROP_SIZE = 64
_CROPPER = CenterCrop(crop_size=_CROP_SIZE)

N_COLS = 11   # col 0 = row label, cols 1-10 = NNs
N_ROWS = 5    # row 0 = query, rows 1-4 = results
ROW_LABELS = ["", "Ours · L2", "Ours · Cos", "AION · L2", "AION · Cos"]

# ---------------------------------------------------------------------------
# Image helpers (adapted from search_lens_neighbors.py)
# ---------------------------------------------------------------------------


def tensor_to_rgb(tensor, channels=(0, 1, 2), percentile_clip=99.5):
    rgb = tensor[list(channels)].cpu().numpy().transpose(1, 2, 0)
    for i in range(3):
        p_low = np.percentile(rgb[:, :, i], 100 - percentile_clip)
        p_high = np.percentile(rgb[:, :, i], percentile_clip)
        rgb[:, :, i] = np.clip(rgb[:, :, i], p_low, p_high)
    for i in range(3):
        ch = rgb[:, :, i]
        mn, mx = ch.min(), ch.max()
        rgb[:, :, i] = (ch - mn) / (mx - mn) if mx > mn else 0.0
    return rgb


def _center_crop(t):
    return _CROPPER(t.unsqueeze(0)).squeeze(0)


def _show_img(ax, t, title=None, title_color="black", is_lens=False):
    try:
        ch = (0, 1, 2) if t.shape[0] >= 3 else (0, 0, 0)
        ax.imshow(tensor_to_rgb(t, ch))
    except Exception:
        x = t[0].numpy()
        ax.imshow((x - x.min()) / (x.max() - x.min() + 1e-8), cmap="gray")
    if title:
        ax.set_title(title, fontsize=7, color=title_color, fontweight="bold")
    ax.set_axis_off()
    if is_lens:
        rect = Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                         fill=False, edgecolor="royalblue", linewidth=4, clip_on=False)
        ax.add_patch(rect)


def _load_nn_display(nb_h5, raw_h5_row, source):
    key = "images_hsc" if source == "hsc" else "images_legacy"
    img = torch.from_numpy(nb_h5[key][raw_h5_row]).float()
    return _center_crop(img)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_our_model(checkpoint_path, device):
    try:
        from galaxy_images.galaxy_model.hierarchical_attention.double_train_fm_neighbors_hier_global_ins import (
            HierarchicalGlobalInstrumentFlowMatchingModule,
        )
        model = HierarchicalGlobalInstrumentFlowMatchingModule.load_from_checkpoint(
            checkpoint_path, map_location="cpu"
        )
    except Exception:
        from double_train_fm_neighbors import ConditionalFlowMatchingModule
        model = ConditionalFlowMatchingModule.load_from_checkpoint(
            checkpoint_path, map_location="cpu"
        )
    model.eval()
    torch.set_grad_enabled(False)
    return model.to(device)


# ---------------------------------------------------------------------------
# Lens encoding
# ---------------------------------------------------------------------------


def encode_lens_ours(raw_hsc, model, device):
    from neighbors import preprocess_raw_image
    img = preprocess_raw_image(raw_hsc, "hsc", 48)[:4].unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encoder_1(img).cpu().flatten(start_dim=1)
    return emb.numpy().astype(np.float32)  # (1, D)


def encode_lens_aion(raw_hsc_np, device_str):
    """Encode a single lens HSC image with AION; returns (1, D) float32."""
    from aion import AION
    from aion.codecs import CodecManager
    from aion.modalities import HSCImage

    device = torch.device(device_str)
    print("Loading AION model for lens encoding ...")
    aion_model = AION.from_pretrained("polymathic-ai/aion-base").to(device)
    codec_manager = CodecManager(device=device_str)
    aion_model.eval()

    t = torch.from_numpy(raw_hsc_np.astype(np.float32))
    if t.dim() == 3:
        t = t.unsqueeze(0)  # (1, 5, H, W)
    t = t.to(device)
    image = HSCImage(flux=t, bands=["HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y"])
    tokens = codec_manager.encode(image)
    with torch.no_grad():
        emb = aion_model.encode(tokens)  # no token limit — average all
    result = emb.mean(dim=1).cpu().numpy().astype(np.float32)  # (1, D)

    del aion_model, codec_manager
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------
# Gallery loading
# ---------------------------------------------------------------------------


def load_ours_gallery(latents_path, mode):
    """Returns (gallery_emb, pos_to_survey_row_fn).

    hsc_only: all source_type∈{0,1} HSC images (~366k)
    combined: source_type=0 only (overlap where both surveys available, ~103k each)
    """
    with h5py.File(latents_path, "r") as f:
        hsc_index = f["hsc_index_mmu"][:]
        hsc_phys = f["hsc_physics"][:]
        hsc_st = f["hsc_source_type"][:]
        if mode == "combined":
            leg_index = f["legacy_index_mmu"][:]
            leg_phys = f["legacy_physics"][:]
            leg_st = f["legacy_source_type"][:]

    if mode == "hsc_only":
        _hi = hsc_index

        def p2sr(pos):
            return "hsc", int(_hi[pos])

        return hsc_phys, p2sr

    # combined: filter to source_type=0 (overlap only)
    hsc_mask = (hsc_st == 0)
    leg_mask = (leg_st == 0)
    hsc_phys_ov = hsc_phys[hsc_mask]
    hsc_idx_ov = hsc_index[hsc_mask]
    leg_phys_ov = leg_phys[leg_mask]
    leg_idx_ov = leg_index[leg_mask]
    print(f"  Ours combined (overlap): {len(hsc_idx_ov):,} HSC + {len(leg_idx_ov):,} Legacy")

    gallery = np.concatenate([hsc_phys_ov, leg_phys_ov], axis=0)
    N_hsc = len(hsc_idx_ov)
    _hi, _li = hsc_idx_ov, leg_idx_ov

    def p2sr(pos):
        if pos < N_hsc:
            return "hsc", int(_hi[pos])
        return "legacy", int(_li[pos - N_hsc])

    return gallery, p2sr


def load_aion_gallery(hsc_path, leg_path, mode):
    """Returns (gallery_emb, pos_to_survey_row_fn).

    hsc_only: hsc_path contains all HSC-valid AION embeddings (~366k)
    combined: hsc_path + leg_path each contain overlap-only embeddings (~103k each)
    """
    with h5py.File(hsc_path, "r") as f:
        aion_hsc_idx = f["raw_index"][:]
        aion_hsc_emb = f["embeddings_mean_hsc"][:]

    if mode == "hsc_only":
        _ai = aion_hsc_idx

        def p2sr(pos):
            return "hsc", int(_ai[pos])

        return aion_hsc_emb, p2sr

    with h5py.File(leg_path, "r") as f:
        aion_leg_idx = f["raw_index"][:]
        aion_leg_emb = f["embeddings_mean_legacy"][:]

    gallery = np.concatenate([aion_hsc_emb, aion_leg_emb], axis=0)
    N_hsc = len(aion_hsc_idx)
    _ai, _li = aion_hsc_idx, aion_leg_idx

    def p2sr(pos):
        if pos < N_hsc:
            return "hsc", int(_ai[pos])
        return "legacy", int(_li[pos - N_hsc])

    return gallery, p2sr


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def _filter_nns(positions, distances, p2sr, k, skip_raw_row=None):
    result = []
    for pos, dist in zip(positions, distances):
        if dist < 1e-6:
            continue
        survey, raw_row = p2sr(int(pos))
        if skip_raw_row is not None and raw_row == skip_raw_row:
            continue
        result.append((int(pos), survey, raw_row))
        if len(result) == k:
            break
    return result


def search(gallery, query, p2sr, k, metric, skip_raw_row=None):
    k_query = min(k + 4, len(gallery))
    nn = NearestNeighbors(n_neighbors=k_query, metric=metric, algorithm="auto")
    nn.fit(gallery)
    dists, inds = nn.kneighbors(query, n_neighbors=k_query)
    return _filter_nns(inds[0], dists[0], p2sr, k, skip_raw_row)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_figure(query_hsc_img, result_rows, result_images,
                user_idx, obj_id, all_lens_h5_indices, out_path, mode, k):
    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(N_COLS * 1.8, N_ROWS * 1.8))
    for ax in axes.flat:
        ax.set_axis_off()

    # Row 0: query image centered at col 5
    _show_img(axes[0, 5], query_hsc_img, "Query HSC")

    # Row label cells (col 0, rows 1-4)
    for row_i, label in enumerate(ROW_LABELS):
        if row_i == 0 or not label:
            continue
        ax = axes[row_i, 0]
        ax.text(0.5, 0.5, label, transform=ax.transAxes,
                ha="center", va="center", fontsize=8, fontweight="bold", rotation=90)

    # Result rows: cols 1-10
    for row_i, (nns, imgs) in enumerate(zip(result_rows, result_images), start=1):
        for j, ((_, src, raw_row), img) in enumerate(zip(nns, imgs)):
            col = 1 + j
            if col >= N_COLS:
                break
            is_lens = raw_row in all_lens_h5_indices
            src_label = "(H)" if src == "hsc" else "(L)"
            _show_img(axes[row_i, col], img,
                      title=f"#{j+1} {src_label}",
                      title_color="black",
                      is_lens=is_lens)
            axes[row_i, col].set_xlabel(f"h5row {raw_row}", fontsize=6)

    # Separator lines
    fig.canvas.draw()
    x0 = min(ax.get_position().x0 for ax in axes[0, :])
    x1 = max(ax.get_position().x1 for ax in axes[0, :])

    def add_sep(ra, rb, style="--", lw=1.5, color="black"):
        y = (axes[ra, 0].get_position().y0 + axes[rb, 0].get_position().y1) / 2
        fig.add_artist(Line2D([x0, x1], [y, y], transform=fig.transFigure,
                               linewidth=lw, color=color, linestyle=style))

    add_sep(0, 1)        # query → results
    add_sep(2, 3, lw=2)  # ours → AION

    mode_label = "HSC-only" if mode == "hsc_only" else "HSC+Legacy combined"
    fig.suptitle(
        f"Lens {user_idx} ({obj_id})  —  HSC query · top-{k} kNN · {mode_label}\n"
        "blue border = known lens  |  (H) = HSC gallery  |  (L) = Legacy gallery",
        fontsize=9,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["hsc_only", "combined"], required=True)
    p.add_argument("--lens-index", type=int, default=48, help="1-based lens index")
    p.add_argument("--ours-latents", type=Path, default=DEFAULT_OURS_LATENTS)
    # hsc_only mode: pass the full HSC-extended AION file
    # combined mode: pass the overlap-only HSC and Legacy AION files
    p.add_argument("--aion-hsc", type=Path, default=None,
                   help="AION HSC gallery file (default: hsc_extended for hsc_only, overlap_hsc for combined)")
    p.add_argument("--aion-legacy", type=Path, default=None,
                   help="AION Legacy gallery file (overlap_legacy for combined; unused for hsc_only)")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--lens-h5", type=Path, default=DEFAULT_LENS_H5)
    p.add_argument("--neighbors-h5", type=Path, default=NEIGHBORS_HDF5)
    p.add_argument("--k", type=int, default=K_NEIGHBORS)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    out_dir = args.out_dir or (_here / f"outputs_{args.mode}")
    lens_0idx = args.lens_index - 1

    # Resolve AION file defaults based on mode
    aion_hsc_path = args.aion_hsc or Path(
        DEFAULT_AION_HSC if args.mode == "hsc_only" else DEFAULT_AION_OVERLAP_HSC
    )
    aion_leg_path = args.aion_legacy or Path(DEFAULT_AION_OVERLAP_LEGACY)

    print(f"Mode: {args.mode}  |  Lens: {args.lens_index}  |  Device: {device_str}")

    # Validate AION latent files exist
    if not aion_hsc_path.is_file():
        raise FileNotFoundError(
            f"AION HSC latents not found: {aion_hsc_path}\n"
            "Run encode_latents_aion_extended.py with the appropriate --survey and --suffix."
        )
    if args.mode == "combined" and not aion_leg_path.is_file():
        raise FileNotFoundError(
            f"AION Legacy latents not found: {aion_leg_path}\n"
            "Run encode_latents_aion_extended.py --survey legacy --source-types 0 --suffix overlap_legacy"
        )

    # Load galleries
    print(f"Loading our model gallery from {args.ours_latents} ...")
    ours_gallery, ours_p2sr = load_ours_gallery(args.ours_latents, args.mode)
    print(f"  Ours gallery: {ours_gallery.shape}")

    print("Loading AION gallery ...")
    aion_gallery, aion_p2sr = load_aion_gallery(aion_hsc_path, aion_leg_path, args.mode)
    print(f"  AION gallery: {aion_gallery.shape}")

    # Load lens dataset
    print(f"Loading lens dataset from {args.lens_h5} ...")
    with h5py.File(args.lens_h5, "r") as f:
        lens_h5_indices = f["h5_index"][:]
        lens_object_ids = f["object_id_hsc"][:]
        raw_hsc_all = f["images_hsc"][:]

    obj_id = lens_object_ids[lens_0idx]
    if isinstance(obj_id, bytes):
        obj_id = obj_id.decode()
    lens_raw_h5_row = int(lens_h5_indices[lens_0idx])
    all_lens_h5_indices = set(int(x) for x in lens_h5_indices)
    print(f"  Lens {args.lens_index}: {obj_id}  (h5_row={lens_raw_h5_row})")

    # Encode with our model
    print(f"Loading our model from {args.checkpoint} ...")
    our_model = _load_our_model(args.checkpoint, device)
    print("Encoding lens HSC with our model ...")
    q_ours = encode_lens_ours(raw_hsc_all[lens_0idx], our_model, device)  # (1, 64)
    del our_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Encode with AION
    print("Encoding lens HSC with AION ...")
    q_aion = encode_lens_aion(raw_hsc_all[lens_0idx], device_str)  # (1, 768)

    # Search all 4 configurations
    print("Searching our model gallery (L2 + Cosine) ...")
    nns_ours_l2  = search(ours_gallery, q_ours, ours_p2sr, args.k, "euclidean", lens_raw_h5_row)
    nns_ours_cos = search(ours_gallery, q_ours, ours_p2sr, args.k, "cosine",    lens_raw_h5_row)

    print("Searching AION gallery (L2 + Cosine) ...")
    nns_aion_l2  = search(aion_gallery, q_aion, aion_p2sr, args.k, "euclidean", lens_raw_h5_row)
    nns_aion_cos = search(aion_gallery, q_aion, aion_p2sr, args.k, "cosine",    lens_raw_h5_row)

    all_nns = [nns_ours_l2, nns_ours_cos, nns_aion_l2, nns_aion_cos]

    # Load query display image (64×64 center crop)
    with h5py.File(args.lens_h5, "r") as f:
        query_hsc_img = _center_crop(torch.from_numpy(f["images_hsc"][lens_0idx]).float())

    # Load NN display images
    print("Loading NN display images ...")
    all_nn_images = []
    with h5py.File(args.neighbors_h5, "r") as nb_h5:
        for nns in all_nns:
            imgs = [_load_nn_display(nb_h5, raw_row, src) for _, src, raw_row in nns]
            all_nn_images.append(imgs)

    # Plot
    out_path = out_dir / f"lens_{args.lens_index:03d}_hsc_aion_{args.mode}.png"
    plot_figure(
        query_hsc_img=query_hsc_img,
        result_rows=all_nns,
        result_images=all_nn_images,
        user_idx=args.lens_index,
        obj_id=obj_id,
        all_lens_h5_indices=all_lens_h5_indices,
        out_path=out_path,
        mode=args.mode,
        k=args.k,
    )
    print(f"\nDone. Figure: {out_path}")


if __name__ == "__main__":
    main()
