"""
neighbors_plot.py

Custom visualization for galaxy query neighbors.
Layout:
  Row 1: HSC Query | (Line) | HSC Phys NNs (1-4) | (Line) | HSC Inst NNs (1-3)
  Row 2: Leg Query | (Line) | Leg Phys NNs (1,2,5,7)| (Line) | Leg Inst NNs (1-3)

Usage:
  python neighbor_search/neighbors_plot.py --index 172 --out query_172_custom.png
"""

import sys
from pathlib import Path
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from sklearn.neighbors import NearestNeighbors

# --- Path Setup ---
_here = Path(__file__).resolve().parent
_src = _here.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

# --- Constants ---
NEIGHBORS_HDF5 = "/data/vision/billf/scratch/pablomer/data/neighbours_v2.h5"
K_FETCH = 30
BAR_COLORS = ["#2E86AB", "#CC546D", "#6CA75D", "#9473C6", "#BF823B"]


def load_images_for_indices(indices, hdf5_path=NEIGHBORS_HDF5):
    """Load (HSC, Legacy) preprocessed images for dataset indices."""
    from neighbors import NeighborsSimpleDataset
    dataset = NeighborsSimpleDataset(hdf5_path=hdf5_path)
    hsc_list, leg_list = [], []
    for idx in indices:
        hsc, leg, _ = dataset[idx]
        hsc_list.append(hsc)
        leg_list.append(leg)
    return hsc_list, leg_list

def load_images_for_neighbors(neighbor_list, hdf5_path=NEIGHBORS_HDF5):
    from neighbors import NeighborsSimpleDataset
    if not neighbor_list:
        return [], [], []

    unique_indices = sorted(set(idx for idx, _ in neighbor_list))
    idx_to_rank = {idx: i for i, idx in enumerate(unique_indices)}

    hsc_list, leg_list = load_images_for_indices(unique_indices, hdf5_path)

    images = []
    sources = []
    indices = []
    for dataset_idx, source in neighbor_list:
        r = idx_to_rank[dataset_idx]
        img = hsc_list[r] if source == "hsc" else leg_list[r]
        images.append(img)
        sources.append(source)
        indices.append(dataset_idx)
    return images, sources, indices

def tensor_to_display(t, channel=0):
    x = t.numpy()
    if x.ndim == 3:
        x = x[channel]
    x = np.clip(x, -3, 3)
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    return x

def tensor_to_rgb_row_scaled(t, vmin, vmax):
    x = t.numpy()
    if x.ndim != 3:
        raise ValueError(f"Expected tensor with shape (C,H,W), got {x.shape}")

    if x.shape[0] < 3:
        gray = tensor_to_display(t)
        return np.stack([gray, gray, gray], axis=-1)

    x = x[:3]
    vmin = np.asarray(vmin).reshape(3, 1, 1)
    vmax = np.asarray(vmax).reshape(3, 1, 1)
    y = (x - vmin) / (vmax - vmin + 1e-8)
    y = np.clip(y, 0.0, 1.0)
    return np.moveaxis(y, 0, 2)

def plot_custom_grid(
    query_hsc, query_legacy,
    hsc_phys_data, hsc_inst_data,
    leg_phys_data, leg_inst_data,
    out_path
):
    """
    Plots the 2x8 grid (Query + 4 Phys + 3 Inst).
    """
    n_rows = 2
    n_cols = 8  # 1 Query + 4 Phys + 3 Inst

    # Larger figure to accommodate larger text
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.8))

    # --- Setup Normalization ---
    q = query_hsc.numpy()
    if q.shape[0] < 3:
        use_rgb = False
        vmin = vmax = None
    else:
        use_rgb = True
        q_ch = q[:3]
        q_flat = q_ch.reshape(3, -1)
        vmin = q_flat.min(axis=1)
        vmax = q_flat.max(axis=1)

    def show_img_with_overlay(ax, t, text=None, text_color="black"):
        """Displays image and puts text in a semi-transparent box."""
        if use_rgb:
            ax.imshow(tensor_to_rgb_row_scaled(t, vmin, vmax))
        else:
            ax.imshow(tensor_to_display(t), cmap="gray")

        if text:
            # Centered at x=0.5, top y=0.96
            ax.text(
                0.5, 0.96, text,
                transform=ax.transAxes,
                fontsize=14,
                fontweight='bold',
                color=text_color,
                verticalalignment='top',
                horizontalalignment='center',
                bbox=dict(boxstyle='square,pad=0.2', facecolor='white', alpha=0.7, linewidth=0)
            )

        ax.set_axis_off()

    def fmt_src(s):
        """Helper to capitalize source labels."""
        if s == "hsc": return "(HSC)"
        if s == "legacy": return "(Legacy)"
        return s

    # --- ROW 0: HSC Query ---
    # Col 0
    show_img_with_overlay(axes[0, 0], query_hsc, text="HSC")

    # Cols 1-4: Physics Neighbors (HSC)
    imgs, srcs, idxs, ranks = hsc_phys_data
    for i in range(4):
        ax = axes[0, 1+i]
        if i < len(imgs):
            if i == 0: color = "indianred"
            elif srcs[i] == "legacy": color = "gold"
            else: color = "black"

            if color == "gold": text_c = "#B8860B"
            else: text_c = color

            # One line label
            lbl = f"NN #{ranks[i]} {fmt_src(srcs[i])}"
            show_img_with_overlay(ax, imgs[i], text=lbl, text_color=text_c)
        else:
            ax.set_axis_off()

    # Cols 5-7: Instrument Neighbors (HSC)
    imgs, srcs, idxs, ranks = hsc_inst_data
    for i in range(3):
        ax = axes[0, 5+i]
        if i < len(imgs):
            if srcs[i] == "legacy": text_c = "#B8860B"
            else: text_c = "black"

            lbl = f"NN #{ranks[i]} {fmt_src(srcs[i])}"
            show_img_with_overlay(ax, imgs[i], text=lbl, text_color=text_c)
        else:
            ax.set_axis_off()

    # --- ROW 1: Legacy Query ---
    # Col 0
    show_img_with_overlay(axes[1, 0], query_legacy, text="Legacy")

    # Cols 1-4: Physics Neighbors (Legacy)
    imgs, srcs, idxs, ranks = leg_phys_data
    for i in range(4):
        ax = axes[1, 1+i]
        if i < len(imgs):
            if i == 0: color = "indianred"
            elif srcs[i] == "hsc": color = "gold"
            else: color = "black"

            if color == "gold": text_c = "#B8860B"
            else: text_c = color

            lbl = f"NN #{ranks[i]} {fmt_src(srcs[i])}"
            show_img_with_overlay(ax, imgs[i], text=lbl, text_color=text_c)
        else:
            ax.set_axis_off()

    # Cols 5-7: Instrument Neighbors (Legacy)
    imgs, srcs, idxs, ranks = leg_inst_data
    for i in range(3):
        ax = axes[1, 5+i]
        if i < len(imgs):
            if srcs[i] == "hsc": text_c = "#B8860B"
            else: text_c = "black"

            lbl = f"NN #{ranks[i]} {fmt_src(srcs[i])}"
            show_img_with_overlay(ax, imgs[i], text=lbl, text_color=text_c)
        else:
            ax.set_axis_off()

    # --- Finalize Layout & Lines/Titles ---
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, wspace=0.1, hspace=0.05)

    fig.canvas.draw()

    # --- Add Background Shades ---
    # We want a background for Row 0 (top half) and Row 1 (bottom half)
    # Using figure coordinates (0,0 bottom-left to 1,1 top-right)
    # Top rect: y from ~0.5 to 1.0
    # Bottom rect: y from 0.0 to ~0.5

    # We can get the exact midpoint between the two rows of axes
    bbox_row0_bot = axes[0, 0].get_position().y0
    bbox_row1_top = axes[1, 0].get_position().y1
    mid_y = (bbox_row0_bot + bbox_row1_top) / 2.0

    # Shade 1: Top (HSC) - Light Blueish
    rect_top = Rectangle((0, mid_y-0.07), 1, 1-mid_y, transform=fig.transFigure,
                         facecolor='lightgray', edgecolor='none', zorder=-10)
    fig.patches.extend([rect_top])

    # Shade 2: Bottom (Legacy) - Light Orangeish
    rect_bot = Rectangle((0, 0), 1, mid_y, transform=fig.transFigure,
                         facecolor='silver', edgecolor='none', zorder=-10)
    fig.patches.extend([rect_bot])


    # --- Draw Vertical Lines ---
    bbox_q = axes[0, 0].get_position()
    bbox_p1 = axes[0, 1].get_position()
    line1_x = (bbox_q.x1 + bbox_p1.x0) / 2.0

    bbox_p4 = axes[0, 4].get_position()
    bbox_i1 = axes[0, 5].get_position()
    line2_x = (bbox_p4.x1 + bbox_i1.x0) / 2.0

    line1 = Line2D([line1_x, line1_x], [0.02, 0.90], transform=fig.transFigure, color="black", linewidth=4, linestyle='--')
    line2 = Line2D([line2_x, line2_x], [0.02, 0.90], transform=fig.transFigure, color="black", linewidth=4, linestyle='--')
    fig.add_artist(line1)
    fig.add_artist(line2)

    # --- Add Supertitles ---
    x_query = (bbox_q.x0 + bbox_q.x1) / 2.0
    x_phys = (bbox_p1.x0 + bbox_p4.x1) / 2.0
    bbox_i3 = axes[0, 7].get_position()
    x_inst = (bbox_i1.x0 + bbox_i3.x1) / 2.0

    y_title = 0.94

    fig.text(x_query, y_title, "Query", fontsize=20, fontweight='bold', ha='center', va='bottom')
    fig.text(x_phys, y_title, "Physics NNs", fontsize=20, fontweight='bold', ha='center', va='bottom', color=BAR_COLORS[0])
    fig.text(x_inst, y_title, "Instrument NNs", fontsize=20, fontweight='bold', ha='center', va='bottom', color=BAR_COLORS[1])

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved custom plot to: {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latents", type=Path, default=None)
    p.add_argument("--index", type=int, default=172)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--neighbors-h5", type=str, default=NEIGHBORS_HDF5)
    args = p.parse_args()

    # --- Load Latents ---
    latents_path = Path(args.latents) if args.latents else None
    if latents_path is None:
        candidates = list(_here.glob("neighbor_latents_*.h5"))
        if len(candidates) == 1:
            latents_path = candidates[0]
        else:
            raise FileNotFoundError("Could not auto-locate neighbor_latents file. Pass --latents.")

    print(f"Loading latents from {latents_path}...")
    with h5py.File(latents_path, "r") as f:
        idx = f["idx"][:]
        hsc_physics_emb = f["physics_embedding"][:]
        hsc_instrument_emb = f["instrument_embedding"][:]
        leg_physics_emb = f["legacy_physics_embedding"][:]
        leg_instrument_emb = f["legacy_instrument_embedding"][:]

    n = len(idx)
    combined_physics = np.concatenate([hsc_physics_emb, leg_physics_emb], axis=0)
    combined_instrument = np.concatenate([hsc_instrument_emb, leg_instrument_emb], axis=0)

    # --- Fit NN ---
    print("Fitting NearestNeighbors...")
    k_use = K_FETCH + 2
    nn_phys = NearestNeighbors(n_neighbors=min(k_use, 2 * n), metric="euclidean").fit(combined_physics)
    nn_inst = NearestNeighbors(n_neighbors=min(k_use, 2 * n), metric="euclidean").fit(combined_instrument)

    def combined_pos_to_dataset_and_source(pos):
        if pos < n: return (int(pos), "hsc")
        return (int(pos - n), "legacy")

    def get_neighbors_indices(query_emb, nn_model, self_pos, selection_ranks):
        _, indices = nn_model.kneighbors(query_emb, n_neighbors=k_use)
        raw_indices = indices[0]
        valid_neighbors = [p for p in raw_indices if p != self_pos]

        selected_info = []
        for r in selection_ranks:
            if (r-1) < len(valid_neighbors):
                pos = valid_neighbors[r-1]
                dataset_idx, source = combined_pos_to_dataset_and_source(pos)
                selected_info.append((dataset_idx, source))
            else:
                print(f"Warning: requested rank {r} but not enough neighbors found.")
        return selected_info

    query_idx = args.index
    out_path = args.out if args.out else Path(f"query_{query_idx}_custom.png")

    # --- INDICES DEFINITION ---
    # 1. HSC Query Logic
    hsc_self_pos = query_idx
    hsc_phys_list = get_neighbors_indices(combined_physics[hsc_self_pos:hsc_self_pos+1], nn_phys, hsc_self_pos, [1, 2, 3, 4])
    hsc_inst_list = get_neighbors_indices(combined_instrument[hsc_self_pos:hsc_self_pos+1], nn_inst, hsc_self_pos, [1, 2, 3])

    # 2. Legacy Query Logic
    leg_self_pos = n + query_idx
    leg_phys_list = get_neighbors_indices(combined_physics[leg_self_pos:leg_self_pos+1], nn_phys, leg_self_pos, [1, 2, 5, 7])
    leg_inst_list = get_neighbors_indices(combined_instrument[leg_self_pos:leg_self_pos+1], nn_inst, leg_self_pos, [1, 2, 3])

    # --- Load Images ---
    print(f"Loading images for Query {query_idx}...")
    q_hsc_img, q_leg_img = load_images_for_indices([query_idx], args.neighbors_h5)
    q_hsc_img = q_hsc_img[0]
    q_leg_img = q_leg_img[0]

    def fetch_data(neighbor_list, rank_labels):
        imgs, srcs, idxs = load_images_for_neighbors(neighbor_list, args.neighbors_h5)
        return (imgs, srcs, idxs, rank_labels)

    # Fetch data
    hsc_phys_data = fetch_data(hsc_phys_list, [1, 2, 3, 4])
    hsc_inst_data = fetch_data(hsc_inst_list, [1, 2, 3])

    leg_phys_data = fetch_data(leg_phys_list, [1, 2, 5, 7])
    leg_inst_data = fetch_data(leg_inst_list, [1, 2, 3])

    # --- Plot ---
    print("Plotting...")
    plot_custom_grid(
        q_hsc_img, q_leg_img,
        hsc_phys_data, hsc_inst_data,
        leg_phys_data, leg_inst_data,
        out_path
    )

if __name__ == "__main__":
    main()
