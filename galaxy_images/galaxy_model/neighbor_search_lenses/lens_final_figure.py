"""
Custom 2-row figure for lenses 33 and 48.

  Lens 33 (row 0): Query HSC | physics NNs #1, 2, 3, 6, 7, 8
  Lens 48 (row 1): Query HSC | physics NNs #1, 3, 4, 5, 6, 8

Images are 64×64 raw center-crops (same as the per-lens figures on Discord).
Visual style matches plot_gem.py: text overlay in semi-transparent box,
alternating row shading, dashed vertical separator, supertitles.

Intermediate NN results are cached to CACHE_PATH so future replots skip
model loading and kNN search entirely.

Run from galaxy_model/:
  python neighbor_search_lenses/lens_final_figure.py
"""
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
_src  = _here.parent          # galaxy_model/
_root = _src.parents[1]       # tess-generative/
for p in (_src, _root):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from sklearn.neighbors import NearestNeighbors

try:
    torch.backends.cuda.preferred_blas_library("hipblas")
except Exception:
    pass

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NEIGHBORS_HDF5 = "/work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5"
LATENTS_PATH   = _src / "neighbor_search/neighbor_latents_103k.h5"
CHECKPOINT     = str(_src / "outputs/neighbors_all_attn/2026-04-05/checkpoints/best-epoch=228-step=87000.ckpt")
LENS_H5        = _src / "lense_reconstruction/lens_reconstruction_dataset.h5"
OUT_PATH       = _here / "outputs" / "lens_neighbors_final_figure.png"
OUT_PDF        = _here / "outputs" / "lens_neighbors_final_figure.pdf"
OUT_TEX        = _here / "outputs" / "lens_neighbors_table.txt"
CACHE_PATH     = _here / "outputs" / "lens_neighbors_final_figure_cache.h5"

LENSES = [
    {"user_idx": 33, "nn_ranks": [2, 3, 4, 7, 8, 9]},
    {"user_idx": 48, "nn_ranks": [2, 4, 5, 6, 7, 9]},
]
K_SEARCH = 12

# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def save_cache(cache_path: Path, lens_data: list):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(cache_path, "w") as f:
        f.attrs["checkpoint"] = CHECKPOINT
        f.attrs["latents"]    = str(LATENTS_PATH)
        for d in lens_data:
            grp = f.create_group(f"lens_{d['user_idx']}")
            grp.attrs["user_idx"]  = d["user_idx"]
            grp.attrs["lens_0idx"] = d["lens_0idx"]
            grp.attrs["obj_id"]    = d["obj_id"]
            grp.attrs["h5_row"]    = d["h5_row"]
            all_nns = d["all_nns"]
            grp.create_dataset("nn_combined_pos", data=np.array([x[0] for x in all_nns], dtype=np.int64))
            grp.create_dataset("nn_survey",       data=np.array([x[1].encode() for x in all_nns]))
            grp.create_dataset("nn_raw_h5_row",   data=np.array([x[2] for x in all_nns], dtype=np.int64))
    print(f"Cache saved: {cache_path}")


def load_cache(cache_path: Path) -> list:
    lens_data = []
    with h5py.File(cache_path, "r") as f:
        for key in sorted(f.keys()):
            grp = f[key]
            all_nns = list(zip(
                grp["nn_combined_pos"][:].tolist(),
                [s.decode() for s in grp["nn_survey"][:]],
                grp["nn_raw_h5_row"][:].tolist(),
            ))
            lens_data.append({
                "user_idx":  int(grp.attrs["user_idx"]),
                "lens_0idx": int(grp.attrs["lens_0idx"]),
                "obj_id":    str(grp.attrs["obj_id"]),
                "h5_row":    int(grp.attrs["h5_row"]),
                "all_nns":   all_nns,
            })
    print(f"Loaded cache: {cache_path}  ({len(lens_data)} lenses)")
    return lens_data

# ---------------------------------------------------------------------------
# NN search helpers
# ---------------------------------------------------------------------------

def encode_lenses(model, device, raw_hsc_list: list) -> np.ndarray:
    from neighbors import preprocess_raw_image
    tensors = torch.stack([preprocess_raw_image(img, "hsc", 48)[:4] for img in raw_hsc_list])
    with torch.no_grad():
        emb = model.encoder_1(tensors.to(device)).cpu().flatten(start_dim=1)
    return emb.numpy().astype(np.float32)


def filter_nns(positions, distances, n_hsc, index_mmu_hsc, index_mmu_leg, k):
    result = []
    for pos, dist in zip(positions, distances):
        if dist < 1e-6:
            continue
        pos = int(pos)
        if pos < n_hsc:
            result.append((pos, "hsc", int(index_mmu_hsc[pos])))
        else:
            result.append((pos, "legacy", int(index_mmu_leg[pos - n_hsc])))
        if len(result) == k:
            break
    return result

# ---------------------------------------------------------------------------
# Image loading — 64×64 raw center-crop
# ---------------------------------------------------------------------------

_cropper64 = None

def load_64px(h5_file: h5py.File, raw_row: int, survey: str) -> torch.Tensor:
    global _cropper64
    if _cropper64 is None:
        from galaxy_images.image_preprocessing import CenterCrop
        _cropper64 = CenterCrop(crop_size=64)
    key = "images_hsc" if survey == "hsc" else "images_legacy"
    t = torch.from_numpy(h5_file[key][raw_row]).float()
    return _cropper64(t.unsqueeze(0)).squeeze(0)

# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def tensor_to_rgb(t: torch.Tensor, percentile_clip: float = 99.5) -> np.ndarray:
    n_ch = t.shape[0]
    c = (0, 1, 2) if n_ch >= 3 else (0, 0, 0)
    rgb = t[list(c)].cpu().numpy().transpose(1, 2, 0)
    for i in range(3):
        lo = np.percentile(rgb[:, :, i], 100 - percentile_clip)
        hi = np.percentile(rgb[:, :, i], percentile_clip)
        rgb[:, :, i] = np.clip(rgb[:, :, i], lo, hi)
    for i in range(3):
        ch = rgb[:, :, i]; lo, hi = ch.min(), ch.max()
        rgb[:, :, i] = (ch - lo) / (hi - lo) if hi > lo else 0.0
    return rgb


def show_img(ax, t: torch.Tensor, text: str = None, text_color: str = "black", text_fontsize: int = 14):
    ax.imshow(tensor_to_rgb(t))
    if text:
        ax.text(
            0.5, 0.96, text,
            transform=ax.transAxes,
            fontsize=text_fontsize, fontweight="bold",
            color=text_color,
            va="top", ha="center",
            bbox=dict(boxstyle="square,pad=0.2", facecolor="white", alpha=0.7, linewidth=0),
        )
    ax.set_axis_off()


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

def save_latex_table(lens_data: list, out_path: Path):
    """Write a NeurIPS-style booktabs LaTeX table of object IDs."""
    n_nn = max(len(d["selected"]) for d in lens_data)

    # Collect IDs from neighbours_v2.h5
    with h5py.File(NEIGHBORS_HDF5, "r") as f:
        obj_id_hsc = f["object_id_hsc"][:]

    def get_hsc_id(raw_row: int) -> str:
        v = obj_id_hsc[raw_row]
        return v.decode() if hasattr(v, "decode") else str(v)

    rows = []
    for d in lens_data:
        query_id = get_hsc_id(d["h5_row"])
        nn_cells = []
        for rank, (_, survey, raw_row) in d["selected"]:
            survey_tag = "H" if survey == "hsc" else "L"
            nn_cells.append(f"\\texttt{{{get_hsc_id(raw_row)}}} (\\#{rank - 1}, {survey_tag})")
        rows.append((d["user_idx"], query_id, nn_cells))

    # Positional column headers (ranks differ per lens — rank noted inside each cell)
    col_header = " & ".join(
        ["Lens", "Query (HSC)"] + [f"Neighbor {i + 1}" for i in range(n_nn)]
    )

    lines = []
    lines.append(r"\begin{table}[h!]")
    lines.append(r"\centering")
    lines.append(r"\caption{Object identifiers for query lenses and their displayed physics nearest neighbours.")
    lines.append(r"  Each neighbour cell shows the object ID followed by its rank and survey")
    lines.append(r"  (\textbf{H}\,=\,HSC, \textbf{L}\,=\,Legacy).}")
    lines.append(r"\label{tab:lens_neighbours}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    col_fmt = "l " + "l " * (1 + n_nn)
    lines.append(f"\\begin{{tabular}}{{{col_fmt.strip()}}}")
    lines.append(r"\toprule")
    lines.append(col_header + r" \\")
    lines.append(r"\midrule")
    for user_idx, query_id, nn_cells in rows:
        cells = [f"Lens {user_idx}", f"\\texttt{{{query_id}}} (H)"] + nn_cells
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Saved: {out_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load or build NN cache
    if CACHE_PATH.exists():
        lens_data = load_cache(CACHE_PATH)
    else:
        print(f"Loading latents from {LATENTS_PATH} ...")
        with h5py.File(LATENTS_PATH, "r") as f:
            if "hsc_index_mmu" in f:
                index_mmu_hsc = f["hsc_index_mmu"][:]
                index_mmu_leg = f["legacy_index_mmu"][:]
                hsc_phys = f["hsc_physics"][:]
                leg_phys  = f["legacy_physics"][:]
            else:
                index_mmu_hsc = f["index_mmu"][:]
                index_mmu_leg = index_mmu_hsc
                hsc_phys = f["physics_embedding"][:]
                leg_phys  = f["legacy_physics_embedding"][:]

        N_hsc    = len(index_mmu_hsc)
        combined = np.concatenate([hsc_phys, leg_phys], axis=0)
        print(f"Combined gallery: {len(combined):,}  (HSC={N_hsc:,})")

        k_query = K_SEARCH + 2
        nn_phys = NearestNeighbors(
            n_neighbors=min(k_query, len(combined)), metric="euclidean"
        ).fit(combined)

        with h5py.File(LENS_H5, "r") as f:
            lens_h5_indices = f["h5_index"][:]
            lens_object_ids = f["object_id_hsc"][:]
            raw_hsc_all     = f["images_hsc"][:]

        print("Loading model ...")
        from double_train_fm_neighbors import ConditionalFlowMatchingModule
        model = ConditionalFlowMatchingModule.load_from_checkpoint(CHECKPOINT, map_location="cpu")
        model.eval()
        torch.set_grad_enabled(False)
        model.to(device)

        lens_0indices = [L["user_idx"] - 1 for L in LENSES]
        lens_embs = encode_lenses(model, device, [raw_hsc_all[i] for i in lens_0indices])

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        lens_data = []
        for spec, lens_0idx, emb in zip(LENSES, lens_0indices, lens_embs):
            dists, inds = nn_phys.kneighbors(emb[None], n_neighbors=k_query)
            all_nns = filter_nns(inds[0], dists[0], N_hsc, index_mmu_hsc, index_mmu_leg, K_SEARCH)
            obj_id = lens_object_ids[lens_0idx]
            if isinstance(obj_id, bytes):
                obj_id = obj_id.decode()
            lens_data.append({
                "user_idx":  spec["user_idx"],
                "lens_0idx": lens_0idx,
                "obj_id":    obj_id,
                "h5_row":    int(lens_h5_indices[lens_0idx]),
                "all_nns":   all_nns,
            })

        save_cache(CACHE_PATH, lens_data)

    # Match lens_data order to LENSES config order (cache sorts by key name)
    lens_by_uid = {d["user_idx"]: d for d in lens_data}
    lens_data = [lens_by_uid[spec["user_idx"]] for spec in LENSES]

    # Apply nn_ranks from config and select NNs
    for d, spec in zip(lens_data, LENSES):
        selected = []
        for rank in spec["nn_ranks"]:
            idx0 = rank - 1
            if idx0 < len(d["all_nns"]):
                selected.append((rank, d["all_nns"][idx0]))
            else:
                print(f"  Warning: rank {rank} not available for lens {d['user_idx']}")
        d["selected"] = selected   # [(rank, (combined_pos, survey, raw_h5_row)), ...]

    # Load images (64×64 raw crop)
    print("Loading images ...")
    with h5py.File(NEIGHBORS_HDF5, "r") as nb_h5:
        for d in lens_data:
            d["query_img"] = load_64px(nb_h5, d["h5_row"], "hsc")
            d["nn_imgs"]   = [load_64px(nb_h5, rr, sv) for _, (_, sv, rr) in d["selected"]]
            d["nn_meta"]   = [(rank, sv) for rank, (_, sv, _) in d["selected"]]

    # Plot
    n_nn_cols = max(len(d["selected"]) for d in lens_data)
    n_cols    = 1 + n_nn_cols
    n_rows    = len(lens_data)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.8))
    if n_rows == 1:
        axes = axes[None, :]
    for ax in axes.flat:
        ax.set_axis_off()

    def src_lbl(s): return "(HSC)" if s == "hsc" else "(Legacy)"

    for r, d in enumerate(lens_data):
        show_img(axes[r, 0], d["query_img"], text=f"HSC {d['obj_id']}", text_fontsize=10)
        for c, (img, (rank, survey)) in enumerate(zip(d["nn_imgs"], d["nn_meta"]), start=1):
            show_img(axes[r, c], img,
                     text=f"NN #{rank - 1} {src_lbl(survey)}",
                     text_color="#B8860B" if survey == "legacy" else "black")

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, wspace=0.1, hspace=0.05)
    fig.canvas.draw()

    # Row background shading
    row_colors = ["lightgray", "silver"]
    for r in range(n_rows):
        y0 = min(axes[r, c].get_position().y0 for c in range(n_cols)) - 0.01
        y1 = max(axes[r, c].get_position().y1 for c in range(n_cols)) + 0.01
        fig.patches.append(Rectangle((0, y0), 1, y1 - y0,
                                     transform=fig.transFigure,
                                     facecolor=row_colors[r % 2], edgecolor="none", zorder=-10))

    # Dashed vertical separator between query and NNs
    bq  = axes[0, 0].get_position()
    bn1 = axes[0, 1].get_position()
    sep_x = (bq.x1 + bn1.x0) / 2
    fig.add_artist(Line2D([sep_x, sep_x], [0.02, 0.90], transform=fig.transFigure,
                          color="black", linewidth=4, linestyle="--"))

    # Supertitles
    fig.text((bq.x0 + bq.x1) / 2, 0.94, "Query",
             ha="center", va="bottom", fontsize=20, fontweight="bold")
    x0n = axes[0, 1].get_position().x0
    x1n = axes[0, -1].get_position().x1
    fig.text((x0n + x1n) / 2, 0.94, "Physics NNs",
             ha="center", va="bottom", fontsize=20, fontweight="bold", color="#2E86AB")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")
    print(f"Saved: {OUT_PDF}")

    save_latex_table(lens_data, OUT_TEX)


if __name__ == "__main__":
    main()
