"""
Cross-space UMAP visualization of anomaly detectors.

Fits a UMAP on --n-background random background points (Option 2B) for each of
three latent spaces (physics 64D, AION 768D, instrument 64D), then projects the
top-N iforest anomalies from each detector into those embeddings.

Produces two figures:
  Figure 1: top-N physics + AION anomalies highlighted across all 3 spaces
  Figure 2: top-N instrument anomalies highlighted across all 3 spaces

Run from galaxy_model/:
  python anomaly_detection/umap_visualizations/plot_umap_cross_space.py \\
    [--ours-latents outputs/anomaly_latents_ours_ours_100k.h5] \\
    [--aion-latents outputs/anomaly_latents_aion_aion_100k.h5] \\
    [--ins-latents  outputs/anomaly_latents_ours_ins_100k.h5]  \\
    [--ours-scores  outputs/anomaly_scores_ours_100k.h5]       \\
    [--aion-scores  outputs/anomaly_scores_aion_100k.h5]       \\
    [--ins-scores   outputs/anomaly_scores_ins_100k.h5]        \\
    [--suffix crossspace_100k] [--top-n 10] [--n-background 10000]
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
# Force regular hipBLAS instead of hipBLASLt — hipBLASLt is buggy on MI210
# for certain matrix shapes and causes HIPBLAS_STATUS_INVALID_VALUE at runtime.
torch.backends.cuda.preferred_blas_library("hipblas")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_here = Path(__file__).resolve().parent
OUTPUT_DIR = _here.parent / "outputs"


def _resolve_path(path_str):
    p = Path(path_str)
    if not p.is_absolute() and not p.exists():
        p = OUTPUT_DIR / path_str
    return p


def _top_n_by_score(scores_path, score_key, top_n, rank_max=None):
    with h5py.File(scores_path, "r") as f:
        scores = f[score_key][:]
    ranked = np.argsort(scores)[::-1]  # position 0 = rank 1 (highest score)
    if rank_max is not None:
        pool = ranked[:rank_max]
        pick = np.round(np.linspace(0, len(pool) - 1, top_n)).astype(int)
        return pool[pick]
    return ranked[:top_n]


def _fit_umap_transform(bg_feats, query_feats, include_anomalies=False):
    try:
        import umap as umap_lib
    except ImportError:
        raise RuntimeError("umap-learn not installed. Run: pip install umap-learn")
    reducer = umap_lib.UMAP(
        n_neighbors=15, min_dist=0.1, n_components=2,
        metric="euclidean", random_state=42, verbose=True,
    )
    if include_anomalies:
        n_bg = len(bg_feats)
        combined = np.vstack([bg_feats, query_feats]).astype(np.float32)
        emb = reducer.fit_transform(combined)
        return emb[:n_bg], emb[n_bg:]
    else:
        bg_emb = reducer.fit_transform(bg_feats.astype(np.float32))
        query_emb = reducer.transform(query_feats.astype(np.float32))
        return bg_emb, query_emb


def _make_figure(bg_embs, space_names, anom_embs_per_set, highlight_defs, out_path, title):
    """
    bg_embs          : list of (N_bg, 2) arrays, one per space
    space_names      : list of str
    anom_embs_per_set: dict[set_name] → list of (n, 2) arrays, one per space
    highlight_defs   : dict[set_name] → (color, marker, label)
    """
    n = len(space_names)
    fig, axes = plt.subplots(1, n, figsize=(n * 5, 5))
    if n == 1:
        axes = [axes]

    for col, (ax, name, bg) in enumerate(zip(axes, space_names, bg_embs)):
        ax.scatter(bg[:, 0], bg[:, 1], c="#cccccc", s=2, alpha=0.4,
                   rasterized=True, linewidths=0)
        for set_name, (color, marker, label) in highlight_defs.items():
            emb = anom_embs_per_set[set_name][col]
            ax.scatter(emb[:, 0], emb[:, 1], c=color, s=140, marker=marker,
                       label=label, zorder=5, edgecolors="black", linewidths=0.6)
        ax.set_title(name, fontsize=11)
        ax.axis("off")
        if col == 0:
            ax.legend(fontsize=8, loc="upper left", framealpha=0.8)

    fig.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours-latents", default="anomaly_latents_ours_ours_100k.h5")
    parser.add_argument("--aion-latents", default="anomaly_latents_aion_aion_100k.h5")
    parser.add_argument("--ins-latents",  default="anomaly_latents_ours_ins_100k.h5")
    parser.add_argument("--ours-scores",  default="anomaly_scores_ours_100k.h5")
    parser.add_argument("--aion-scores",  default="anomaly_scores_aion_100k.h5")
    parser.add_argument("--ins-scores",   default="anomaly_scores_ins_100k.h5")
    parser.add_argument("--suffix",            default="crossspace_100k")
    parser.add_argument("--n-background",      type=int, default=10000)
    parser.add_argument("--top-n",             type=int, default=10)
    parser.add_argument("--rank-max",          type=int, default=None,
                        help="If set, sample --top-n points uniformly spread across ranks 1..rank-max")
    parser.add_argument("--include-anomalies", action="store_true",
                        help="Include anomaly points in the UMAP fit (joint embedding)")
    args = parser.parse_args()

    out_dir = OUTPUT_DIR / f"figures_{args.suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    ours_lat = _resolve_path(args.ours_latents)
    aion_lat = _resolve_path(args.aion_latents)
    ins_lat  = _resolve_path(args.ins_latents)
    ours_scr = _resolve_path(args.ours_scores)
    aion_scr = _resolve_path(args.aion_scores)
    ins_scr  = _resolve_path(args.ins_scores)

    # ── Load all latents into memory ─────────────────────────────────────────
    print("Loading latents...")
    with h5py.File(ours_lat, "r") as f:
        physics_feats = f["hsc_flat"][:]          # (N, 64)
    with h5py.File(aion_lat, "r") as f:
        aion_feats = f["embeddings_mean_hsc"][:]  # (N, 768)
    with h5py.File(ins_lat, "r") as f:
        ins_feats = f["hsc_flat"][:]              # (N, 64)

    n_total = len(physics_feats)
    print(f"  N={n_total}  physics={physics_feats.shape}  "
          f"aion={aion_feats.shape}  ins={ins_feats.shape}")

    # ── Top-N anomaly positions per detector ─────────────────────────────────
    rank_desc = f"ranks 1–{args.rank_max}" if args.rank_max else f"top-{args.top_n}"
    print(f"\nSelecting {args.top_n} anomaly points ({rank_desc}) per detector...")
    phys_top = _top_n_by_score(ours_scr, "ours/hsc_flat/iforest", args.top_n, args.rank_max)
    aion_top = _top_n_by_score(aion_scr, "aion/hsc_mean_pca64/iforest", args.top_n, args.rank_max)
    ins_top  = _top_n_by_score(ins_scr,  "ours/hsc_flat/iforest", args.top_n, args.rank_max)
    print(f"  physics:    {phys_top}")
    print(f"  aion:       {aion_top}")
    print(f"  instrument: {ins_top}")

    # ── Sample background excluding all anomalies ────────────────────────────
    all_anom_pos = np.unique(np.concatenate([phys_top, aion_top, ins_top]))
    anom_set = set(all_anom_pos.tolist())
    candidates = np.array([i for i in range(n_total) if i not in anom_set])
    rng = np.random.default_rng(42)
    bg_pos = np.sort(rng.choice(candidates, size=min(args.n_background, len(candidates)), replace=False))
    print(f"\nBackground: {len(bg_pos)} points (excluded {len(anom_set)} anomaly positions)")

    # ── Build query array (all unique anomalies, sorted) ────────────────────
    anom_sorted = np.sort(all_anom_pos)
    pos_to_row = {int(p): r for r, p in enumerate(anom_sorted)}

    def _anom_rows(pos_arr):
        return np.array([pos_to_row[int(p)] for p in pos_arr])

    # ── Fit UMAP per space, collect embeddings ───────────────────────────────
    space_configs = [
        ("Physics (64D)",    physics_feats),
        ("AION (768D)",      aion_feats),
        ("Instrument (64D)", ins_feats),
    ]

    bg_embs    = []
    anom_embs_all_spaces = {k: [] for k in ("physics", "aion", "instrument")}

    for space_name, feats in space_configs:
        print(f"\n=== UMAP: {space_name} ===")
        bg_feats    = feats[bg_pos]
        query_feats = feats[anom_sorted]
        n_fit = bg_feats.shape[0] + (query_feats.shape[0] if args.include_anomalies else 0)
        print(f"  fit on {n_fit} pts × {bg_feats.shape[1]}D ...")
        bg_emb, query_emb = _fit_umap_transform(bg_feats, query_feats, args.include_anomalies)
        bg_embs.append(bg_emb)
        anom_embs_all_spaces["physics"].append(query_emb[_anom_rows(phys_top)])
        anom_embs_all_spaces["aion"].append(query_emb[_anom_rows(aion_top)])
        anom_embs_all_spaces["instrument"].append(query_emb[_anom_rows(ins_top)])

    space_names = [cfg[0] for cfg in space_configs]

    # ── Figure 1: physics + AION anomalies ──────────────────────────────────
    print("\nGenerating Figure 1: physics + AION anomalies across all spaces...")
    _make_figure(
        bg_embs,
        space_names,
        {k: anom_embs_all_spaces[k] for k in ("physics", "aion")},
        {
            "physics": ("#e74c3c", "o", f"Physics top-{args.top_n} (iforest)"),
            "aion":    ("#3498db", "^", f"AION top-{args.top_n} (iforest)"),
        },
        out_dir / f"umap_crossspace_physics_aion_{args.suffix}.png",
        "Cross-space UMAP — physics & AION anomalies",
    )

    # ── Figure 2: instrument anomalies ──────────────────────────────────────
    print("\nGenerating Figure 2: instrument anomalies across all spaces...")
    _make_figure(
        bg_embs,
        space_names,
        {"instrument": anom_embs_all_spaces["instrument"]},
        {"instrument": ("#2ecc71", "*", f"Instrument top-{args.top_n} (iforest)")},
        out_dir / f"umap_crossspace_instrument_{args.suffix}.png",
        "Cross-space UMAP — instrument anomalies",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
