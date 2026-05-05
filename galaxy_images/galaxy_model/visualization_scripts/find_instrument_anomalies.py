"""Find Legacy points sitting inside the HSC island on the Instrument-latent UMAP
of a saved run, then send their actual HSC + Legacy thumbnails to Discord.

For each Legacy point on the Instrument panel, score its 'HSC-ness' as the
fraction of its k=20 nearest neighbors (in the 2D UMAP plane) that are HSC.
The top-K Legacy points by score are the ones embedded deepest into the HSC
cluster — those are the suspects.

Usage:
    python find_instrument_anomalies.py --tag base_n4096_s314 --top-k 16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

SCRIPT_DIR = Path(__file__).resolve().parent
GM_DIR = SCRIPT_DIR.parent
REPO_ROOT = GM_DIR.parent.parent
LATENT_SPACE_DIR = SCRIPT_DIR / "neighbors_visualization" / "latent_space"
for p in (str(REPO_ROOT), str(GM_DIR), str(SCRIPT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from galaxy_images.image_preprocessing import preprocess_image_v2
from galaxy_images.galaxy_model.data import zoom_legacy_image
from galaxy_images.galaxy_model.neighbors import NORM_DICT
from discord_notify import notify
import plot_umap_from_file as plotter
import torch


def _read_rows_from_bin(bin_path: Path, row_indices, n_total: int, c: int, h: int, w: int):
    """Read specific rows of a (N, C, H, W) float16 binary by seek+read — no mmap."""
    bytes_per_row = c * h * w * 2  # float16
    out = np.empty((len(row_indices), c, h, w), dtype=np.float16)
    with open(bin_path, "rb") as f:
        for i, r in enumerate(row_indices):
            f.seek(int(r) * bytes_per_row)
            buf = f.read(bytes_per_row)
            out[i] = np.frombuffer(buf, dtype=np.float16).reshape(c, h, w)
    return out


def _preprocess_one(raw: np.ndarray, survey: str, crop_size: int = 48) -> np.ndarray:
    """Match NeighborsEfficientDataset preprocessing: crop, scale, zoom (legacy), normalize."""
    img = torch.from_numpy(raw.astype(np.float32))
    img = preprocess_image_v2(img, crop_size=crop_size, survey=survey)
    if survey == "legacy":
        img = zoom_legacy_image(img)
        mean, std = NORM_DICT["legacy_zoom"]
    else:
        mean, std = NORM_DICT["hsc"]
    img = (img - mean) / std
    return img.numpy()


def _row_scale_rgb(img_chw: np.ndarray) -> np.ndarray:
    """img_chw: (>=3, H, W) -> (H, W, 3) in [0,1] using per-channel min/max stretch."""
    x = img_chw[:3].astype(np.float32)
    vmin = x.reshape(3, -1).min(axis=1).reshape(3, 1, 1)
    vmax = x.reshape(3, -1).max(axis=1).reshape(3, 1, 1)
    y = (x - vmin) / (vmax - vmin + 1e-8)
    return np.clip(y, 0, 1).transpose(1, 2, 0)


def _resolve_npz(tag: str) -> Path:
    cands = sorted(LATENT_SPACE_DIR.glob(f"umap_both_encoders_zdim*_zoom_flat_{tag}_data.npz"))
    if not cands:
        raise FileNotFoundError(f"No saved UMAP for tag '{tag}' under {LATENT_SPACE_DIR}")
    return cands[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="base_n4096_s314")
    parser.add_argument("--data-dir", type=Path,
                        default=Path("/work1/jeroenaudenaert/pablomer/data/neighbors_efficient"))
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--knn-k", type=int, default=20)
    parser.add_argument("--webhook", type=str, default=os.environ.get("DISCORD_WEBHOOK", ""))
    args = parser.parse_args()

    npz_path = _resolve_npz(args.tag)
    print(f"[anom] loading {npz_path}")
    with np.load(npz_path, allow_pickle=False) as z:
        hsc_umap_1 = z["hsc_umap_1"]
        legacy_umap_1 = z["legacy_umap_1"]
        hsc_umap_2 = z["hsc_umap_2"]
        legacy_umap_2 = z["legacy_umap_2"]
    n = hsc_umap_2.shape[0]
    print(f"[anom] {n} pairs")

    # --- kNN-based anomaly score on the Instrument UMAP plane ---
    pts = np.concatenate([hsc_umap_2, legacy_umap_2], axis=0)
    is_legacy = np.concatenate([np.zeros(n, bool), np.ones(n, bool)], axis=0)
    nbrs = NearestNeighbors(n_neighbors=args.knn_k + 1).fit(pts)
    _, idx = nbrs.kneighbors(pts)
    nn_idx = idx[:, 1:]   # drop self
    nn_is_legacy = is_legacy[nn_idx]
    hsc_frac = (~nn_is_legacy).mean(axis=1)

    legacy_hsc_frac = hsc_frac[n:]                      # for the n legacy points
    order = np.argsort(-legacy_hsc_frac)
    top_pair_idx = order[: args.top_k].astype(int)
    top_scores = legacy_hsc_frac[top_pair_idx]
    print(f"[anom] top-{args.top_k} Legacy HSC-fractions: "
          f"min={top_scores.min():.2f} max={top_scores.max():.2f}")

    # --- Re-derive images for those pair indices (direct seek+read; no full mmap) ---
    catalog = pd.read_parquet(args.data_dir / "catalog.parquet")
    anchor_indices = catalog.index[catalog["source_type"] == 0].to_numpy()
    with open(args.data_dir / "meta.json") as f:
        meta_disk = json.load(f)
    h_total, h_c, h_h, h_w = meta_disk["hsc_image"]["shape"]
    l_total, l_c, l_h, l_w = meta_disk["legacy_image"]["shape"]

    selected = sorted(top_pair_idx.tolist())
    rows_needed = [int(anchor_indices[i]) for i in selected]
    hsc_raw = _read_rows_from_bin(
        args.data_dir / "hsc_image.bin", rows_needed, h_total, h_c, h_h, h_w,
    )
    legacy_raw = _read_rows_from_bin(
        args.data_dir / "legacy_image.bin", rows_needed, l_total, l_c, l_h, l_w,
    )

    pair_to_imgs = {}
    pair_to_meta = {}
    for k, i in enumerate(selected):
        # Match dataset preprocessing: drop y band on HSC after preprocess.
        hsc_img = _preprocess_one(hsc_raw[k], "hsc", crop_size=48)[:4]
        leg_img = _preprocess_one(legacy_raw[k], "legacy", crop_size=48)
        pair_to_imgs[i] = (hsc_img, leg_img)
        row_idx = rows_needed[k]
        catrow = catalog.loc[row_idx]
        pair_to_meta[i] = {
            "row_idx": int(row_idx),
            "ra": float(catrow["ra"]),
            "dec": float(catrow["dec"]),
            "object_id_hsc": str(catrow["object_id_hsc"]),
            "object_id_legacy": str(catrow["object_id_legacy"]),
        }

    # ordered by score (most anomalous first)
    ordered = list(top_pair_idx)

    # --- Composite figure: 2 rows (HSC top, Legacy bottom) x K cols ---
    K = len(ordered)
    fig, axes = plt.subplots(2, K, figsize=(1.6 * K, 3.6))
    if K == 1:
        axes = axes.reshape(2, 1)
    for col, pair_idx in enumerate(ordered):
        hsc_img, leg_img = pair_to_imgs[pair_idx]
        m = pair_to_meta[pair_idx]
        s = legacy_hsc_frac[pair_idx]
        axes[0, col].imshow(_row_scale_rgb(hsc_img))
        axes[0, col].set_xticks([]); axes[0, col].set_yticks([])
        axes[0, col].set_title(
            f"#{col+1}\nrow={m['row_idx']}\nscore={s:.2f}",
            fontsize=8,
        )
        axes[1, col].imshow(_row_scale_rgb(leg_img))
        axes[1, col].set_xticks([]); axes[1, col].set_yticks([])
        axes[1, col].set_title(f"ra={m['ra']:.2f}\ndec={m['dec']:.2f}", fontsize=7)
    axes[0, 0].set_ylabel("HSC", fontsize=11)
    axes[1, 0].set_ylabel("Legacy", fontsize=11)
    plt.suptitle(
        f"Top-{K} Legacy points inside the HSC island — Instrument latent ({args.tag})\n"
        f"score = fraction of k={args.knn_k} nearest UMAP neighbors that are HSC",
        fontsize=11,
    )
    plt.tight_layout()
    composite_path = LATENT_SPACE_DIR / f"anomalies_{args.tag}_top{K}.png"
    plt.savefig(composite_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[anom] saved {composite_path}")

    # --- Overlay: suspects + their HSC pair on BOTH panels (Physics + Instrument) ---
    SUSPECT_COLOR = "red"      # the Legacy suspect (anomaly)
    PARTNER_COLOR = "#2ca02c"  # its HSC partner (same galaxy)
    LINK_COLOR = "#444444"

    def _draw_overlay(ax, hsc_xy, legacy_xy, title):
        ax.scatter(hsc_xy[:, 0], hsc_xy[:, 1], s=20, c=plotter.COLOR_HSC,
                   alpha=0.35, label="HSC")
        ax.scatter(legacy_xy[:, 0], legacy_xy[:, 1], s=20, c=plotter.COLOR_LEGACY,
                   alpha=0.25, label="Legacy")
        for col, pair_idx in enumerate(ordered):
            xL, yL = legacy_xy[pair_idx]
            xH, yH = hsc_xy[pair_idx]
            # Connecting line first (under markers).
            ax.plot([xL, xH], [yL, yH], color=LINK_COLOR, linewidth=1.0,
                    alpha=0.7, linestyle="-", zorder=4)
            # HSC partner (green ring).
            ax.scatter([xH], [yH], s=220, facecolors="none",
                       edgecolors=PARTNER_COLOR, linewidths=2.2, zorder=5)
            # Legacy suspect (red ring) with rank label.
            ax.scatter([xL], [yL], s=240, facecolors="none",
                       edgecolors=SUSPECT_COLOR, linewidths=2.5, zorder=5)
            ax.text(xL, yL, str(col + 1), fontsize=9, fontweight="bold",
                    color=SUSPECT_COLOR, ha="center", va="center", zorder=6)
            ax.text(xH, yH, str(col + 1), fontsize=8, fontweight="bold",
                    color=PARTNER_COLOR, ha="center", va="center", zorder=6)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("UMAP Component 1")
        ax.set_ylabel("UMAP Component 2")
        # Custom legend including the suspect/partner markers.
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=plotter.COLOR_HSC,
                   markeredgecolor='black', markersize=8, label='HSC'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=plotter.COLOR_LEGACY,
                   markeredgecolor='black', markersize=8, label='Legacy'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                   markeredgecolor=SUSPECT_COLOR, markeredgewidth=2.2, markersize=12,
                   label='Suspect (Legacy)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                   markeredgecolor=PARTNER_COLOR, markeredgewidth=2.2, markersize=12,
                   label='HSC partner'),
        ]
        ax.legend(handles=handles, loc='best', fontsize=9)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(20, 8))
    _draw_overlay(axL, hsc_umap_1, legacy_umap_1,
                  f"Physics latent — top-{K} suspects + pairs")
    _draw_overlay(axR, hsc_umap_2, legacy_umap_2,
                  f"Instrument latent — top-{K} Legacy-in-HSC suspects + pairs")
    plt.suptitle(f"Anomaly suspects ({args.tag}) — red = Legacy suspect, green = HSC partner",
                 fontsize=12)
    plt.tight_layout()
    overlay_path = LATENT_SPACE_DIR / f"anomalies_{args.tag}_top{K}_umap.png"
    plt.savefig(overlay_path, dpi=150)
    plt.close()
    print(f"[anom] saved {overlay_path}")

    # --- Stdout table + save provenance JSON ---
    table_rows = []
    print("rank\tpair_idx\trow_idx\tscore\tra\tdec\tobject_id_hsc\tobject_id_legacy")
    for col, pair_idx in enumerate(ordered):
        m = pair_to_meta[pair_idx]
        s = float(legacy_hsc_frac[pair_idx])
        line = f"{col+1}\t{pair_idx}\t{m['row_idx']}\t{s:.3f}\t{m['ra']:.5f}\t{m['dec']:.5f}\t{m['object_id_hsc']}\t{m['object_id_legacy']}"
        print(line)
        table_rows.append({
            "rank": col + 1, "pair_idx": int(pair_idx), "row_idx": m["row_idx"],
            "score": s, "ra": m["ra"], "dec": m["dec"],
            "object_id_hsc": m["object_id_hsc"], "object_id_legacy": m["object_id_legacy"],
        })
    json_path = LATENT_SPACE_DIR / f"anomalies_{args.tag}_top{K}.json"
    json_path.write_text(json.dumps(table_rows, indent=2))

    # --- Discord push: overlay first, composite next ---
    if args.webhook:
        notify(args.webhook,
               f"🔎 **Instrument-latent anomalies** ({args.tag}) — top {K} Legacy points "
               f"deepest in the HSC island. score = HSC fraction among k={args.knn_k} "
               f"nearest UMAP neighbors.\nrow_idx ordered (rank 1→{K}): "
               f"{[pair_to_meta[i]['row_idx'] for i in ordered]}",
               file_path=overlay_path)
        notify(args.webhook,
               f"📸 HSC + Legacy thumbnails for the {K} suspects (rank-ordered, "
               f"top row HSC / bottom row Legacy):",
               file_path=composite_path)


if __name__ == "__main__":
    main()
