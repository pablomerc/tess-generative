"""For the new redshift-overlap UMAP, find the top-K Legacy points sitting deepest
in the HSC island (Instrument latent, kNN HSC-fraction score), and produce:

  1. A 1x2 figure: Physics / Instrument UMAP (colored by redshift) with the
     suspect Legacy points + their HSC partners ringed and connected.
  2. A redshift histogram of the full sample with the suspects' redshifts
     overplotted as red ticks/markers.

Posts both PNGs to Discord.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.neighbors import NearestNeighbors

SCRIPT_DIR = Path(__file__).resolve().parent
LATENT_SPACE_DIR = SCRIPT_DIR / "neighbors_visualization" / "latent_space"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import plot_umap_from_file as plotter
from discord_notify import notify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path,
                        default=LATENT_SPACE_DIR /
                        "umap_redshift_zdim16_overlap_redshift_data.npz")
    parser.add_argument("--tag", type=str, default="overlap_redshift")
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--knn-k", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--point-size", type=float, default=14)
    parser.add_argument("--webhook", type=str, default=os.environ.get("DISCORD_WEBHOOK", ""))
    args = parser.parse_args()

    print(f"[anom-z] loading {args.npz}")
    with np.load(args.npz, allow_pickle=False) as z:
        hsc_u1 = z["hsc_umap_1"]; leg_u1 = z["legacy_umap_1"]
        hsc_u2 = z["hsc_umap_2"]; leg_u2 = z["legacy_umap_2"]
        redshift = z["redshift"]
        ra = z["ra"] if "ra" in z.files else None
        dec = z["dec"] if "dec" in z.files else None
    n = hsc_u2.shape[0]
    print(f"[anom-z] N={n}, redshift range=[{redshift.min():.3f}, {redshift.max():.3f}]")

    # --- kNN HSC-fraction on the Instrument plane ---
    pts = np.concatenate([hsc_u2, leg_u2], axis=0)
    is_legacy = np.concatenate([np.zeros(n, bool), np.ones(n, bool)], axis=0)
    nbrs = NearestNeighbors(n_neighbors=args.knn_k + 1).fit(pts)
    _, idx = nbrs.kneighbors(pts)
    nn_idx = idx[:, 1:]
    nn_is_legacy = is_legacy[nn_idx]
    hsc_frac = (~nn_is_legacy).mean(axis=1)
    legacy_hsc_frac = hsc_frac[n:]
    order = np.argsort(-legacy_hsc_frac)
    top_pair_idx = order[: args.top_k].astype(int)
    top_scores = legacy_hsc_frac[top_pair_idx]
    print(f"[anom-z] top-{args.top_k} HSC-fractions: "
          f"min={top_scores.min():.2f} max={top_scores.max():.2f}")

    suspect_z = redshift[top_pair_idx]
    print(f"[anom-z] suspect redshifts: "
          f"min={suspect_z.min():.4f} max={suspect_z.max():.4f} median={np.median(suspect_z):.4f}")

    # --- Plot 1: Physics + Instrument UMAP, colored by survey, suspects ringed ---
    SUSPECT_COLOR = "red"
    PARTNER_COLOR = "#2ca02c"
    LINK_COLOR = "#444444"

    def _draw(ax, hsc_xy, legacy_xy, title):
        ax.scatter(hsc_xy[:, 0], hsc_xy[:, 1], s=args.point_size,
                   c=plotter.COLOR_HSC, alpha=args.alpha, label="HSC")
        ax.scatter(legacy_xy[:, 0], legacy_xy[:, 1], s=args.point_size,
                   c=plotter.COLOR_LEGACY, alpha=args.alpha, label="Legacy")
        for col, pi in enumerate(top_pair_idx):
            xL, yL = legacy_xy[pi]
            xH, yH = hsc_xy[pi]
            ax.plot([xL, xH], [yL, yH], color=LINK_COLOR, lw=1.0, alpha=0.7,
                    zorder=4)
            ax.scatter([xH], [yH], s=200, facecolors="none",
                       edgecolors=PARTNER_COLOR, linewidths=2.2, zorder=5)
            ax.scatter([xL], [yL], s=220, facecolors="none",
                       edgecolors=SUSPECT_COLOR, linewidths=2.5, zorder=5)
            ax.text(xL, yL, str(col + 1), fontsize=9, fontweight="bold",
                    color=SUSPECT_COLOR, ha="center", va="center", zorder=6)
            ax.text(xH, yH, str(col + 1), fontsize=8, fontweight="bold",
                    color=PARTNER_COLOR, ha="center", va="center", zorder=6)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("UMAP Component 1")
        ax.set_ylabel("UMAP Component 2")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(20, 8))
    _draw(axL, hsc_u1, leg_u1, f"Physics latent — top-{args.top_k} suspects")
    _draw(axR, hsc_u2, leg_u2, f"Instrument latent — top-{args.top_k} suspects")
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
    axR.legend(handles=handles, loc='best', fontsize=9)
    plt.suptitle(f"Anomalies on overlap dataset ({args.tag}) — colored by survey",
                 fontsize=13)
    plt.tight_layout()
    overlay_path = LATENT_SPACE_DIR / f"anomalies_{args.tag}_top{args.top_k}_umap.png"
    plt.savefig(overlay_path, dpi=150)
    plt.close()
    print(f"[anom-z] saved {overlay_path}")

    # --- Plot 2: Redshift histogram with suspects overplotted ---
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    bins = np.linspace(redshift.min(), redshift.max(), 60)
    ax.hist(redshift, bins=bins, color="#8eb8e8", alpha=0.85,
            edgecolor="black", linewidth=0.4, label=f"All overlap galaxies (N={n})")
    # Mark suspects: rug at the bottom + red histogram on twin axis
    y_top = ax.get_ylim()[1]
    rug_y = -y_top * 0.04
    ax.set_ylim(rug_y * 1.2, y_top)
    ax.scatter(suspect_z, np.full_like(suspect_z, rug_y),
               marker="|", s=400, c="red", linewidths=2.0, zorder=5,
               label=f"Top-{args.top_k} suspects")
    for k, sz in enumerate(suspect_z):
        ax.text(sz, rug_y * 1.7, str(k + 1), fontsize=8, color="red",
                ha="center", va="top")
    # Also overplot a normalized red histogram for the suspects on a twin axis.
    ax2 = ax.twinx()
    ax2.hist(suspect_z, bins=bins, color="red", alpha=0.55,
             edgecolor="darkred", linewidth=0.4)
    ax2.set_ylabel(f"Suspect count (top-{args.top_k})", color="red")
    ax2.tick_params(axis="y", colors="red")

    ax.set_xlabel("DESI redshift")
    ax.set_ylabel("# galaxies (full sample)")
    ax.set_title(
        f"Redshift distribution — overlap dataset (N={n}); suspects overlaid in red\n"
        f"all: median={np.median(redshift):.3f} 5/95={np.percentile(redshift, 5):.3f}/"
        f"{np.percentile(redshift, 95):.3f}  |  "
        f"suspects: median={np.median(suspect_z):.3f} min/max="
        f"{suspect_z.min():.3f}/{suspect_z.max():.3f}",
        fontsize=11,
    )
    ax.legend(loc="upper right")
    plt.tight_layout()
    hist_path = LATENT_SPACE_DIR / f"anomalies_{args.tag}_top{args.top_k}_redshift_hist.png"
    plt.savefig(hist_path, dpi=150)
    plt.close()
    print(f"[anom-z] saved {hist_path}")

    # --- Save table ---
    table = []
    for k, pi in enumerate(top_pair_idx):
        row = {
            "rank": k + 1, "pair_idx": int(pi),
            "score": float(legacy_hsc_frac[pi]),
            "redshift": float(redshift[pi]),
        }
        if ra is not None:
            row["ra"] = float(ra[pi]); row["dec"] = float(dec[pi])
        table.append(row)
    table_path = LATENT_SPACE_DIR / f"anomalies_{args.tag}_top{args.top_k}.json"
    table_path.write_text(json.dumps(table, indent=2))
    print(f"[anom-z] saved {table_path}")
    for r in table:
        print(r)

    # --- Discord push: overlay first, histogram second ---
    if args.webhook:
        notify(args.webhook,
               f"🔎 **Suspects on overlap dataset** ({args.tag}) — top {args.top_k} "
               f"Legacy points deepest in HSC island. score=HSC-frac among k={args.knn_k} "
               f"nearest UMAP neighbors. suspect z: median={np.median(suspect_z):.3f}, "
               f"range=[{suspect_z.min():.3f}, {suspect_z.max():.3f}]",
               file_path=overlay_path)
        notify(args.webhook,
               f"📊 Redshift distribution of all {n} galaxies (blue) with the "
               f"top-{args.top_k} suspects overlaid in red.",
               file_path=hist_path)


if __name__ == "__main__":
    main()
