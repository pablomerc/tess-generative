"""Render a 1x3 UMAP figure: ours-Physics | ours-Instrument | AION.

Loads:
  - latent_space/umap_both_encoders_zdim{dim}_zoom_flat_{tag}_data.npz  (panels 1+2)
  - <aion-npz> with keys hsc_umap, legacy_umap                          (panel 3)

The pair markers (selected_indices) are taken from OUR npz so all three panels
share the same pair set by construction.

Reuses _draw_panel from plot_umap_from_file.py and the existing style constants.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
LATENT_SPACE_DIR = SCRIPT_DIR / "neighbors_visualization" / "latent_space"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import plot_umap_from_file as plotter
from discord_notify import notify


def _resolve_ours_npz(tag: str) -> Path:
    candidates = sorted(LATENT_SPACE_DIR.glob(f"umap_both_encoders_zdim*_zoom_flat_{tag}_data.npz"))
    if not candidates:
        raise FileNotFoundError(f"No saved UMAP data for tag '{tag}' in {LATENT_SPACE_DIR}")
    if len(candidates) > 1:
        raise RuntimeError(f"Ambiguous tag '{tag}', got {len(candidates)} candidates: {candidates}")
    return candidates[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, type=str,
                        help="Tag of the saved 'ours' UMAP, e.g. base_n4096_s314")
    parser.add_argument("--aion-npz", required=True, type=Path,
                        help="Path to the AION UMAP npz (keys: hsc_umap, legacy_umap, anchor_row_idx)")
    parser.add_argument("--no-pair-lines", action="store_true",
                        help="Hide connecting lines between pair members on all panels")
    parser.add_argument("--alpha-e1", type=float, default=plotter.ALPHA)
    parser.add_argument("--alpha-e2", type=float, default=plotter.ALPHA_E2)
    parser.add_argument("--alpha-aion", type=float, default=plotter.ALPHA,
                        help="Alpha for AION panel (defaults to encoder-1 alpha)")
    parser.add_argument("--point-size", type=float, default=plotter.POINT_SIZE)
    parser.add_argument("--pair-line-width", type=float, default=plotter.PAIR_LINE_WIDTH)
    parser.add_argument("--pair-line-alpha", type=float, default=plotter.PAIR_LINE_ALPHA)
    parser.add_argument("--pair-line-style", type=str, default=plotter.PAIR_LINE_STYLE)
    parser.add_argument("--science", nargs="*", default=None,
                        help="Apply SciencePlots styles, e.g. --science science no-latex")
    parser.add_argument("--suffix", type=str, default="_three_panels")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=plotter.DPI)
    parser.add_argument("--webhook", type=str, default=os.environ.get("DISCORD_WEBHOOK", ""))
    args = parser.parse_args()

    science_style = None
    if args.science is not None:
        try:
            import scienceplots  # noqa: F401
        except ImportError:
            print("[render-3p] scienceplots not installed; --science ignored", file=sys.stderr)
        else:
            science_style = list(args.science) if args.science else ["science"]

    # --- Load OURS npz (panels 1+2 + selected_indices) ---
    ours_npz = _resolve_ours_npz(args.tag)
    print(f"[render-3p] ours: {ours_npz}")
    with np.load(ours_npz, allow_pickle=False) as z:
        hsc_umap_1 = z["hsc_umap_1"]
        legacy_umap_1 = z["legacy_umap_1"]
        hsc_umap_2 = z["hsc_umap_2"]
        legacy_umap_2 = z["legacy_umap_2"]
        selected_indices = z["selected_indices"]
    if selected_indices.size == 0:
        selected_indices = None

    # --- Load AION npz (panel 3) ---
    aion_npz_path = Path(args.aion_npz)
    print(f"[render-3p] aion: {aion_npz_path}")
    with np.load(aion_npz_path, allow_pickle=False) as z:
        hsc_umap_aion = z["hsc_umap"]
        legacy_umap_aion = z["legacy_umap"]
        if "anchor_row_idx" in z.files:
            anchor_row_idx = z["anchor_row_idx"]
        else:
            anchor_row_idx = None

    # Sanity: same number of points
    n_ours = hsc_umap_1.shape[0]
    n_aion = hsc_umap_aion.shape[0]
    if n_ours != n_aion:
        raise ValueError(f"Point count mismatch: ours has {n_ours} pairs, AION has {n_aion}")

    # --- Build figure ---
    pair_markers = ["x", "s", "o", "^"]
    pair_colors = plotter.PAIR_COLORS

    style_ctx = plt.style.context(science_style) if science_style else plt.style.context("default")
    with style_ctx:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 8))

        plotter._draw_panel(
            ax1, hsc_umap_1, legacy_umap_1, selected_indices,
            color_hsc=plotter.COLOR_HSC, color_legacy=plotter.COLOR_LEGACY,
            point_size=args.point_size, alpha=args.alpha_e1,
            show_pairs=selected_indices is not None,
            pair_colors=pair_colors, pair_markers=pair_markers,
            pair_marker_size=plotter.PAIR_MARKER_SIZE, pair_linewidths=plotter.PAIR_LINEWIDTHS,
            show_pair_lines=not args.no_pair_lines, pair_line_width=args.pair_line_width,
            pair_line_alpha=args.pair_line_alpha, pair_line_style=args.pair_line_style,
            title="Physics Latent Space",
        )
        plotter._draw_panel(
            ax2, hsc_umap_2, legacy_umap_2, selected_indices,
            color_hsc=plotter.COLOR_HSC, color_legacy=plotter.COLOR_LEGACY,
            point_size=args.point_size, alpha=args.alpha_e2,
            show_pairs=selected_indices is not None,
            pair_colors=pair_colors, pair_markers=pair_markers,
            pair_marker_size=plotter.PAIR_MARKER_SIZE, pair_linewidths=plotter.PAIR_LINEWIDTHS,
            show_pair_lines=not args.no_pair_lines, pair_line_width=args.pair_line_width,
            pair_line_alpha=args.pair_line_alpha, pair_line_style=args.pair_line_style,
            title="Instrument Latent Space",
        )
        plotter._draw_panel(
            ax3, hsc_umap_aion, legacy_umap_aion, selected_indices,
            color_hsc=plotter.COLOR_HSC, color_legacy=plotter.COLOR_LEGACY,
            point_size=args.point_size, alpha=args.alpha_aion,
            show_pairs=selected_indices is not None,
            pair_colors=pair_colors, pair_markers=pair_markers,
            pair_marker_size=plotter.PAIR_MARKER_SIZE, pair_linewidths=plotter.PAIR_LINEWIDTHS,
            show_pair_lines=not args.no_pair_lines, pair_line_width=args.pair_line_width,
            pair_line_alpha=args.pair_line_alpha, pair_line_style=args.pair_line_style,
            title="AION Latent Space",
        )

        plt.tight_layout()
        if args.out is None:
            out = LATENT_SPACE_DIR / f"umap_three_panels_{args.tag}{args.suffix}.png"
        else:
            out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=args.dpi)
        plt.close()
    print(f"[render-3p] saved {out}")

    if args.webhook:
        msg = (f"🖼️ 3-panel UMAP for `{args.tag}` "
               f"(ours-Physics | ours-Instrument | AION). "
               f"science={'on' if science_style else 'off'}, "
               f"pair_lines={'off' if args.no_pair_lines else 'on'}.")
        notify(args.webhook, msg, file_path=out)


if __name__ == "__main__":
    main()
