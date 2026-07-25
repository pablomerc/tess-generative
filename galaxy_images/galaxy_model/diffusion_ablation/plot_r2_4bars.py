#!/usr/bin/env python3
"""Paper-style R2 figure for the diffusion ablation: 4 bars per property.

Clones the layout of downstream_evaluation/engaging/outputs/plot_paper/make_plot.py
(group shading, gaps, band-averaged targets, fonts, legend) with the series
replaced by: FM (Physics), DDPM (Physics), FM (Instrument), DDPM (Instrument),
read from two predict_*.csv probe outputs. Hue encodes the latent (blue=physics,
red=instrument, matching the paper); tint + hatch encode the model (DDPM hatched).

Usage:
    python plot_r2_4bars.py --fm-csv <predict_fm...csv> --ddpm-csv <predict_ddpm...csv> \
        --out-stem results/r2_4bars_74k58k [--style scienceplots]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# --- frozen copies of the paper figure's style assets -----------------------
GROUP_PHYSICS = "Physics-like Properties"
GROUP_LEGACY = "Legacy"
GROUP_HSC = "HSC"
GAP_SIZE = 0.5
BG_COLORS = {GROUP_PHYSICS: "#c0c0c0", GROUP_LEGACY: "#8eb8e8", GROUP_HSC: "#e8c4a0"}
BG_ALPHA = 0.85
GROUP_LABEL_COLORS = {GROUP_LEGACY: "#2563a8", GROUP_HSC: "#996515"}

LABEL_MAPPING = {
    "desi_z": r"$z$",
    "provabgs_logmstar": r"$\log M_\star$",
    "provabgs_tage_mw": r"$t_{\mathrm{age}}$",
    "provabgs_z_mw": r"$\log Z_{\mathrm{Met}}$",
    "ssfr": r"sSFR",
    "legacy_shape_e1": r"Ellipticity $e_1$",
    "legacy_shape_e2": r"Ellipticity $e_2$",
    "hsc_a": "Extinction a",
    "hsc_psf_fwhm": "PSF size",
    "legacy_PSFSIZE": "PSF size",
    "legacy_PSFDEPTH": "PSF Depth",
    "legacy_GALDEPTH": "Galaxy Depth",
    "legacy_NOBS": "# Observations",
    "legacy_EBV": "Extinction E(B-V)",
}

# Paper row order (from plot_paper/data/merged_simplified.csv) + the band sets
# each row averages over, mapped to the raw target names in our predict CSVs.
ROWS = [
    ("legacy_shape_e2", GROUP_PHYSICS, ["legacy_shape_e2"]),
    ("legacy_shape_e1", GROUP_PHYSICS, ["legacy_shape_e1"]),
    ("desi_z", GROUP_PHYSICS, ["desi_z"]),
    ("provabgs_logmstar", GROUP_PHYSICS, ["provabgs_logmstar"]),
    ("ssfr", GROUP_PHYSICS, ["ssfr"]),
    ("provabgs_tage_mw", GROUP_PHYSICS, ["provabgs_tage_mw"]),
    ("provabgs_z_mw", GROUP_PHYSICS, ["provabgs_z_mw"]),
    ("legacy_GALDEPTH", GROUP_LEGACY, [f"GALDEPTH_{b}" for b in "GRIZ"]),
    ("legacy_PSFDEPTH", GROUP_LEGACY, [f"PSFDEPTH_{b}" for b in "GRIZ"]),
    ("legacy_PSFSIZE", GROUP_LEGACY, [f"PSFSIZE_{b}" for b in "GRIZ"]),
    ("legacy_EBV", GROUP_LEGACY, ["EBV"]),
    ("legacy_NOBS", GROUP_LEGACY, [f"NOBS_{b}" for b in "GRIZ"]),
    ("hsc_a", GROUP_HSC, [f"a_{b}" for b in "grizy"]),
    ("hsc_psf_fwhm", GROUP_HSC, [f"psf_fwhm_{b}" for b in "grizy"]),
]

# (label, csv-role, latent, color, hatch) — hue = latent (paper colors),
# tint+hatch = model, so adjacent same-hue bars stay separable in print/CVD.
SERIES = [
    ("FM (Physics)", "fm", "combined_e1", "#2E86AB", False),
    ("DDPM (Physics)", "ddpm", "combined_e1", "#8FC3DB", True),
    ("FM (Instrument)", "fm", "combined_e2", "#CC546D", False),
    ("DDPM (Instrument)", "ddpm", "combined_e2", "#EBA9B8", True),
]


def load_scores(csv_path: Path) -> dict[tuple[str, str], float]:
    df = pd.read_csv(csv_path)
    out = {}
    for _, r in df.iterrows():
        try:
            out[(str(r["target"]), str(r["latent_variant"]))] = float(r["score"])
        except (KeyError, ValueError):
            continue
    return out


def row_value(scores: dict, latent: str, raw_targets: list[str]) -> float:
    vals = [scores[(t, latent)] for t in raw_targets if (t, latent) in scores]
    return float(np.mean(vals)) if vals else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fm-csv", type=Path, required=True)
    ap.add_argument("--ddpm-csv", type=Path, required=True)
    ap.add_argument("--out-stem", type=Path, required=True)
    ap.add_argument("--style", choices=["scienceplots", "default"], default="scienceplots")
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    if args.style == "scienceplots":
        try:
            import scienceplots  # noqa: F401

            plt.style.use(["science", "no-latex"])
        except Exception as e:  # noqa: BLE001
            print(f"[warn] scienceplots not available ({e}); falling back to default")

    scores = {"fm": load_scores(args.fm_csv), "ddpm": load_scores(args.ddpm_csv)}

    targets = [t for t, _, _ in ROWS]
    groups = [g for _, g, _ in ROWS]
    display_labels = [LABEL_MAPPING.get(t, t) for t in targets]

    # X positions with a gap between groups (identical to make_plot.py).
    x_positions, current_x, last_group, group_bounds = [], 0.0, None, {}
    for g in groups:
        if last_group is not None and g != last_group:
            current_x += GAP_SIZE
        group_bounds.setdefault(g, {"min": current_x, "max": current_x})["max"] = current_x
        x_positions.append(current_x)
        last_group = g
        current_x += 1.25
    x = np.array(x_positions)

    n = len(SERIES)
    width = 0.95 / n
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

    fig_width = max(8, (x[-1] - x[0] + 2) * 0.7)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    for i, (label, role, latent, color, hatch) in enumerate(SERIES):
        vals = np.array([row_value(scores[role], latent, raw) for _, _, raw in ROWS])
        ax.bar(
            x + offsets[i], vals, width, label=label, color=color, alpha=0.88,
            edgecolor="black", linewidth=0.5, hatch="///" if hatch else None, zorder=3,
        )

    for g in (GROUP_PHYSICS, GROUP_LEGACY, GROUP_HSC):
        if g in group_bounds:
            ax.axvspan(group_bounds[g]["min"] - 0.5, group_bounds[g]["max"] + 0.5,
                       facecolor=BG_COLORS[g], alpha=BG_ALPHA, zorder=0)

    SUPER_TITLE_Y, SUB_TITLE_Y = 1.03, 0.83
    SUPER_FS, SUB_FS = 21, 19
    gb = group_bounds[GROUP_PHYSICS]
    ax.text((gb["min"] + gb["max"]) / 2, SUPER_TITLE_Y, "Physics-Related Properties",
            ha="center", va="bottom", fontsize=SUPER_FS, fontweight="bold",
            color="black", transform=ax.get_xaxis_transform())
    for g, sub in ((GROUP_LEGACY, "Legacy"), (GROUP_HSC, "HSC")):
        gb = group_bounds[g]
        ax.text((gb["min"] + gb["max"]) / 2, SUB_TITLE_Y, sub, ha="center", va="center",
                fontsize=SUB_FS, fontweight="bold", color=GROUP_LABEL_COLORS[g],
                transform=ax.get_xaxis_transform())
    starts = [group_bounds[g]["min"] - 0.5 for g in (GROUP_LEGACY, GROUP_HSC)]
    ends = [group_bounds[g]["max"] + 0.5 for g in (GROUP_LEGACY, GROUP_HSC)]
    ax.text((min(starts) + max(ends)) / 2, SUPER_TITLE_Y, "Instrument Properties",
            ha="center", va="bottom", fontsize=SUPER_FS, fontweight="bold",
            color="black", transform=ax.get_xaxis_transform())

    margin = max(np.abs(offsets)) + width / 2 + 0.15
    ax.set_xlim(x[0] - margin, x[-1] + margin)
    ax.set_ylabel("R²", fontsize=19)
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=17)
    ax.tick_params(axis="x", which="both", length=0, top=False, bottom=False)
    ax.tick_params(axis="y", labelsize=15)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.00), ncol=n,
              fontsize=15, frameon=True)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(axis="y", which="major", alpha=0.9, linestyle="--", linewidth=0.9,
            color="#888888", zorder=1)
    ax.grid(axis="y", which="minor", alpha=0.7, linestyle=":", linewidth=0.6,
            color="#888888", zorder=1)
    ax.axhline(0, color="gray", linewidth=0.7, zorder=2)
    if args.title:
        ax.set_title(args.title, fontsize=13, pad=36)

    plt.tight_layout()
    args.out_stem.parent.mkdir(parents=True, exist_ok=True)
    png = args.out_stem.with_suffix(".png")
    plt.savefig(png, dpi=150, bbox_inches="tight")
    plt.savefig(args.out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png} and .pdf")


if __name__ == "__main__":
    main()
