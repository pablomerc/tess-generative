"""
Compare downstream instrument-property predictions across multiple contrastive model variants.
Band-specific properties (G/R/I/Z) are averaged into a single value per property.

Usage:
  python compare_variants.py \
    --csv predict_all_contrastive_contrastive_300k.csv      --label "Neighbors (orig)" \
    --csv predict_all_contrastive_contrastive_varA.csv      --label "Neighbors + phys-neg" \
    --csv predict_all_contrastive_contrastive_varB.csv      --label "Random pos" \
    --csv predict_all_contrastive_contrastive_varC.csv      --label "Random pos + phys-neg" \
    --output-dir .

Add --diagnostic to also produce a single-model breakdown for the first CSV.
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


# ── Property groups (members are averaged across bands) ───────────────────────

LEGACY_INSTRUMENT_GROUPS = [
    ("Galaxy Depth", ["legacy_GALDEPTH_G", "legacy_GALDEPTH_R", "legacy_GALDEPTH_I", "legacy_GALDEPTH_Z"]),
    ("PSF Depth",    ["legacy_PSFDEPTH_G", "legacy_PSFDEPTH_R", "legacy_PSFDEPTH_I", "legacy_PSFDEPTH_Z"]),
    ("PSF Size",     ["legacy_PSFSIZE_G",  "legacy_PSFSIZE_R",  "legacy_PSFSIZE_I",  "legacy_PSFSIZE_Z"]),
    ("# Obs",        ["legacy_NOBS_G",     "legacy_NOBS_R",     "legacy_NOBS_I",     "legacy_NOBS_Z"]),
]

HSC_INSTRUMENT_GROUPS = [
    ("PSF FWHM", ["hsc_g_psf_fwhm", "hsc_i_psf_fwhm", "hsc_r_psf_fwhm", "hsc_z_psf_fwhm"]),
    ("Variance",  ["hsc_i_variance_value", "hsc_r_variance_value", "hsc_z_variance_value"]),
]

# ── Colors ────────────────────────────────────────────────────────────────────

PHYSICS_COLOR    = "#5B9BD5"
INSTRUMENT_COLOR = "#E05252"
UNTRAINED_COLOR  = "#AAAAAA"
RANDOM_COLOR     = "#CCCCCC"

HATCHES = ["", "//", "xx", ".."]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_csv(path):
    df = pd.read_csv(path)
    for col in ["r2_physics", "r2_instrument", "r2_untrained", "r2_mean"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def group_r2(df, members, col):
    """Average R² across band members. Returns (mean_r2, any_negative)."""
    vals = []
    for t in members:
        row = df[df["target"] == t]
        if not row.empty and col in row.columns:
            v = row.iloc[0][col]
            if pd.notna(v):
                vals.append(float(v))
    if not vals:
        return np.nan, False
    mean_val = float(np.mean(vals))
    return mean_val, mean_val < 0.0


def clip_r2(val):
    if np.isnan(val):
        return 0.0
    return max(0.0, val)


# ── Multi-variant comparison plot ─────────────────────────────────────────────

def plot_comparison(variant_dfs, variant_labels, output_path, show_legacy=True, show_hsc=True):
    groups = []
    group_survey = []
    if show_legacy:
        for name, members in LEGACY_INSTRUMENT_GROUPS:
            groups.append((name, members))
            group_survey.append("legacy")
    if show_hsc:
        for name, members in HSC_INSTRUMENT_GROUPS:
            groups.append((name, members))
            group_survey.append("hsc")

    if not groups:
        print("[compare_variants] No groups to plot.")
        return

    n_variants = len(variant_dfs)
    n_groups   = len(groups)
    bars_per_group = 3  # physics / instrument / untrained
    group_width = bars_per_group * n_variants + 1

    fig_w = max(12, n_groups * group_width * 0.22)
    fig, ax = plt.subplots(figsize=(fig_w, 6))

    group_positions = np.arange(n_groups) * group_width
    bar_width = 0.75
    legend_handles = []

    for vi, (df, label) in enumerate(zip(variant_dfs, variant_labels)):
        hatch = HATCHES[vi % len(HATCHES)]
        phys_vals, inst_vals, untr_vals = [], [], []
        phys_neg,  inst_neg,  untr_neg  = [], [], []

        for name, members in groups:
            pv, pn = group_r2(df, members, "r2_physics")
            iv, in_ = group_r2(df, members, "r2_instrument")
            uv, un = group_r2(df, members, "r2_untrained")
            phys_vals.append(clip_r2(pv)); phys_neg.append(pn)
            inst_vals.append(clip_r2(iv)); inst_neg.append(in_)
            untr_vals.append(clip_r2(uv)); untr_neg.append(un)

        offset_phys = group_positions + (vi * bars_per_group + 0) * bar_width
        offset_inst = group_positions + (vi * bars_per_group + 1) * bar_width
        offset_untr = group_positions + (vi * bars_per_group + 2) * bar_width

        ax.bar(offset_phys, phys_vals, bar_width, color=PHYSICS_COLOR,    hatch=hatch, edgecolor="white", linewidth=0.5, alpha=0.85)
        ax.bar(offset_inst, inst_vals, bar_width, color=INSTRUMENT_COLOR,  hatch=hatch, edgecolor="white", linewidth=0.5, alpha=0.85)
        ax.bar(offset_untr, untr_vals, bar_width, color=UNTRAINED_COLOR,   hatch=hatch, edgecolor="white", linewidth=0.5, alpha=0.85)

        for i in range(n_groups):
            if phys_neg[i]: ax.text(offset_phys[i], 0.01, "▼", ha="center", va="bottom", fontsize=6, color=PHYSICS_COLOR)
            if inst_neg[i]: ax.text(offset_inst[i], 0.01, "▼", ha="center", va="bottom", fontsize=6, color=INSTRUMENT_COLOR)
            if untr_neg[i]: ax.text(offset_untr[i], 0.01, "▼", ha="center", va="bottom", fontsize=6, color=UNTRAINED_COLOR)

        patch = mpatches.Patch(facecolor="#888888", hatch=hatch, edgecolor="white", label=label)
        legend_handles.append(patch)

    legend_handles += [
        mpatches.Patch(facecolor=PHYSICS_COLOR,    edgecolor="grey", label="Physics encoder"),
        mpatches.Patch(facecolor=INSTRUMENT_COLOR,  edgecolor="grey", label="Instrument encoder"),
        mpatches.Patch(facecolor=UNTRAINED_COLOR,   edgecolor="grey", label="Untrained (sanity check)"),
    ]

    group_centers = group_positions + (n_variants * bars_per_group / 2 - 0.5) * bar_width
    ax.set_xticks(group_centers)
    ax.set_xticklabels([g[0] for g in groups], rotation=30, ha="right", fontsize=10)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("R² (band-averaged)")
    ax.set_title("Instrument Property Prediction — Contrastive Variants")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, ncol=2)

    # Background shading per survey section.
    if show_legacy and show_hsc:
        n_leg = len(LEGACY_INSTRUMENT_GROUPS)
        if n_leg > 0:
            x0 = group_positions[0] - bar_width
            x1 = group_positions[n_leg - 1] + n_variants * bars_per_group * bar_width + bar_width
            ax.axvspan(x0, x1, alpha=0.04, color="steelblue", zorder=0)
            ax.text((x0 + x1) / 2, 1.02, "Legacy Survey", ha="center", fontsize=9,
                    color="steelblue", transform=ax.get_xaxis_transform())
        if len(groups) > n_leg:
            x0 = group_positions[n_leg] - bar_width
            x1 = group_positions[-1] + n_variants * bars_per_group * bar_width + bar_width
            ax.axvspan(x0, x1, alpha=0.04, color="goldenrod", zorder=0)
            ax.text((x0 + x1) / 2, 1.02, "HSC", ha="center", fontsize=9,
                    color="goldenrod", transform=ax.get_xaxis_transform())

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ── Single-model diagnostic plot ──────────────────────────────────────────────

def plot_diagnostic(df, label, output_path):
    all_groups = LEGACY_INSTRUMENT_GROUPS + HSC_INSTRUMENT_GROUPS
    survey_labels = (["Legacy"] * len(LEGACY_INSTRUMENT_GROUPS) +
                     ["HSC"]    * len(HSC_INSTRUMENT_GROUPS))

    names = [g[0] for g in all_groups]
    n = len(all_groups)

    phys_vals, inst_vals, untr_vals, rand_vals = [], [], [], []
    phys_neg,  inst_neg,  untr_neg,  rand_neg  = [], [], [], []

    for _, members in all_groups:
        for col, lst, neg_lst in [
            ("r2_physics",    phys_vals, phys_neg),
            ("r2_instrument", inst_vals, inst_neg),
            ("r2_untrained",  untr_vals, untr_neg),
            ("r2_mean",       rand_vals, rand_neg),
        ]:
            v, neg = group_r2(df, members, col)
            lst.append(clip_r2(v))
            neg_lst.append(neg)

    x = np.arange(n)
    width = 0.2
    fig, ax = plt.subplots(figsize=(max(10, n * 0.9), 6))

    ax.bar(x - 1.5 * width, phys_vals, width, label="Physics encoder",       color=PHYSICS_COLOR)
    ax.bar(x - 0.5 * width, inst_vals, width, label="Instrument encoder",     color=INSTRUMENT_COLOR)
    ax.bar(x + 0.5 * width, untr_vals, width, label="Untrained (same arch)",  color=UNTRAINED_COLOR)
    ax.bar(x + 1.5 * width, rand_vals, width, label="Mean baseline",          color=RANDOM_COLOR)

    for i in range(n):
        if phys_neg[i]: ax.text(x[i] - 1.5 * width, 0.01, "▼", ha="center", fontsize=6, color=PHYSICS_COLOR)
        if inst_neg[i]: ax.text(x[i] - 0.5 * width, 0.01, "▼", ha="center", fontsize=6, color=INSTRUMENT_COLOR)
        if untr_neg[i]: ax.text(x[i] + 0.5 * width, 0.01, "▼", ha="center", fontsize=6, color=UNTRAINED_COLOR)
        if rand_neg[i]: ax.text(x[i] + 1.5 * width, 0.01, "▼", ha="center", fontsize=6, color="grey")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=10)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("R² (band-averaged)")
    ax.set_title(f"Instrument Property Prediction Diagnostic — {label}")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    # Survey shading
    n_leg = len(LEGACY_INSTRUMENT_GROUPS)
    if n_leg > 0:
        ax.axvspan(-0.5, n_leg - 0.5, alpha=0.04, color="steelblue", zorder=0)
        ax.text((n_leg - 1) / 2, 1.02, "Legacy Survey", ha="center", fontsize=9,
                color="steelblue", transform=ax.get_xaxis_transform())
    if n > n_leg:
        ax.axvspan(n_leg - 0.5, n - 0.5, alpha=0.04, color="goldenrod", zorder=0)
        ax.text(n_leg + (n - n_leg - 1) / 2, 1.02, "HSC", ha="center", fontsize=9,
                color="goldenrod", transform=ax.get_xaxis_transform())

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Compare instrument-property R² (band-averaged) across contrastive model variants.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--csv",   action="append", dest="csvs",   metavar="PATH",  required=True)
    p.add_argument("--label", action="append", dest="labels", metavar="LABEL", required=True)
    p.add_argument("--output-dir", type=Path, default=Path("."))
    p.add_argument("--diagnostic", action="store_true",
                   help="Also produce a single-model diagnostic plot for the first CSV")
    p.add_argument("--no-legacy", action="store_true")
    p.add_argument("--no-hsc",    action="store_true")
    args = p.parse_args()

    if len(args.csvs) != len(args.labels):
        p.error("Number of --csv and --label arguments must match.")

    dfs = [load_csv(c) for c in args.csvs]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    suffix = "_".join(re.sub(r"[^a-zA-Z0-9]", "", lbl)[:8] for lbl in args.labels)
    plot_comparison(dfs, args.labels,
                    args.output_dir / f"compare_variants_{suffix}.png",
                    show_legacy=not args.no_legacy,
                    show_hsc=not args.no_hsc)

    if args.diagnostic or len(dfs) == 1:
        slug = re.sub(r"[^a-zA-Z0-9]", "", args.labels[0])[:20]
        plot_diagnostic(dfs[0], args.labels[0],
                        args.output_dir / f"diagnostic_{slug}.png")

    print("Done.")


if __name__ == "__main__":
    main()
