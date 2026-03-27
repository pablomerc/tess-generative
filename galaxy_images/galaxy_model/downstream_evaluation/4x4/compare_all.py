"""
Compare Ours (instrument-4x4) vs AION vs Baseline on downstream R².

Reads:
  - Ours:    4x4/predict_all_ours_ours_instrument4x4.csv
  - AION:    aion_benchmark/r2_comparisons_aion/downstream_aion_512-256-128.csv
  - AION legacy: aion_benchmark/aion_legacy/r2_comparisons_aion_legacy_v2/downstream_aion_legacy_train_512-256-128.csv
  - AION MMU: aion_benchmark/aion_mmu/r2_comparisons_aion_mmu_v2/downstream_aion_mmu_matched_512-256-128.csv
  - Baseline: downstream_evaluation/final/predict_baseline.csv

Produces:
  - compare_all.png          grouped bar chart
  - compare_all_merged.csv   merged table

Usage:
  python compare_all.py
  python compare_all.py --baseline          # include resnet baseline
  python compare_all.py --no-scienceplots   # skip scienceplots if not installed
"""
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

_here = Path(__file__).resolve().parent
_downstream = _here.parent
_galaxy_model = _downstream.parent
_aion = _galaxy_model / "aion_benchmark"
_final = _downstream / "final"

OURS_CSV = _here / "predict_all_ours_ours_instrument4x4.csv"

AION_CSV = _aion / "r2_comparisons_aion" / "downstream_aion_512-256-128.csv"
AION_LEGACY_CSV = _aion / "aion_legacy" / "r2_comparisons_aion_legacy_v2" / "downstream_aion_legacy_train_512-256-128.csv"
AION_MMU_CSV = _aion / "aion_mmu" / "r2_comparisons_aion_mmu_v2" / "downstream_aion_mmu_matched_512-256-128.csv"
BASELINE_CSV = _final / "predict_baseline.csv"

OUT_CSV = _here / "compare_all_merged.csv"
OUT_PNG = _here / "compare_all.png"

AVERAGE_PATTERNS = {
    r"^a_": "a",
    r"^legacy_GALDEPTH_": "legacy_GALDEPTH",
    r"^legacy_NOBS_": "legacy_NOBS",
    r"^legacy_PSFSIZE_": "legacy_PSFSIZE",
    r"^legacy_PSFDEPTH_": "legacy_PSFDEPTH",
    r"^hsc_.*_variance_value$": "hsc_variance_value",
    r"^hsc_.*_psf_fwhm$": "hsc_psf_fwhm",
}

EXCLUDE_TARGETS = {"hsc_z_psf_fwhm", "hsc_variance_value", "DEC",
                   "hsc_g_extendedness_value", "i_extendedness_value"}

BAR_COLORS = ["#2E86AB", "#CC546D", "#6CA75D", "#9473C6", "#BF823B"]

GROUP_PHYSICS = "Physics-like Properties"
GROUP_LEGACY = "Legacy"
GROUP_HSC = "HSC"
GAP_SIZE = 0.5

BG_COLORS = {
    GROUP_PHYSICS: "#c0c0c0",
    GROUP_LEGACY: "#8eb8e8",
    GROUP_HSC: "#e8c4a0",
}
BG_ALPHA = 0.85

GROUP_LABEL_COLORS = {
    GROUP_PHYSICS: "#888888",
    GROUP_LEGACY: "#2563a8",
    GROUP_HSC: "#996515",
}

LABEL_MAPPING = {
    "desi_Z": r"$Z$",
    "LOG_MSTAR": r"$M_\star$",
    "TAGE_MW": r"$t_{\mathrm{age}}$",
    "LOG_Z_MW": r"$\log Z_{\mathrm{Met}}$",
    "sSFR": r"$\mathrm{sSFR}$",
    "SHAPE_E1": r"Ellipticity $e_1$",
    "SHAPE_E2": r"Ellipticity $e_2$",
    "SHAPE_R": r"$R_{\mathrm{eff}}$",
    "a": "Extinction a",
    "hsc_variance_value": "Variance",
    "hsc_psf_fwhm": "PSF Size",
    "legacy_GALDEPTH": "Galaxy Depth",
    "legacy_NOBS": "# Observations",
    "legacy_PSFSIZE": "PSF Size",
    "legacy_PSFDEPTH": "PSF Depth",
    "EBV": "Extinction E(B-V)",
}


def _get_group(target):
    t = str(target)
    if t in ("a", "hsc_variance_value", "hsc_psf_fwhm") or t.startswith("hsc_"):
        return GROUP_HSC
    if t.lower() == "ebv" or t.startswith("legacy_"):
        return GROUP_LEGACY
    return GROUP_PHYSICS


def load_ours():
    df = pd.read_csv(OURS_CSV)
    df = df[~df["target"].isin(EXCLUDE_TARGETS)]
    agg = df.groupby("target", as_index=False).agg({
        "r2_physics": "mean",
        "r2_instrument": "mean",
        "r2_untrained": "mean",
    })
    return agg.rename(columns={
        "r2_physics": "Physics_latents",
        "r2_instrument": "Instrument_latents",
        "r2_untrained": "Untrained",
    })


def load_aion():
    parts = []
    for path in [AION_MMU_CSV, AION_CSV, AION_LEGACY_CSV]:
        if path.exists():
            parts.append(pd.read_csv(path)[["target", "r2_mean"]].rename(columns={"r2_mean": "AION"}))
    if not parts:
        return None
    return pd.concat(parts, ignore_index=True).drop_duplicates(subset=["target"], keep="first")


def load_baseline():
    if not BASELINE_CSV.exists():
        return None
    return pd.read_csv(BASELINE_CSV)[["target", "r2_baseline"]].rename(columns={"r2_baseline": "Baseline"})


def average_by_patterns(df):
    numeric_cols = [c for c in ("Physics_latents", "Instrument_latents", "Untrained", "AION", "Baseline") if c in df.columns]
    to_drop, new_rows = [], []
    for pattern, new_name in AVERAGE_PATTERNS.items():
        mask = df["target"].astype(str).str.contains(pattern, regex=True)
        subset = df.loc[mask].copy()
        if new_name == "hsc_psf_fwhm":
            subset = subset[subset["target"] != "hsc_z_psf_fwhm"]
        if subset.empty:
            continue
        to_drop.extend(subset.index.tolist())
        row = subset[numeric_cols].mean(numeric_only=True).to_dict()
        row["target"] = new_name
        new_rows.append(row)
    df = df.drop(index=to_drop)
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df["group"] = df["target"].apply(_get_group)
    return df


def make_plot(merged, include_baseline):
    group_order = [GROUP_PHYSICS, GROUP_LEGACY, GROUP_HSC]
    merged["group_rank"] = merged["group"].map({g: i for i, g in enumerate(group_order)})
    merged["_sort"] = np.where(
        merged["group_rank"] == 0,
        -merged["Physics_latents"].fillna(-1),
        -merged["Instrument_latents"].fillna(-1),
    )
    merged = merged.sort_values(["group_rank", "_sort"]).reset_index(drop=True)
    merged = merged[~merged["target"].isin(EXCLUDE_TARGETS)].reset_index(drop=True)

    targets = merged["target"].tolist()
    labels = [LABEL_MAPPING.get(t, t) for t in targets]

    x_pos, cur_x, last_g = [], 0.0, None
    group_bounds = {}
    for _, row in merged.iterrows():
        g = row["group"]
        if last_g is not None and g != last_g:
            cur_x += GAP_SIZE
        group_bounds.setdefault(g, {"min": cur_x, "max": cur_x})
        group_bounds[g]["max"] = cur_x
        x_pos.append(cur_x)
        last_g = g
        cur_x += 1.25
    x = np.array(x_pos)

    has_baseline = include_baseline and "Baseline" in merged.columns
    n_series = 5 if has_baseline else 4
    w = 0.18 if n_series == 5 else 0.22
    offsets = np.linspace(-(n_series - 1) / 2, (n_series - 1) / 2, n_series) * w

    fig_w = max(10, cur_x * 0.6)
    fig, ax = plt.subplots(figsize=(fig_w, 6))

    series = [
        ("Physics latents", merged["Physics_latents"].values, BAR_COLORS[0], False),
        ("Instrument latents", merged["Instrument_latents"].values, BAR_COLORS[1], False),
        ("AION-1", merged.get("AION", pd.Series(dtype=float)).reindex(merged.index).values, BAR_COLORS[2], False),
        ("Untrained ResNet-18", merged["Untrained"].values, None, True),
    ]
    if has_baseline:
        series.append(("Baseline", merged["Baseline"].values, BAR_COLORS[4], False))

    for i, (label, vals, color, is_ref) in enumerate(series):
        vals = np.where(np.isfinite(vals), vals, 0.0)
        if is_ref:
            ax.bar(x + offsets[i], vals, w, label=label, color="gray",
                   edgecolor="black", linewidth=0.5, hatch="///", zorder=3)
        else:
            ax.bar(x + offsets[i], vals, w, label=label, color=color,
                   alpha=0.85, edgecolor="black", linewidth=0.5, zorder=3)

    for g in group_order:
        if g not in group_bounds:
            continue
        s, e = group_bounds[g]["min"] - 0.5, group_bounds[g]["max"] + 0.5
        ax.axvspan(s, e, facecolor=BG_COLORS[g], alpha=BG_ALPHA, zorder=0)

    SUPER_Y, SUB_Y = 1.06, 0.86
    SUPER_FS, SUB_FS = 18, 16
    for g in group_order:
        if g not in group_bounds:
            continue
        s, e = group_bounds[g]["min"] - 0.5, group_bounds[g]["max"] + 0.5
        mid = (s + e) / 2
        color = GROUP_LABEL_COLORS[g]
        if g == GROUP_PHYSICS:
            ax.text(mid, SUPER_Y, "Physics-Related Properties", ha="center", va="bottom",
                    fontsize=SUPER_FS, fontweight="bold", color="black",
                    transform=ax.get_xaxis_transform())
        else:
            ax.text(mid, SUB_Y, g, ha="center", va="center",
                    fontsize=SUB_FS, fontweight="bold", color=color,
                    transform=ax.get_xaxis_transform())
    if GROUP_LEGACY in group_bounds and GROUP_HSC in group_bounds:
        inst_s = group_bounds[GROUP_LEGACY]["min"] - 0.5
        inst_e = group_bounds[GROUP_HSC]["max"] + 0.5
        ax.text((inst_s + inst_e) / 2, SUPER_Y, "Instrument Properties",
                ha="center", va="bottom", fontsize=SUPER_FS, fontweight="bold",
                color="black", transform=ax.get_xaxis_transform())
    elif GROUP_LEGACY in group_bounds:
        s, e = group_bounds[GROUP_LEGACY]["min"] - 0.5, group_bounds[GROUP_LEGACY]["max"] + 0.5
        ax.text((s + e) / 2, SUPER_Y, "Instrument Properties",
                ha="center", va="bottom", fontsize=SUPER_FS, fontweight="bold",
                color="black", transform=ax.get_xaxis_transform())

    margin = max(np.abs(offsets)) + w / 2 + 0.15
    ax.set_xlim(x[0] - margin, x[-1] + margin)
    ax.set_ylabel("R²", fontsize=14)
    ax.set_xticks(x + 0.22)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=12)
    ax.tick_params(axis="x", which="both", length=0, top=False, bottom=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.00), ncol=n_series,
              fontsize=12, frameon=True)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(axis="y", which="major", alpha=0.75, linestyle="--", zorder=1)
    ax.grid(axis="y", which="minor", alpha=0.6, linestyle=":", zorder=1)
    ax.axhline(0, color="gray", linewidth=0.7, zorder=2)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {OUT_PNG}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", action="store_true", help="Include resnet baseline series")
    p.add_argument("--no-scienceplots", action="store_true", help="Skip scienceplots style")
    args = p.parse_args()

    if not args.no_scienceplots:
        try:
            import scienceplots  # noqa: F401
            plt.style.use(["science", "no-latex"])
        except ImportError:
            print("scienceplots not installed, using default style")

    ours = load_ours()
    aion = load_aion()
    baseline = load_baseline() if args.baseline else None

    merged = ours.copy()
    if aion is not None:
        merged = merged.merge(aion, on="target", how="left")
    if baseline is not None:
        merged = merged.merge(baseline, on="target", how="left")

    merged = average_by_patterns(merged)
    merged.to_csv(OUT_CSV, index=False)
    print(f"Merged CSV: {OUT_CSV}")

    make_plot(merged, args.baseline)


if __name__ == "__main__":
    main()
