"""
Compare Contrastive baseline vs Ours on downstream R².

Reads:
  - Contrastive: contrastive_baseline/downstream_evaluation/predict_all_contrastive_{suffix}.csv
  - Ours: downstream_evaluation/final/predict_all_zdim16_nogeom_neighbors_table.csv (or long CSV)

Produces:
  - Merged CSV
  - Grouped bar plot

Run from galaxy_model:
  python contrastive_baseline/downstream_evaluation/contrastive_vs_ours_all.py
  python contrastive_baseline/downstream_evaluation/contrastive_vs_ours_all.py --suffix contrastive_latest
  python contrastive_baseline/downstream_evaluation/contrastive_vs_ours_all.py --include-untrained
"""
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import scienceplots

plt.style.use(["science", "no-latex"])

_script_dir = Path(__file__).resolve().parent
_galaxy_model = _script_dir.parent.parent
_ours_final = _galaxy_model / "downstream_evaluation" / "final"

OURS_TABLE_CSV = _ours_final / "predict_all_zdim16_nogeom_neighbors_table.csv"
OURS_LONG_CSV = _ours_final / "predict_all_zdim16_nogeom_neighbors.csv"

AVERAGE_PATTERNS = {
    r"^a_": "a",
    r"^legacy_GALDEPTH_": "legacy_GALDEPTH",
    r"^legacy_NOBS_": "legacy_NOBS",
    r"^legacy_PSFSIZE_": "legacy_PSFSIZE",
    r"^legacy_PSFDEPTH_": "legacy_PSFDEPTH",
    r"^hsc_.*_variance_value$": "hsc_variance_value",
    r"^hsc_.*_psf_fwhm$": "hsc_psf_fwhm",
}

GROUP_PHYSICS = "Physics-like Properties"
GROUP_LEGACY = "Legacy"
GROUP_HSC = "HSC"
GAP_SIZE = 0.5

EXCLUDE_TARGETS = ["hsc_z_psf_fwhm", "hsc_variance_value"]

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
    "Z": "Redshift",
    "logMstar": "Stellar Mass",
}


def _get_group(target_name):
    t = str(target_name)
    if t in ["a", "hsc_variance_value", "hsc_psf_fwhm"] or t.startswith("hsc_"):
        return GROUP_HSC
    if t.lower() == "ebv" or t.startswith("legacy_"):
        return GROUP_LEGACY
    return GROUP_PHYSICS


def average_by_patterns(df):
    df = df[df["target"] != "DEC"].copy()
    numeric_cols = [
        c
        for c in [
            "Ours_Physics",
            "Ours_Instrument",
            "Contrastive_Physics",
            "Contrastive_Instrument",
            "Ours_Untrained",
            "Contrastive_Untrained",
        ]
        if c in df.columns
    ]
    indices_to_drop = []
    new_rows = []

    for pattern, new_name in AVERAGE_PATTERNS.items():
        mask = df["target"].astype(str).str.contains(pattern, regex=True)
        subset = df.loc[mask].copy()
        if new_name == "hsc_psf_fwhm":
            subset = subset[subset["target"] != "hsc_z_psf_fwhm"]
        if subset.empty:
            continue
        indices_to_drop.extend(subset.index.tolist())
        averaged = subset[numeric_cols].mean(numeric_only=True).to_dict()
        averaged["target"] = new_name
        if "group" in df.columns:
            averaged["group"] = subset.iloc[0]["group"]
        new_rows.append(averaged)

    df = df.drop(index=indices_to_drop)
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df["group"] = df["target"].apply(_get_group)
    return df


def load_ours():
    if OURS_TABLE_CSV.exists():
        df = pd.read_csv(OURS_TABLE_CSV)
        df = df.dropna(subset=["Physics latents", "Instrument latents"], how="all")
        df = df[df["target"].astype(str).str.strip().astype(bool)]
        out = pd.DataFrame(
            {
                "target": df["target"].values,
                "Ours_Physics": df["Physics latents"].values,
                "Ours_Instrument": df["Instrument latents"].values,
                "Ours_Untrained": df["Untrained"].values,
            }
        )
        return out

    if OURS_LONG_CSV.exists():
        df = pd.read_csv(OURS_LONG_CSV).drop_duplicates(subset=["target"], keep="first")
        out = pd.DataFrame(
            {
                "target": df["target"].values,
                "Ours_Physics": df["r2_physics"].values,
                "Ours_Instrument": df["r2_instrument"].values,
                "Ours_Untrained": df["r2_untrained"].values,
            }
        )
        return out

    raise FileNotFoundError(f"Ours CSV not found at {OURS_TABLE_CSV} or {OURS_LONG_CSV}")


def load_contrastive(contrastive_csv, include_hsc_provabgs=False):
    df = pd.read_csv(contrastive_csv)
    if "objective" in df.columns and not include_hsc_provabgs:
        df = df[df["objective"] != "hsc_provabgs"].copy()

    numeric_cols = [c for c in ["r2_physics", "r2_instrument", "r2_untrained"] if c in df.columns]
    # hsc_provabgs overlaps legacy_provabgs targets; collapse to a single target row.
    df = df.groupby("target", as_index=False)[numeric_cols].mean(numeric_only=True)

    return df.rename(
        columns={
            "r2_physics": "Contrastive_Physics",
            "r2_instrument": "Contrastive_Instrument",
            "r2_untrained": "Contrastive_Untrained",
        }
    )


def main():
    p = argparse.ArgumentParser(description="Contrastive baseline vs Ours downstream R² comparison")
    p.add_argument("--suffix", default="contrastive_latest", help="Suffix in predict_all_contrastive_{suffix}.csv")
    p.add_argument("--include-hsc-provabgs", action="store_true", help="Also include hsc_provabgs rows")
    p.add_argument("--include-untrained", action="store_true", help="Also plot both untrained series")
    args = p.parse_args()

    contrastive_csv = _script_dir / f"predict_all_contrastive_{args.suffix}.csv"
    if not contrastive_csv.exists():
        raise FileNotFoundError(f"Missing contrastive CSV: {contrastive_csv}")

    ours = load_ours()
    contrastive = load_contrastive(contrastive_csv, include_hsc_provabgs=args.include_hsc_provabgs)
    merged = ours.merge(contrastive, on="target", how="outer")
    merged = average_by_patterns(merged)

    group_order = [GROUP_PHYSICS, GROUP_LEGACY, GROUP_HSC]
    merged["group_rank"] = merged["group"].map({g: i for i, g in enumerate(group_order)})
    merged["_sort_r2"] = np.where(
        merged["group_rank"] == 0,
        -merged["Ours_Physics"].fillna(-1),
        -merged["Ours_Instrument"].fillna(-1),
    )
    merged = merged.sort_values(by=["group_rank", "_sort_r2"]).reset_index(drop=True)
    merged = merged.drop(columns=["_sort_r2"])
    merged = merged[~merged["target"].isin(EXCLUDE_TARGETS)].reset_index(drop=True)

    out_csv = _script_dir / f"contrastive_vs_ours_all_{args.suffix}_merged.csv"
    out_png = _script_dir / f"contrastive_vs_ours_all_{args.suffix}.png"
    merged.to_csv(out_csv, index=False)
    print(f"Merged CSV saved: {out_csv}")

    targets = merged["target"].tolist()
    display_labels = [LABEL_MAPPING.get(t, t) for t in targets]

    x_positions = []
    current_x = 0.0
    last_group = None
    group_bounds = {}
    for _, row in merged.iterrows():
        group = row["group"]
        if last_group is not None and group != last_group:
            current_x += GAP_SIZE
        if group not in group_bounds:
            group_bounds[group] = {"min": current_x, "max": current_x}
        else:
            group_bounds[group]["max"] = current_x
        x_positions.append(current_x)
        last_group = group
        current_x += 1.25
    x = np.array(x_positions)
    total_width = current_x

    series = [
        ("Ours (Physics)", merged["Ours_Physics"].values, "#2E86AB", False, None),
        ("Ours (Instrument)", merged["Ours_Instrument"].values, "#CC546D", False, None),
        ("Contrastive (Physics)", merged["Contrastive_Physics"].values, "#3B7A57", False, None),
        ("Contrastive (Instrument)", merged["Contrastive_Instrument"].values, "#BF823B", False, None),
    ]
    if args.include_untrained:
        series.append(("Ours Untrained", merged["Ours_Untrained"].values, "gray", True, "///"))
        series.append(("Contrastive Untrained", merged["Contrastive_Untrained"].values, "white", True, "\\\\\\"))

    n_series = len(series)
    width = 0.15 if n_series >= 6 else 0.2
    offsets = np.linspace(-(n_series - 1) / 2, (n_series - 1) / 2, n_series) * width

    fig_width = max(10, total_width * 0.6)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    for i, (label, vals, color, hatched, hatch) in enumerate(series):
        vals = np.where(np.isnan(vals), 0.0, vals)
        if hatched:
            ax.bar(
                x + offsets[i],
                vals,
                width,
                label=label,
                color=color,
                edgecolor="black",
                linewidth=0.5,
                hatch=hatch,
                zorder=3,
            )
        else:
            ax.bar(
                x + offsets[i],
                vals,
                width,
                label=label,
                color=color,
                alpha=0.85,
                edgecolor="black",
                linewidth=0.5,
                zorder=3,
            )

    for group in group_order:
        if group not in group_bounds:
            continue
        start = group_bounds[group]["min"] - 0.5
        end = group_bounds[group]["max"] + 0.5
        ax.axvspan(start, end, facecolor=BG_COLORS[group], alpha=BG_ALPHA, zorder=0)

    super_title_y = 1.06
    sub_title_y = 0.86
    for group in group_order:
        if group not in group_bounds:
            continue
        start = group_bounds[group]["min"] - 0.5
        end = group_bounds[group]["max"] + 0.5
        mid = (start + end) / 2
        color = GROUP_LABEL_COLORS[group]
        if group == GROUP_PHYSICS:
            ax.text(
                mid,
                super_title_y,
                "Physics-Related Properties",
                ha="center",
                va="bottom",
                fontsize=18,
                fontweight="bold",
                color="black",
                transform=ax.get_xaxis_transform(),
            )
        else:
            ax.text(
                mid,
                sub_title_y,
                group,
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold",
                color=color,
                transform=ax.get_xaxis_transform(),
            )

    if GROUP_LEGACY in group_bounds and GROUP_HSC in group_bounds:
        inst_start = group_bounds[GROUP_LEGACY]["min"] - 0.5
        inst_end = group_bounds[GROUP_HSC]["max"] + 0.5
        inst_mid = (inst_start + inst_end) / 2
        ax.text(
            inst_mid,
            super_title_y,
            "Instrument Properties",
            ha="center",
            va="bottom",
            fontsize=18,
            fontweight="bold",
            color="black",
            transform=ax.get_xaxis_transform(),
        )

    margin = max(np.abs(offsets)) + width / 2 + 0.15
    ax.set_xlim(x[0] - margin, x[-1] + margin)
    ax.set_ylabel("R²", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=12)
    ax.tick_params(axis="x", which="both", length=0, top=False, bottom=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.00), ncol=min(n_series, 6), fontsize=11, frameon=True)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(axis="y", which="major", alpha=0.75, linestyle="--", zorder=1)
    ax.grid(axis="y", which="minor", alpha=0.6, linestyle=":", zorder=1)
    ax.axhline(0, color="gray", linewidth=0.7, zorder=2)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {out_png}")


if __name__ == "__main__":
    main()
