"""
Compare a debug contrastive CSV against Ours downstream R^2.

Example:
  python compare_debug_contrastive_vs_ours.py \
    --debug-csv outputs/predict_neighbors_debug_contrastive_neighbors_debug_shards_hsc_only.csv
"""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots

plt.style.use(["science", "no-latex"])

_here = Path(__file__).resolve().parent
_galaxy_model = _here.parents[2]
_ours_final = _galaxy_model / "downstream_evaluation" / "final"

DEFAULT_OURS_TABLE = _ours_final / "predict_all_zdim16_nogeom_neighbors_table.csv"
DEFAULT_OURS_LONG = _ours_final / "predict_all_zdim16_nogeom_neighbors.csv"

AVERAGE_PATTERNS = {
    r"^legacy_GALDEPTH_": "legacy_GALDEPTH",
    r"^legacy_NOBS_": "legacy_NOBS",
    r"^legacy_PSFSIZE_": "legacy_PSFSIZE",
    r"^legacy_PSFDEPTH_": "legacy_PSFDEPTH",
    r"^hsc_.*_variance_value$": "hsc_variance_value",
    r"^hsc_.*_psf_fwhm$": "hsc_psf_fwhm",
}

GROUP_LEGACY = "Legacy"
GROUP_HSC = "HSC"
GAP_SIZE = 0.5

BG_COLORS = {
    GROUP_LEGACY: "#8eb8e8",
    GROUP_HSC: "#e8c4a0",
}

GROUP_LABEL_COLORS = {
    GROUP_LEGACY: "#2563a8",
    GROUP_HSC: "#996515",
}

LABEL_MAPPING = {
    "hsc_variance_value": "Variance",
    "hsc_psf_fwhm": "PSF Size",
    "legacy_GALDEPTH": "Galaxy Depth",
    "legacy_NOBS": "# Observations",
    "legacy_PSFSIZE": "PSF Size",
    "legacy_PSFDEPTH": "PSF Depth",
    "EBV": "Extinction E(B-V)",
}


def _get_group(target_name):
    t = str(target_name)
    if t in ["hsc_variance_value", "hsc_psf_fwhm"] or t.startswith("hsc_"):
        return GROUP_HSC
    return GROUP_LEGACY


def average_by_patterns(df):
    numeric_cols = [
        c
        for c in ["Ours_Physics", "Ours_Instrument", "Debug_Physics", "Debug_Instrument"]
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
        new_rows.append(averaged)

    df = df.drop(index=indices_to_drop)
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    df["group"] = df["target"].apply(_get_group)
    return df


def load_ours(path_table=None, path_long=None):
    table_path = Path(path_table) if path_table else DEFAULT_OURS_TABLE
    long_path = Path(path_long) if path_long else DEFAULT_OURS_LONG

    if table_path.exists():
        df = pd.read_csv(table_path)
        df = df.dropna(subset=["Physics latents", "Instrument latents"], how="all")
        df = df[df["target"].astype(str).str.strip().astype(bool)]
        return pd.DataFrame(
            {
                "target": df["target"].values,
                "Ours_Physics": df["Physics latents"].values,
                "Ours_Instrument": df["Instrument latents"].values,
            }
        )

    if long_path.exists():
        df = pd.read_csv(long_path).drop_duplicates(subset=["target"], keep="first")
        return pd.DataFrame(
            {
                "target": df["target"].values,
                "Ours_Physics": df["r2_physics"].values,
                "Ours_Instrument": df["r2_instrument"].values,
            }
        )

    raise FileNotFoundError(
        f"Ours CSV not found at {table_path} or {long_path}. "
        "Pass --ours-table-csv or --ours-long-csv."
    )


def load_debug(debug_csv):
    df = pd.read_csv(debug_csv)
    required = {"target", "r2_physics", "r2_instrument"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Debug CSV missing required columns: {missing}")

    out = (
        df.groupby("target", as_index=False)[["r2_physics", "r2_instrument"]]
        .mean(numeric_only=True)
        .rename(columns={"r2_physics": "Debug_Physics", "r2_instrument": "Debug_Instrument"})
    )
    return out


def make_plot(df, out_png, title):
    group_order = [GROUP_LEGACY, GROUP_HSC]
    df["group_rank"] = df["group"].map({g: i for i, g in enumerate(group_order)})
    df["_sort_r2"] = np.where(
        df["group_rank"] == 0,
        -df["Ours_Instrument"].fillna(-1),
        -df["Ours_Instrument"].fillna(-1),
    )
    df = df.sort_values(by=["group_rank", "_sort_r2"]).reset_index(drop=True)

    labels = [LABEL_MAPPING.get(t, t) for t in df["target"].tolist()]

    x_positions = []
    current_x = 0.0
    last_group = None
    group_bounds = {}
    for _, row in df.iterrows():
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
    fig_width = max(10, current_x * 0.6)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    series = [
        ("Ours (Physics)", df["Ours_Physics"].values, "#2E86AB"),
        ("Ours (Instrument)", df["Ours_Instrument"].values, "#CC546D"),
        ("Debug (Physics)", df["Debug_Physics"].values, "#3B7A57"),
        ("Debug (Instrument)", df["Debug_Instrument"].values, "#BF823B"),
    ]

    width = 0.2
    offsets = np.linspace(-1.5, 1.5, len(series)) * width

    for i, (name, vals, color) in enumerate(series):
        vals = np.where(np.isnan(vals), 0.0, vals)
        ax.bar(
            x + offsets[i],
            vals,
            width,
            label=name,
            color=color,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )

    for group in group_order:
        if group not in group_bounds:
            continue
        start = group_bounds[group]["min"] - 0.7
        end = group_bounds[group]["max"] + 0.7
        ax.axvspan(start, end, facecolor=BG_COLORS[group], alpha=0.6, zorder=0)
        mid = (start + end) / 2
        ax.text(
            mid,
            1.03,
            group,
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color=GROUP_LABEL_COLORS[group],
            transform=ax.get_xaxis_transform(),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("R²")
    ax.set_title(title)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="upper left")

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {out_png}")


def main():
    p = argparse.ArgumentParser(description="Compare debug contrastive CSV vs Ours")
    p.add_argument("--debug-csv", type=Path, required=True, help="Path to debug contrastive prediction CSV")
    p.add_argument("--ours-table-csv", type=Path, default=None, help="Optional override for ours table CSV")
    p.add_argument("--ours-long-csv", type=Path, default=None, help="Optional override for ours long CSV")
    p.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: debug CSV directory)")
    args = p.parse_args()

    if not args.debug_csv.exists():
        raise FileNotFoundError(f"Missing debug CSV: {args.debug_csv}")

    output_dir = args.output_dir or args.debug_csv.parent
    stem = args.debug_csv.stem

    ours = load_ours(args.ours_table_csv, args.ours_long_csv)
    debug = load_debug(args.debug_csv)

    merged = ours.merge(debug, on="target", how="inner")
    merged = average_by_patterns(merged)

    out_csv = output_dir / f"compare_debug_vs_ours_{stem}_merged.csv"
    out_png = output_dir / f"compare_debug_vs_ours_{stem}.png"

    merged.to_csv(out_csv, index=False)
    print(f"Merged CSV saved: {out_csv}")

    title = f"Ours vs Debug Contrastive ({stem})"
    make_plot(merged, out_png, title)


if __name__ == "__main__":
    main()
