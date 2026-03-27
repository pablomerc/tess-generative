"""
Make simplified plots from contrastive debug neighbors CSVs by averaging over bands.

Example:
  python makeplot_contrastive.py \
    --csv outputs/predict_neighbors_debug_contrastive_neighbors_debug_shards_hsc_only.csv
"""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import scienceplots  # noqa: F401
except Exception:
    scienceplots = None

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

LABEL_MAPPING = {
    "hsc_variance_value": "HSC Variance",
    "hsc_psf_fwhm": "HSC PSF Size",
    "legacy_GALDEPTH": "Legacy Galaxy Depth",
    "legacy_NOBS": "Legacy # Obs",
    "legacy_PSFSIZE": "Legacy PSF Size",
    "legacy_PSFDEPTH": "Legacy PSF Depth",
}

BG_COLORS = {
    GROUP_LEGACY: "#e6f2ff",
    GROUP_HSC: "#fff0e6",
}

BAR_LABELS = ["Physics latents", "Instrument latents", "Untrained"]
BAR_KEYS = ["r2_physics", "r2_instrument", "r2_untrained"]
BAR_COLORS = ["#8AC3EE", "#E5254E", "#B19221"]


def _get_group(target_name):
    t = str(target_name)
    if t in ["hsc_variance_value", "hsc_psf_fwhm"] or t.startswith("hsc_"):
        return GROUP_HSC
    return GROUP_LEGACY


def _average_by_patterns(df):
    df = df.copy()
    numeric_cols = [c for c in ["r2_physics", "r2_instrument", "r2_untrained"] if c in df.columns]

    drop_idx = []
    new_rows = []
    for pattern, new_name in AVERAGE_PATTERNS.items():
        mask = df["target"].astype(str).str.contains(pattern, regex=True)
        subset = df.loc[mask].copy()
        if new_name == "hsc_psf_fwhm":
            subset = subset[subset["target"] != "hsc_z_psf_fwhm"]
        if subset.empty:
            continue
        drop_idx.extend(subset.index.tolist())
        avg = subset[numeric_cols].mean(numeric_only=True).to_dict()
        avg["target"] = new_name
        avg["task"] = subset["task"].iloc[0] if "task" in subset.columns and len(subset) else "averaged"
        new_rows.append(avg)

    df = df.drop(index=drop_idx)
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    return df


def load_and_process(csv_path):
    df = pd.read_csv(csv_path)
    if "target" not in df.columns:
        raise ValueError("CSV missing required 'target' column")

    # Keep one row per target if duplicates exist (e.g., across tasks/objectives)
    keep_cols = [c for c in ["r2_physics", "r2_instrument", "r2_untrained"] if c in df.columns]
    if keep_cols:
        df = df.groupby("target", as_index=False)[keep_cols].mean(numeric_only=True)

    df = _average_by_patterns(df)
    df["group"] = df["target"].apply(_get_group)

    group_order = [GROUP_LEGACY, GROUP_HSC]
    df["group_rank"] = df["group"].map({g: i for i, g in enumerate(group_order)})
    df = df.sort_values(by=["group_rank", "target"]).reset_index(drop=True)
    return df


def make_plot(df, out_path, title):
    if scienceplots is not None:
        plt.style.use(["science", "no-latex"])

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
        current_x += 1.0

    x = np.array(x_positions)
    labels = [LABEL_MAPPING.get(t, t) for t in df["target"].tolist()]

    fig_width = max(8, current_x * 0.9)
    fig, ax = plt.subplots(figsize=(fig_width, 5.5))

    width = 0.25
    offsets = [-width, 0.0, width]
    for i, (key, label, color) in enumerate(zip(BAR_KEYS, BAR_LABELS, BAR_COLORS)):
        if key not in df.columns:
            continue
        vals = pd.to_numeric(df[key], errors="coerce").to_numpy()
        vals = np.nan_to_num(vals, nan=0.0)
        ax.bar(x + offsets[i], vals, width, label=label, color=color, edgecolor="gray", linewidth=0.5, zorder=3)

    for group, bounds in group_bounds.items():
        start = bounds["min"] - 0.5
        end = bounds["max"] + 0.5
        ax.axvspan(start, end, facecolor=BG_COLORS[group], alpha=1.0, zorder=0)
        ax.text((start + end) / 2, 1.03, group, ha="center", va="bottom", fontsize=11, fontweight="bold",
                transform=ax.get_xaxis_transform())

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("R²")
    ax.set_title(title)
    ax.axhline(0, color="black", linewidth=0.8, zorder=2)
    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=1)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {out_path}")


def main():
    p = argparse.ArgumentParser(description="Simplified band-averaged plot for contrastive debug neighbors CSV")
    p.add_argument("--csv", type=Path, required=True, help="Input predict_neighbors_debug_contrastive_*.csv")
    p.add_argument("--output", type=Path, default=None, help="Output PNG path (default: next to CSV)")
    args = p.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"Missing CSV: {args.csv}")

    df = load_and_process(args.csv)
    output = args.output or args.csv.with_name(args.csv.stem + "_plot_simplified.png")
    title = f"Contrastive Debug (Simplified): {args.csv.stem}"
    make_plot(df, output, title)


if __name__ == "__main__":
    main()
