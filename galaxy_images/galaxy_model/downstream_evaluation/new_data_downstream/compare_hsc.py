"""
Publication-quality comparison plot for the HSC downstream evaluation.

Mirrors the style of downstream_evaluation/4x4/compare_all.py but for the
HSC-only setting (no Legacy group, no AION). Produces a single figure with
two semantic groups:
  - Physics-Related Properties   (PROVABGS physical props + galaxy morphology)
  - HSC Instrument Properties    (extinction, PSF size, sky variance — band-averaged)

Usage:
  python compare_hsc.py --suffix <suffix>
  python compare_hsc.py --suffix <suffix> --output-dir /path/to/dir
  python compare_hsc.py --suffix <suffix> --no-scienceplots
"""
from pathlib import Path
import argparse
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

_here = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------

GROUP_PHYSICS = "Physics-Related Properties"
GROUP_HSC     = "HSC Instrument Properties"
GAP_SIZE      = 0.5

BG_COLORS = {
    GROUP_PHYSICS: "#c0c0c0",
    GROUP_HSC:     "#e8c4a0",
}
BG_ALPHA = 0.85

GROUP_LABEL_COLORS = {
    GROUP_PHYSICS: "#888888",
    GROUP_HSC:     "#996615",
}

BAR_COLORS = ["#2E86AB", "#CC546D"]  # physics, instrument

# Targets that live in the physics group
PHYSICS_TARGETS = frozenset({
    "desi_z", "provabgs_logmstar", "provabgs_tage_mw",
    "provabgs_avg_sfr", "provabgs_z_mw",
    "i_sdssshape_shape11", "i_sdssshape_shape22", "i_sdssshape_shape12",
})

# ---------------------------------------------------------------------------
# Band averaging: collapse per-band rows into a single representative value
# ---------------------------------------------------------------------------

AVERAGE_PATTERNS = {
    r"^a_":              "a",              # a_g, a_r, a_i, a_z, a_y  → single Extinction bar
    r"^psf_fwhm_":       "psf_fwhm",       # psf_fwhm_g/r/i/z          → single PSF Size bar
    r"_variance_value$": "variance_value", # g/r/i/z_variance_value    → single Sky Variance bar
}

EXCLUDE_TARGETS: frozenset = frozenset()

# ---------------------------------------------------------------------------
# Display labels (LaTeX where available)
# ---------------------------------------------------------------------------

LABEL_MAPPING = {
    # Physics / PROVABGS
    "desi_z":              r"$z_{\rm spec}$",
    "provabgs_logmstar":   r"$\log M_\star$",
    "provabgs_tage_mw":    r"$t_{\rm age}$",
    "provabgs_avg_sfr":    r"$\langle{\rm SFR}\rangle$",
    "provabgs_z_mw":       r"$\log Z_\star$",
    # Galaxy morphology
    "i_sdssshape_shape11": r"$I_{11}$",
    "i_sdssshape_shape22": r"$I_{22}$",
    "i_sdssshape_shape12": r"$I_{12}$",
    # HSC Instrument (averaged)
    "a":               "Extinction",
    "psf_fwhm":        "PSF Size",
    "variance_value":  "Sky Variance",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_group(target: str) -> str:
    return GROUP_PHYSICS if target in PHYSICS_TARGETS else GROUP_HSC


def load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[~df["target"].isin(EXCLUDE_TARGETS)].copy()
    # Coerce metrics to numeric (empty strings → NaN)
    for col in ("r2_physics", "r2_instrument", "r2_untrained", "r2_mean"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # If a target somehow appears in multiple tasks, take the mean
    return df.groupby("target", as_index=False).agg({
        "r2_physics":    "mean",
        "r2_instrument": "mean",
        "r2_untrained":  "mean",
    })


def average_by_patterns(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [c for c in ("r2_physics", "r2_instrument", "r2_untrained") if c in df.columns]
    to_drop, new_rows = [], []
    for pattern, new_name in AVERAGE_PATTERNS.items():
        mask   = df["target"].astype(str).str.contains(pattern, regex=True)
        subset = df.loc[mask].copy()
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
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_plot(merged: pd.DataFrame, out_path: Path) -> None:
    group_order = [GROUP_PHYSICS, GROUP_HSC]
    merged = merged.copy()
    merged["group_rank"] = merged["group"].map({g: i for i, g in enumerate(group_order)})
    # Sort physics by r2_physics desc, instrument by r2_instrument desc
    merged["_sort"] = np.where(
        merged["group_rank"] == 0,
        -merged["r2_physics"].fillna(-1),
        -merged["r2_instrument"].fillna(-1),
    )
    merged = merged.sort_values(["group_rank", "_sort"]).reset_index(drop=True)
    merged = merged[~merged["target"].isin(EXCLUDE_TARGETS)].reset_index(drop=True)

    targets = merged["target"].tolist()
    labels  = [LABEL_MAPPING.get(t, t) for t in targets]

    # Compute x positions with a gap between groups
    x_pos, cur_x, last_g = [], 0.0, None
    group_bounds: dict = {}
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

    # 3 series: physics, instrument, untrained
    w       = 0.28
    offsets = np.array([-w, 0.0, w])

    fig_w = max(10, cur_x * 0.65)
    fig, ax = plt.subplots(figsize=(fig_w, 6))

    series = [
        ("Physics latents",    merged["r2_physics"].to_numpy(),    BAR_COLORS[0], False),
        ("Instrument latents", merged["r2_instrument"].to_numpy(),  BAR_COLORS[1], False),
        ("Untrained",          merged["r2_untrained"].to_numpy(),   None,          True),
    ]

    for i, (label, vals, color, is_ref) in enumerate(series):
        vals = np.where(np.isfinite(vals), vals, 0.0)
        if is_ref:
            ax.bar(x + offsets[i], vals, w, label=label,
                   color="gray", edgecolor="black", linewidth=0.5, hatch="///", zorder=3)
        else:
            ax.bar(x + offsets[i], vals, w, label=label,
                   color=color, alpha=0.85, edgecolor="black", linewidth=0.5, zorder=3)

    # Colored background bands
    for g in group_order:
        if g not in group_bounds:
            continue
        s = group_bounds[g]["min"] - 0.5
        e = group_bounds[g]["max"] + 0.5
        ax.axvspan(s, e, facecolor=BG_COLORS[g], alpha=BG_ALPHA, zorder=0)

    # Group header text
    SUPER_Y = 1.06
    SUPER_FS = 17
    for g in group_order:
        if g not in group_bounds:
            continue
        s   = group_bounds[g]["min"] - 0.5
        e   = group_bounds[g]["max"] + 0.5
        mid = (s + e) / 2
        ax.text(mid, SUPER_Y, g, ha="center", va="bottom",
                fontsize=SUPER_FS, fontweight="bold", color="black",
                transform=ax.get_xaxis_transform())

    margin = w + 0.2
    ax.set_xlim(x[0] - margin, x[-1] + margin)
    ax.set_ylabel("R²", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=12)
    ax.tick_params(axis="x", which="both", length=0, top=False, bottom=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.00), ncol=3,
              fontsize=12, frameon=True)
    ax.set_ylim(0, 1.18)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(axis="y", which="major", alpha=0.75, linestyle="--", zorder=1)
    ax.grid(axis="y", which="minor", alpha=0.6,  linestyle=":",  zorder=1)
    ax.axhline(0, color="gray", linewidth=0.7, zorder=2)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Publication-quality HSC downstream comparison plot.")
    p.add_argument("--suffix",          required=True,
                   help="Must match the suffix used in predict_hsc_downstream.py")
    p.add_argument("--output-dir",      type=Path, default=_here)
    p.add_argument("--no-scienceplots", action="store_true",
                   help="Skip scienceplots style even if installed")
    args = p.parse_args()

    if not args.no_scienceplots:
        try:
            import scienceplots  # noqa: F401
            plt.style.use(["science", "no-latex"])
        except ImportError:
            pass  # fall back to matplotlib default

    output_dir = args.output_dir
    csv_path   = output_dir / f"predict_hsc_downstream_{args.suffix}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found: {csv_path}\n"
            "Run predict_hsc_downstream.py first."
        )

    df     = load_csv(csv_path)
    merged = average_by_patterns(df)

    out_csv = output_dir / f"compare_hsc_{args.suffix}.csv"
    merged.to_csv(out_csv, index=False)
    print(f"Averaged CSV: {out_csv}")

    make_plot(merged, output_dir / f"compare_hsc_{args.suffix}.png")


if __name__ == "__main__":
    main()
