"""
Per-variant publication-style downstream-task R² comparison plot.

Reads engaging-pipeline long-form CSVs:
    outputs/predictions/predict_<variant>.csv         # this variant's probe scores
    outputs/predictions/predict_aion-base.csv         # AION-base reference (constant)
    outputs/predictions/predict_untrained-resnet.csv  # random-init ResNet-18 reference (constant)
    outputs/predictions/predict_instrument_baseline.csv  # cross-predict baseline (physics only, optional)

Builds a wide target-by-series table, runs the same band-averaging / group-shading
style as downstream_evaluation/final/aion_vs_ours_all.py, writes
    outputs/plots_publication/aion_vs_<variant>_all.png
and pushes the PNG to Discord with a short caption.

Single-encoder variants (has_e2=False, only `combined_e1` in their CSV) collapse
to a 4-series plot: Latents | AION-1 | Untrained ResNet-18 | Baseline.

Usage (from galaxy_model/):
    python downstream_evaluation/engaging/plot_aion_vs_variant.py --variant base-15NB
    python downstream_evaluation/engaging/plot_aion_vs_variant.py --variant base-15NB --no-discord
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

try:
    import scienceplots  # noqa: F401  (registers styles)
    plt.style.use(["science", "no-latex"])
except Exception:
    pass


# Hardcoded webhook (per user direction). Caveat: do not push this file to a public remote
# without rotating the URL first.
WEBHOOK_URL = "https://discord.com/api/webhooks/1500242707522519183/NqqAxuo5n0Ex4X81vMcq7gDOjrP3oAURAVZtgrZT5CramGNfBGPxuEvC2NnxwYBkfs9j"

_here = Path(__file__).resolve().parent
_galaxy_model = _here.parents[1]
_repo_root = _galaxy_model.parent.parent
_predictions_dir = _here / "outputs" / "predictions"
_plots_dir = _here / "outputs" / "plots_publication"

# Wire discord_notify in.
sys.path.insert(0, str(_galaxy_model / "visualization_scripts"))
try:
    from discord_notify import notify  # type: ignore
except Exception:
    def notify(webhook_url, message, file_path=None):  # noqa: ARG001
        print(f"[plot_aion_vs_variant] discord_notify import failed; would send: {message}")


# Style assets lifted from final/aion_vs_ours_all.py (kept identical).
GROUP_PHYSICS = "Physics-like Properties"
GROUP_LEGACY = "Legacy"
GROUP_HSC = "HSC"
GAP_SIZE = 0.5
BAR_COLORS = ["#2E86AB", "#CC546D", "#6CA75D", "#9473C6", "#BF823B"]
BG_COLORS = {GROUP_PHYSICS: "#c0c0c0", GROUP_LEGACY: "#8eb8e8", GROUP_HSC: "#e8c4a0"}
BG_ALPHA = 0.85
GROUP_LABEL_COLORS = {GROUP_PHYSICS: "#888888", GROUP_LEGACY: "#2563a8", GROUP_HSC: "#996515"}

# Targets to drop from plot (broken / redundant).
EXCLUDE_TARGETS = {
    "hsc_z_psf_fwhm",        # only present in legacy schema; engaging never has this
    "hsc_variance_value",    # noisy, drop by default
    "legacy_MW_TRANSMISSION", # not very informative
}


def _get_group(target_name: str) -> str:
    t = str(target_name)
    if t.startswith("hsc_"):
        return GROUP_HSC
    if t.startswith("legacy_") or t.lower() == "ebv":
        return GROUP_LEGACY
    return GROUP_PHYSICS


# Band-averaging on the *prefixed* target names emitted by `_long_to_wide` below.
AVERAGE_PATTERNS = {
    r"^hsc_a_[grizy]$": "hsc_a",
    r"^hsc_psf_fwhm_[grizy]$": "hsc_psf_fwhm",
    r"^hsc_.*_variance_value$": "hsc_variance_value",
    r"^legacy_PSFSIZE_[GRIZ]$": "legacy_PSFSIZE",
    r"^legacy_PSFDEPTH_[GRIZ]$": "legacy_PSFDEPTH",
    r"^legacy_GALDEPTH_[GRIZ]$": "legacy_GALDEPTH",
    r"^legacy_NOBS_[GRIZ]$": "legacy_NOBS",
    r"^legacy_MW_TRANSMISSION_[GRIZ]$": "legacy_MW_TRANSMISSION",
    r"^legacy_psf_fwhm_[grizy]$": "legacy_psf_fwhm",
}

# Display labels (mathtext for subscripts/symbols).
LABEL_MAPPING = {
    "desi_z": r"$z$",
    "provabgs_logmstar": r"$\log M_\star$",
    "provabgs_tage_mw": r"$t_{\mathrm{age}}$",
    "provabgs_avg_sfr": r"SFR",
    "provabgs_z_mw": r"$\log Z_{\mathrm{Met}}$",
    "ssfr": r"sSFR",
    "hsc_a": "Extinction a",
    "hsc_psf_fwhm": "PSF Size",
    "hsc_variance_value": "Variance",
    "legacy_GALDEPTH": "Galaxy Depth",
    "legacy_NOBS": "# Observations",
    "legacy_PSFSIZE": "PSF Size",
    "legacy_PSFDEPTH": "PSF Depth",
    "legacy_psf_fwhm": "PSF FWHM",
    "legacy_EBV": "Extinction E(B-V)",
}


def _long_to_wide(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Translate a single long-form CSV (one variant or AION/untrained) into target,
    physics_latents, instrument_latents columns. Targets are prefixed with task family
    so HSC/Legacy/Physics targets never collide.
    """
    df = df[df["score_metric"] == "r2"].copy()
    if df.empty:
        return pd.DataFrame(columns=["target", score_col + "_e1", score_col + "_e2"])

    def _prefix(row):
        fam = row["task_family"]
        t = row["target"]
        if fam == "instrument_hsc":
            return f"hsc_{t}"
        if fam == "instrument_legacy":
            return f"legacy_{t}"
        return t

    df["target_pref"] = df.apply(_prefix, axis=1)
    e1 = df[df["latent_variant"] == "combined_e1"][["target_pref", "score"]].rename(
        columns={"target_pref": "target", "score": score_col + "_e1"}
    )
    e2 = df[df["latent_variant"] == "combined_e2"][["target_pref", "score"]].rename(
        columns={"target_pref": "target", "score": score_col + "_e2"}
    )
    if e2.empty:
        return e1
    return e1.merge(e2, on="target", how="outer")


def _load_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"[skip] missing CSV: {path}")
        return None
    return pd.read_csv(path)


def _baseline_long_to_wide(df: pd.DataFrame, col_name: str = "Baseline") -> pd.DataFrame:
    """Cross-predict baseline CSV may have multiple seeds — average per (task_family, target)."""
    df = df[df["score_metric"] == "r2"].copy()
    if df.empty:
        return pd.DataFrame(columns=["target", col_name])

    def _prefix(row):
        fam = row["task_family"]
        t = row["target"]
        if fam == "instrument_hsc":
            return f"hsc_{t}"
        if fam == "instrument_legacy":
            return f"legacy_{t}"
        return t

    df["target_pref"] = df.apply(_prefix, axis=1)
    out = df.groupby("target_pref", as_index=False)["score"].mean().rename(
        columns={"target_pref": "target", "score": col_name}
    )
    return out


def average_by_patterns(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """Collapse multi-band targets (a_g/r/i/z, psf_fwhm_*, etc.) by regex; same logic as final/."""
    df = df.copy()
    indices_to_drop = []
    new_rows = []
    for pattern, new_name in AVERAGE_PATTERNS.items():
        mask = df["target"].astype(str).str.contains(pattern, regex=True)
        subset = df.loc[mask].copy()
        if not subset.empty:
            indices_to_drop.extend(subset.index.tolist())
            averaged = subset[numeric_cols].mean(numeric_only=True).to_dict()
            averaged["target"] = new_name
            new_rows.append(averaged)
    df = df.drop(index=indices_to_drop)
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df["group"] = df["target"].apply(_get_group)
    return df


def build_wide_table(variant: str) -> tuple[pd.DataFrame, bool]:
    """Return (wide DataFrame ready for plotting, has_instrument_latents_split)."""
    ours_csv = _predictions_dir / f"predict_{variant}.csv"
    aion_csv = _predictions_dir / "predict_aion-base.csv"
    untr_csv = _predictions_dir / "predict_untrained-resnet.csv"
    base_csv = _predictions_dir / "predict_instrument_baseline.csv"
    base_v2_csv = _predictions_dir / "predict_instrument_baseline_v2.csv"

    if not ours_csv.exists():
        raise FileNotFoundError(f"Missing variant CSV: {ours_csv}")

    ours = _long_to_wide(pd.read_csv(ours_csv), "ours")
    aion_df = _load_optional_csv(aion_csv)
    untr_df = _load_optional_csv(untr_csv)
    base_df = _load_optional_csv(base_csv)
    base_v2_df = _load_optional_csv(base_v2_csv)

    has_e2 = "ours_e2" in ours.columns

    out = ours.rename(columns={"ours_e1": "Physics_latents"})
    if has_e2:
        out = out.rename(columns={"ours_e2": "Instrument_latents"})
    else:
        # Single-encoder variant: collapse to one "Latents" column, also surfaced as Instrument_latents=NaN.
        out["Instrument_latents"] = np.nan

    if aion_df is not None:
        aion_w = _long_to_wide(aion_df, "aion")
        out = out.merge(aion_w[["target", "aion_e1"]].rename(columns={"aion_e1": "AION"}),
                        on="target", how="left")
    else:
        out["AION"] = np.nan

    if untr_df is not None:
        untr_w = _long_to_wide(untr_df, "untr")
        out = out.merge(untr_w[["target", "untr_e1"]].rename(columns={"untr_e1": "Untrained"}),
                        on="target", how="left")
    else:
        out["Untrained"] = np.nan

    if base_df is not None:
        out = out.merge(_baseline_long_to_wide(base_df, "Baseline"), on="target", how="left")
    else:
        out["Baseline"] = np.nan

    if base_v2_df is not None:
        out = out.merge(_baseline_long_to_wide(base_v2_df, "Baseline_v2"), on="target", how="left")
    else:
        out["Baseline_v2"] = np.nan

    out = out[~out["target"].isin(EXCLUDE_TARGETS)].reset_index(drop=True)
    numeric_cols = [c for c in ["Physics_latents", "Instrument_latents", "AION", "Untrained",
                                "Baseline", "Baseline_v2"]
                    if c in out.columns]
    out = average_by_patterns(out, numeric_cols)

    # Order: Physics → Legacy → HSC; within each group sort by best Latents R².
    group_order = [GROUP_PHYSICS, GROUP_LEGACY, GROUP_HSC]
    out["group_rank"] = out["group"].map({g: i for i, g in enumerate(group_order)})
    sort_metric = np.where(
        out["group_rank"] == 0,
        -out["Physics_latents"].fillna(-1),
        -out["Instrument_latents"].fillna(out["Physics_latents"]).fillna(-1),
    )
    out["_sort"] = sort_metric
    out = out.sort_values(["group_rank", "_sort"]).drop(columns=["_sort"]).reset_index(drop=True)
    return out, has_e2


def render_plot(merged: pd.DataFrame, variant: str, has_e2: bool, out_png: Path) -> Path:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    targets = merged["target"].tolist()
    display_labels = [LABEL_MAPPING.get(t, t) for t in targets]

    # X positions with gaps between groups.
    x_positions, current_x, last_group, group_bounds = [], 0.0, None, {}
    for _, row in merged.iterrows():
        g = row["group"]
        if last_group is not None and g != last_group:
            current_x += GAP_SIZE
        group_bounds.setdefault(g, {"min": current_x, "max": current_x})["max"] = current_x
        x_positions.append(current_x)
        last_group = g
        current_x += 1.25
    x = np.array(x_positions)

    has_baseline = "Baseline" in merged.columns and merged["Baseline"].notna().any()
    has_baseline_v2 = "Baseline_v2" in merged.columns and merged["Baseline_v2"].notna().any()
    if has_e2:
        series = [
            ("Physics latents", merged["Physics_latents"].values, BAR_COLORS[0], False),
            ("Instrument latents", merged["Instrument_latents"].values, BAR_COLORS[1], False),
            ("AION-1", merged["AION"].values, BAR_COLORS[2], False),
            ("Untrained ResNet-18", merged["Untrained"].values, None, True),
        ]
    else:
        series = [
            ("Latents", merged["Physics_latents"].values, BAR_COLORS[0], False),
            ("AION-1", merged["AION"].values, BAR_COLORS[2], False),
            ("Untrained ResNet-18", merged["Untrained"].values, None, True),
        ]
    if has_baseline:
        series.append(("Baseline (raw)", merged["Baseline"].values, BAR_COLORS[4], False))
    if has_baseline_v2:
        series.append(("Baseline (log+std)", merged["Baseline_v2"].values, BAR_COLORS[3], False))

    n_series = len(series)
    # Bar group occupies ~0.95 of the slot regardless of how many series we plot.
    width = 0.95 / max(n_series, 1)
    offsets = np.linspace(-(n_series - 1) / 2, (n_series - 1) / 2, n_series) * width

    fig_width = max(10, (x[-1] - x[0] + 2) * 0.6)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    for i, (label, vals, color, is_ref) in enumerate(series):
        vals = np.where(np.isnan(vals), 0.0, vals)
        if is_ref:
            ax.bar(x + offsets[i], vals, width, label=label, color="gray",
                   edgecolor="black", linewidth=0.5, hatch="///", zorder=3)
        else:
            ax.bar(x + offsets[i], vals, width, label=label, color=color, alpha=0.85,
                   edgecolor="black", linewidth=0.5, zorder=3)

    group_order = [GROUP_PHYSICS, GROUP_LEGACY, GROUP_HSC]
    for g in group_order:
        if g in group_bounds:
            start = group_bounds[g]["min"] - 0.5
            end = group_bounds[g]["max"] + 0.5
            ax.axvspan(start, end, facecolor=BG_COLORS[g], alpha=BG_ALPHA, zorder=0)

    SUPER_TITLE_Y, SUB_TITLE_Y = 1.06, 0.86
    SUPER_FS, SUB_FS = 18, 16
    if GROUP_PHYSICS in group_bounds:
        gb = group_bounds[GROUP_PHYSICS]
        mid = (gb["min"] - 0.5 + gb["max"] + 0.5) / 2
        ax.text(mid, SUPER_TITLE_Y, "Physics-Related Properties", ha="center", va="bottom",
                fontsize=SUPER_FS, fontweight="bold", color="black",
                transform=ax.get_xaxis_transform())
    if GROUP_LEGACY in group_bounds:
        gb = group_bounds[GROUP_LEGACY]
        mid = (gb["min"] - 0.5 + gb["max"] + 0.5) / 2
        ax.text(mid, SUB_TITLE_Y, "Legacy", ha="center", va="center",
                fontsize=SUB_FS, fontweight="bold", color=GROUP_LABEL_COLORS[GROUP_LEGACY],
                transform=ax.get_xaxis_transform())
    if GROUP_HSC in group_bounds:
        gb = group_bounds[GROUP_HSC]
        mid = (gb["min"] - 0.5 + gb["max"] + 0.5) / 2
        ax.text(mid, SUB_TITLE_Y, "HSC", ha="center", va="center",
                fontsize=SUB_FS, fontweight="bold", color=GROUP_LABEL_COLORS[GROUP_HSC],
                transform=ax.get_xaxis_transform())
    if GROUP_LEGACY in group_bounds or GROUP_HSC in group_bounds:
        starts = [group_bounds[g]["min"] - 0.5 for g in (GROUP_LEGACY, GROUP_HSC) if g in group_bounds]
        ends = [group_bounds[g]["max"] + 0.5 for g in (GROUP_LEGACY, GROUP_HSC) if g in group_bounds]
        ax.text((min(starts) + max(ends)) / 2, SUPER_TITLE_Y, "Instrument Properties",
                ha="center", va="bottom", fontsize=SUPER_FS, fontweight="bold", color="black",
                transform=ax.get_xaxis_transform())

    margin = max(np.abs(offsets)) + width / 2 + 0.15
    ax.set_xlim(x[0] - margin, x[-1] + margin)
    ax.set_ylabel("R²", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=12)
    ax.tick_params(axis="x", which="both", length=0, top=False, bottom=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.00), ncol=n_series, fontsize=12, frameon=True)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(axis="y", which="major", alpha=0.75, linestyle="--", zorder=1)
    ax.grid(axis="y", which="minor", alpha=0.6, linestyle=":", zorder=1)
    ax.axhline(0, color="gray", linewidth=0.7, zorder=2)
    fig.suptitle(f"Variant: {variant}", fontsize=14, y=1.13)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--no-discord", action="store_true", help="Skip Discord push (for local testing)")
    args = ap.parse_args()

    merged, has_e2 = build_wide_table(args.variant)
    out_png = _plots_dir / f"aion_vs_{args.variant}_all.png"
    out_csv = _plots_dir / f"aion_vs_{args.variant}_all_merged.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    render_plot(merged, args.variant, has_e2, out_png)
    print(f"Plot: {out_png}")
    print(f"Wide CSV: {out_csv}")

    if not args.no_discord:
        # Short caption with mean physics R².
        phys = merged[merged["group"] == GROUP_PHYSICS]
        msg = f"engaging downstream **{args.variant}** — physics latents mean R² = {phys['Physics_latents'].mean():.3f}"
        notify(WEBHOOK_URL, msg, str(out_png))


if __name__ == "__main__":
    main()
