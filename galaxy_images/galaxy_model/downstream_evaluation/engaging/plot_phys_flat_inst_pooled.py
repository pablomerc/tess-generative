"""
Per-target R² comparison with **four series**:

    Ours (physics)              base.csv          combined_e1   flat 128-D
    Ours (instr, pooled)        base-meanpool.csv combined_e2   pooled 32-D   *
    AION                        aion-base.csv     combined_e1
    ResNet baseline (matched)   untrained-resnet-matched.csv  combined_e1

  *Token-mean over the 4 spatial tokens of the (B, 4, 16) instrument encoder.

The HSC instrument group is dropped entirely — only Physics and Legacy
groups remain. Output:
    outputs/plots_publication/phys_flat_inst_pooled_all.png
Pushes the PNG to Discord on success.
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
    import scienceplots  # noqa: F401
    plt.style.use(["science", "no-latex"])
except Exception:
    pass

WEBHOOK_URL = "https://discord.com/api/webhooks/1500242707522519183/NqqAxuo5n0Ex4X81vMcq7gDOjrP3oAURAVZtgrZT5CramGNfBGPxuEvC2NnxwYBkfs9j"

_here = Path(__file__).resolve().parent
_galaxy_model = _here.parents[1]
_predictions_dir = _here / "outputs" / "predictions"
_plots_dir = _here / "outputs" / "plots_publication"

sys.path.insert(0, str(_galaxy_model / "visualization_scripts"))
try:
    from discord_notify import notify  # type: ignore
except Exception:
    def notify(webhook_url, message, file_path=None):
        print(f"[plot_phys_flat_inst_pooled] discord_notify import failed; would send: {message}")


GROUP_PHYSICS = "Physics-like Properties"
GROUP_LEGACY = "Legacy"
GROUP_HSC = "HSC"
GAP_SIZE = 0.5
BG_COLORS = {GROUP_PHYSICS: "#c0c0c0", GROUP_LEGACY: "#8eb8e8", GROUP_HSC: "#e8c4a0"}
BG_ALPHA = 0.85
GROUP_LABEL_COLORS = {GROUP_PHYSICS: "#888888", GROUP_LEGACY: "#2563a8", GROUP_HSC: "#996515"}

EXCLUDE_TARGETS = {
    "hsc_z_psf_fwhm",
    "hsc_variance_value",
    "legacy_MW_TRANSMISSION",
}

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

LABEL_MAPPING = {
    "desi_z": r"$z$",
    "provabgs_logmstar": r"$\log M_\star$",
    "provabgs_tage_mw": r"$t_{\mathrm{age}}$",
    "provabgs_avg_sfr": r"SFR",
    "provabgs_z_mw": r"$\log Z_{\mathrm{Met}}$",
    "ssfr": r"sSFR",
    "hsc_a": "Extinction a",
    "hsc_psf_fwhm": "PSF FWHM",
    "legacy_GALDEPTH": "Galaxy Depth",
    "legacy_NOBS": "# Observations",
    "legacy_PSFSIZE": "PSF Size",
    "legacy_PSFDEPTH": "PSF Depth",
    "legacy_psf_fwhm": "PSF FWHM",
    "legacy_EBV": "Extinction E(B-V)",
}


def _prefix(row):
    fam = row["task_family"]
    t = row["target"]
    if fam == "instrument_hsc":
        return f"hsc_{t}"
    if fam == "instrument_legacy":
        return f"legacy_{t}"
    return t


def _get_group(target_name: str) -> str:
    t = str(target_name)
    if t.startswith("hsc_"):
        return GROUP_HSC
    if t.startswith("legacy_") or t.lower() == "ebv":
        return GROUP_LEGACY
    return GROUP_PHYSICS


def _load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"[skip] {path} not found")
        return None
    return pd.read_csv(path)


def _series_from_variant(variant: str, latent_variant: str, score_name: str) -> pd.DataFrame:
    """Pull a single (target, score) series from `predict_<variant>.csv`,
    using the rows whose `latent_variant` column matches `latent_variant`
    (e.g. 'combined_e1' or 'combined_e2'). R² rows only."""
    df = _load_csv(_predictions_dir / f"predict_{variant}.csv")
    if df is None:
        return pd.DataFrame(columns=["target", score_name])
    df = df[(df["score_metric"] == "r2") & (df["latent_variant"] == latent_variant)].copy()
    if df.empty:
        return pd.DataFrame(columns=["target", score_name])
    df["target_pref"] = df.apply(_prefix, axis=1)
    return df[["target_pref", "score"]].rename(
        columns={"target_pref": "target", "score": score_name}
    )


def average_by_patterns(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    drops, new_rows = [], []
    for pattern, new_name in AVERAGE_PATTERNS.items():
        mask = df["target"].astype(str).str.contains(pattern, regex=True)
        sub = df.loc[mask].copy()
        if not sub.empty:
            drops.extend(sub.index.tolist())
            avg = sub[numeric_cols].mean(numeric_only=True).to_dict()
            avg["target"] = new_name
            new_rows.append(avg)
    df = df.drop(index=drops)
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df["group"] = df["target"].apply(_get_group)
    return df


def build_table() -> pd.DataFrame:
    ours_phys = _series_from_variant("base", "combined_e1", "Ours (physics)")
    ours_instr = _series_from_variant("base-meanpool", "combined_e2", "Ours (instr, pooled)")
    aion = _series_from_variant("aion-base", "combined_e1", "AION")
    untr_matched = _series_from_variant("untrained-resnet-matched", "combined_e1", "ResNet baseline (matched)")

    out = ours_phys.merge(ours_instr, on="target", how="outer")
    if not aion.empty:
        out = out.merge(aion, on="target", how="outer")
    else:
        out["AION"] = np.nan
    if not untr_matched.empty:
        out = out.merge(untr_matched, on="target", how="outer")
    else:
        out["ResNet baseline (matched)"] = np.nan

    out = out[~out["target"].isin(EXCLUDE_TARGETS)].reset_index(drop=True)

    numeric_cols = [c for c in
                    ["Ours (physics)", "Ours (instr, pooled)", "AION", "ResNet baseline (matched)"]
                    if c in out.columns]
    out = average_by_patterns(out, numeric_cols)
    # Apply exclusion again after band-averaging — post-averaged names like
    # "hsc_variance_value" / "legacy_MW_TRANSMISSION" only exist now.
    out = out[~out["target"].isin(EXCLUDE_TARGETS)].reset_index(drop=True)

    group_order = [GROUP_PHYSICS, GROUP_LEGACY, GROUP_HSC]
    out["group_rank"] = out["group"].map({g: i for i, g in enumerate(group_order)})
    # Within group, sort by best of Ours.
    sort_metric = -np.fmax(
        out["Ours (physics)"].fillna(-1).values,
        out["Ours (instr, pooled)"].fillna(-1).values,
    )
    out["_sort"] = sort_metric
    out = out.sort_values(["group_rank", "_sort"]).drop(columns=["_sort"]).reset_index(drop=True)
    return out


def render(df: pd.DataFrame, out_png: Path) -> Path:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    targets = df["target"].tolist()
    display_labels = [LABEL_MAPPING.get(t, t) for t in targets]

    x_positions, current_x, last_group, group_bounds = [], 0.0, None, {}
    for _, row in df.iterrows():
        g = row["group"]
        if last_group is not None and g != last_group:
            current_x += GAP_SIZE
        group_bounds.setdefault(g, {"min": current_x, "max": current_x})["max"] = current_x
        x_positions.append(current_x)
        last_group = g
        current_x += 1.25
    x = np.array(x_positions)

    label_specs = [
        ("Ours (physics)",            "#2E86AB", False),
        ("Ours (instr, pooled)",      "#CC546D", False),
        ("AION",                      "#6CA75D", False),
        ("ResNet baseline (matched)", None,      True),
    ]
    series = [(lbl, df[lbl].values, color, hatch)
              for (lbl, color, hatch) in label_specs if lbl in df.columns]

    n = len(series)
    width = 0.20 if n == 4 else (0.24 if n == 3 else 0.30)
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

    fig_width = max(8, (x[-1] - x[0] + 2) * 0.7)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    for i, (label, vals, color, hatch) in enumerate(series):
        vals = np.where(np.isnan(vals), 0.0, vals)
        if hatch:
            ax.bar(x + offsets[i], vals, width, label=label, color="gray",
                   edgecolor="black", linewidth=0.5, hatch="///", zorder=3)
        else:
            ax.bar(x + offsets[i], vals, width, label=label, color=color, alpha=0.88,
                   edgecolor="black", linewidth=0.5, zorder=3)

    for g in [GROUP_PHYSICS, GROUP_LEGACY, GROUP_HSC]:
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
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.00), ncol=n, fontsize=11, frameon=True)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(axis="y", which="major", alpha=0.75, linestyle="--", zorder=1)
    ax.grid(axis="y", which="minor", alpha=0.6, linestyle=":", zorder=1)
    ax.axhline(0, color="gray", linewidth=0.7, zorder=2)
    fig.suptitle("Ours (physics flat 64D, instr pooled 16D) vs AION vs matched-arch random ResNet",
                 fontsize=13, y=1.13)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-discord", action="store_true")
    args = ap.parse_args()

    df = build_table()
    out_png = _plots_dir / "phys_flat_inst_pooled_all.png"
    out_csv = _plots_dir / "phys_flat_inst_pooled_all_merged.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    render(df, out_png)
    print(f"Plot: {out_png}")
    print(f"CSV : {out_csv}")
    if not args.no_discord:
        msg = ("engaging downstream — Ours(phys) + Ours(instr,pooled) + AION + matched-resnet baseline. "
               "HSC group dropped.")
        notify(WEBHOOK_URL, msg, str(out_png))


if __name__ == "__main__":
    main()
