"""
Unified per-target R² plot for the 3-probes-per-encoder setup.

Reads four long-form CSVs produced with the new `--mlp-arch` flag and writes
a single PNG (Physics + Legacy + HSC groups on one x-axis).

Series:
    Ours (physics)              base                       combined_e1   flat 128-D
    Ours (instr, pooled)        base-meanpool              combined_e2   pooled 32-D   *
    AION                        aion-base                  combined_e1
    ResNet baseline (matched)   untrained-resnet-matched   combined_e1

Per-arch CSV file naming: predict_<variant>__<arch>.csv
PNG: outputs/plots_publication/unified_three_probes_<arch>.png
Convergence summary: outputs/plots_publication/unified_three_probes_<arch>__convergence.txt

Pushes both files to Discord on success.
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
_xpred_dir = _galaxy_model / "cross_predict_experiment"

sys.path.insert(0, str(_galaxy_model / "visualization_scripts"))
try:
    from discord_notify import notify  # type: ignore
except Exception:
    def notify(webhook_url, message, file_path=None):
        print(f"[plot_unified_three_probes] discord_notify import failed; would send: {message}")


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
    # HSC-derived ellipticity dropped from the plot — Legacy ellipticity stands in
    # for both surveys (the user only wants one ellipticity bar pair).
    "hsc_shape_e1_i",
    "hsc_shape_e2_i",
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
}

# R² reference values from AION-1 paper Table 1.
# AION-1-B with Photometry+Imaging inputs (Ph+Im row).
AION_B_PH_IM = {
    "desi_z":            0.93,
    "provabgs_logmstar": 0.89,
    "provabgs_tage_mw":  0.45,
    "provabgs_z_mw":     0.49,
    "ssfr":              0.64,
}
# AstroCLIP (Parker et al. 2024) image-only with Legacy {g,r,z}.
ASTROCLIP_IM = {
    "desi_z":            0.78,
    "provabgs_logmstar": 0.73,
    "provabgs_tage_mw":  0.29,
    "provabgs_z_mw":     0.36,
    "ssfr":              0.42,
}

LABEL_MAPPING = {
    "desi_z": r"$z$",
    "provabgs_logmstar": r"$\log M_\star$",
    "provabgs_tage_mw": r"$t_{\mathrm{age}}$",
    "provabgs_z_mw": r"$\log Z_{\mathrm{Met}}$",
    "ssfr": r"sSFR",
    "legacy_shape_e1": r"$e_1$",
    "legacy_shape_e2": r"$e_2$",
    "hsc_a": "Extinction a",
    "hsc_psf_fwhm": "PSF size",
    "legacy_PSFSIZE": "PSF size",
    "legacy_PSFDEPTH": "PSF Depth",
    "legacy_GALDEPTH": "Galaxy Depth",
    "legacy_NOBS": "# Observations",
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
    # HSC-derived ellipticity stays in physics group (galaxy property, not instrument).
    if t in ("hsc_shape_e1_i", "hsc_shape_e2_i"):
        return GROUP_PHYSICS
    if t.startswith("hsc_"):
        return GROUP_HSC
    if t.startswith("legacy_") or t.lower() == "ebv":
        # Legacy SHAPE_E1/E2 → physics group.
        if t in ("legacy_shape_e1", "legacy_shape_e2"):
            return GROUP_PHYSICS
        return GROUP_LEGACY
    return GROUP_PHYSICS


def _load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"[skip] {path} not found")
        return None
    return pd.read_csv(path)


def _series_from_variant(variant: str, latent_variant: str, score_name: str, arch: str) -> pd.DataFrame:
    df = _load_csv(_predictions_dir / f"predict_{variant}__{arch}.csv")
    if df is None:
        return pd.DataFrame(columns=["target", score_name])
    df = df[(df["score_metric"] == "r2") & (df["latent_variant"] == latent_variant)].copy()
    if df.empty:
        return pd.DataFrame(columns=["target", score_name])
    df["target_pref"] = df.apply(_prefix, axis=1)
    return df[["target_pref", "score"]].rename(
        columns={"target_pref": "target", "score": score_name}
    )


def _xpred_rename(t: str) -> str:
    """cross_predict.py uses `hsc_<band>_psf_fwhm` while the rest of the engaging
    pipeline uses `hsc_psf_fwhm_<band>`. Translate so band-averaging matches."""
    if t.startswith("hsc_") and t.endswith("_psf_fwhm"):
        band = t[len("hsc_"):-len("_psf_fwhm")]
        if band in ("g", "r", "i", "z", "y"):
            return f"hsc_psf_fwhm_{band}"
    return t


def _series_from_crosspredict() -> pd.DataFrame:
    """Read the two cross_predict_experiment CSVs (one per direction) and return
    a single (target, 'Cross-predict') frame, with target names normalised to
    match the engaging-pipeline conventions (so band-averaging via AVERAGE_PATTERNS
    folds them into legacy_PSFSIZE / legacy_PSFDEPTH / legacy_GALDEPTH / legacy_NOBS /
    hsc_psf_fwhm)."""
    frames = []
    for fname in ("r2_results_hsc_to_legacy.csv", "r2_results_legacy_to_hsc.csv"):
        path = _xpred_dir / fname
        df = _load_csv(path)
        if df is None:
            continue
        if not {"target", "r2_cross_predict"}.issubset(df.columns):
            print(f"[skip] {path}: unexpected columns {list(df.columns)}")
            continue
        df = df.copy()
        df["target"] = df["target"].astype(str).map(_xpred_rename)
        frames.append(df.rename(columns={"r2_cross_predict": "Cross-predict"})[
            ["target", "Cross-predict"]
        ])
    if not frames:
        return pd.DataFrame(columns=["target", "Cross-predict"])
    out = pd.concat(frames, ignore_index=True)
    return out.groupby("target", as_index=False)["Cross-predict"].mean()


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


def extra_column_names(label: str) -> tuple[str, str]:
    """(physics_col, instrument_col) for an overlaid extra variant with `label`."""
    return f"{label} (physics)", f"{label} (instr)"


def build_table(arch: str, instr_source: str, phys_source: str = "flat",
                extra_variants: list[tuple[str, str]] | None = None) -> pd.DataFrame:
    instr_variant = "base-meanpool" if instr_source == "pooled" else "base"
    phys_variant  = "base-meanpool" if phys_source  == "pooled" else "base"
    ours_phys   = _series_from_variant(phys_variant,                "combined_e1", "Ours (physics)",   arch)
    ours_instr  = _series_from_variant(instr_variant,               "combined_e2", "Ours (instr)",     arch)
    aion        = _series_from_variant("aion-base",                 "combined_e1", "AION",             arch)
    untr_match  = _series_from_variant("untrained-resnet-matched",  "combined_e1", "ResNet baseline",  arch)

    out = ours_phys.merge(ours_instr, on="target", how="outer")
    if not aion.empty:
        out = out.merge(aion, on="target", how="outer")
    else:
        out["AION"] = np.nan
    if not untr_match.empty:
        out = out.merge(untr_match, on="target", how="outer")
    else:
        out["ResNet baseline"] = np.nan

    xpred = _series_from_crosspredict()
    if not xpred.empty:
        out = out.merge(xpred, on="target", how="outer")
    else:
        out["Cross-predict"] = np.nan

    # Overlay any extra variants (e.g. the contrastive baseline): a physics series
    # from combined_e1 and an instrument series from combined_e2, one column each.
    extra_cols: list[str] = []
    for (variant, label) in (extra_variants or []):
        phys_col, inst_col = extra_column_names(label)
        ep = _series_from_variant(variant, "combined_e1", phys_col, arch)
        ei = _series_from_variant(variant, "combined_e2", inst_col, arch)
        if not ep.empty:
            out = out.merge(ep, on="target", how="outer")
            extra_cols.append(phys_col)
        if not ei.empty:
            out = out.merge(ei, on="target", how="outer")
            extra_cols.append(inst_col)

    out = out[~out["target"].isin(EXCLUDE_TARGETS)].reset_index(drop=True)

    numeric_cols = [c for c in
                    (["Ours (physics)", "Ours (instr)", "AION", "ResNet baseline", "Cross-predict"]
                     + extra_cols)
                    if c in out.columns]
    out = average_by_patterns(out, numeric_cols)
    out = out[~out["target"].isin(EXCLUDE_TARGETS)].reset_index(drop=True)

    group_order = [GROUP_PHYSICS, GROUP_LEGACY, GROUP_HSC]
    out["group_rank"] = out["group"].map({g: i for i, g in enumerate(group_order)})
    sort_metric = -np.fmax(
        out["Ours (physics)"].fillna(-1).values,
        out["Ours (instr)"].fillna(-1).values,
    )
    out["_sort"] = sort_metric
    out = out.sort_values(["group_rank", "_sort"]).drop(columns=["_sort"]).reset_index(drop=True)
    return out


# Palette for overlaid extra series (distinct from the 5 built-in bar colors).
EXTRA_PALETTE = ["#7D3C98", "#17A2A2", "#D4A017", "#C0392B",
                 "#2C3E50", "#E67E22", "#1ABC9C", "#8E44AD"]


def _extra_specs(extra_variants: list[tuple[str, str]] | None):
    """Return render specs (col, legend, color, hatch) for overlaid variants."""
    specs = []
    ci = 0
    for (_variant, label) in (extra_variants or []):
        phys_col, inst_col = extra_column_names(label)
        specs.append((phys_col, f"{label} (Physics)", EXTRA_PALETTE[ci % len(EXTRA_PALETTE)], False))
        ci += 1
        specs.append((inst_col, f"{label} (Instrument)", EXTRA_PALETTE[ci % len(EXTRA_PALETTE)], False))
        ci += 1
    return specs


def render(df: pd.DataFrame, arch: str, instr_source: str, out_png: Path,
           phys_source: str = "flat", extra_specs: list | None = None) -> Path:
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

    # (column_name_in_df, legend_label, color, hatch?)
    label_specs = [
        ("Ours (physics)",  "Ours (physics)",                 "#2E86AB", False),
        ("Ours (instr)",    "Ours (Instrument)",              "#CC546D", False),
        ("AION",            "AION-1",                         "#6CA75D", False),
        ("ResNet baseline", "Rand. Init. ResNet Baseline",    None,      True),
        ("Cross-predict",   "Cross-predict",                  "#BF823B", False),
    ]
    label_specs += (extra_specs or [])
    series = []
    for (col_name, label, color, hatch) in label_specs:
        if col_name not in df.columns:
            continue
        # Drop a column with only NaN (e.g. Cross-predict before the SLURM run finishes).
        if df[col_name].isna().all():
            continue
        series.append((label, df[col_name].values, color, hatch))

    n = len(series)
    width = 0.95 / max(n, 1)
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

    fig_width = max(8, (x[-1] - x[0] + 2) * 0.7)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    for i, (label, vals, color, hatch) in enumerate(series):
        # Leave NaN as NaN — matplotlib silently skips them, so a sparse
        # series (e.g. Cross-predict, only defined on HSC targets) doesn't
        # render fake zero-height bars on every non-covered target.
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

    # Reference lines on physics targets — span the bar group's width per target.
    half_span = max(np.abs(offsets)) + width / 2
    aion_color = "#1F3A68"
    astroclip_color = "#8B0000"
    for i, t in enumerate(targets):
        x_left, x_right = x[i] - half_span, x[i] + half_span
        if t in AION_B_PH_IM:
            ax.hlines(AION_B_PH_IM[t], x_left, x_right,
                      colors=aion_color, linestyles="--", linewidth=2.0, zorder=5)
        if t in ASTROCLIP_IM:
            ax.hlines(ASTROCLIP_IM[t], x_left, x_right,
                      colors=astroclip_color, linestyles=":", linewidth=2.0, zorder=5)

    margin = max(np.abs(offsets)) + width / 2 + 0.15
    ax.set_xlim(x[0] - margin, x[-1] + margin)
    ax.set_ylabel("R²", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=12)
    ax.tick_params(axis="x", which="both", length=0, top=False, bottom=False)

    from matplotlib.lines import Line2D
    bar_handles, bar_labels = ax.get_legend_handles_labels()
    ref_handles = [
        Line2D([0], [0], color=aion_color, linestyle="--", linewidth=2.0,
               label="AION-1-B (Ph+Im)"),
        Line2D([0], [0], color=astroclip_color, linestyle=":", linewidth=2.0,
               label="AstroCLIP (Parker '24, Im)"),
    ]
    all_handles = bar_handles + ref_handles
    all_labels = bar_labels + [h.get_label() for h in ref_handles]
    ax.legend(all_handles, all_labels,
              loc="upper center", bbox_to_anchor=(0.5, 1.00),
              ncol=min(len(all_handles), 3), fontsize=10, frameon=True)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(axis="y", which="major", alpha=0.75, linestyle="--", zorder=1)
    ax.grid(axis="y", which="minor", alpha=0.6, linestyle=":", zorder=1)
    ax.axhline(0, color="gray", linewidth=0.7, zorder=2)
    arch_label = {"aion": "strict AION (Linear→GELU→Linear, hidden=256)",
                  "aion_reg": "AION + LN + Dropout(0.2)"}.get(arch, arch)
    fig.suptitle(f"3-probe-per-encoder  ·  MLP arch: {arch_label}  ·  "
                 f"physics={phys_source}  ·  instrument={instr_source}",
                 fontsize=11, y=1.13)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    out_pdf = out_png.with_suffix(".pdf")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_png


def build_convergence_summary(arch: str, out_txt: Path) -> Path:
    """Write a per-target convergence dump across all 4 variants."""
    variants = ["base", "base-meanpool", "aion-base", "untrained-resnet-matched"]
    chunks = []
    for v in variants:
        df = _load_csv(_predictions_dir / f"predict_{v}__{arch}.csv")
        if df is None:
            continue
        df = df.copy()
        df.insert(0, "variant", v)
        chunks.append(df)
    if not chunks:
        out_txt.write_text("(no CSVs found for arch=" + arch + ")\n")
        return out_txt
    all_df = pd.concat(chunks, ignore_index=True)

    lines = [f"=== Convergence summary (mlp_arch={arch}) ==="]
    grouped = all_df.groupby(["variant", "task_family", "latent_variant"])
    lines.append("")
    lines.append(f"{'variant':28s}  {'family':18s}  {'latent':14s}  "
                 f"{'best_ep':>7s}  {'epochs_run':>10s}  {'max_ep':>6s}  {'early_stop':>10s}  "
                 f"{'mean_R²':>7s}  n_targets")
    for (v, fam, lat), g in grouped:
        be = int(g["best_epoch"].iloc[0])
        er = int(g["epochs_run"].iloc[0])
        mx = int(g["max_epochs"].iloc[0])
        es = bool(g["early_stopped"].iloc[0])
        rmean = g[g["score_metric"] == "r2"]["score"].mean()
        n_targ = len(g[g["score_metric"] == "r2"])
        lines.append(f"{v:28s}  {fam:18s}  {lat:14s}  "
                     f"{be:>7d}  {er:>10d}  {mx:>6d}  {str(es):>10s}  "
                     f"{rmean:>7.3f}  {n_targ}")

    lines.append("")
    lines.append("--- Per-target detail (R² · best_epoch · early_stopped) ---")
    for (v, fam, lat), g in grouped:
        lines.append(f"\n[{v} :: {fam} :: {lat}]")
        for _, row in g.iterrows():
            if row["score_metric"] != "r2":
                continue
            lines.append(f"   {row['target']:32s}  R²={row['score']:.3f}  "
                         f"best_epoch={int(row['best_epoch']):>3d}  "
                         f"epochs_run={int(row['epochs_run']):>3d}  "
                         f"early={str(bool(row['early_stopped']))}")

    out_txt.write_text("\n".join(lines) + "\n")
    return out_txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlp-arch", choices=["aion", "aion_reg"], required=True)
    ap.add_argument("--instr-source", choices=["pooled", "flat"], default="pooled",
                    help="Where to read the instrument bar from. "
                         "pooled → base-meanpool combined_e2 (16D per encoder). "
                         "flat   → base combined_e2 (64D per encoder).")
    ap.add_argument("--phys-source", choices=["pooled", "flat"], default="flat",
                    help="Where to read the physics bar from. "
                         "flat   → base combined_e1 (64D per encoder, default). "
                         "pooled → base-meanpool combined_e1 (16D per encoder).")
    ap.add_argument("--no-discord", action="store_true")
    ap.add_argument("--no-cross-predict", action="store_true",
                    help="Drop the Cross-predict series even if its CSVs exist "
                         "(use for paper-clean plots).")
    ap.add_argument("--suffix", default=None,
                    help="Override the output filename suffix (default derives "
                         "from mlp_arch/phys/instr).")
    ap.add_argument("--extra-variant", action="append", default=[],
                    help="Overlay an extra model as 'variant:label' — adds its "
                         "physics (combined_e1) and instrument (combined_e2) series. "
                         "Repeatable. Reads predict_<variant>__<arch>.csv.")
    args = ap.parse_args()

    extra_variants = []
    for spec in args.extra_variant:
        variant, _, label = spec.partition(":")
        extra_variants.append((variant, label or variant))

    df = build_table(args.mlp_arch, args.instr_source, args.phys_source,
                     extra_variants=extra_variants)
    if args.no_cross_predict and "Cross-predict" in df.columns:
        df = df.drop(columns=["Cross-predict"])
    suffix = args.suffix or f"{args.mlp_arch}_phys-{args.phys_source}_instr-{args.instr_source}"
    out_png = _plots_dir / f"unified_three_probes_{suffix}.png"
    out_csv = _plots_dir / f"unified_three_probes_{suffix}__merged.csv"
    out_txt = _plots_dir / f"unified_three_probes_{args.mlp_arch}__convergence.txt"

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    render(df, args.mlp_arch, args.instr_source, out_png, phys_source=args.phys_source,
           extra_specs=_extra_specs(extra_variants))
    build_convergence_summary(args.mlp_arch, out_txt)
    out_pdf = out_png.with_suffix(".pdf")
    print(f"Plot (PNG):  {out_png}")
    print(f"Plot (PDF):  {out_pdf}")
    print(f"Merged CSV:  {out_csv}")
    print(f"Convergence: {out_txt}")

    if not args.no_discord:
        bars = "Ours(physics) + Ours(Instrument) + AION-1 + Rand. Init. ResNet Baseline"
        if not args.no_cross_predict:
            bars += " + Cross-predict"
        for (_v, label) in extra_variants:
            bars += f" + {label}(Phys/Instr)"
        msg = (f"R² per-property · mlp_arch=**{args.mlp_arch}** · "
               f"physics={args.phys_source} · instr={args.instr_source}. {bars}.")
        notify(WEBHOOK_URL, msg, str(out_pdf if out_pdf.exists() else out_png))


if __name__ == "__main__":
    main()
