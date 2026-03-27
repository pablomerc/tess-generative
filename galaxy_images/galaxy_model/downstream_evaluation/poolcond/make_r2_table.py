"""
Build a combined R² table across five model variants.

Sources
-------
  - 4x4:         4x4/predict_all_ours_ours_instrument4x4.csv
  - 4x16:        final/predict_all_zdim16_nogeom_neighbors_table.csv
  - Pooled 1x16: poolcond/predict_all_ours_poolcond.csv
  - AION-1:      aion_benchmark/ (MMU → standard → legacy, first match wins)
  - Untrained:   taken from poolcond CSV (same resnet for all)

Outputs
-------
  - r2_table.csv    machine-readable, columns per model × physics/instrument
  - r2_table.tex    LaTeX booktabs table
  - r2_table.png    styled heatmap-style table figure

Usage
-----
  python make_r2_table.py
  python make_r2_table.py --no-scienceplots
"""
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_here = Path(__file__).resolve().parent
_downstream = _here.parent
_galaxy_model = _downstream.parent
_aion = _galaxy_model / "aion_benchmark"
_final = _downstream / "final"

CSV_4X4       = _downstream / "4x4" / "predict_all_ours_ours_instrument4x4.csv"
CSV_4X16      = _final / "predict_all_zdim16_nogeom_neighbors_table.csv"
CSV_POOLCOND  = _here / "predict_all_ours_poolcond.csv"

AION_CSVS = [
    _aion / "aion_mmu"    / "r2_comparisons_aion_mmu_v2"        / "downstream_aion_mmu_matched_512-256-128.csv",
    _aion / "r2_comparisons_aion"                               / "downstream_aion_512-256-128.csv",
    _aion / "aion_legacy" / "r2_comparisons_aion_legacy_v2"     / "downstream_aion_legacy_train_512-256-128.csv",
]

OUT_CSV = _here / "r2_table.csv"
OUT_TEX = _here / "r2_table.tex"
OUT_PNG = _here / "r2_table.png"

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

LABEL_MAPPING = {
    "desi_Z": "Z (redshift)",
    "LOG_MSTAR": "log M★",
    "TAGE_MW": "t_age",
    "LOG_Z_MW": "log Z_Met",
    "sSFR": "sSFR",
    "SHAPE_E1": "Ellipticity e1",
    "SHAPE_E2": "Ellipticity e2",
    "SHAPE_R": "R_eff",
    "a": "Extinction a",
    "hsc_variance_value": "Variance",
    "hsc_psf_fwhm": "PSF Size (HSC)",
    "legacy_GALDEPTH": "Galaxy Depth",
    "legacy_NOBS": "# Observations",
    "legacy_PSFSIZE": "PSF Size (Legacy)",
    "legacy_PSFDEPTH": "PSF Depth",
    "EBV": "Extinction E(B-V)",
}

GROUP_ORDER = ["physics", "legacy", "hsc"]
GROUP_PHYSICS = "physics"
GROUP_LEGACY  = "legacy"
GROUP_HSC     = "hsc"

GROUP_DISPLAY = {
    GROUP_PHYSICS: "Physics-like",
    GROUP_LEGACY:  "Legacy",
    GROUP_HSC:     "HSC",
}


def _get_group(target):
    t = str(target)
    if t in ("a", "hsc_variance_value", "hsc_psf_fwhm") or t.startswith("hsc_"):
        return GROUP_HSC
    if t.lower() == "ebv" or t.startswith("legacy_"):
        return GROUP_LEGACY
    return GROUP_PHYSICS


def _apply_patterns(df, numeric_cols):
    """Collapse band-specific rows into averages per pattern."""
    to_drop, new_rows = [], []
    for pattern, new_name in AVERAGE_PATTERNS.items():
        mask = df["target"].astype(str).str.contains(pattern, regex=True)
        subset = df.loc[mask].copy()
        if new_name == "hsc_psf_fwhm":
            subset = subset[subset["target"] != "hsc_z_psf_fwhm"]
        if subset.empty:
            continue
        to_drop.extend(subset.index.tolist())
        row = subset[[c for c in numeric_cols if c in subset.columns]].mean(numeric_only=True).to_dict()
        row["target"] = new_name
        new_rows.append(row)
    df = df.drop(index=to_drop)
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    return df


def load_raw_csv(path, phys_col, inst_col, untr_col=None):
    """Load a raw (multi-row-per-target) CSV and return per-target means."""
    df = pd.read_csv(path)
    df = df[~df["target"].isin(EXCLUDE_TARGETS)]
    agg_cols = {phys_col: "mean", inst_col: "mean"}
    if untr_col and untr_col in df.columns:
        agg_cols[untr_col] = "mean"
    return df.groupby("target", as_index=False).agg(agg_cols)


def load_4x4():
    df = load_raw_csv(CSV_4X4, "r2_physics", "r2_instrument", "r2_untrained")
    return df.rename(columns={
        "r2_physics":    "Phys_4x4",
        "r2_instrument": "Inst_4x4",
        "r2_untrained":  "Untrained",
    })


def load_4x16():
    df = pd.read_csv(CSV_4X16)
    df = df[~df["target"].isin(EXCLUDE_TARGETS)]
    return df[["target", "Physics latents", "Instrument latents"]].rename(columns={
        "Physics latents":    "Phys_4x16",
        "Instrument latents": "Inst_4x16",
    })


def load_poolcond():
    df = load_raw_csv(CSV_POOLCOND, "r2_physics", "r2_instrument")
    return df.rename(columns={
        "r2_physics":    "Phys_pooled",
        "r2_instrument": "Inst_pooled",
    })


def load_aion():
    parts = []
    for path in AION_CSVS:
        if path.exists():
            parts.append(pd.read_csv(path)[["target", "r2_mean"]].rename(columns={"r2_mean": "AION"}))
    if not parts:
        return None
    return pd.concat(parts, ignore_index=True).drop_duplicates(subset=["target"], keep="first")


def build_table():
    base     = load_4x4()          # has Untrained too
    t4x16    = load_4x16()
    poolcond = load_poolcond()
    aion     = load_aion()

    df = base.merge(t4x16,    on="target", how="outer")
    df = df.merge(poolcond,   on="target", how="outer")
    if aion is not None:
        df = df.merge(aion,   on="target", how="left")

    numeric_cols = [c for c in ("Phys_4x4", "Inst_4x4",
                                "Phys_4x16", "Inst_4x16",
                                "Phys_pooled", "Inst_pooled",
                                "Untrained", "AION") if c in df.columns]

    df = df[~df["target"].isin(EXCLUDE_TARGETS)]
    df = _apply_patterns(df, numeric_cols)
    df["group"] = df["target"].apply(_get_group)
    df["group_rank"] = df["group"].map({g: i for i, g in enumerate(GROUP_ORDER)})
    df["_sort"] = -df["Phys_4x16"].fillna(-1)
    df = df.sort_values(["group_rank", "_sort"]).reset_index(drop=True)
    df["label"] = df["target"].map(LABEL_MAPPING).fillna(df["target"])
    return df, numeric_cols


DISPLAY_COLS = [
    ("Phys_4x4",    "Physics\n4x4"),
    ("Inst_4x4",    "Instrument\n4x4"),
    ("Phys_4x16",   "Physics\n4x16"),
    ("Inst_4x16",   "Instrument\n4x16"),
    ("Phys_pooled", "Physics\npooled 1x16"),
    ("Inst_pooled", "Instrument\npooled 1x16"),
    ("AION",        "AION-1"),
    ("Untrained",   "Untrained\nResNet-18"),
]


def save_csv(df):
    cols = ["group", "target", "label"] + [c for c, _ in DISPLAY_COLS if c in df.columns]
    df[cols].to_csv(OUT_CSV, index=False, float_format="%.4f")
    print(f"CSV saved: {OUT_CSV}")


def save_latex(df):
    present = [(c, h) for c, h in DISPLAY_COLS if c in df.columns]
    header_line = " & ".join(["Group", "Property"] + [h.replace("\n", " ") for _, h in present]) + r" \\"
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{ll" + "r" * len(present) + "}",
        r"\toprule",
        header_line,
        r"\midrule",
    ]
    last_group = None
    for _, row in df.iterrows():
        g = row["group"]
        if g != last_group:
            if last_group is not None:
                lines.append(r"\midrule")
            last_group = g
        vals = []
        for c, _ in present:
            v = row.get(c, np.nan)
            vals.append(f"{v:.3f}" if pd.notna(v) else "—")
        lines.append(f"{GROUP_DISPLAY.get(g, g)} & {row['label']} & " + " & ".join(vals) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{R\textsuperscript{2} comparison across model variants.}",
        r"\label{tab:r2_comparison}",
        r"\end{table}",
    ]
    OUT_TEX.write_text("\n".join(lines))
    print(f"LaTeX saved: {OUT_TEX}")


def save_png(df):
    present = [(c, h) for c, h in DISPLAY_COLS if c in df.columns]
    col_keys   = [c for c, _ in present]
    col_labels = [h for _, h in present]

    data = df[col_keys].values.astype(float)

    n_rows, n_cols = data.shape
    cell_w, cell_h = 1.1, 0.38
    label_w = 3.0
    group_w = 1.1
    fig_w = group_w + label_w + n_cols * cell_w + 0.4
    fig_h = 0.5 + n_rows * cell_h + 0.6

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    # Column headers
    header_y = fig_h - 0.3
    for j, lbl in enumerate(col_labels):
        cx = group_w + label_w + (j + 0.5) * cell_w
        ax.text(cx, header_y, lbl, ha="center", va="center",
                fontsize=7.5, fontweight="bold", linespacing=1.2)

    GROUP_BG = {
        GROUP_PHYSICS: "#e8e8e8",
        GROUP_LEGACY:  "#dce8f5",
        GROUP_HSC:     "#f5e8d8",
    }
    GROUP_FG = {
        GROUP_PHYSICS: "#555555",
        GROUP_LEGACY:  "#2563a8",
        GROUP_HSC:     "#996515",
    }

    # Per-column min/max for colouring (ignoring NaN)
    col_min = np.nanmin(data, axis=0)
    col_max = np.nanmax(data, axis=0)

    last_group = None
    group_start_y = None
    group_label_info = []  # (group, center_y)

    for i, (_, row) in enumerate(df.iterrows()):
        row_y = fig_h - 0.6 - (i + 0.5) * cell_h
        g = row["group"]

        # Group background stripe
        stripe_y = fig_h - 0.6 - i * cell_h
        ax.add_patch(plt.Rectangle(
            (0, stripe_y - cell_h), fig_w, cell_h,
            facecolor=GROUP_BG.get(g, "#ffffff"), edgecolor="none", zorder=0,
        ))

        # Group label (emit once per group block)
        if g != last_group:
            if last_group is not None:
                mid_y = (group_start_y + row_y + cell_h / 2) / 2
                group_label_info.append((last_group, mid_y))
                # divider line
                div_y = fig_h - 0.6 - i * cell_h
                ax.axhline(div_y, color="#aaaaaa", linewidth=0.7, zorder=2)
            group_start_y = row_y + cell_h / 2
            last_group = g

        # Property label
        ax.text(group_w + 0.1, row_y, row["label"], ha="left", va="center",
                fontsize=8.5)

        # Value cells
        for j, key in enumerate(col_keys):
            v = row.get(key, np.nan)
            cx = group_w + label_w + (j + 0.5) * cell_w
            cell_bg = "#ffffff"
            if pd.notna(v):
                span = col_max[j] - col_min[j]
                t = (v - col_min[j]) / span if span > 1e-9 else 0.5
                # white → green gradient
                r = int(255 - t * (255 - 200))
                g_ch = int(255 - t * (255 - 230))
                b_ch = int(255 - t * (255 - 200))
                cell_bg = f"#{r:02x}{g_ch:02x}{b_ch:02x}"
            ax.add_patch(plt.Rectangle(
                (group_w + label_w + j * cell_w, stripe_y - cell_h),
                cell_w, cell_h,
                facecolor=cell_bg, edgecolor="#cccccc", linewidth=0.4, zorder=1,
            ))
            txt = f"{v:.3f}" if pd.notna(v) else "—"
            ax.text(cx, row_y, txt, ha="center", va="center", fontsize=8)

    # Last group label
    if last_group is not None:
        bottom_y = fig_h - 0.6 - len(df) * cell_h
        mid_y = (group_start_y + bottom_y + cell_h / 2) / 2
        group_label_info.append((last_group, mid_y))

    for g, mid_y in group_label_info:
        ax.text(group_w / 2, mid_y, GROUP_DISPLAY.get(g, g),
                ha="center", va="center", fontsize=8.5, fontweight="bold",
                color=GROUP_FG.get(g, "black"), rotation=90)

    # Outer border
    border_y_top = fig_h - 0.45
    border_y_bot = fig_h - 0.6 - len(df) * cell_h
    ax.add_patch(plt.Rectangle(
        (0, border_y_bot), fig_w, border_y_top - border_y_bot,
        fill=False, edgecolor="#888888", linewidth=1.0, zorder=3,
    ))

    plt.tight_layout(pad=0.2)
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"PNG saved: {OUT_PNG}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-scienceplots", action="store_true")
    args = p.parse_args()

    if not args.no_scienceplots:
        try:
            import scienceplots  # noqa: F401
            plt.style.use(["science", "no-latex"])
        except ImportError:
            print("scienceplots not installed, using default style")

    df, _ = build_table()
    save_csv(df)
    save_latex(df)
    save_png(df)


if __name__ == "__main__":
    main()
