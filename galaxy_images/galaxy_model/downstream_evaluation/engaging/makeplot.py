"""
Stage 4 — read all per-checkpoint prediction CSVs and produce two plot artefacts:

1. compare_<ckpt>.png  (one per checkpoint): horizontal-bar layout with
   - rows grouped by task family (band-averaged target labels via regex)
   - each row has up to 3 bars hued by latent variant (HSC / Legacy / Combined / e2)
   - shaded background bands per task family

2. compare_all_checkpoints.png: cross-checkpoint comparison, one bar per
   checkpoint × latent variant per (task, target).

CLI shape (locked):

    python makeplot.py \
        --predictions-dir outputs/predictions \
        --out-dir         outputs/plots \
        [--checkpoints base hier hier-small single-baseline]
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Try to use scienceplots if present (pretty), fall back silently.
try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "no-latex"])
except Exception:
    pass


# ---- color/styling ----------------------------------------------------------
LATENT_COLORS = {
    "combined_e1": "#5DA271",  # green   — physics latent (e1, HSC+Legacy concat)
    "combined_e2": "#9C5DA2",  # purple  — instrument latent (e2, HSC+Legacy concat)
}

LATENT_LABELS = {
    "combined_e1": "Combined physics (e1)",
    "combined_e2": "Combined instrument (e2)",
}

# Reference baselines pulled from external CSVs:
#   OURS-FEB : prior pipeline run (Feb), Physics_latents → combined_e1, Instrument_latents → combined_e2
#   AION-FEB : prior AION-base eval on the OLD mixed-source dataset → combined_e1 slot
#              (kept here for context only; the *current* AION-base run goes through the
#              regular pipeline as a checkpoint named "aion-base")
REFERENCE_CHECKPOINTS = ("OURS-FEB", "AION-FEB")
EXTERNAL_TARGET_MAP = {
    "desi_Z":          ("physics_provabgs",  "desi_z"),
    "LOG_MSTAR":       ("physics_provabgs",  "provabgs_logmstar"),
    "sSFR":            ("physics_provabgs",  "ssfr"),
    "LOG_Z_MW":        ("physics_provabgs",  "provabgs_z_mw"),
    "TAGE_MW":         ("physics_provabgs",  "provabgs_tage_mw"),
    "legacy_GALDEPTH": ("instrument_legacy", "Gal depth"),
    "legacy_PSFDEPTH": ("instrument_legacy", "PSF depth"),
    "legacy_NOBS":     ("instrument_legacy", "# obs"),
    "legacy_PSFSIZE":  ("instrument_legacy", "PSF size"),
    "EBV":             ("instrument_legacy", "EBV"),
    "hsc_psf_fwhm":    ("instrument_hsc",    "PSF FWHM"),
    "a":               ("instrument_hsc",    "extinction (a_*)"),
    # SHAPE_E1 / SHAPE_E2 / SHAPE_R are AION-only — no equivalent in our pipeline.
}

TASK_BG = {
    "physics_provabgs":   "#f0f0f0",
    "morphology_gz10":    "#fff4e0",
    "instrument_hsc":     "#e6f2ff",
    "instrument_legacy":  "#fbe8ee",
}

TASK_ORDER = ["physics_provabgs", "morphology_gz10", "instrument_hsc", "instrument_legacy"]

TASK_TITLES = {
    "physics_provabgs":   "Physics (PROVABGS)",
    "morphology_gz10":    "Morphology (GZ10)",
    "instrument_hsc":     "HSC instrument",
    "instrument_legacy":  "Legacy instrument",
}

# Regex bands → averaged label.
AVERAGE_PATTERNS = [
    (re.compile(r"^a_[grizy]$"),                          "extinction (a_*)"),
    (re.compile(r"^[grizy]_variance_value$"),             "variance"),
    (re.compile(r"^psf_fwhm_[grizy]$"),                   "PSF FWHM"),
    (re.compile(r"^MW_TRANSMISSION_[GRIZ]$"),             "MW transmission"),
    (re.compile(r"^PSFSIZE_[GRIZ]$"),                     "PSF size"),
    (re.compile(r"^PSFDEPTH_[GRIZ]$"),                    "PSF depth"),
    (re.compile(r"^GALDEPTH_[GRIZ]$"),                    "Gal depth"),
    (re.compile(r"^NOBS_[GRIZ]$"),                        "# obs"),
]

LABEL_MAP = {
    "desi_z":              "redshift",
    "provabgs_logmstar":   "log M*",
    "provabgs_tage_mw":    "age (mw)",
    "provabgs_avg_sfr":    "avg SFR",
    "provabgs_z_mw":       "Z (mw)",
    "ssfr":                "log sSFR",
    "gz10_label":          "GZ10 class",
    "EBV":                 "E(B-V)",
}


def _avg_target(target: str) -> str:
    for pat, name in AVERAGE_PATTERNS:
        if pat.match(target):
            return name
    return target


def _pretty_target(name: str) -> str:
    return LABEL_MAP.get(name, name)


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def _load_predictions(pred_dir: Path, checkpoints: list[str]) -> pd.DataFrame:
    frames = []
    for c in checkpoints:
        path = pred_dir / f"predict_{c}.csv"
        if not path.exists():
            print(f"  [skip] {path} not found")
            continue
        df = pd.read_csv(path)
        if df.empty:
            print(f"  [skip] {path} is empty")
            continue
        # Some entries (classification) have two metrics for the same target — keep both rows.
        df["checkpoint"] = c
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_external_rows(merged_csv: Path) -> pd.DataFrame:
    """Read the merged comparison CSV and emit rows in the same schema as the
    band-averaged predictions df: [checkpoint, task_family, target,
    latent_variant, score_metric, score]."""
    if not merged_csv.exists():
        print(f"  [skip] external CSV {merged_csv} not found")
        return pd.DataFrame()
    src = pd.read_csv(merged_csv)
    out_rows = []
    for _, row in src.iterrows():
        t = row["target"]
        if t not in EXTERNAL_TARGET_MAP:
            continue
        task, our_target = EXTERNAL_TARGET_MAP[t]
        phys = row.get("Physics_latents", np.nan)
        inst = row.get("Instrument_latents", np.nan)
        aion = row.get("AION", np.nan)
        if pd.notna(phys):
            out_rows.append(dict(checkpoint="OURS-FEB", task_family=task,
                                 target=our_target, latent_variant="combined_e1",
                                 score_metric="r2", score=float(phys)))
        if pd.notna(inst):
            out_rows.append(dict(checkpoint="OURS-FEB", task_family=task,
                                 target=our_target, latent_variant="combined_e2",
                                 score_metric="r2", score=float(inst)))
        if pd.notna(aion):
            out_rows.append(dict(checkpoint="AION-FEB", task_family=task,
                                 target=our_target, latent_variant="combined_e1",
                                 score_metric="r2", score=float(aion)))
    if out_rows:
        print(f"  [ext] loaded {len(out_rows)} reference rows from {merged_csv.name}")
    return pd.DataFrame(out_rows)


def _band_average(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse multi-band targets into single rows per averaging pattern."""
    df = df.copy()
    df["target_avg"] = df["target"].apply(_avg_target)
    # Keep only the primary metric per task: r2 / accuracy / f1_macro all become a single
    # 'score' column already; for morph we want both, so retain (task, target_avg, latent, ckpt, metric).
    grouped = (
        df.groupby(["checkpoint", "task_family", "target_avg", "latent_variant", "score_metric"],
                   as_index=False)["score"].mean()
    )
    return grouped.rename(columns={"target_avg": "target"})


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _row_order(df: pd.DataFrame) -> list[tuple[str, str, str]]:
    """Return ordered list of (task_family, target, score_metric) rows for the plot."""
    rows = []
    for task in TASK_ORDER:
        sub = df[df["task_family"] == task]
        if sub.empty:
            continue
        for (target, metric), _g in sorted(sub.groupby(["target", "score_metric"])):
            rows.append((task, target, metric))
    return rows


def _plot_per_checkpoint(df_avg: pd.DataFrame, ckpt: str, out_path: Path):
    df = df_avg[df_avg["checkpoint"] == ckpt].copy()
    if df.empty:
        print(f"  [skip] no rows for {ckpt}")
        return
    rows = _row_order(df)
    if not rows:
        return

    variants = [v for v in ["combined_e1", "combined_e2"]
                if v in df["latent_variant"].unique()]

    n = len(rows)
    fig_h = max(4, 0.45 * n + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    # Group rows by task, with a small visual gap between task groups.
    y_positions = []
    task_bounds = {}
    cur = 0.0
    last_task = None
    for task, target, metric in rows:
        if last_task is not None and task != last_task:
            cur += 0.5
        if task not in task_bounds:
            task_bounds[task] = [cur, cur]
        task_bounds[task][1] = cur
        y_positions.append(cur)
        last_task = task
        cur += 1.0

    # Background shading per task.
    for task, (a, b) in task_bounds.items():
        ax.axhspan(a - 0.5, b + 0.5, color=TASK_BG.get(task, "#eeeeee"), alpha=0.6, zorder=0)
        ax.text(1.02, (a + b) / 2, TASK_TITLES.get(task, task),
                transform=ax.get_yaxis_transform(), va="center", ha="left",
                fontsize=9, fontweight="bold", color="#333", rotation=0)

    width = 0.8 / max(len(variants), 1)
    for vi, variant in enumerate(variants):
        scores = []
        for (task, target, metric), y in zip(rows, y_positions):
            sub = df[(df["task_family"] == task) & (df["target"] == target) &
                     (df["latent_variant"] == variant) & (df["score_metric"] == metric)]
            scores.append(float(sub["score"].iloc[0]) if not sub.empty else np.nan)
        offsets = (vi - (len(variants) - 1) / 2) * width
        ys = np.array(y_positions) + offsets
        ax.barh(ys, np.nan_to_num(scores, nan=0.0), height=width,
                color=LATENT_COLORS.get(variant, "#888"),
                edgecolor="gray", linewidth=0.4,
                label=LATENT_LABELS.get(variant, variant), zorder=3)

    yticks = list(y_positions)
    yticklabels = [f"{_pretty_target(t)} [{m}]" if m != "r2" else _pretty_target(t)
                   for (_task, t, m) in rows]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.7, zorder=2)
    ax.set_xlabel("Score (R² for regression, accuracy/F1 for morphology)")
    ax.set_title(f"{ckpt}", fontsize=12)
    ax.grid(axis="x", linestyle="--", alpha=0.3, zorder=1)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out_path}")


def _plot_cross_checkpoint(df_avg: pd.DataFrame, out_path: Path, checkpoints: list[str]):
    df = df_avg[df_avg["checkpoint"].isin(checkpoints)].copy()
    if df.empty:
        print(f"  [skip] no rows for cross-checkpoint plot")
        return
    rows = _row_order(df)
    if not rows:
        return

    # palette: one color per (latent_variant, checkpoint) pair, cycled by checkpoint.
    ckpt_alpha = {c: 0.45 + 0.5 * i / max(len(checkpoints) - 1, 1)
                  for i, c in enumerate(checkpoints)}

    n = len(rows)
    fig_h = max(5, 0.55 * n + 1.5)
    fig_w = max(11, 1.2 + 1.0 * len(checkpoints))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    y_positions = []
    task_bounds = {}
    cur = 0.0
    last_task = None
    for task, target, metric in rows:
        if last_task is not None and task != last_task:
            cur += 0.5
        if task not in task_bounds:
            task_bounds[task] = [cur, cur]
        task_bounds[task][1] = cur
        y_positions.append(cur)
        last_task = task
        cur += 1.0

    for task, (a, b) in task_bounds.items():
        ax.axhspan(a - 0.5, b + 0.5, color=TASK_BG.get(task, "#eeeeee"), alpha=0.5, zorder=0)
        ax.text(1.02, (a + b) / 2, TASK_TITLES.get(task, task),
                transform=ax.get_yaxis_transform(), va="center", ha="left",
                fontsize=9, fontweight="bold", color="#333")

    variants = [v for v in ["combined_e1", "combined_e2"]
                if v in df["latent_variant"].unique()]

    # Skip (variant, ckpt) pairs with no values at all — keeps the legend tidy
    # (e.g. AION has no combined_e2 entry).
    bar_groups = []
    for v in variants:
        for c in checkpoints:
            if not df[(df["checkpoint"] == c) & (df["latent_variant"] == v)].empty:
                bar_groups.append((v, c))
    width = 0.85 / max(len(bar_groups), 1)
    for gi, (variant, ckpt) in enumerate(bar_groups):
        color = LATENT_COLORS.get(variant, "#888")
        alpha = ckpt_alpha[ckpt]
        scores = []
        for (task, target, metric), y in zip(rows, y_positions):
            sub = df[(df["checkpoint"] == ckpt) & (df["task_family"] == task) &
                     (df["target"] == target) & (df["latent_variant"] == variant) &
                     (df["score_metric"] == metric)]
            scores.append(float(sub["score"].iloc[0]) if not sub.empty else np.nan)
        offsets = (gi - (len(bar_groups) - 1) / 2) * width
        ys = np.array(y_positions) + offsets
        # Reference baselines (OURS-FEB, AION) get a thicker dark edge to stand out.
        is_ref = ckpt in REFERENCE_CHECKPOINTS
        ax.barh(ys, np.nan_to_num(scores, nan=0.0), height=width,
                color=color, alpha=(0.95 if is_ref else alpha),
                edgecolor=("#111" if is_ref else "black"),
                linewidth=(0.9 if is_ref else 0.3),
                hatch=("//" if ckpt == "AION-FEB" else None),
                zorder=3,
                label=f"{LATENT_LABELS.get(variant, variant)} · {ckpt}")

    yticks = list(y_positions)
    yticklabels = [f"{_pretty_target(t)} [{m}]" if m != "r2" else _pretty_target(t)
                   for (_task, t, m) in rows]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.7, zorder=2)
    ax.set_xlabel("Score (R² for regression, accuracy/F1 for morphology)")
    ax.set_title("Cross-checkpoint comparison", fontsize=12)
    ax.grid(axis="x", linestyle="--", alpha=0.3, zorder=1)
    ax.legend(loc="upper left", bbox_to_anchor=(1.12, 1), fontsize=7, ncol=1, framealpha=0.95)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--checkpoints", nargs="+",
                   default=["base", "hier", "hier-small", "single-baseline"])
    p.add_argument("--external-csv", type=Path,
                   default=Path("/home/pablomer/orcd/pool/aion_benchmark_csvs/r2_comparisons_aion/aion_vs_ours_all_merged.csv"),
                   help="Optional merged CSV with OURS-FEB + AION reference R²s.")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = _load_predictions(args.predictions_dir, args.checkpoints)
    if df.empty:
        print("No prediction CSVs found — nothing to plot.")
        return
    df_avg = _band_average(df)

    found = sorted(df_avg["checkpoint"].unique().tolist())
    print(f"Plotting checkpoints: {found}")
    for c in found:
        _plot_per_checkpoint(df_avg, c, args.out_dir / f"compare_{c}.png")

    if len(found) >= 2:
        _plot_cross_checkpoint(df_avg, args.out_dir / "compare_all_checkpoints.png", found)

    # Reference baselines (prior pipeline run + AION) — written as a separate plot.
    ext = _load_external_rows(args.external_csv)
    if not ext.empty:
        df_with_ref = pd.concat([df_avg, ext], ignore_index=True)
        ref_ckpts = [c for c in REFERENCE_CHECKPOINTS if c in ext["checkpoint"].unique()]
        all_ckpts = found + ref_ckpts
        _plot_cross_checkpoint(df_with_ref,
                               args.out_dir / "compare_all_with_baselines.png",
                               all_ckpts)


if __name__ == "__main__":
    main()
