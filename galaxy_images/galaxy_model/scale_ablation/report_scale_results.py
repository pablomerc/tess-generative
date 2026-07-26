#!/usr/bin/env python
"""Tables + plots for the data-scale ablation: recon MSE vs N and downstream R^2 vs N.

Subcommands
-----------
  mse   read results/recon_mse_scale.csv  -> results/mse_table.md + results/mse_vs_n.png
  r2    read downstream predict_*.csv     -> results/r2_table.md  + results/r2_vs_n.png
                                             + results/r2_per_target.csv (full raw table)

Pure reader: consumes CSVs, writes markdown/PNG/CSV. Posting to Discord is done by the
caller (so plots get eyeballed before they are sent).

Conventions (deliberate, to match the published table exactly)
--------------------------------------------------------------
* Physics average = mean R^2 of the PHYSICS latent (combined_e1, flat 128-d) over the
  paper table's SEVEN physics rows: legacy_shape_e1/e2, desi_z, provabgs_logmstar, ssfr,
  provabgs_tage_mw, provabgs_z_mw. The raw physics family also contains hsc_shape_e1_i /
  e2_i, but the paper table omits them (make_table.py LABEL list), so the average here
  does too. The full per-target CSV keeps them.
* Instrument averages are reported **per survey, separately** — Legacy over its FIVE
  band-averaged rows (GALDEPTH, PSFDEPTH, PSFSIZE, EBV, NOBS) and HSC over its TWO
  (hsc_a, hsc_psf_fwhm). They are deliberately NOT pooled into one headline number: the
  two surveys differ hugely in depth and resolution, the paper's own table groups them as
  separate blocks, and a pooled mean would silently weight Legacy 5:2 and hide which
  survey drives the trend. A pooled figure is still emitted, as a non-bold reference row.
  Band-averaging mirrors plot_aion_vs_variant.AVERAGE_PATTERNS (mean over channels within
  a quantity); legacy_MW_TRANSMISSION is excluded exactly as there.
* Instrument latent source: the MEANPOOL twin (predict_<v>-meanpool.csv, pooled 32-d
  combined_e2) when available -- the paper's "Ours (Instrument)" bar is base-meanpool.
  Falls back to the flat combined_e2 with a visible flag if a twin is missing.
* Self-check: the aggregation is validated by recomputing the paper's own `base` column
  from predict_base__aion.csv and diffing against outputs/plot_paper/data/
  merged_simplified.csv. A mismatch prints loudly but does not block.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
ENGAGING = HERE.parent / "downstream_evaluation" / "engaging"
PREDICTIONS = ENGAGING / "outputs" / "predictions"
PAPER_SIMPLIFIED = ENGAGING / "outputs" / "plot_paper" / "data" / "merged_simplified.csv"

ARMS = {"scale-1000": 1000, "scale-3162": 3162, "scale-10000": 10000, "scale-31622": 31622}

# --- paper-table row definitions -----------------------------------------------------
PHYSICS_ROWS = [
    "legacy_shape_e1", "legacy_shape_e2", "desi_z",
    "provabgs_logmstar", "ssfr", "provabgs_tage_mw", "provabgs_z_mw",
]
INSTR_ROWS_LEGACY = ["legacy_GALDEPTH", "legacy_PSFDEPTH", "legacy_PSFSIZE", "legacy_EBV", "legacy_NOBS"]
INSTR_ROWS_HSC = ["hsc_a", "hsc_psf_fwhm"]
INSTR_ROWS = INSTR_ROWS_LEGACY + INSTR_ROWS_HSC

# Band-averaging on prefixed names — mirrors plot_aion_vs_variant.AVERAGE_PATTERNS.
AVERAGE_PATTERNS = {
    r"^hsc_a_[grizy]$": "hsc_a",
    r"^hsc_psf_fwhm_[grizy]$": "hsc_psf_fwhm",
    r"^legacy_PSFSIZE_[GRIZ]$": "legacy_PSFSIZE",
    r"^legacy_PSFDEPTH_[GRIZ]$": "legacy_PSFDEPTH",
    r"^legacy_GALDEPTH_[GRIZ]$": "legacy_GALDEPTH",
    r"^legacy_NOBS_[GRIZ]$": "legacy_NOBS",
    r"^legacy_MW_TRANSMISSION_[GRIZ]$": "legacy_MW_TRANSMISSION",
}
EXCLUDED_INSTR = {"legacy_MW_TRANSMISSION"}

# --- dataviz reference palette (validated order, slots 1-2 + chrome tokens) ----------
C_HSC = "#2a78d6"       # categorical slot 1 (blue)   — HSC anchor survey, MSE chart
C_LEGACY = "#eb6834"    # categorical slot 2 (orange) — Legacy anchor survey, MSE chart
C_SERIES = "#2a78d6"    # slot 1, shared by the R² small multiples (one series per panel)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASELINE)
        ax.spines[s].set_linewidth(1.0)
    ax.grid(True, axis="y", color=GRID, linewidth=1.0, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK2)


def _log_x(ax, ns):
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([f"{n:,}" for n in ns])
    ax.minorticks_off()


# ======================================================================================
# MSE
# ======================================================================================

def cmd_mse(post_note: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # The eval is split across CSVs when a run hits its wall clock and the remaining
    # models are finished in a second job. Every part is pinned to the SAME
    # recon_eval_manifest.json (identical 256 anchors, identical per-batch x0 noise seeds),
    # so concatenating parts is exactly equivalent to one run. First occurrence wins.
    parts = sorted(RESULTS.glob("recon_mse_scale*.csv"))
    if not parts:
        sys.exit(f"[mse] no recon_mse_scale*.csv in {RESULTS}")
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    df = df.drop_duplicates(subset=["model_label", "anchor_survey"], keep="first")
    print(f"[mse] read {len(parts)} part(s): {[p.name for p in parts]} -> "
          f"{df['model_label'].nunique()} models")
    manifest = json.loads((HERE / "subsets" / "manifest.json").read_text())
    imgs = {r["n_anchors"]: r["distinct_images_total"] for r in manifest["rungs"]}

    wide = df.pivot_table(index="model_label", columns="anchor_survey",
                          values=["mse_mean", "mse_sem"], aggfunc="first")

    order = ["scale-1000", "scale-3162", "scale-10000", "scale-31622",
             "fm-control-75k", "base-87k", "constlr-35k"]
    order = [m for m in order if m in wide.index]

    label = {
        "scale-1000": "N=1,000", "scale-3162": "N=3,162",
        "scale-10000": "N=10,000", "scale-31622": "N=31,622",
        "fm-control-75k": "full data, 75k (fm-control) *",
        "base-87k": "full data, 87k (paper ckpt) *",
        "constlr-35k": "full data, constant-LR 35k *",
    }

    lines = [
        "## Reconstruction MSE vs # crossmatches (C0 / intact, 256 held-out anchors, 250 Euler steps)",
        "",
        "| model | N crossmatches | distinct images | MSE HSC | MSE Legacy |",
        "|---|---|---|---|---|",
    ]
    for m in order:
        n = ARMS.get(m)
        ncol = f"{n:,}" if n else "93,362 (full)*"
        icol = f"{imgs[n]:,}" if n in imgs else "—"
        h = wide.loc[m, ("mse_mean", "hsc")]
        hs = wide.loc[m, ("mse_sem", "hsc")]
        l = wide.loc[m, ("mse_mean", "legacy")]
        ls = wide.loc[m, ("mse_sem", "legacy")]
        lines.append(f"| {label.get(m, m)} | {ncol} | {icol} | {h:.4f} ± {hs:.4f} | {l:.4f} ± {ls:.4f} |")
    lines += [
        "",
        "\\* trained **without** the downstream holdout → saw ~95% of these eval galaxies "
        "during training (unlabeled); their MSE is optimistic. The four scale arms never saw "
        "any eval galaxy — enforced at load time. Conservative direction for the scaling claim.",
    ]
    if post_note:
        lines += ["", post_note]
    md = "\n".join(lines)
    (RESULTS / "mse_table.md").write_text(md + "\n")
    print(md)

    # ---- plot ----
    ns = [ARMS[m] for m in order if m in ARMS]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)

    for survey, color, name in (("hsc", C_HSC, "HSC"), ("legacy", C_LEGACY, "Legacy")):
        y = [wide.loc[m, ("mse_mean", survey)] for m in order if m in ARMS]
        e = [wide.loc[m, ("mse_sem", survey)] for m in order if m in ARMS]
        ax.errorbar(ns, y, yerr=e, color=color, linewidth=2, marker="o", markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=2, capsize=3, label=name)
        if "fm-control-75k" in wide.index:
            ref = wide.loc["fm-control-75k", ("mse_mean", survey)]
            ax.axhline(ref, color=color, linewidth=1.4, linestyle=(0, (4, 3)), alpha=0.8)
            ax.annotate(f"full data ({name})*", xy=(ns[-1], ref), xytext=(4, 3),
                        textcoords="offset points", fontsize=8, color=INK2,
                        ha="right", va="bottom")

    _log_x(ax, ns)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("cross-matched training galaxies N (log scale)", color=INK2, fontsize=10)
    ax.set_ylabel("reconstruction MSE (held-out, 250-step)", color=INK2, fontsize=10)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper right")
    ax.set_title("Counterfactual-objective reconstruction vs crossmatch count",
                 color=INK, fontsize=12, loc="left", pad=14)
    fig.text(0.005, 0.005,
             "* full-data reference trained without holdout (saw ~95% of eval galaxies); "
             "dashed = fm-control-75k, step-matched",
             fontsize=7, color=MUTED)
    fig.tight_layout()
    out = RESULTS / "mse_vs_n.png"
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    print(f"\n[wrote] {out}")


# ======================================================================================
# R^2
# ======================================================================================

def _load_predict(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["score_metric"] == "r2"].copy()
    def _prefix(row):
        fam, t = row["task_family"], str(row["target"])
        if fam == "instrument_hsc" and not t.startswith("hsc_"):
            return "hsc_" + t
        if fam == "instrument_legacy" and not t.startswith("legacy_"):
            return "legacy_" + t
        return t
    df["ptarget"] = df.apply(_prefix, axis=1)
    return df


def _band_average(series: pd.Series) -> pd.Series:
    """Collapse per-channel targets to paper quantities (mean over channels)."""
    groups: dict[str, list[float]] = {}
    for t, v in series.items():
        out_name = t
        for pat, name in AVERAGE_PATTERNS.items():
            if re.match(pat, t):
                out_name = name
                break
        groups.setdefault(out_name, []).append(v)
    return pd.Series({k: float(np.mean(v)) for k, v in groups.items()})


def _column_for(name: str, flat_csv: Path, pool_csv: Path | None):
    """Return (banded physics+instr series, physics_mean, instr_means dict, pooled_flag)."""
    flat = _load_predict(flat_csv)
    e1 = flat[flat["latent_variant"] == "combined_e1"].set_index("ptarget")["score"]
    pooled = False
    if pool_csv is not None and pool_csv.exists():
        pool = _load_predict(pool_csv)
        e2 = pool[pool["latent_variant"] == "combined_e2"].set_index("ptarget")["score"]
        pooled = True
    else:
        e2 = flat[flat["latent_variant"] == "combined_e2"].set_index("ptarget")["score"]
    b1, b2 = _band_average(e1), _band_average(e2)

    rows = {}
    for r in PHYSICS_ROWS:
        rows[r] = b1.get(r, np.nan)
    for r in INSTR_ROWS:
        rows[r] = b2.get(r, np.nan)
    phys_mean = float(np.nanmean([rows[r] for r in PHYSICS_ROWS]))
    instr = {
        "all7": float(np.nanmean([rows[r] for r in INSTR_ROWS])),
        "legacy": float(np.nanmean([rows[r] for r in INSTR_ROWS_LEGACY])),
        "hsc": float(np.nanmean([rows[r] for r in INSTR_ROWS_HSC])),
    }
    return rows, phys_mean, instr, pooled


def _selfcheck(base_rows: dict) -> str:
    """Diff our recomputed base column against the paper's merged_simplified.csv."""
    if not PAPER_SIMPLIFIED.exists():
        return "self-check SKIPPED (merged_simplified.csv not found)"
    paper = pd.read_csv(PAPER_SIMPLIFIED).set_index("target")
    diffs = []
    for r in PHYSICS_ROWS:
        if r in paper.index:
            diffs.append(abs(base_rows[r] - float(paper.loc[r, "Ours (physics)"])))
    for r in INSTR_ROWS:
        if r in paper.index:
            diffs.append(abs(base_rows[r] - float(paper.loc[r, "Ours (instr)"])))
    if not diffs:
        return "self-check SKIPPED (no overlapping rows)"
    worst = max(diffs)
    verdict = "OK" if worst < 5e-3 else "MISMATCH — inspect before trusting the table"
    return f"self-check vs paper merged_simplified.csv: {len(diffs)} rows, max |Δ| = {worst:.2e} → {verdict}"


def cmd_r2(post_note: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variants = [(v, PREDICTIONS / f"predict_{v}.csv", PREDICTIONS / f"predict_{v}-meanpool.csv")
                for v in ARMS]
    variants += [
        ("fm-control", PREDICTIONS / "predict_fm-control.csv", PREDICTIONS / "predict_fm-control-meanpool.csv"),
        ("base", PREDICTIONS / "predict_base__aion.csv", PREDICTIONS / "predict_base-meanpool__aion.csv"),
        ("untrained", PREDICTIONS / "predict_untrained-resnet-matched__aion.csv",
         PREDICTIONS / "predict_untrained-resnet-matched-meanpool.csv"),
    ]

    cols, missing = {}, []
    for name, flat, pool in variants:
        if not flat.exists():
            missing.append(name)
            continue
        cols[name] = _column_for(name, flat, pool)
    if missing:
        print(f"[r2] missing predict CSVs (skipped): {missing}")
    if not cols:
        sys.exit("[r2] nothing to report yet")

    # full per-target CSV artifact (raw, un-banded, both latents)
    frames = []
    for name, (rows, *_rest) in cols.items():
        flat = _load_predict(dict(variants_index(variants))[name][0])
        flat["variant"] = name
        frames.append(flat[["variant", "task_family", "ptarget", "latent_variant", "score", "n_train", "n_test"]])
    pd.concat(frames).to_csv(RESULTS / "r2_per_target.csv", index=False)

    row_labels = {
        "legacy_shape_e1": "Ellipticity e1", "legacy_shape_e2": "Ellipticity e2",
        "desi_z": "Redshift z", "provabgs_logmstar": "log M*", "ssfr": "sSFR",
        "provabgs_tage_mw": "t_age", "provabgs_z_mw": "log Z_Met",
        "legacy_GALDEPTH": "Galaxy Depth (L)", "legacy_PSFDEPTH": "PSF Depth (L)",
        "legacy_PSFSIZE": "PSF Size (L)", "legacy_EBV": "Extinction E(B-V) (L)",
        "legacy_NOBS": "# Observations (L)", "hsc_a": "Extinction a (H)",
        "hsc_psf_fwhm": "PSF FWHM (H)",
    }
    names = list(cols)
    lines = [
        "## Downstream R² vs # crossmatches (n=5469 crossmatch, 90/10 MLP probes, fixed-75k ckpts)",
        "",
        "physics rows → physics latent (combined_e1, flat); instrument rows → instrument latent "
        "(combined_e2, **meanpool** twin where available — flagged † when flat fallback was used)",
        "",
        "| property | " + " | ".join(
            n + ("" if cols[n][3] else " †") for n in names) + " |",
        "|---|" + "|".join(["---"] * len(names)) + "|",
    ]
    blank = " | ".join([""] * len(names))

    def _block(header: str, rows: list[str], mean_key, bold_label: str):
        out = [f"| **{header}** | {blank} |"]
        for r in rows:
            out.append(f"| {row_labels[r]} | " + " | ".join(f"{cols[n][0][r]:.3f}" for n in names) + " |")
        out.append(f"| **{bold_label}** | " + " | ".join(f"**{mean_key(cols[n]):.3f}**" for n in names) + " |")
        return out

    lines += _block("Physics-related properties", PHYSICS_ROWS,
                    lambda c: c[1], "mean physics (7)")
    lines += _block("Legacy instrument properties", INSTR_ROWS_LEGACY,
                    lambda c: c[2]["legacy"], "mean Legacy instrument (5)")
    lines += _block("HSC instrument properties", INSTR_ROWS_HSC,
                    lambda c: c[2]["hsc"], "mean HSC instrument (2)")
    lines += [
        "| _pooled instrument (7), Legacy-weighted 5:2 — reference only_ | "
        + " | ".join(f"_{cols[n][2]['all7']:.3f}_" for n in names) + " |",
        "",
        "_Averaging follows the paper table exactly: 7 physics rows (HSC ellipticities excluded "
        "as in the paper), band-averaged instrument rows (channels averaged within each quantity, "
        "MW_TRANSMISSION excluded). **Legacy and HSC instrument means are reported separately** — "
        "the surveys differ in depth/resolution and the paper groups them separately; the pooled "
        "row is shown only for reference because it silently weights Legacy 5:2._",
    ]
    if "base" in cols:
        lines += ["", "_" + _selfcheck(cols["base"][0]) + "_"]
    if post_note:
        lines += ["", post_note]
    md = "\n".join(lines)
    (RESULTS / "r2_table.md").write_text(md + "\n")
    print(md)

    # ---- plot: three panels, HSC and Legacy instrument kept separate ----
    # Small multiples of the same metric type, one series each: the panel title carries
    # identity, so all three share categorical slot 1 rather than spending three hues on
    # information the titles already give (and no legend box is needed for one series).
    ns = [ARMS[v] for v in ARMS if v in cols]
    if len(ns) >= 2:
        fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), dpi=200)
        fig.patch.set_facecolor(SURFACE)
        panels = [
            ("mean physics R²  (7 properties)", lambda c: c[1], C_SERIES),
            ("mean Legacy instrument R²  (5)", lambda c: c[2]["legacy"], C_SERIES),
            ("mean HSC instrument R²  (2)", lambda c: c[2]["hsc"], C_SERIES),
        ]
        for ax, (title, get, color) in zip(axes, panels):
            _style(ax)
            y = [get(cols[v]) for v in ARMS if v in cols]
            ax.plot(ns, y, color=color, linewidth=2, marker="o", markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=2)
            for ref, ls, lab in (("fm-control", (0, (4, 3)), "full data*"),
                                 ("untrained", (0, (1, 2)), "rand-init floor")):
                if ref in cols:
                    v = get(cols[ref])
                    ax.axhline(v, color=MUTED, linewidth=1.4, linestyle=ls)
                    ax.annotate(lab, xy=(ns[0], v), xytext=(2, 3), textcoords="offset points",
                                fontsize=8, color=INK2, va="bottom")
            _log_x(ax, ns)
            ax.set_ylim(0, 1)
            ax.set_xlabel("cross-matched training galaxies N", color=INK2, fontsize=10)
            ax.set_title(title, color=INK, fontsize=11, loc="left")
        fig.suptitle("Downstream probe R² vs crossmatch count (fixed-75k checkpoints)",
                     color=INK, fontsize=12, x=0.01, ha="left")
        fig.text(0.005, 0.005,
                 "* full-data references trained without holdout; instrument panel uses the "
                 "paper's meanpool convention where available",
                 fontsize=7, color=MUTED)
        fig.tight_layout(rect=(0, 0.02, 1, 0.95))
        out = RESULTS / "r2_vs_n.png"
        fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
        print(f"\n[wrote] {out}")


def variants_index(variants):
    for name, flat, pool in variants:
        yield name, (flat, pool)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["mse", "r2"])
    ap.add_argument("--note", default="", help="extra line appended to the markdown table")
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.mode == "mse":
        cmd_mse(args.note)
    else:
        cmd_r2(args.note)


if __name__ == "__main__":
    main()
