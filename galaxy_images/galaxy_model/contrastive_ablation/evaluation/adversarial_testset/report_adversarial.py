#!/usr/bin/env python3
"""Turn evaluate_adversarial.py outputs into the two tables we actually report.

Table A - delta_survey: R2(HSC-derived latent) - R2(Legacy-derived latent), per model.
          This is the resolution-collapse statistic. If contrastive aligns the two
          views onto their lowest common denominator, its delta_survey should be small
          even where HSC is much sharper; Ours should keep a gap. The model-by-survey
          interaction (Ours - Contrastive) is read with its object-paired bootstrap CI.

Table B - joint-embedding R2: physics R2 from the combined (128-d) latent, per model.
          This is the headline probe number restricted to each stratum.

Both are reported on the frozen strata: top_n (adversarial), random_n (plain uniform
same-size draw), matched_control (1:1, matched on the other two quality axes), and all.

Usage:
    python report_adversarial.py --eval-dir outputs/adversarial_eval --axis resolution
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PHYSICS = [
    "desi_z",
    "provabgs_logmstar",
    "provabgs_tage_mw",
    "provabgs_z_mw",
    "ssfr",
    "legacy_shape_e1",
    "legacy_shape_e2",
    "hsc_shape_e1_i",
    "hsc_shape_e2_i",
]
PRETTY = {
    "desi_z": "Redshift z",
    "provabgs_logmstar": "log M*",
    "provabgs_tage_mw": "t_age",
    "provabgs_z_mw": "log Z_met",
    "ssfr": "sSFR",
    "legacy_shape_e1": "Ellipticity e1 (Legacy)",
    "legacy_shape_e2": "Ellipticity e2 (Legacy)",
    "hsc_shape_e1_i": "Ellipticity e1 (HSC)",
    "hsc_shape_e2_i": "Ellipticity e2 (HSC)",
}
STRATA = ["top_n", "random_n", "matched_control", "all"]
STRATUM_LABEL = {
    "top_n": "top 200 (adversarial)",
    "random_n": "random 200",
    "matched_control": "matched control 200",
    "all": "all 5,469",
}


def _pivot(df: pd.DataFrame, value: str) -> pd.DataFrame:
    return df.pivot_table(index="target", columns="stratum", values=value, aggfunc="first")


def table_delta_survey(metrics: pd.DataFrame, axis: str) -> str:
    sub = metrics[(metrics.axis == axis) & metrics.view.isin(["hsc", "legacy"])]
    wide = sub.pivot_table(
        index=["model", "target", "stratum"], columns="view", values="r2", aggfunc="first"
    ).reset_index()
    wide["delta_survey"] = wide["hsc"] - wide["legacy"]
    lines = []
    for model in sorted(wide.model.unique()):
        m = wide[wide.model == model]
        piv = m.pivot_table(index="target", columns="stratum", values="delta_survey")
        cols = [c for c in STRATA if c in piv.columns]
        piv = piv.reindex([t for t in PHYSICS if t in piv.index])[cols]
        lines.append(f"\n**{model}** — delta_survey = R²(HSC latent) − R²(Legacy latent)\n")
        lines.append("| property | " + " | ".join(STRATUM_LABEL[c] for c in cols) + " |")
        lines.append("|---" * (len(cols) + 1) + "|")
        for t, row in piv.iterrows():
            lines.append(
                f"| {PRETTY.get(t, t)} | " + " | ".join(f"{row[c]:+.3f}" for c in cols) + " |"
            )
        lines.append(
            "| **mean (9)** | "
            + " | ".join(f"**{piv[c].mean():+.3f}**" for c in cols)
            + " |"
        )
    return "\n".join(lines)


def table_joint_r2(metrics: pd.DataFrame, axis: str) -> str:
    sub = metrics[(metrics.axis == axis) & (metrics.view == "combined")]
    if sub.empty:
        return "\n_No `combined` view present — rerun with `--pred MODEL:combined=...`._\n"
    # Order explicitly: "ours" first so the delta column really is Ours - Contrastive.
    # sorted() would put "contrastive" first and silently flip the sign of the label.
    present = list(sub.model.unique())
    models = [m for m in ("ours", "contrastive") if m in present]
    models += [m for m in sorted(present) if m not in models]
    lines = []
    for stratum in STRATA:
        s = sub[sub.stratum == stratum]
        if s.empty:
            continue
        piv = s.pivot_table(index="target", columns="model", values="r2", aggfunc="first")
        piv = piv.reindex([t for t in PHYSICS if t in piv.index])
        n = int(s["n"].max()) if "n" in s else -1
        lines.append(f"\n**{STRATUM_LABEL[stratum]}** (n≈{n} objects) — physics R², joint 128-d latent\n")
        lines.append("| property | " + " | ".join(models) + " | Ours − Contr |")
        lines.append("|---" * (len(models) + 2) + "|")
        for t, row in piv.iterrows():
            vals = " | ".join(f"{row[m]:.3f}" for m in models)
            d = row[models[0]] - row[models[1]] if len(models) == 2 else float("nan")
            lines.append(f"| {PRETTY.get(t, t)} | {vals} | {d:+.3f} |")
        means = [piv[m].mean() for m in models]
        dm = means[0] - means[1] if len(means) == 2 else float("nan")
        lines.append(
            "| **mean (9)** | " + " | ".join(f"**{v:.3f}**" for v in means) + f" | **{dm:+.3f}** |"
        )
    return "\n".join(lines)


def table_interaction(inter: pd.DataFrame, axis: str) -> str:
    if inter.empty:
        return "\n_No interaction rows._\n"
    sub = inter[(inter.axis == axis) & (inter.metric == "r2")]
    est = [c for c in ("interaction", "estimate", "a_minus_b", "value") if c in sub.columns]
    if not est:
        return "\n_Interaction column not recognised: " + str(list(sub.columns)) + "_\n"
    est = est[0]
    lines = [
        "\ninteraction = [R²(Ours,HSC) − R²(Ours,Legacy)] − [R²(Contr,HSC) − R²(Contr,Legacy)]",
        "\n> Positive ⇒ Ours exploits the sharper survey more than contrastive does,",
        "> i.e. contrastive has collapsed toward the lower-resolution view.\n",
        "| property | " + " | ".join(STRATUM_LABEL[s] for s in STRATA if s in set(sub.stratum)) + " |",
    ]
    cols = [s for s in STRATA if s in set(sub.stratum)]
    lines.append("|---" * (len(cols) + 1) + "|")
    piv = sub.pivot_table(index="target", columns="stratum", values=est, aggfunc="first")
    lo = sub.pivot_table(index="target", columns="stratum", values="ci_low", aggfunc="first")
    hi = sub.pivot_table(index="target", columns="stratum", values="ci_high", aggfunc="first")
    piv = piv.reindex([t for t in PHYSICS if t in piv.index])
    for t, row in piv.iterrows():
        cells = []
        for c in cols:
            cells.append(f"{row[c]:+.3f} [{lo.loc[t, c]:+.2f},{hi.loc[t, c]:+.2f}]")
        lines.append(f"| {PRETTY.get(t, t)} | " + " | ".join(cells) + " |")
    lines.append(
        "| **mean (9)** | " + " | ".join(f"**{piv[c].mean():+.3f}**" for c in cols) + " |"
    )
    return "\n".join(lines)


def table_headline(boot: pd.DataFrame, axis: str) -> str:
    """Ours - Contrastive on the joint latent, per stratum, with paired bootstrap CIs.

    R2 is normalised by WITHIN-stratum target variance, so R2 must not be compared
    across strata. The legitimate comparison is the Ours-Contrastive gap computed
    inside each stratum. nRMSE (residual in units of that stratum's sigma) is carried
    alongside as the variance-independent cross-check.
    """
    if boot.empty:
        return "\n_No bootstrap rows._\n"
    sub = boot[(boot.view == "combined") & (boot.axis == axis)]
    if sub.empty:
        return "\n_No combined-view bootstrap rows._\n"
    sign = 1 if sub.model_a.iloc[0] == "ours" else -1
    lines = [
        "\nOurs − Contrastive **within** each stratum (positive R² = Ours better;",
        "negative nRMSE = Ours better). 95% object-paired bootstrap, 2000 replicates.\n",
        "| stratum | n | mean ΔR² (9) | targets w/ CI≠0 | mean ΔnRMSE (9) |",
        "|---|---|---|---|---|",
    ]
    for st in STRATA:
        r2 = sub[(sub.stratum == st) & (sub.metric == "r2")]
        nr = sub[(sub.stratum == st) & (sub.metric == "nrmse")]
        if r2.empty:
            continue
        lo, hi = sign * r2.ci_low, sign * r2.ci_high
        if sign < 0:
            lo, hi = sign * r2.ci_high, sign * r2.ci_low
        nsig = int(((lo > 0) | (hi < 0)).sum())
        lines.append(
            f"| {STRATUM_LABEL[st]} | {int(r2.n.max())} | "
            f"**{sign * r2.a_minus_b.mean():+.3f}** | {nsig}/{len(r2)} | "
            f"{sign * nr.a_minus_b.mean():+.4f} |"
        )
    lines.append("\nPer-target, spectroscopic properties only (ΔR² [95% CI]):\n")
    spectro = ["desi_z", "provabgs_logmstar", "provabgs_tage_mw", "provabgs_z_mw"]
    cols = [s for s in STRATA if s in set(sub.stratum)]
    lines.append("| property | " + " | ".join(STRATUM_LABEL[c] for c in cols) + " |")
    lines.append("|---" * (len(cols) + 1) + "|")
    for t in spectro:
        cells = []
        for c in cols:
            r = sub[(sub.stratum == c) & (sub.metric == "r2") & (sub.target == t)]
            if r.empty:
                cells.append("—")
                continue
            r = r.iloc[0]
            lo, hi = (sign * r.ci_low, sign * r.ci_high) if sign > 0 else (sign * r.ci_high, sign * r.ci_low)
            star = "*" if (lo > 0 or hi < 0) else " "
            cells.append(f"{sign * r.a_minus_b:+.3f} [{lo:+.2f},{hi:+.2f}]{star}")
        lines.append(f"| {PRETTY.get(t, t)} | " + " | ".join(cells) + " |")
    lines.append("\n`*` = 95% CI excludes zero.")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-dir", required=True, type=Path)
    ap.add_argument("--axis", default="resolution", choices=["resolution", "snr", "joint"])
    ap.add_argument("--out-md", type=Path)
    args = ap.parse_args()

    metrics = pd.read_csv(args.eval_dir / "stratified_metrics.csv")
    inter_path = args.eval_dir / "survey_interaction_bootstrap.csv"
    inter = pd.read_csv(inter_path) if inter_path.exists() else pd.DataFrame()
    boot_path = args.eval_dir / "paired_bootstrap.csv"
    boot = pd.read_csv(boot_path) if boot_path.exists() else pd.DataFrame()

    parts = [
        f"# Adversarial resolution-collapse result — axis: `{args.axis}`",
        "",
        "Selection is quality-only and was frozen before any prediction was read "
        "(`adversarial_manifest.json: selection_uses_predictions = false`). "
        "Probes are out-of-fold over 5 disjoint folds. `top_n` = 200 galaxies where HSC "
        "resolves well and Legacy resolves poorly; `random_n` = uniform 200; "
        "`matched_control` = 200 outside top-N matched on the other two quality axes.",
        "",
        "## Headline — Ours vs Contrastive on the joint latent, by stratum",
        table_headline(boot, args.axis),
        "",
        "## Table A — delta_survey (per-model HSC-vs-Legacy latent asymmetry)",
        table_delta_survey(metrics, args.axis),
        "",
        "## Table B — model-by-survey interaction, object-paired bootstrap 95% CI",
        table_interaction(inter, args.axis),
        "",
        "## Table C — physics R² from the joint (combined 128-d) latent",
        table_joint_r2(metrics, args.axis),
        "",
    ]
    text = "\n".join(parts)
    if args.out_md:
        args.out_md.write_text(text)
        print(f"wrote {args.out_md}")
    print(text)


if __name__ == "__main__":
    main()
