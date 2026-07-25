"""
Quantify the SSL-contamination effect for the contrastive baseline.

Reads two `predict_*__aion.csv` files produced by
`downstream_evaluation/engaging/predict_combined.py` — one from a run that trained
on all anchors, one from the DWNVAL run that excluded the n=5,469 downstream-eval
galaxies — and reports the per-cell R² difference.

The question it answers: does contrastive's strong R² asymmetry survive when the
probe galaxies were never seen by the SSL objective? If |ΔR²| stays at the level
already measured for the flow-matching arm (mean ±0.005 across 88 cells, from
`predict_base-5NB.csv` vs `predict_base-5NB-DWNVAL.csv`), contamination is not the
explanation and the comparison in the paper stands for both methods.

Usage:
  python compare_dwnval.py \
      --holdout-csv   .../predict_contrastive-spatial-conv1x1-DWNVAL__aion.csv \
      --reference-csv .../predict_contrastive-spatial-conv1x1__aion.csv \
      --out-md        contrastive_ablation/DWNVAL_RESULT.md
"""

import argparse
import csv
import statistics
from pathlib import Path

# Reference: the same comparison already measured on the flow-matching side.
FM_REFERENCE = {
    "base-5NB": -0.0047,
    "base-15NB": +0.0052,
}

# The cells that appear in the paper's Table 5 / Fig. 4, so the report leads with
# the numbers a reviewer would actually look up.
HEADLINE = [
    ("physics_provabgs", "desi_z", "combined_e1", "Redshift z (physics latent)"),
    ("physics_provabgs", "provabgs_logmstar", "combined_e1", "log M* (physics latent)"),
    ("physics_provabgs", "legacy_shape_e1", "combined_e1", "Ellipticity e1 (physics latent)"),
    ("physics_provabgs", "hsc_shape_e1_i", "combined_e1", "HSC ellipticity e1 (physics latent)"),
    ("instrument_legacy", "GALDEPTH_R", "combined_e2", "Galaxy Depth R (instrument latent)"),
    ("instrument_legacy", "PSFDEPTH_R", "combined_e2", "PSF Depth R (instrument latent)"),
    ("instrument_legacy", "NOBS_R", "combined_e2", "# Observations R (instrument latent)"),
    ("instrument_legacy", "EBV", "combined_e2", "E(B-V) (instrument latent)"),
    ("instrument_legacy", "GALDEPTH_R", "combined_e1", "Galaxy Depth R (physics latent — expect ~0)"),
    ("physics_provabgs", "desi_z", "combined_e2", "Redshift z (instrument latent — the leak)"),
]


def load(path: Path) -> dict:
    rows = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r.get("score_metric") != "r2":
                continue
            try:
                rows[(r["task_family"], r["target"], r["latent_variant"])] = float(r["score"])
            except (TypeError, ValueError):
                continue
    if not rows:
        raise SystemExit(f"no r2 rows parsed from {path}")
    return rows


def fmt(v) -> str:
    return "  n/a " if v is None else f"{v:+.3f}"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--holdout-csv", required=True, type=Path)
    p.add_argument("--reference-csv", required=True, type=Path)
    p.add_argument("--out-md", type=Path, default=None)
    p.add_argument("--top-movers", type=int, default=8)
    args = p.parse_args()

    hold = load(args.holdout_csv)
    ref = load(args.reference_csv)
    shared = sorted(set(hold) & set(ref))
    if not shared:
        raise SystemExit("the two CSVs share no (task_family, target, latent_variant) cells")

    deltas = {k: hold[k] - ref[k] for k in shared}
    vals = list(deltas.values())

    lines = []
    w = lines.append
    w("# Contrastive DWNVAL — SSL-contamination result\n")
    w(f"- holdout run   : `{args.holdout_csv.name}`")
    w(f"- reference run : `{args.reference_csv.name}`")
    w(f"- shared cells  : {len(shared)}\n")

    w("## Aggregate ΔR² (holdout − reference)\n")
    w("| statistic | value |")
    w("|---|---|")
    w(f"| mean | {statistics.mean(vals):+.4f} |")
    w(f"| median | {statistics.median(vals):+.4f} |")
    w(f"| stdev | {statistics.pstdev(vals):.4f} |")
    w(f"| max abs | {max(vals, key=abs):+.4f} |")
    w(f"| cells with abs(Δ) > 0.05 | {sum(1 for v in vals if abs(v) > 0.05)} / {len(vals)} |")
    w("")
    w("Flow-matching reference for the same comparison (already measured): " +
      ", ".join(f"`{k}` mean {v:+.4f}" for k, v in FM_REFERENCE.items()) + ".")
    w("")

    w("## By encoder x task family\n")
    w("| latent | task family | n | mean Δ | median Δ | max abs Δ |")
    w("|---|---|---|---|---|---|")
    groups = {}
    for (fam, _tgt, lat), d in deltas.items():
        groups.setdefault((lat, fam), []).append(d)
    for (lat, fam) in sorted(groups):
        g = groups[(lat, fam)]
        w(f"| {lat} | {fam} | {len(g)} | {statistics.mean(g):+.4f} | "
          f"{statistics.median(g):+.4f} | {max(g, key=abs):+.4f} |")
    w("")

    w("## Headline cells (paper Table 5 / Fig. 4)\n")
    w("| cell | reference | holdout | Δ |")
    w("|---|---|---|---|")
    for fam, tgt, lat, label in HEADLINE:
        k = (fam, tgt, lat)
        if k in ref or k in hold:
            w(f"| {label} | {fmt(ref.get(k))} | {fmt(hold.get(k))} | "
              f"{fmt(deltas.get(k))} |")
    w("")

    w(f"## Top {args.top_movers} movers\n")
    w("| task family | target | latent | reference | holdout | Δ |")
    w("|---|---|---|---|---|---|")
    for k in sorted(shared, key=lambda k: abs(deltas[k]), reverse=True)[: args.top_movers]:
        fam, tgt, lat = k
        w(f"| {fam} | {tgt} | {lat} | {ref[k]:+.3f} | {hold[k]:+.3f} | {deltas[k]:+.3f} |")
    w("")

    mean_abs = statistics.mean(abs(v) for v in vals)
    fm_bound = max(abs(v) for v in FM_REFERENCE.values())
    if mean_abs <= 2 * fm_bound:
        verdict = (
            f"Mean |ΔR²| = {mean_abs:.4f}, within the flow-matching arm's already-measured "
            f"contamination bound ({fm_bound:.4f}). **Seeing the eval galaxies during SSL "
            f"training does not explain the contrastive baseline's R² pattern**; the "
            f"generative-vs-contrastive comparison is unaffected."
        )
    else:
        verdict = (
            f"Mean |ΔR²| = {mean_abs:.4f}, LARGER than the flow-matching arm's bound "
            f"({fm_bound:.4f}). Contamination affects the contrastive arm more than the "
            f"generative one — report the holdout numbers as the headline and say so."
        )
    w("## Verdict\n")
    w(verdict)
    w("")

    text = "\n".join(lines)
    print(text)
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(text)
        print(f"\n[compare] wrote {args.out_md}")


if __name__ == "__main__":
    main()
