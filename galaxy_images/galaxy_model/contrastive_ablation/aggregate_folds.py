"""
Aggregate per-fold probe CSVs into mean +- std R2, and print the headline comparison.

Input: the CSVs written by `predict_combined.py --n-folds N --fold i`, which carry
`n_folds`/`fold` columns. Because the folds are a disjoint partition, each galaxy is
tested exactly once across the set, so the spread across folds is a genuine error bar
on the probe rather than the resampling noise you get from re-drawing 90/10 splits.

Output: a long-form CSV (one row per variant x task_family x target x latent with
mean/std/n_folds) and a markdown report leading with the cells the rebuttal quotes.

  python aggregate_folds.py --folds-dir <dir> --n-folds 5 \
      --out-csv folds_r2_summary.csv --out-md RESULT_folds.md
"""

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

# Filename shape written by folds_engaging.slurm.
FNAME = re.compile(r"^predict_(?P<variant>.+)__aion_f(?P<fold>\d+)of(?P<nfolds>\d+)\.csv$")

# (label, variant, latent, [target prefixes]) -- the dim-matched comparison of PLAN.md 1.
HEADLINE_TARGETS = [
    ("Redshift z", ["desi_z"], "combined_e1"),
    ("log M*", ["provabgs_logmstar"], "combined_e1"),
    ("sSFR", ["ssfr"], "combined_e1"),
    ("Ellipticity (legacy)", ["legacy_shape"], "combined_e1"),
    ("Ellipticity (HSC)", ["hsc_shape"], "combined_e1"),
    ("Galaxy Depth", ["GALDEPTH"], "combined_e2"),
    ("PSF Depth", ["PSFDEPTH"], "combined_e2"),
    ("# Observations", ["NOBS"], "combined_e2"),
    ("E(B-V)", ["EBV"], "combined_e2"),
    ("PSF FWHM (HSC)", ["psf_fwhm"], "combined_e2"),
    ("LEAK: z from instrument", ["desi_z"], "combined_e2"),
    ("LEAK: depth from physics", ["GALDEPTH"], "combined_e1"),
]


def fmt(m, s):
    if m is None:
        return "   -   "
    return f"{m:+.3f}±{s:.3f}" if s is not None and not math.isnan(s) else f"{m:+.3f}"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folds-dir", required=True, type=Path)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--out-csv", type=Path, default=None)
    p.add_argument("--out-md", type=Path, default=None)
    args = p.parse_args()

    # scores[(variant, family, target, latent)] -> {fold: r2}
    scores = defaultdict(dict)
    files = sorted(args.folds_dir.glob("predict_*__aion_f*of*.csv"))
    if not files:
        raise SystemExit(f"no fold CSVs found in {args.folds_dir}")

    for f in files:
        m = FNAME.match(f.name)
        if not m or int(m["nfolds"]) != args.n_folds:
            continue
        variant, fold = m["variant"], int(m["fold"])
        for r in csv.DictReader(open(f)):
            if r.get("score_metric") != "r2":
                continue
            try:
                scores[(variant, r["task_family"], r["target"], r["latent_variant"])][fold] = \
                    float(r["score"])
            except (TypeError, ValueError):
                continue

    variants = sorted({k[0] for k in scores})
    complete = sum(1 for v in scores.values() if len(v) == args.n_folds)
    print(f"[aggregate] {len(files)} fold CSVs | {len(variants)} variants | "
          f"{len(scores)} cells ({complete} with all {args.n_folds} folds)")

    rows = []
    for key in sorted(scores):
        variant, family, target, latent = key
        vals = [scores[key][f] for f in sorted(scores[key])]
        rows.append(dict(
            variant=variant, task_family=family, target=target, latent_variant=latent,
            r2_mean=statistics.mean(vals),
            r2_std=statistics.pstdev(vals) if len(vals) > 1 else float("nan"),
            r2_min=min(vals), r2_max=max(vals), n_folds_present=len(vals),
        ))

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[aggregate] wrote {args.out_csv} ({len(rows)} rows)")

    # ---- markdown report --------------------------------------------------------
    def group(variant, latent, prefixes):
        vals = [(r["r2_mean"], r["r2_std"]) for r in rows
                if r["variant"] == variant and r["latent_variant"] == latent
                and any(r["target"].startswith(pre) for pre in prefixes)]
        if not vals:
            return None, None
        # Average the band-level means; combine spreads in quadrature over the band group.
        means = [v[0] for v in vals]
        stds = [v[1] for v in vals if not math.isnan(v[1])]
        s = (math.sqrt(sum(x * x for x in stds)) / len(stds)) if stds else float("nan")
        return statistics.mean(means), s

    lines = [f"# {args.n_folds}-fold R² with error bars\n",
             f"Disjoint {args.n_folds}-way partition (seed 0), `--mlp-arch aion`; every galaxy "
             f"tested exactly once. Values are mean ± std across folds.\n",
             f"Variants: {', '.join(f'`{v}`' for v in variants)}\n"]

    lines.append("| cell | " + " | ".join(variants) + " |")
    lines.append("|---" * (len(variants) + 1) + "|")
    for label, prefixes, latent in HEADLINE_TARGETS:
        cells = [fmt(*group(v, latent, prefixes)) for v in variants]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")

    # How big is the typical error bar? That is the number that decides which gaps are real.
    all_std = [r["r2_std"] for r in rows if not math.isnan(r["r2_std"])]
    if all_std:
        lines.append(f"**Typical fold-to-fold spread:** median std "
                     f"{statistics.median(all_std):.3f}, "
                     f"90th pct {sorted(all_std)[int(0.9 * len(all_std))]:.3f}. "
                     f"Differences smaller than ~2x this are not resolvable with n=5,469.\n")

    text = "\n".join(lines)
    print("\n" + text)
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(text)
        print(f"[aggregate] wrote {args.out_md}")


if __name__ == "__main__":
    main()
