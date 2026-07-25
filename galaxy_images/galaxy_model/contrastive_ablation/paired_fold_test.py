"""
Paired per-fold comparison of two variants — the statistically correct way to read
the cross-validation output.

Why paired: every fold tests the SAME galaxies for both variants, so the fold-to-fold
noise (which galaxies landed in this fold) is common to both and cancels in the
difference. Combining the two variants' independent std's instead throws that away and
badly understates significance — e.g. the redshift gap reads 0.032 ± 0.008 unpaired
but 0.032 ± 0.002 paired.

Reports mean difference, standard error over folds, and a t statistic, with a verdict
threshold deliberately set high (|t| >= 4) because with only 5 folds a t of 2-3 is not
worth quoting in a rebuttal.

  python paired_fold_test.py --folds-dir <dir> --a base --b contrastive-spatial-conv1x1

Also runs the differential test that matters most for the paper's "collapse toward the
lowest-resolution observation" claim: is the contrastive deficit on HSC-measured shapes
LARGER than on Legacy-measured shapes? That is a difference-of-differences, so it is
immune to any overall offset between the two models.
"""

import argparse
import csv
import glob
import math
import os
import re
import statistics
from collections import defaultdict
from pathlib import Path

FN = re.compile(r"^predict_(?P<v>.+)__aion_f(?P<f>\d+)of(?P<n>\d+)\.csv$")

CELLS = [
    ("Redshift z (phys)", ["desi_z"], "combined_e1"),
    ("log M* (phys)", ["provabgs_logmstar"], "combined_e1"),
    ("sSFR (phys)", ["ssfr"], "combined_e1"),
    ("Ellipticity LEGACY (phys)", ["legacy_shape"], "combined_e1"),
    ("Ellipticity HSC (phys)", ["hsc_shape"], "combined_e1"),
    ("Galaxy Depth (instr)", ["GALDEPTH"], "combined_e2"),
    ("PSF Depth (instr)", ["PSFDEPTH"], "combined_e2"),
    ("# Observations (instr)", ["NOBS"], "combined_e2"),
    ("E(B-V) (instr)", ["EBV"], "combined_e2"),
    ("PSF FWHM HSC (instr)", ["psf_fwhm"], "combined_e2"),
    ("LEAK z from instr", ["desi_z"], "combined_e2"),
    ("LEAK depth from phys", ["GALDEPTH"], "combined_e1"),
]


def load(folds_dir: Path, n_folds: int):
    scores = defaultdict(dict)
    for p in glob.glob(str(folds_dir / f"predict_*__aion_f*of{n_folds}.csv")):
        m = FN.match(os.path.basename(p))
        if not m or int(m["n"]) != n_folds:
            continue
        v, f = m["v"], int(m["f"])
        for r in csv.DictReader(open(p)):
            if r.get("score_metric") == "r2":
                try:
                    scores[(v, r["target"], r["latent_variant"])][f] = float(r["score"])
                except (TypeError, ValueError):
                    pass
    return scores


def group(scores, variant, latent, prefixes, n_folds):
    """Per-fold mean over a band group (e.g. GALDEPTH_{G,R,I,Z})."""
    per = {}
    for f in range(n_folds):
        vals = [v[f] for k, v in scores.items()
                if k[0] == variant and k[2] == latent
                and any(k[1].startswith(pre) for pre in prefixes) and f in v]
        if vals:
            per[f] = statistics.mean(vals)
    return per


def paired(pa, pb):
    fs = sorted(set(pa) & set(pb))
    if len(fs) < 2:
        return None
    d = [pa[f] - pb[f] for f in fs]
    m = statistics.mean(d)
    se = statistics.stdev(d) / math.sqrt(len(d))
    return m, se, (m / se if se > 0 else float("nan")), len(d)


def verdict(t):
    return "RESOLVED" if abs(t) >= 4 else ("marginal" if abs(t) >= 2 else "NOT resolved")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folds-dir", required=True, type=Path)
    p.add_argument("--a", required=True, help="variant A (e.g. base)")
    p.add_argument("--b", required=True, help="variant B (e.g. contrastive-spatial-conv1x1)")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--out-md", type=Path, default=None)
    args = p.parse_args()

    scores = load(args.folds_dir, args.n_folds)
    lines = [f"# Paired {args.n_folds}-fold comparison: `{args.a}` vs `{args.b}`\n",
             f"Positive = **{args.a}** higher. SE is over folds, paired (same test galaxies "
             f"both sides). |t| >= 4 required to call a difference resolved.\n",
             "| cell | A − B | SE | t | verdict |", "|---|---|---|---|---|"]
    for label, pref, lat in CELLS:
        r = paired(group(scores, args.a, lat, pref, args.n_folds),
                   group(scores, args.b, lat, pref, args.n_folds))
        if r is None:
            continue
        m, se, t, _ = r
        lines.append(f"| {label} | {m:+.3f} | {se:.3f} | {t:+.1f} | {verdict(t)} |")
    lines.append("")

    # Difference-of-differences: high-resolution vs low-resolution shape retention.
    leg = paired(group(scores, args.a, "combined_e1", ["legacy_shape"], args.n_folds),
                 group(scores, args.b, "combined_e1", ["legacy_shape"], args.n_folds))
    hsc = paired(group(scores, args.a, "combined_e1", ["hsc_shape"], args.n_folds),
                 group(scores, args.b, "combined_e1", ["hsc_shape"], args.n_folds))
    if leg and hsc:
        pl_a = group(scores, args.a, "combined_e1", ["legacy_shape"], args.n_folds)
        pl_b = group(scores, args.b, "combined_e1", ["legacy_shape"], args.n_folds)
        ph_a = group(scores, args.a, "combined_e1", ["hsc_shape"], args.n_folds)
        ph_b = group(scores, args.b, "combined_e1", ["hsc_shape"], args.n_folds)
        fs = sorted(set(pl_a) & set(pl_b) & set(ph_a) & set(ph_b))
        dd = [(ph_a[f] - ph_b[f]) - (pl_a[f] - pl_b[f]) for f in fs]
        m = statistics.mean(dd)
        se = statistics.stdev(dd) / math.sqrt(len(dd))
        lines += [
            "## Resolution-retention test (difference of differences)\n",
            f"(A−B on **HSC** shapes) − (A−B on **Legacy** shapes) = "
            f"**{m:+.3f} ± {se:.3f}** (t = {m / se:+.1f}, {verdict(m / se)})\n",
            "Positive means B loses more on the higher-resolution survey's shapes than on "
            "the lower-resolution one's — the signature of an objective that keeps only "
            "what the two views share. Being a difference of differences, it cannot be "
            "explained by either model simply being better overall.\n",
        ]

    text = "\n".join(lines)
    print(text)
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(text)
        print(f"[paired] wrote {args.out_md}")


if __name__ == "__main__":
    main()
