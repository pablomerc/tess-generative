"""
Build the rebuttal instrument-property table: published columns + the new
concat(physics, instrument) columns, over the same folds.

Reads two fold-CSV directories:
  --pub-folds   ../engaging/outputs/predictions/folds        (READ ONLY, published)
  --new-folds   outputs/predictions/folds                    (this experiment)

and reproduces the published aggregation math *exactly*, in the same order as
contrastive_ablation/aggregate_folds.py — this matters, because doing it in the
other order gives a different error bar:

  1. per (variant, family, target, latent): mean and **population** std over folds
     (aggregate_folds.py:88-91)
  2. per band group: mean of the band means; band stds combined in quadrature and
     divided by the number of bands (aggregate_folds.py:104-114)

With a single fold the std is undefined, so cells print bare means -- which is why
--folds 0 gives an honest apples-to-apples preliminary table: the published columns
are recomputed on fold 0 only, not copied from the 5-fold RESULT_folds.md.

Identity gate (--check-identity, on by default): every derived H5 carries
`combined_e1` through verbatim, so its combined_e1 probe scores must reproduce the
published `base` / `contrastive-spatial-conv1x1` combined_e1 scores. Any real
deviation means the derived-H5 path is not reproducing the published pipeline and
the new numbers are not comparable.

CLI:

    python aggregate_rebuttal_table.py --folds 0                 # preliminary
    python aggregate_rebuttal_table.py --folds 0,1,2,3,4         # final
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PUB_FOLDS = HERE.parent / "engaging" / "outputs" / "predictions" / "folds"
NEW_FOLDS = HERE / "outputs" / "predictions" / "folds"

FNAME_RE = re.compile(r"^predict_(?P<variant>.+)__aion_f(?P<fold>\d+)of(?P<nf>\d+)\.csv$")

# Table rows: (display label, target prefixes, task family). Same prefixes as
# aggregate_folds.py:28-41. The prefix determines the family uniquely, which is what
# lets the survey-matched columns pick their source variant.
ROWS = [
    ("Galaxy Depth",   ("GALDEPTH",), "instrument_legacy"),
    ("PSF Depth",      ("PSFDEPTH",), "instrument_legacy"),
    ("# Observations", ("NOBS",),     "instrument_legacy"),
    ("E(B-V)",         ("EBV",),      "instrument_legacy"),
    ("PSF FWHM (HSC)", ("psf_fwhm",), "instrument_hsc"),
]

# A column is (label, source, {family: variant}, latent, is_main).
# `source`: "pub" = published fold CSVs, "new" = this experiment's.
COLUMNS = [
    # ---- main table -------------------------------------------------------------
    ("Ours (instr, flat 128-D)", "pub",
     {"*": "base"}, "combined_e2", True),
    ("Contrastive (instr, flat 128-D)", "pub",
     {"*": "contrastive-spatial-conv1x1"}, "combined_e2", True),
    ("Ours phys+instr (256-D)", "new",
     {"*": "base-e1e2-256"}, "combined_e2", True),
    ("Contrastive phys+instr (256-D)", "new",
     {"*": "contr-e1e2-256"}, "combined_e2", True),

    # ---- supplementary controls -------------------------------------------------
    ("Ours phys+instr matched (128-D)", "new",
     {"instrument_hsc": "base-e1e2m-hsc", "instrument_legacy": "base-e1e2m-leg"},
     "combined_e2", False),
    ("Contrastive phys+instr matched (128-D)", "new",
     {"instrument_hsc": "contr-e1e2m-hsc", "instrument_legacy": "contr-e1e2m-leg"},
     "combined_e2", False),
    ("Ours instr matched (64-D)", "new",
     {"instrument_hsc": "base-e2m-hsc", "instrument_legacy": "base-e2m-leg"},
     "combined_e2", False),
    ("Contrastive instr matched (64-D)", "new",
     {"instrument_hsc": "contr-e2m-hsc", "instrument_legacy": "contr-e2m-leg"},
     "combined_e2", False),
    ("Ours instr pooled (32-D)", "pub",
     {"*": "base-meanpool"}, "combined_e2", False),
    ("Contrastive instr pooled (32-D)", "pub",
     {"*": "contrastive-spatial-conv1x1-meanpool"}, "combined_e2", False),
    ("Ours physics only (128-D)", "pub",
     {"*": "base"}, "combined_e1", False),
    ("Contrastive physics only (128-D)", "pub",
     {"*": "contrastive-spatial-conv1x1"}, "combined_e1", False),

]

# Scale control, promoted into the main table: each 64-D block rescaled to unit RMS
# before concatenating. Run because the raw 256-D concat came in *below* the 128-D
# instrument-only baseline, which is exactly the signature a block-scale imbalance
# leaves — and it recovered most of the drop, so the raw number understates the
# information content and the balanced one is the fair measurement.
COLUMNS[4:4] = [
    ("Ours phys+instr (256-D, balanced)", "new",
     {"*": "base-e1e2-256-bal"}, "combined_e2", True),
    ("Contrastive phys+instr (256-D, balanced)", "new",
     {"*": "contr-e1e2-256-bal"}, "combined_e2", True),
]

# Derived variant -> the published variant its combined_e1 must reproduce.
_RECIPES = ("e1e2-256", "e1e2m-hsc", "e1e2m-leg", "e2m-hsc", "e2m-leg", "e1e2-256-bal")
IDENTITY_REF = {
    **{f"base-{r}": "base" for r in _RECIPES},
    **{f"contr-{r}": "contrastive-spatial-conv1x1" for r in _RECIPES},
}

IDENTITY_PASS = 1e-6   # numerically identical
IDENTITY_WARN = 1e-3   # thread/BLAS-order noise; still comparable
# above IDENTITY_WARN -> hard failure


def load_folds(folds_dir: Path, wanted_folds: set[int]) -> dict:
    """(variant, family, target, latent) -> {fold: score}."""
    scores: dict[tuple, dict[int, float]] = defaultdict(dict)
    if not folds_dir.is_dir():
        return scores
    for path in sorted(folds_dir.glob("predict_*__aion_f*of*.csv")):
        m = FNAME_RE.match(path.name)
        if not m:
            continue
        fold = int(m.group("fold"))
        if fold not in wanted_folds:
            continue
        variant = m.group("variant")
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                if r.get("score_metric") != "r2":
                    continue
                try:
                    v = float(r["score"])
                except (TypeError, ValueError):
                    continue
                if math.isnan(v):
                    continue
                key = (variant, r["task_family"], r["target"], r["latent_variant"])
                scores[key][fold] = v
    return scores


def cell(scores: dict, variant: str, family: str, latent: str,
         prefixes: tuple[str, ...]) -> tuple[float | None, float | None, int, int]:
    """aggregate_folds.py's math, in its order. Returns (mean, std, n_bands, n_folds)."""
    per_target = []
    n_folds_seen = 0
    for (v, fam, target, lat), by_fold in scores.items():
        if v != variant or fam != family or lat != latent:
            continue
        if not any(target.startswith(p) for p in prefixes):
            continue
        vals = [by_fold[f] for f in sorted(by_fold)]
        if not vals:
            continue
        n_folds_seen = max(n_folds_seen, len(vals))
        per_target.append((st.mean(vals),
                           st.pstdev(vals) if len(vals) > 1 else float("nan")))
    if not per_target:
        return None, None, 0, 0
    means = [m for m, _ in per_target]
    stds = [s for _, s in per_target if not math.isnan(s)]
    std = (math.sqrt(sum(s * s for s in stds)) / len(stds)) if stds else float("nan")
    return st.mean(means), std, len(per_target), n_folds_seen


def fmt(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "--"
    if std is None or math.isnan(std):
        return f"{mean:+.3f}"
    return f"{mean:+.3f} ± {std:.3f}"


def check_identity(pub: dict, new: dict, folds: set[int]) -> tuple[str, list[str]]:
    """Derived combined_e1 scores must reproduce the published ones."""
    lines, worst, n_cmp, missing = [], 0.0, 0, []
    for derived, ref in sorted(IDENTITY_REF.items()):
        deltas = []
        for (v, fam, target, lat), by_fold in new.items():
            if v != derived or lat != "combined_e1":
                continue
            ref_by_fold = pub.get((ref, fam, target, "combined_e1"))
            if not ref_by_fold:
                continue
            for f, val in by_fold.items():
                if f in ref_by_fold:
                    deltas.append(abs(val - ref_by_fold[f]))
        if not deltas:
            missing.append(derived)
            continue
        mx = max(deltas)
        n_cmp += len(deltas)
        worst = max(worst, mx)
        verdict = "PASS" if mx <= IDENTITY_PASS else ("WARN" if mx <= IDENTITY_WARN else "FAIL")
        lines.append(f"| `{derived}` | `{ref}` | {len(deltas)} | {mx:.2e} | {verdict} |")
    if not lines:
        return "NO DATA", lines
    status = "PASS" if worst <= IDENTITY_PASS else ("WARN" if worst <= IDENTITY_WARN else "FAIL")
    if missing:
        lines.append(f"| _(no combined_e1 rows yet: {', '.join(missing)})_ | | | | |")
    return status, lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", default="0",
                    help="Comma-separated folds to aggregate, e.g. '0' or '0,1,2,3,4'.")
    ap.add_argument("--pub-folds", type=Path, default=PUB_FOLDS)
    ap.add_argument("--new-folds", type=Path, default=NEW_FOLDS)
    ap.add_argument("--out-md", type=Path, default=HERE / "RESULT_instrument_concat.md")
    ap.add_argument("--out-csv", type=Path, default=HERE / "RESULT_instrument_concat.csv")
    ap.add_argument("--no-check-identity", action="store_true")
    args = ap.parse_args()

    folds = {int(x) for x in args.folds.split(",") if x.strip() != ""}
    pub = load_folds(args.pub_folds, folds)
    new = load_folds(args.new_folds, folds)
    both = {**pub, **new}
    srcmap = {"pub": pub, "new": new}
    print(f"[aggregate] folds={sorted(folds)}  pub_cells={len(pub)}  new_cells={len(new)}")

    # ---- build the table ----------------------------------------------------
    table: dict[str, dict[str, str]] = {}
    rows_csv = []
    for label, src, vmap, latent, _main in COLUMNS:
        scores = srcmap[src]
        table[label] = {}
        for row_label, prefixes, family in ROWS:
            variant = vmap.get(family, vmap.get("*"))
            if variant is None:
                table[label][row_label] = "--"
                continue
            mean, std, n_bands, n_folds_seen = cell(scores, variant, family, latent, prefixes)
            table[label][row_label] = fmt(mean, std)
            rows_csv.append(dict(column=label, property=row_label, source=src,
                                 variant=variant, latent=latent, family=family,
                                 r2_mean="" if mean is None else f"{mean:.6f}",
                                 r2_err="" if std is None or math.isnan(std) else f"{std:.6f}",
                                 n_bands=n_bands, n_folds=n_folds_seen))

    # ---- identity gate ------------------------------------------------------
    id_status, id_lines = ("SKIPPED", [])
    if not args.no_check_identity:
        id_status, id_lines = check_identity(pub, new, folds)

    # ---- markdown -----------------------------------------------------------
    nf = len(folds)
    single = nf == 1
    L = [
        "# Instrument properties from concat(physics, instrument)",
        "",
        f"Folds aggregated: **{sorted(folds)}** ({nf} of 5).",
        "",
        "Frozen checkpoints, no training: **Ours** = the paper base model",
        "(`checkpoints_new/base/snapshot.ckpt`, `neighbors_all_attn/2026-04-05`, "
        "`best-epoch=228-step=87000`); **Contrastive** = "
        "`checkpoints_new/contrastive-spatial-conv1x1/best-epoch=53-step=83000.ckpt`.",
        "Probe: 2-layer MLP (`Linear(d,256)->GELU->Linear`), `--mlp-arch aion --seed 0 "
        "--n-folds 5`, `max_epochs=50`, early stopping on `val/loss`; R2 on the held-out fold.",
        "",
        "The 256-D feature is **64 x 4** — 64 per image per latent x 2 surveys x 2 latents:",
        "",
        "```",
        "[ hsc_e1(64) | legacy_e1(64) | hsc_e2(64) | legacy_e2(64) ]  = 256",
        "  physics      physics         instrument   instrument",
        "```",
        "",
        "and each 64 = 4 spatial tokens x 16 channels (the 2x2 grid of a 48x48 input,",
        "flattened). The published \"Ours\" column is the right-hand half only: 128-D.",
        "",
    ]
    if single:
        L += [
            f"> **Preliminary — single fold ({sorted(folds)[0]}), so no error bars.** Every column,",
            "> including the published ones, is recomputed on this fold alone, so the comparison",
            "> is apples-to-apples. Folds 1-4 add the error bars.",
            "",
        ]

    def emit(title: str, cols: list[tuple], note: str = "") -> None:
        L.append(f"## {title}")
        L.append("")
        if note:
            L.extend([note, ""])   # not `L += ...`: that would rebind L as a local
        L.append("| property | " + " | ".join(c[0] for c in cols) + " |")
        L.append("|---" * (len(cols) + 1) + "|")
        for row_label, _p, _f in ROWS:
            L.append(f"| {row_label} | " +
                     " | ".join(table[c[0]][row_label] for c in cols) + " |")
        L.append("")

    emit("Main table", [c for c in COLUMNS if c[4]],
         "The four right-hand columns are new. `balanced` rescales each 64-D block to unit "
         "RMS before concatenating; it is the fair measurement, because the physics blocks "
         "carry ~1.9x the RMS of the instrument blocks and the raw concat therefore "
         "down-weights the informative half. Caveat: the 256-D concat has **2x the probe "
         "input width** of the 128-D columns — see the dimension-matched control below.")
    emit("Supplementary — controls", [c for c in COLUMNS if not c[4]],
         "`matched` = the survey providing the target only (Legacy latents for Legacy "
         "targets, HSC for HSC), so no cross-survey information. `pooled 32-D` and "
         "`physics only` are published cells, recomputed here on the same folds.")

    L += ["## Validation — `combined_e1` identity gate", "",
          "Every derived H5 carries `combined_e1` through verbatim, so its probe scores must "
          "reproduce the published pipeline's. This validates the derived-feature path "
          "end-to-end.", "",
          f"**Status: {id_status}**", ""]
    if id_lines:
        L += ["| derived variant | reference | comparisons | max abs delta | verdict |",
              "|---|---|---|---|---|"] + id_lines + [""]
    L += [f"Thresholds: PASS <= {IDENTITY_PASS:g}, WARN <= {IDENTITY_WARN:g} "
          "(BLAS reduction-order noise), FAIL above.", "",
          "## Provenance", "",
          f"- published fold CSVs (read-only): `{args.pub_folds}`",
          f"- new fold CSVs: `{args.new_folds}`",
          "- derived features: `build_derived_h5.py` (post-hoc H5 transform of the cached "
          "embeddings; no re-encoding, no GPU)",
          "- probe driver: `../engaging/predict_combined.py`, **unmodified**",
          "- aggregation math replicates `contrastive_ablation/aggregate_folds.py:88-114`",
          ""]

    args.out_md.write_text("\n".join(L))
    print(f"[aggregate] wrote {args.out_md}")

    if rows_csv:
        with open(args.out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_csv[0].keys()))
            w.writeheader()
            w.writerows(rows_csv)
        print(f"[aggregate] wrote {args.out_csv} ({len(rows_csv)} rows)")

    print(f"[aggregate] identity gate: {id_status}")
    if id_status == "FAIL":
        raise SystemExit("identity gate FAILED — derived path does not reproduce published scores")


if __name__ == "__main__":
    main()
