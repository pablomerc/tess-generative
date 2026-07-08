"""
Read-only audit of the metadata-only baseline's input features and physics targets.

Computes per-column summary stats on the `overlap` subset of one
prepare_combined.py H5 (default: outputs/embeddings/base.h5 — labels are
identical across checkpoints), and a complete-case row report so we can
decide on (a) which columns to log-transform and (b) the NaN policy.

Outputs:
    outputs/diagnostics/baseline_feature_stats.csv
    outputs/diagnostics/baseline_complete_case_report.txt

Usage:
    python audit_baseline_features.py \
        --h5  outputs/embeddings/base.h5 \
        --out-dir outputs/diagnostics
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from predict_combined import (
    CkptH5,
    HSC_INSTRUMENT_TARGETS,
    LEGACY_INSTRUMENT_TARGETS,
    PHYSICS_TARGETS,
    _physics_label,
)

SUBSET = "overlap"

# Coordinate columns are stored as `hsc_ra`, `hsc_dec`, `legacy_ra`, `legacy_dec`.
COORD_COLS_PREFIXED = ["hsc_ra", "hsc_dec", "legacy_ra", "legacy_dec"]


def _candidate_input_columns() -> list[tuple[str, str]]:
    """Return list of (h5_key, group) where group is 'hsc' / 'legacy' / 'coord'."""
    cols: list[tuple[str, str]] = []
    for c in HSC_INSTRUMENT_TARGETS:
        cols.append((f"hsc_{c}", "hsc"))
    for c in LEGACY_INSTRUMENT_TARGETS:
        cols.append((f"legacy_{c}", "legacy"))
    for c in COORD_COLS_PREFIXED:
        cols.append((c, "coord"))
    return cols


def _summarize(name: str, group: str, x: np.ndarray) -> dict:
    """One-row summary of an array. Robust to all-NaN columns."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    finite_mask = np.isfinite(x)
    n_nan = int(np.isnan(x).sum())
    n_inf = int(np.isinf(x).sum())
    xf = x[finite_mask]
    if xf.size == 0:
        return dict(
            column=name, group=group, n=n, n_finite=0, n_nan=n_nan, n_inf=n_inf,
            min=np.nan, p1=np.nan, p50=np.nan, p99=np.nan, max=np.nan,
            mean=np.nan, std=np.nan, skew=np.nan,
            frac_zero=np.nan, frac_negative=np.nan,
        )
    return dict(
        column=name,
        group=group,
        n=n,
        n_finite=int(xf.size),
        n_nan=n_nan,
        n_inf=n_inf,
        min=float(np.min(xf)),
        p1=float(np.percentile(xf, 1)),
        p50=float(np.percentile(xf, 50)),
        p99=float(np.percentile(xf, 99)),
        max=float(np.max(xf)),
        mean=float(np.mean(xf)),
        std=float(np.std(xf)),
        skew=float(stats.skew(xf)),
        frac_zero=float(np.mean(xf == 0.0)),
        frac_negative=float(np.mean(xf < 0.0)),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5", type=Path, required=True,
                   help="Any prepare_combined.py output H5 (labels are identical across checkpoints).")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Auditing {args.h5}")
    h5 = CkptH5(args.h5)
    if not h5.has_subset(SUBSET):
        raise RuntimeError(f"H5 has no /{SUBSET} group")

    available_label_keys = set(h5.label_keys(SUBSET))
    print(f"  /{SUBSET}/labels has {len(available_label_keys)} keys")

    # --- Input feature stats -------------------------------------------------
    input_rows: list[dict] = []
    input_arrays: dict[str, np.ndarray] = {}  # for complete-case computation
    missing: list[str] = []
    for key, group in _candidate_input_columns():
        if key not in available_label_keys:
            missing.append(key)
            continue
        arr = h5.label(SUBSET, key)
        if arr is None:
            missing.append(key)
            continue
        arr = np.asarray(arr, dtype=np.float64).ravel()
        input_arrays[key] = arr
        input_rows.append(_summarize(key, group, arr))

    if missing:
        print(f"  warn: {len(missing)} candidate input columns not found: "
              f"{missing[:6]}{' ...' if len(missing) > 6 else ''}")

    # --- Target stats --------------------------------------------------------
    target_rows: list[dict] = []
    for tgt in PHYSICS_TARGETS:
        y = _physics_label(h5, SUBSET, tgt)
        if y is None:
            print(f"  [skip] target {tgt} not present")
            continue
        target_rows.append(_summarize(tgt, "target", np.asarray(y, dtype=np.float64)))
    # Bonus: log10(provabgs_avg_sfr) for the proposed log-transform discussion.
    sfr_raw = _physics_label(h5, SUBSET, "provabgs_avg_sfr")
    if sfr_raw is not None:
        sfr_arr = np.asarray(sfr_raw, dtype=np.float64).ravel()
        with np.errstate(invalid="ignore", divide="ignore"):
            log_sfr = np.where(sfr_arr > 0, np.log10(sfr_arr), np.nan)
        target_rows.append(_summarize("provabgs_avg_sfr_log10", "target_derived", log_sfr))

    # --- Write feature_stats.csv --------------------------------------------
    df = pd.DataFrame(input_rows + target_rows)
    feature_csv = args.out_dir / "baseline_feature_stats.csv"
    df.to_csv(feature_csv, index=False)
    print(f"\nWrote {feature_csv}  ({len(df)} rows)")

    # --- Complete-case report -----------------------------------------------
    n_overlap = next(iter(input_arrays.values())).size if input_arrays else 0
    keys = list(input_arrays.keys())
    if n_overlap == 0 or not keys:
        report = "no input columns found — cannot compute complete-case report\n"
    else:
        # Build (N, F) finite-mask matrix.
        finite_mat = np.column_stack([np.isfinite(input_arrays[k]) for k in keys])
        complete_all = int(finite_mat.all(axis=1).sum())

        hsc_keys = [k for k in keys if k.startswith("hsc_") and k not in COORD_COLS_PREFIXED]
        leg_keys = [k for k in keys if k.startswith("legacy_") and k not in COORD_COLS_PREFIXED]

        hsc_mask = np.column_stack([np.isfinite(input_arrays[k]) for k in hsc_keys]) if hsc_keys else np.empty((n_overlap, 0))
        leg_mask = np.column_stack([np.isfinite(input_arrays[k]) for k in leg_keys]) if leg_keys else np.empty((n_overlap, 0))
        complete_hsc = int(hsc_mask.all(axis=1).sum()) if hsc_mask.shape[1] else n_overlap
        complete_leg = int(leg_mask.all(axis=1).sum()) if leg_mask.shape[1] else n_overlap

        # Per-column n_nan ranking — biggest offenders first.
        nan_counts = sorted(
            ((k, int((~np.isfinite(input_arrays[k])).sum())) for k in keys),
            key=lambda kv: kv[1], reverse=True,
        )

        lines = []
        lines.append(f"H5: {args.h5}")
        lines.append(f"Subset: /{SUBSET}")
        lines.append(f"N rows in /{SUBSET}: {n_overlap}")
        lines.append("")
        lines.append("Complete-case (all candidate inputs finite, no NaN/inf):")
        lines.append(f"  All inputs ({len(keys)} cols incl. coords): "
                     f"{complete_all} / {n_overlap}  "
                     f"({100.0 * complete_all / n_overlap:.2f}%)")
        lines.append(f"  HSC instrument cols only ({len(hsc_keys)} cols): "
                     f"{complete_hsc} / {n_overlap}  "
                     f"({100.0 * complete_hsc / n_overlap:.2f}%)")
        lines.append(f"  Legacy instrument cols only ({len(leg_keys)} cols): "
                     f"{complete_leg} / {n_overlap}  "
                     f"({100.0 * complete_leg / n_overlap:.2f}%)")
        lines.append("")
        lines.append("Per-column NaN/Inf counts (biggest offenders first):")
        for k, c in nan_counts:
            if c == 0:
                continue
            lines.append(f"  {c:8d}  {k}")
        if all(c == 0 for _, c in nan_counts):
            lines.append("  (none — every input column is fully observed)")
        report = "\n".join(lines) + "\n"

    report_path = args.out_dir / "baseline_complete_case_report.txt"
    report_path.write_text(report)
    print(f"Wrote {report_path}")
    print()
    print(report)

    h5.close()


if __name__ == "__main__":
    main()
