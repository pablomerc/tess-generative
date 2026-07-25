#!/usr/bin/env python3
"""Condition × survey summary tables + paired deltas for the shuffle ablation.

Reads results/shuffle_per_anchor.csv and writes:
  - results/tables/summary_mean_sem.csv
  - results/tables/summary_mean_sem.md
  - results/tables/paired_deltas.csv
  - results/tables/sigma_tracking.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

ABL_DIR = Path(__file__).resolve().parent
CONDITIONS = ("C0", "C1", "C2", "C3")
CONDITION_NAMES = {
    "C0": "intact",
    "C1": "shuffle-phy",
    "C2": "shuffle-ins",
    "C3": "shuffle-both",
}
METRICS = (
    "mse_own",
    "mse_donor",
    "mse_own_32",
    "mse_donor_32",
    "sigma_corner",
    "sigma_adjdiff",
    "sigma_mad",
    "high_k_power",
)


def _sem(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return float("nan")
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(var / n)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _f(row: Dict[str, str], key: str) -> float:
    return float(row[key])


def aggregate_mean_sem(
    rows: List[Dict[str, str]],
    *,
    m_only: Optional[int] = 0,
) -> List[Dict[str, object]]:
    buckets: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(
        lambda: {m: [] for m in METRICS}
    )
    for r in rows:
        if m_only is not None and int(r["m"]) != m_only:
            continue
        key = (r["condition"], r["survey"])
        for m in METRICS:
            buckets[key][m].append(_f(r, m))

    out: List[Dict[str, object]] = []
    for cond in CONDITIONS:
        for survey in ("hsc", "legacy"):
            d = buckets.get((cond, survey))
            if not d or not d["mse_own"]:
                continue
            row: Dict[str, object] = {
                "condition": cond,
                "condition_name": CONDITION_NAMES[cond],
                "survey": survey,
                "n": len(d["mse_own"]),
                "m_filter": "" if m_only is None else m_only,
            }
            for m in METRICS:
                vals = d[m]
                row[f"{m}_mean"] = sum(vals) / len(vals)
                row[f"{m}_sem"] = _sem(vals)
            out.append(row)
    return out


def paired_deltas(
    rows: List[Dict[str, str]],
    *,
    m_only: int = 0,
    baselines: Sequence[str] = ("C1", "C2", "C3"),
) -> List[Dict[str, object]]:
    """Per-anchor deltas vs C0, then mean±sem by survey."""
    # index: (anchor_id, condition) -> row (m filter)
    by_key: Dict[Tuple[int, str], Dict[str, str]] = {}
    for r in rows:
        if int(r["m"]) != m_only:
            continue
        by_key[(int(r["anchor_id"]), r["condition"])] = r

    buckets: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(
        lambda: {m: [] for m in METRICS}
    )
    for (aid, cond), r in by_key.items():
        if cond not in baselines:
            continue
        base = by_key.get((aid, "C0"))
        if base is None:
            continue
        key = (cond, r["survey"])
        for m in METRICS:
            buckets[key][m].append(_f(r, m) - _f(base, m))

    out: List[Dict[str, object]] = []
    for cond in baselines:
        for survey in ("hsc", "legacy"):
            d = buckets.get((cond, survey))
            if not d or not d["mse_own"]:
                continue
            row: Dict[str, object] = {
                "delta": f"{cond}-C0",
                "condition": cond,
                "survey": survey,
                "n": len(d["mse_own"]),
                "m": m_only,
            }
            for m in METRICS:
                vals = d[m]
                row[f"d_{m}_mean"] = sum(vals) / len(vals)
                row[f"d_{m}_sem"] = _sem(vals)
            out.append(row)
    return out


def sigma_tracking(
    rows: List[Dict[str, str]],
    *,
    m_only: int = 0,
    sigma_key: str = "sigma_mad",
) -> List[Dict[str, object]]:
    """Pearson r of σ_gen vs σ_own and σ_donor, by condition × survey."""
    buckets: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(
        lambda: {"gen": [], "own": [], "donor": []}
    )
    own_col = f"{sigma_key}_own"
    donor_col = f"{sigma_key}_donor"
    for r in rows:
        if int(r["m"]) != m_only:
            continue
        key = (r["condition"], r["survey"])
        buckets[key]["gen"].append(_f(r, sigma_key))
        buckets[key]["own"].append(_f(r, own_col))
        buckets[key]["donor"].append(_f(r, donor_col))

    out: List[Dict[str, object]] = []
    for cond in CONDITIONS:
        for survey in ("hsc", "legacy"):
            d = buckets.get((cond, survey))
            if not d or len(d["gen"]) < 3:
                continue
            g = np.asarray(d["gen"], dtype=np.float64)
            o = np.asarray(d["own"], dtype=np.float64)
            don = np.asarray(d["donor"], dtype=np.float64)
            out.append(
                {
                    "condition": cond,
                    "condition_name": CONDITION_NAMES[cond],
                    "survey": survey,
                    "sigma": sigma_key,
                    "n": len(g),
                    "r_gen_vs_own": float(np.corrcoef(g, o)[0, 1]),
                    "r_gen_vs_donor": float(np.corrcoef(g, don)[0, 1]),
                    "m": m_only,
                }
            )
    return out


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _summary_md(rows: List[Dict[str, object]]) -> str:
    lines = [
        "# Shuffle ablation — mean ± SEM (m=0)",
        "",
        "| condition | survey | n | MSE_own | MSE_donor | σ_mad | high-k |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['condition']} ({r['condition_name']}) | {r['survey']} | {r['n']} | "
            f"{r['mse_own_mean']:.6f}±{r['mse_own_sem']:.6f} | "
            f"{r['mse_donor_mean']:.6f}±{r['mse_donor_sem']:.6f} | "
            f"{r['sigma_mad_mean']:.4f}±{r['sigma_mad_sem']:.4f} | "
            f"{r['high_k_power_mean']:.3g}±{r['high_k_power_sem']:.3g} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=ABL_DIR / "results" / "shuffle_per_anchor.csv")
    p.add_argument("--out-dir", type=Path, default=ABL_DIR / "results" / "tables")
    p.add_argument("--m", type=int, default=0, help="Which posterior sample index to tabulate")
    args = p.parse_args(argv)

    rows = _read_csv(args.csv)
    summary = aggregate_mean_sem(rows, m_only=args.m)
    deltas = paired_deltas(rows, m_only=args.m)
    tracking = []
    for sk in ("sigma_corner", "sigma_adjdiff", "sigma_mad", "high_k_power"):
        tracking.extend(sigma_tracking(rows, m_only=args.m, sigma_key=sk))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "summary_mean_sem.csv", summary)
    _write_csv(args.out_dir / "paired_deltas.csv", deltas)
    _write_csv(args.out_dir / "sigma_tracking.csv", tracking)
    md = _summary_md(summary)
    (args.out_dir / "summary_mean_sem.md").write_text(md)
    print(md)
    print(f"[tables] wrote under {args.out_dir}")


if __name__ == "__main__":
    main()
