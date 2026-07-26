#!/usr/bin/env python3
"""Freeze quality-only adversarial selections before inspecting model errors."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from common import dataframe_sha256, file_sha256, require_file, write_json

AXES = {
    "resolution": "resolution_gap",
    "snr": "snr_gap",
    "joint": "joint_score",
}
ID_COLUMNS = ["hsc_object_id", "legacy_object_id", "hsc_image_idx", "legacy_image_idx"]


def quantile_bin(values: pd.Series, n_bins: int) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    valid = x.notna()
    out = pd.Series(pd.NA, index=x.index, dtype="Int64")
    if valid.sum() < n_bins:
        return out
    # Rank first so tied catalog values still produce deterministic, balanced bins.
    ranked = x[valid].rank(method="first")
    out.loc[valid] = pd.qcut(ranked, q=n_bins, labels=False).astype(int).to_numpy()
    return out


def build_manifest(df: pd.DataFrame, top_n: int, n_bins: int) -> pd.DataFrame:
    required = set(ID_COLUMNS) | set(AXES.values()) | {"metric_version"}
    missing = required - set(df)
    if missing:
        raise ValueError(f"quality table missing columns: {sorted(missing)}")
    if df[ID_COLUMNS].duplicated().any():
        raise ValueError("quality table contains duplicate paired-object keys")
    versions = df["metric_version"].dropna().unique()
    if len(versions) != 1:
        raise ValueError(f"expected exactly one metric_version, found {versions}")

    keep = ID_COLUMNS + [
        c
        for c in (
            "ra",
            "dec",
            "sep_arcsec",
            "resolution_gap",
            "snr_gap",
            "joint_score",
            "coverage_resolution",
            "coverage_snr",
            "coverage_joint",
            "metric_version",
        )
        if c in df
    ]
    out = df[keep].copy()
    for axis, column in AXES.items():
        finite = pd.to_numeric(out[column], errors="coerce").notna()
        rank = out.loc[finite, column].rank(ascending=False, method="first").astype(int)
        out[f"{axis}_rank"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
        out.loc[finite, f"{axis}_rank"] = rank
        out[f"{axis}_top_n"] = finite & (out[f"{axis}_rank"] <= top_n)
        out[f"{axis}_quantile_bin"] = quantile_bin(out[column], n_bins)
    return out.sort_values("joint_rank", na_position="last").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quality-table", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--metadata", type=Path)
    ap.add_argument("--top-n", type=int, default=200)
    ap.add_argument("--n-quantile-bins", type=int, default=5)
    args = ap.parse_args()
    require_file(args.quality_table, "quality table")
    if args.out.exists():
        raise FileExistsError(
            f"refusing to overwrite frozen manifest: {args.out}; choose a new path"
        )
    if args.top_n <= 0 or args.n_quantile_bins < 2:
        raise ValueError("--top-n must be positive and --n-quantile-bins must be >=2")

    source = pd.read_parquet(args.quality_table)
    manifest = build_manifest(source, args.top_n, args.n_quantile_bins)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.out, index=False)
    versions = manifest["metric_version"].dropna().unique()
    metadata_path = args.metadata or args.out.with_suffix(".json")
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "quality_table": str(args.quality_table.resolve()),
        "quality_table_sha256": file_sha256(args.quality_table),
        "manifest_selection_sha256": dataframe_sha256(
            manifest,
            ID_COLUMNS
            + [f"{axis}_{suffix}" for axis in AXES for suffix in ("rank", "top_n", "quantile_bin")],
        ),
        "metric_version": str(versions[0]),
        "top_n": args.top_n,
        "n_quantile_bins": args.n_quantile_bins,
        "rows": len(manifest),
        "selection_uses_predictions": False,
        "axes": AXES,
    }
    write_json(metadata_path, metadata)
    print(f"froze {args.out} ({len(manifest)} paired objects)")
    print(f"wrote {metadata_path}")


if __name__ == "__main__":
    main()
