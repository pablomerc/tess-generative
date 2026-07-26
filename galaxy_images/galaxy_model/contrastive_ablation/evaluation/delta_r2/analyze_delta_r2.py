#!/usr/bin/env python3
"""Paired fold and object-bootstrap analysis for delta-R2 model pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

VARIANTS = ("hsc_e1", "legacy_e1", "combined_e1")


def _read_scores(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"target", "latent_variant", "fold", "r2", "split_sha256", "finite_mask_sha256"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing score columns {sorted(missing)}")
    return frame


def _read_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"row_index", "target", "latent_variant", "fold", "y_true", "y_pred"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing prediction columns {sorted(missing)}")
    if frame.duplicated(["target", "latent_variant", "row_index"]).any():
        raise ValueError(f"{path}: duplicate OOF prediction keys")
    return frame


def _check_pair(ours: pd.DataFrame, contrastive: pd.DataFrame, name: str) -> None:
    keys = ["target", "latent_variant", "fold"]
    left = ours[keys + ["split_sha256", "finite_mask_sha256"]]
    right = contrastive[keys + ["split_sha256", "finite_mask_sha256"]]
    check = left.merge(right, on=keys, how="outer", suffixes=("_ours", "_contrastive"), indicator=True)
    if not (check["_merge"] == "both").all():
        raise ValueError(f"{name}: score grids differ between models")
    for column in ("split_sha256", "finite_mask_sha256"):
        if not (check[f"{column}_ours"] == check[f"{column}_contrastive"]).all():
            raise ValueError(f"{name}: {column} differs between models")


def _quantities(r2: dict[str, float], prefix: str) -> dict[str, float]:
    survey = r2["hsc_e1"] - r2["legacy_e1"]
    combined_gain = r2["combined_e1"] - max(r2["hsc_e1"], r2["legacy_e1"])
    return {
        f"r2_hsc_{prefix}": r2["hsc_e1"],
        f"r2_legacy_{prefix}": r2["legacy_e1"],
        f"r2_combined_{prefix}": r2["combined_e1"],
        f"delta_survey_{prefix}": survey,
        f"combined_gain_{prefix}": combined_gain,
    }


def fold_results(name: str, ours: pd.DataFrame, contrastive: pd.DataFrame) -> pd.DataFrame:
    rows = []
    targets = sorted(set(ours.target) & set(contrastive.target))
    for target in targets:
        for fold in sorted(set(ours.loc[ours.target == target, "fold"])):
            values = {}
            for label, frame in (("ours", ours), ("contrastive", contrastive)):
                subset = frame[(frame.target == target) & (frame.fold == fold)]
                mapping = dict(zip(subset.latent_variant, subset.r2))
                if set(mapping) != set(VARIANTS):
                    raise ValueError(f"{name}/{target}/fold {fold}: incomplete variant grid")
                values.update(_quantities(mapping, label))
            values["interaction"] = values["delta_survey_ours"] - values["delta_survey_contrastive"]
            values["combined_gain_difference"] = (
                values["combined_gain_ours"] - values["combined_gain_contrastive"]
            )
            rows.append({"pair": name, "target": target, "fold": int(fold), **values})
    return pd.DataFrame(rows)


def summarize_folds(folds: pd.DataFrame) -> pd.DataFrame:
    id_columns = {"pair", "target", "fold"}
    value_columns = [column for column in folds.columns if column not in id_columns]
    rows = []
    for (pair, target), group in folds.groupby(["pair", "target"], sort=True):
        for quantity in value_columns:
            values = group[quantity].to_numpy(float)
            rows.append({
                "pair": pair, "target": target, "quantity": quantity,
                "mean": float(values.mean()), "std": float(values.std(ddof=1)),
                "se": float(values.std(ddof=1) / np.sqrt(len(values))), "n_folds": len(values),
            })
    return pd.DataFrame(rows)


def _prediction_matrix(frame: pd.DataFrame, target: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subset = frame[frame.target == target]
    truth = subset.pivot(index="row_index", columns="latent_variant", values="y_true")
    pred = subset.pivot(index="row_index", columns="latent_variant", values="y_pred")
    if set(pred.columns) != set(VARIANTS) or pred.isna().any().any():
        raise ValueError(f"{target}: incomplete OOF prediction matrix")
    if not np.allclose(truth.to_numpy(), truth.iloc[:, [0]].to_numpy(), rtol=1e-6, atol=1e-7):
        raise ValueError(f"{target}: y_true differs across variants")
    return truth.index.to_numpy(), truth.iloc[:, 0].to_numpy(float), pred.loc[:, VARIANTS].to_numpy(float)


def _bootstrap_quantities(y: np.ndarray, ours: np.ndarray, contrastive: np.ndarray, rows: np.ndarray) -> dict:
    ours_r2 = {variant: float(r2_score(y[rows], ours[rows, i])) for i, variant in enumerate(VARIANTS)}
    con_r2 = {
        variant: float(r2_score(y[rows], contrastive[rows, i])) for i, variant in enumerate(VARIANTS)
    }
    values = {**_quantities(ours_r2, "ours"), **_quantities(con_r2, "contrastive")}
    values["interaction"] = values["delta_survey_ours"] - values["delta_survey_contrastive"]
    values["combined_gain_difference"] = values["combined_gain_ours"] - values["combined_gain_contrastive"]
    return values


def bootstrap_pair(
    name: str,
    ours: pd.DataFrame,
    contrastive: pd.DataFrame,
    *,
    samples: int,
    seed: int,
    ci: float,
) -> pd.DataFrame:
    rows = []
    targets = sorted(set(ours.target) & set(contrastive.target))
    for target_index, target in enumerate(targets):
        ours_index, y, ours_pred = _prediction_matrix(ours, target)
        con_index, con_y, con_pred = _prediction_matrix(contrastive, target)
        if not np.array_equal(ours_index, con_index):
            raise ValueError(f"{name}/{target}: model OOF row_index sets differ")
        if not np.allclose(y, con_y, rtol=1e-6, atol=1e-7):
            raise ValueError(f"{name}/{target}: model y_true values differ")
        n = len(y)
        point = _bootstrap_quantities(y, ours_pred, con_pred, np.arange(n))
        rng = np.random.default_rng(seed + target_index * 1009)
        draws = {quantity: np.empty(samples) for quantity in point}
        for draw in range(samples):
            sampled = rng.integers(0, n, size=n)
            result = _bootstrap_quantities(y, ours_pred, con_pred, sampled)
            for quantity, value in result.items():
                draws[quantity][draw] = value
        alpha = (100.0 - ci) / 2.0
        for quantity, estimate in point.items():
            low, high = np.percentile(draws[quantity], [alpha, 100.0 - alpha])
            rows.append({
                "pair": name, "target": target, "quantity": quantity,
                "estimate": estimate, "ci_low": float(low), "ci_high": float(high),
                "ci_percent": ci, "bootstrap_samples": samples, "n_objects": n,
            })
    return pd.DataFrame(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair", action="append", nargs=5, metavar=("NAME", "OURS_SCORES", "OURS_PREDS", "CON_SCORES", "CON_PREDS"),
        required=True, help="Repeat for primary, secondary, or additional model pairs.",
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260725)
    parser.add_argument("--ci", type=float, default=95.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.bootstrap_samples < 1 or not 0 < args.ci < 100:
        raise SystemExit("bootstrap samples must be positive and CI must be between 0 and 100")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_folds, all_bootstrap = [], []
    input_hashes = {}
    for pair_index, values in enumerate(args.pair):
        name, ours_score_path, ours_pred_path, con_score_path, con_pred_path = values
        paths = [Path(path) for path in values[1:]]
        ours_scores, con_scores = _read_scores(paths[0]), _read_scores(paths[2])
        ours_preds, con_preds = _read_predictions(paths[1]), _read_predictions(paths[3])
        _check_pair(ours_scores, con_scores, name)
        all_folds.append(fold_results(name, ours_scores, con_scores))
        all_bootstrap.append(bootstrap_pair(
            name, ours_preds, con_preds, samples=args.bootstrap_samples,
            seed=args.bootstrap_seed + pair_index * 100_003, ci=args.ci,
        ))
        for path in paths:
            input_hashes[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()

    folds = pd.concat(all_folds, ignore_index=True)
    summaries = summarize_folds(folds)
    bootstrap = pd.concat(all_bootstrap, ignore_index=True)
    outputs = {
        "paired_fold_values.csv": folds,
        "paired_fold_summary.csv": summaries,
        "paired_object_bootstrap.csv": bootstrap,
    }
    output_hashes = {}
    for filename, frame in outputs.items():
        path = args.out_dir / filename
        frame.to_csv(path, index=False)
        output_hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    provenance = {
        "command": sys.argv, "input_sha256": input_hashes, "output_sha256": output_hashes,
        "definitions": {
            "delta_survey": "R2(hsc_e1) - R2(legacy_e1)",
            "interaction": "delta_survey(ours) - delta_survey(contrastive)",
            "combined_gain": "R2(combined_e1) - max(R2(hsc_e1), R2(legacy_e1))",
            "bootstrap": "paired object resampling shared across both models and all three variants",
        },
    }
    (args.out_dir / "analysis_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(outputs)} analysis CSVs and analysis_provenance.json to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
