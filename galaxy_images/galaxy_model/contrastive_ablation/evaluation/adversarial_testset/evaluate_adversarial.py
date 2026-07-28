#!/usr/bin/env python3
"""Evaluate out-of-fold predictions on frozen quality strata and controls."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import normalized_id, require_file, write_json

AXES = ("resolution", "snr", "joint")
TRUE_ALIASES = ("y_true", "true", "target_value", "label", "ground_truth")
PRED_ALIASES = ("y_pred", "pred", "prediction", "predicted")
TARGET_ALIASES = ("target", "property", "task", "label_name")
ID_ALIASES = {
    "hsc_object_id": ("hsc_object_id", "object_id_hsc"),
    "legacy_object_id": ("legacy_object_id", "object_id_legacy"),
}


def _first(columns, aliases):
    lower = {str(c).lower(): c for c in columns}
    hits = [lower[a.lower()] for a in aliases if a.lower() in lower]
    return hits[0] if len(set(hits)) == 1 else None


def parse_prediction_spec(spec: str) -> tuple[str, str, Path]:
    if "=" not in spec or ":" not in spec.split("=", 1)[0]:
        raise ValueError(
            f"invalid --pred {spec!r}; expected MODEL:VIEW=/path/to/oof.csv"
        )
    label, raw_path = spec.split("=", 1)
    model, view = label.split(":", 1)
    if not model or not view:
        raise ValueError(f"empty model/view in --pred {spec!r}")
    return model, view, Path(raw_path)


def load_prediction(spec: str) -> pd.DataFrame:
    model, view, path = parse_prediction_spec(spec)
    require_file(path, f"OOF predictions for {model}:{view}")
    df = pd.read_csv(path)
    if "latent_variant" in df:
        expected = {
            "hsc": "hsc_e1",
            "legacy": "legacy_e1",
            "combined": "combined_e1",
        }.get(view, view)
        df = df[df["latent_variant"].astype(str) == expected].copy()
        if df.empty:
            raise ValueError(
                f"{path} has no latent_variant={expected!r} rows for view {view!r}"
            )
    true_col, pred_col = _first(df.columns, TRUE_ALIASES), _first(df.columns, PRED_ALIASES)
    target_col = _first(df.columns, TARGET_ALIASES)
    resolved_ids = []
    for kind, aliases in ID_ALIASES.items():
        found = _first(df.columns, aliases)
        if found:
            resolved_ids.append((kind, found))
    # delta_r2/oof_predictions.csv deliberately carries both IDs. Prefer HSC
    # as the stable pair key; either key still validates against the manifest.
    id_kind, id_col = resolved_ids[0] if resolved_ids else (None, None)
    missing = [
        name
        for name, value in (("truth", true_col), ("prediction", pred_col), ("object ID", id_col))
        if value is None
    ]
    if missing:
        raise ValueError(
            f"{path} cannot resolve {missing}; columns are {list(df.columns)}"
        )
    out = pd.DataFrame(
        {
            "pair_id_kind": id_kind,
            "pair_id": normalized_id(df[id_col]).to_numpy(),
            "target": df[target_col].astype(str) if target_col else "target",
            "y_true": pd.to_numeric(df[true_col], errors="coerce"),
            "y_pred": pd.to_numeric(df[pred_col], errors="coerce"),
            "model": model,
            "view": view,
            "source_csv": str(path),
        }
    )
    out = out[np.isfinite(out.y_true) & np.isfinite(out.y_pred)].copy()
    if out.empty:
        raise ValueError(f"{path} has no finite truth/prediction pairs")
    dup = out.duplicated(["pair_id", "target"]).sum()
    if dup:
        raise ValueError(f"{path} has {dup} duplicate (object,target) OOF rows")
    return out


def load_manifest(path: Path) -> pd.DataFrame:
    require_file(path, "frozen manifest")
    df = pd.read_csv(path, dtype={"hsc_object_id": str, "legacy_object_id": str})
    needed = {
        "hsc_object_id",
        "legacy_object_id",
        "metric_version",
        *[f"{a}_top_n" for a in AXES],
        *[f"{a}_quantile_bin" for a in AXES],
    }
    missing = needed - set(df)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")
    versions = df.metric_version.dropna().unique()
    if len(versions) != 1:
        raise ValueError(f"manifest must contain one metric_version, found {versions}")
    for col in ("hsc_object_id", "legacy_object_id"):
        df[col] = normalized_id(df[col]).to_numpy()
    for axis in AXES:
        col = f"{axis}_top_n"
        if df[col].dtype != bool:
            df[col] = df[col].astype(str).str.lower().map({"true": True, "false": False})
        if df[col].isna().any():
            raise ValueError(f"manifest has invalid boolean values in {col}")
    if df[["hsc_object_id", "legacy_object_id"]].duplicated().any():
        raise ValueError("manifest has duplicate paired-object keys")
    return df


def attach_manifest(pred: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    kinds = pred.pair_id_kind.unique()
    if len(kinds) != 1:
        raise ValueError("a prediction input resolved to multiple ID kinds")
    key = str(kinds[0])
    merged = pred.merge(
        manifest,
        left_on="pair_id",
        right_on=key,
        how="inner",
        validate="many_to_one",
    )
    if merged.empty:
        raise ValueError(
            f"no {key} values from predictions matched the frozen manifest"
        )
    coverage = len(merged) / len(pred)
    if coverage < 0.95:
        raise ValueError(
            f"only {coverage:.1%} of OOF rows matched the manifest; likely wrong ID encoding/sample"
        )
    return merged


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    valid = np.isfinite(y) & np.isfinite(p)
    y, p = y[valid], p[valid]
    if len(y) < 2:
        return {"n": len(y), "r2": np.nan, "nrmse": np.nan, "nmae": np.nan}
    residual = p - y
    scale = np.std(y)
    sse = np.sum(residual * residual)
    sst = np.sum((y - np.mean(y)) ** 2)
    return {
        "n": len(y),
        "r2": 1.0 - sse / sst if sst > 0 else np.nan,
        "nrmse": np.sqrt(np.mean(residual * residual)) / scale if scale > 0 else np.nan,
        "nmae": np.mean(np.abs(residual)) / scale if scale > 0 else np.nan,
    }


def make_matched_controls(
    manifest: pd.DataFrame, axis: str, seed: int
) -> tuple[pd.Series, dict]:
    """Match top-N objects on the other two quality-axis quantile bins."""
    top_col = f"{axis}_top_n"
    selected = manifest[manifest[top_col]].copy()
    pool = manifest[~manifest[top_col]].copy()
    other = [a for a in AXES if a != axis]
    rng = np.random.default_rng(seed)
    chosen = []
    exact = 0
    used = set()
    for idx, row in selected.iterrows():
        candidates = pool[~pool.index.isin(used)]
        exact_pool = candidates
        for alt in other:
            exact_pool = exact_pool[
                exact_pool[f"{alt}_quantile_bin"] == row[f"{alt}_quantile_bin"]
            ]
        if len(exact_pool):
            pick_pool, exact = exact_pool, exact + 1
        else:
            # Graceful fallback: match whichever alternate axis has candidates.
            one_axis = []
            for alt in other:
                hit = candidates[
                    candidates[f"{alt}_quantile_bin"] == row[f"{alt}_quantile_bin"]
                ]
                if len(hit):
                    one_axis.append(hit)
            pick_pool = min(one_axis, key=len) if one_axis else candidates
        if not len(pick_pool):
            break
        pick = int(rng.choice(pick_pool.index.to_numpy()))
        used.add(pick)
        chosen.append(pick)
    mask = pd.Series(False, index=manifest.index)
    mask.loc[chosen] = True
    return mask, {
        "axis": axis,
        "requested": len(selected),
        "selected": len(chosen),
        "exact_two_axis_matches": exact,
        "without_replacement": True,
    }


def make_random_draw(manifest: pd.DataFrame, n: int, seed: int) -> tuple[pd.Series, dict]:
    """Uniform random sample of n paired objects from the WHOLE manifest.

    Deliberately unmatched and not top-N-excluded: this is the plain "n random
    galaxies" baseline, i.e. an unbiased same-size sample of the population. It
    answers a different question from make_matched_controls, which holds the other
    two quality axes fixed. Report both; the matched control is the causal contrast,
    this one is the "is n=200 of anything special?" sanity check.

    Axis-independent by construction, so the same draw is reused for every axis.
    """
    n = min(int(n), len(manifest))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(manifest.index.to_numpy(), size=n, replace=False)
    mask = pd.Series(False, index=manifest.index)
    mask.loc[chosen] = True
    return mask, {"requested": int(n), "selected": int(mask.sum()), "matched": False,
                  "drawn_from": "all objects", "without_replacement": True}


def summarize_one(group: pd.DataFrame, axis: str, stratum: str) -> dict:
    result = metrics(group.y_true.to_numpy(), group.y_pred.to_numpy())
    result.update(axis=axis, stratum=stratum)
    return result


def summaries(
    data: pd.DataFrame,
    control_ids: dict[str, set[str]],
    random_ids: set[str] | None = None,
) -> pd.DataFrame:
    rows = []
    keys = ["model", "view", "target"]
    for key, group in data.groupby(keys, sort=True):
        base = dict(zip(keys, key))
        for axis in AXES:
            selections = {
                "all": pd.Series(True, index=group.index),
                "top_n": group[f"{axis}_top_n"],
                "matched_control": group["pair_id"].isin(control_ids[axis]),
            }
            if random_ids:
                selections["random_n"] = group["pair_id"].isin(random_ids)
            bins = sorted(group[f"{axis}_quantile_bin"].dropna().unique())
            selections.update(
                {
                    f"quantile_{int(q)}": group[f"{axis}_quantile_bin"] == q
                    for q in bins
                }
            )
            for name, mask in selections.items():
                row = summarize_one(group[mask], axis, name)
                row.update(base)
                rows.append(row)
    return pd.DataFrame(rows)


def paired_losses(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (view, target), group in data.groupby(["view", "target"]):
        models = list(group.model.drop_duplicates())
        for a, b in itertools.combinations(models, 2):
            aa = group[group.model == a][["pair_id", "y_true", "y_pred"]]
            bb = group[group.model == b][["pair_id", "y_true", "y_pred"]]
            pair = aa.merge(bb, on="pair_id", suffixes=("_a", "_b"), validate="one_to_one")
            truth_diff = np.abs(pair.y_true_a - pair.y_true_b)
            if len(pair) and np.nanmax(truth_diff) > 1e-6:
                raise ValueError(f"truth mismatch between {a} and {b} for {view}/{target}")
            ea, eb = pair.y_pred_a - pair.y_true_a, pair.y_pred_b - pair.y_true_b
            pair["model_a"], pair["model_b"] = a, b
            pair["view"], pair["target"] = view, target
            pair["squared_loss_a"], pair["squared_loss_b"] = ea * ea, eb * eb
            pair["absolute_loss_a"], pair["absolute_loss_b"] = np.abs(ea), np.abs(eb)
            pair["delta_squared_loss_a_minus_b"] = pair.squared_loss_a - pair.squared_loss_b
            pair["delta_absolute_loss_a_minus_b"] = pair.absolute_loss_a - pair.absolute_loss_b
            rows.append(pair)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _bootstrap_seed(seed: int, *parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode()).digest()
    return (seed + int.from_bytes(digest[:4], "little")) % (2**32)


def paired_bootstrap(
    data: pd.DataFrame,
    control_ids: dict[str, set[str]],
    replicates: int,
    seed: int,
    random_ids: set[str] | None = None,
) -> pd.DataFrame:
    rows = []
    for (view, target), group in data.groupby(["view", "target"]):
        models = list(group.model.drop_duplicates())
        for a, b in itertools.combinations(models, 2):
            aa = group[group.model == a][["pair_id", "y_true", "y_pred"]]
            bb = group[group.model == b][["pair_id", "y_true", "y_pred"]]
            pair = aa.merge(bb, on="pair_id", suffixes=("_a", "_b"), validate="one_to_one")
            pair = pair.merge(
                group.drop_duplicates("pair_id")[
                    ["pair_id"]
                    + [f"{axis}_top_n" for axis in AXES]
                    + [f"{axis}_quantile_bin" for axis in AXES]
                ],
                on="pair_id",
                validate="one_to_one",
            )
            for axis in AXES:
                strata = {
                    "all": np.ones(len(pair), dtype=bool),
                    "top_n": pair[f"{axis}_top_n"].to_numpy(bool),
                    "matched_control": pair.pair_id.isin(control_ids[axis]).to_numpy(),
                }
                if random_ids:
                    strata["random_n"] = pair.pair_id.isin(random_ids).to_numpy()
                for q in sorted(pair[f"{axis}_quantile_bin"].dropna().unique()):
                    strata[f"quantile_{int(q)}"] = (
                        pair[f"{axis}_quantile_bin"] == q
                    ).to_numpy()
                for stratum, mask in strata.items():
                    sub = pair[mask]
                    if len(sub) < 3:
                        continue
                    rng = np.random.default_rng(
                        _bootstrap_seed(seed, view, target, a, b, axis, stratum)
                    )
                    draws = {metric: [] for metric in ("r2", "nrmse", "nmae")}
                    for _ in range(replicates):
                        idx = rng.integers(0, len(sub), len(sub))
                        sample = sub.iloc[idx]
                        ma = metrics(sample.y_true_a, sample.y_pred_a)
                        mb = metrics(sample.y_true_b, sample.y_pred_b)
                        for metric in draws:
                            draws[metric].append(ma[metric] - mb[metric])
                    point_a = metrics(sub.y_true_a, sub.y_pred_a)
                    point_b = metrics(sub.y_true_b, sub.y_pred_b)
                    for metric, values in draws.items():
                        values = np.asarray(values, float)
                        values = values[np.isfinite(values)]
                        rows.append(
                            {
                                "model_a": a,
                                "model_b": b,
                                "view": view,
                                "target": target,
                                "axis": axis,
                                "stratum": stratum,
                                "metric": metric,
                                "n": len(sub),
                                "a_minus_b": point_a[metric] - point_b[metric],
                                "ci_low": np.quantile(values, 0.025) if len(values) else np.nan,
                                "ci_high": np.quantile(values, 0.975) if len(values) else np.nan,
                                "bootstrap_replicates": replicates,
                            }
                        )
    return pd.DataFrame(rows)


def survey_interaction_bootstrap(
    data: pd.DataFrame,
    control_ids: dict[str, set[str]],
    replicates: int,
    seed: int,
    random_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Bootstrap the model-by-survey difference of differences.

    interaction = [metric(ours,HSC)-metric(ours,Legacy)]
                - [metric(contrastive,HSC)-metric(contrastive,Legacy)]
    """
    rows = []
    model_order = list(data.model.drop_duplicates())
    if len(model_order) != 2 or not {"hsc", "legacy"}.issubset(set(data.view)):
        return pd.DataFrame()
    model_a, model_b = model_order
    for target, group in data.groupby("target", sort=True):
        pieces = {}
        for model in (model_a, model_b):
            for view in ("hsc", "legacy"):
                part = group[(group.model == model) & (group.view == view)][
                    ["pair_id", "y_true", "y_pred"]
                ].rename(
                    columns={
                        "y_true": f"y_{model}_{view}",
                        "y_pred": f"p_{model}_{view}",
                    }
                )
                pieces[(model, view)] = part
        merged = None
        for part in pieces.values():
            merged = part if merged is None else merged.merge(
                part, on="pair_id", how="inner", validate="one_to_one"
            )
        if merged is None or merged.empty:
            continue
        truth_cols = [column for column in merged if column.startswith("y_")]
        truth = merged[truth_cols].to_numpy(float)
        if np.nanmax(np.ptp(truth, axis=1)) > 1e-6:
            raise ValueError(f"truth mismatch across model/view inputs for {target}")
        strata_source = group.drop_duplicates("pair_id")[
            ["pair_id"]
            + [f"{axis}_top_n" for axis in AXES]
            + [f"{axis}_quantile_bin" for axis in AXES]
        ]
        merged = merged.merge(strata_source, on="pair_id", validate="one_to_one")

        def interaction(frame: pd.DataFrame, metric: str) -> float:
            values = {}
            for model in (model_a, model_b):
                for view in ("hsc", "legacy"):
                    values[(model, view)] = metrics(
                        frame[f"y_{model}_{view}"], frame[f"p_{model}_{view}"]
                    )[metric]
            return (
                values[(model_a, "hsc")]
                - values[(model_a, "legacy")]
                - values[(model_b, "hsc")]
                + values[(model_b, "legacy")]
            )

        for axis in AXES:
            strata = {
                "all": np.ones(len(merged), dtype=bool),
                "top_n": merged[f"{axis}_top_n"].to_numpy(bool),
                "matched_control": merged.pair_id.isin(control_ids[axis]).to_numpy(),
            }
            if random_ids:
                strata["random_n"] = merged.pair_id.isin(random_ids).to_numpy()
            for q in sorted(merged[f"{axis}_quantile_bin"].dropna().unique()):
                strata[f"quantile_{int(q)}"] = (
                    merged[f"{axis}_quantile_bin"] == q
                ).to_numpy()
            for stratum, mask in strata.items():
                sub = merged[mask]
                if len(sub) < 3:
                    continue
                rng = np.random.default_rng(
                    _bootstrap_seed(
                        seed, "survey_interaction", target, model_a, model_b, axis, stratum
                    )
                )
                for metric_name in ("r2", "nrmse", "nmae"):
                    estimate = interaction(sub, metric_name)
                    draws = []
                    for _ in range(replicates):
                        sample = sub.iloc[rng.integers(0, len(sub), len(sub))]
                        value = interaction(sample, metric_name)
                        if np.isfinite(value):
                            draws.append(value)
                    rows.append(
                        {
                            "model_a": model_a,
                            "model_b": model_b,
                            "target": target,
                            "axis": axis,
                            "stratum": stratum,
                            "metric": metric_name,
                            "n": len(sub),
                            "interaction": estimate,
                            "ci_low": np.quantile(draws, 0.025) if draws else np.nan,
                            "ci_high": np.quantile(draws, 0.975) if draws else np.nan,
                            "bootstrap_replicates": replicates,
                        }
                    )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument(
        "--pred",
        required=True,
        action="append",
        help="repeat MODEL:VIEW=/path/to/oof.csv; CSV needs object ID, truth, prediction",
    )
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--bootstrap-replicates", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.bootstrap_replicates < 1:
        raise ValueError("--bootstrap-replicates must be positive")
    manifest = load_manifest(args.manifest)
    loaded = [attach_manifest(load_prediction(spec), manifest) for spec in args.pred]
    data = pd.concat(loaded, ignore_index=True)
    duplicates = data.duplicated(["model", "view", "target", "pair_id"]).sum()
    if duplicates:
        raise ValueError(f"{duplicates} duplicate model/view/target/object OOF rows")

    controls, control_meta = {}, {}
    for i, axis in enumerate(AXES):
        mask, meta = make_matched_controls(manifest, axis, args.seed + i)
        controls[axis] = set(manifest.loc[mask, "hsc_object_id"]) | set(
            manifest.loc[mask, "legacy_object_id"]
        )
        control_meta[axis] = meta

    # Plain uniform "n random galaxies" baseline, same size as top-N, drawn once and
    # shared across axes. Complements (does not replace) the matched controls.
    random_n = int(manifest[[f"{axis}_top_n" for axis in AXES]].sum().max())
    random_mask, random_meta = make_random_draw(
        manifest, random_n, args.seed + len(AXES)
    )
    random_ids = set(manifest.loc[random_mask, "hsc_object_id"]) | set(
        manifest.loc[random_mask, "legacy_object_id"]
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = summaries(data, controls, random_ids)
    losses = paired_losses(data)
    bootstrap = paired_bootstrap(
        data, controls, args.bootstrap_replicates, args.seed, random_ids
    )
    interactions = survey_interaction_bootstrap(
        data, controls, args.bootstrap_replicates, args.seed, random_ids
    )
    summary.to_csv(args.out_dir / "stratified_metrics.csv", index=False)
    losses.to_csv(args.out_dir / "paired_object_losses.csv", index=False)
    bootstrap.to_csv(args.out_dir / "paired_bootstrap.csv", index=False)
    interactions.to_csv(args.out_dir / "survey_interaction_bootstrap.csv", index=False)
    manifest_version = str(manifest.metric_version.dropna().iloc[0])
    write_json(
        args.out_dir / "evaluation_metadata.json",
        {
            "manifest": str(args.manifest.resolve()),
            "metric_version": manifest_version,
            "prediction_specs": args.pred,
            "bootstrap_replicates": args.bootstrap_replicates,
            "seed": args.seed,
            "normalization": "nRMSE and nMAE divide by within-stratum std(y_true)",
            "paired_difference_convention": "model_a minus model_b",
            "survey_interaction_convention": (
                "[metric(model_a,HSC)-metric(model_a,Legacy)] - "
                "[metric(model_b,HSC)-metric(model_b,Legacy)]"
            ),
            "matched_controls": control_meta,
            "random_draw": random_meta,
            "strata": (
                "top_n = frozen quality top-N; matched_control = 1:1 outside top-N, "
                "matched on the other two axes' quantile bins; random_n = uniform "
                "same-size draw from all objects (unmatched); quantile_* = dose-response"
            ),
            "selection_uses_predictions": False,
        },
    )
    print(f"wrote {args.out_dir / 'stratified_metrics.csv'} ({len(summary)} rows)")
    print(f"wrote {args.out_dir / 'paired_object_losses.csv'} ({len(losses)} rows)")
    print(f"wrote {args.out_dir / 'paired_bootstrap.csv'} ({len(bootstrap)} rows)")
    print(
        f"wrote {args.out_dir / 'survey_interaction_bootstrap.csv'} "
        f"({len(interactions)} rows)"
    )


if __name__ == "__main__":
    main()
