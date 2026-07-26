#!/usr/bin/env python3
"""Five-fold survey-view physics probes on prepare_combined HDF5 embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

VARIANTS = ("hsc_e1", "legacy_e1", "combined_e1")
DEFAULT_TARGETS = (
    "desi_z",
    "provabgs_logmstar",
    "provabgs_tage_mw",
    "provabgs_z_mw",
    "ssfr",
    "legacy_shape_e1",
    "legacy_shape_e2",
    "hsc_shape_e1_i",
    "hsc_shape_e2_i",
)
VERSION = "1.0"


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def hash_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def make_folds(n_rows: int, n_folds: int, seed: int) -> np.ndarray:
    """Assign folds from absolute row positions, before applying finite masks."""
    if n_folds < 2 or n_rows < n_folds:
        raise ValueError(f"need n_rows >= n_folds >= 2; got {n_rows}, {n_folds}")
    order = np.random.default_rng(seed).permutation(n_rows)
    assignment = np.empty(n_rows, dtype=np.int16)
    for fold, rows in enumerate(np.array_split(order, n_folds)):
        assignment[rows] = fold
    return assignment


def _label(group: h5py.Group, *names: str) -> np.ndarray | None:
    for name in names:
        key = f"labels/{name}"
        if key in group:
            return np.asarray(group[key][:], dtype=np.float64)
    return None


def resolve_target(group: h5py.Group, target: str) -> np.ndarray | None:
    if target == "ssfr":
        sfr = _label(group, "hsc_provabgs_avg_sfr", "legacy_provabgs_avg_sfr")
        mass = _label(group, "hsc_provabgs_logmstar", "legacy_provabgs_logmstar")
        if sfr is None or mass is None:
            return None
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            return np.where(sfr > 0, np.log10(sfr) - mass, np.nan)
    if target in {"legacy_shape_e1", "legacy_shape_e2"}:
        return _label(group, f"legacy_SHAPE_{target[-2:].upper()}")
    if target in {"hsc_shape_e1_i", "hsc_shape_e2_i"}:
        s11 = _label(group, "hsc_i_sdssshape_shape11")
        s22 = _label(group, "hsc_i_sdssshape_shape22")
        if s11 is None or s22 is None:
            return None
        trace = s11 + s22
        with np.errstate(divide="ignore", invalid="ignore"):
            if target == "hsc_shape_e1_i":
                return np.where(trace > 0, (s11 - s22) / trace, np.nan)
            s12 = _label(group, "hsc_i_sdssshape_shape12")
            return None if s12 is None else np.where(trace > 0, 2 * s12 / trace, np.nan)
    return _label(group, f"hsc_{target}", f"legacy_{target}")


class AionMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _predict(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    output = []
    loader = DataLoader(torch.from_numpy(x.astype(np.float32)), batch_size=batch_size)
    with torch.no_grad():
        for batch in loader:
            output.append(model(batch.to(device)).cpu().numpy())
    return np.concatenate(output)


def fit_predict(
    x: np.ndarray,
    y: np.ndarray,
    train_rows: np.ndarray,
    test_rows: np.ndarray,
    *,
    seed: int,
    hidden: int,
    max_epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device_name: str,
) -> tuple[np.ndarray, dict]:
    """Train on one outer fold, using a deterministic inner validation subset."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    inner = np.random.default_rng(seed).permutation(train_rows)
    n_val = max(1, int(round(0.1 * len(inner))))
    if len(inner) - n_val < 2:
        raise ValueError("too few training rows for an inner validation split")
    val_rows, fit_rows = inner[:n_val], inner[n_val:]

    # Scaling may use the complete outer training fold (including inner validation),
    # but never the held-out outer test fold.
    y_mean = float(np.mean(y[train_rows]))
    y_scale = float(np.std(y[train_rows]))
    if not np.isfinite(y_scale) or y_scale < 1e-8:
        raise ValueError("target is constant on the outer training fold")
    y_scaled = ((y - y_mean) / y_scale).astype(np.float32)
    x32 = x.astype(np.float32, copy=False)

    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x32[fit_rows]), torch.from_numpy(y_scaled[fit_rows])),
        batch_size=min(batch_size, len(fit_rows)),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    device = torch.device("cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    model = AionMLP(x.shape[1], hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    loss_fn = nn.SmoothL1Loss(beta=1.0)
    best_state, best_loss, stale, best_epoch = None, float("inf"), 0, -1

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb.to(device)), yb.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        val_pred = _predict(model, x32[val_rows], device, batch_size)
        residual = np.abs(val_pred - y_scaled[val_rows])
        # Match nn.SmoothL1Loss(beta=1) used by the published AION probe.
        val_loss = float(np.mean(np.where(residual < 1.0, 0.5 * residual**2, residual - 0.5)))
        if val_loss < best_loss - 1e-7:
            best_loss, stale, best_epoch = val_loss, 0, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("training produced no finite validation result")
    model.load_state_dict(best_state)
    prediction = _predict(model, x32[test_rows], device, batch_size) * y_scale + y_mean
    return prediction.astype(np.float64), {
        "best_epoch": best_epoch,
        "epochs_run": epoch + 1,
        "target_train_mean": y_mean,
        "target_train_scale": y_scale,
        "device": str(device),
    }


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    scale = float(np.std(y_true))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "normalized_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred)) / scale),
        "normalized_mae": float(mean_absolute_error(y_true, y_pred) / scale),
        "test_target_std": scale,
    }


def _id_columns(frame: pd.DataFrame, requested: str | None) -> list[str]:
    if requested:
        columns = [item.strip() for item in requested.split(",") if item.strip()]
    else:
        columns = [column for column in frame.columns if column.lower().endswith(("_id", "id"))]
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"ID columns absent from overlap parquet: {missing}")
    return columns


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", required=True, type=Path)
    parser.add_argument("--overlap-parquet", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    parser.add_argument("--id-columns", default=None, help="Comma-separated; default: all columns ending in id.")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--min-samples", type=int, default=50)
    parser.add_argument("--skip-input-file-hashes", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.n_folds != 5:
        raise SystemExit("--n-folds must be 5 for the locked delta-R2 protocol")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    overlap = pd.read_parquet(args.overlap_parquet).reset_index(drop=True)
    ids = _id_columns(overlap, args.id_columns)
    if overlap[ids].duplicated().any() if ids else False:
        raise ValueError("selected overlap ID tuple is not unique")
    n_rows = len(overlap)
    fold_assignment = make_folds(n_rows, args.n_folds, args.seed)
    targets = [item.strip() for item in args.targets.split(",") if item.strip()]
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    score_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    mask_hashes: dict[str, str] = {}

    with h5py.File(args.embeddings, "r") as handle:
        if "overlap" not in handle:
            raise ValueError("H5 has no overlap group")
        group = handle["overlap"]
        missing_variants = [name for name in VARIANTS if name not in group]
        if missing_variants:
            raise ValueError(f"H5 missing required embeddings: {missing_variants}")
        features = {}
        for variant in VARIANTS:
            values = np.asarray(group[variant][:])
            values = values.reshape(values.shape[0], -1)
            if len(values) != n_rows:
                raise ValueError(f"{variant}: H5 has {len(values)} rows, overlap has {n_rows}")
            if not np.isfinite(values).all():
                raise ValueError(f"{variant}: non-finite embeddings would make masks variant-dependent")
            features[variant] = values
        model_name = args.model_name or handle.attrs.get("checkpoint_name", args.embeddings.stem)
        if isinstance(model_name, bytes):
            model_name = model_name.decode()

        for target in targets:
            y = resolve_target(group, target)
            if y is None:
                print(f"warning: skipping unavailable target {target}", file=sys.stderr)
                continue
            if len(y) != n_rows:
                raise ValueError(f"{target}: label length {len(y)} != overlap length {n_rows}")
            finite = np.isfinite(y)
            mask_hashes[target] = hash_array(finite.astype(np.uint8))
            if finite.sum() < args.min_samples:
                print(f"warning: skipping {target}; only {finite.sum()} finite labels", file=sys.stderr)
                continue
            for fold in range(args.n_folds):
                test_rows = np.flatnonzero(finite & (fold_assignment == fold))
                train_rows = np.flatnonzero(finite & (fold_assignment != fold))
                if len(test_rows) < 2 or len(train_rows) < args.min_samples:
                    raise ValueError(f"{target} fold {fold}: insufficient train/test labels")
                for variant in VARIANTS:
                    # Same fold-level seed removes avoidable initialization and minibatch
                    # differences between the two equal-dimensional single-survey probes.
                    run_seed = args.seed + fold * 1009
                    pred, training = fit_predict(
                        features[variant], y, train_rows, test_rows,
                        seed=run_seed, hidden=args.hidden, max_epochs=args.max_epochs,
                        patience=args.patience, batch_size=args.batch_size,
                        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
                        device_name=args.device,
                    )
                    values = metrics(y[test_rows], pred)
                    common = {
                        "model": str(model_name), "target": target, "latent_variant": variant,
                        "fold": fold, "n_folds": args.n_folds, "seed": args.seed,
                        "n_train": len(train_rows), "n_test": len(test_rows),
                        "finite_mask_sha256": mask_hashes[target],
                        "split_sha256": hash_array(fold_assignment),
                        "config_sha256": config_hash,
                    }
                    score_rows.append({**common, **values, **training})
                    frame = overlap.loc[test_rows, ids].copy() if ids else pd.DataFrame(index=test_rows)
                    frame.insert(0, "row_index", test_rows)
                    frame["model"] = str(model_name)
                    frame["target"] = target
                    frame["latent_variant"] = variant
                    frame["fold"] = fold
                    frame["y_true"] = y[test_rows]
                    frame["y_pred"] = pred
                    frame["finite_mask_sha256"] = mask_hashes[target]
                    frame["split_sha256"] = hash_array(fold_assignment)
                    prediction_frames.append(frame)

    if not score_rows:
        raise RuntimeError("no targets were evaluated")
    scores = pd.DataFrame(score_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    score_path = args.out_dir / "fold_scores.csv"
    prediction_path = args.out_dir / "oof_predictions.csv"
    scores.to_csv(score_path, index=False)
    predictions.to_csv(prediction_path, index=False)

    # OOF coverage and cross-variant mask checks.
    for target in scores["target"].unique():
        subset = predictions[predictions.target == target]
        groups = subset.groupby("latent_variant")["row_index"].apply(set)
        if len(groups) != 3 or any(rows != groups.iloc[0] for rows in groups.iloc[1:]):
            raise AssertionError(f"{target}: OOF row coverage differs across variants")
        if subset.groupby(["latent_variant", "row_index"]).size().max() != 1:
            raise AssertionError(f"{target}: an object was predicted more than once per variant")

    provenance = {
        "protocol_version": VERSION,
        "created_unix": time.time(),
        "command": sys.argv,
        "model": str(scores["model"].iloc[0]),
        "python": sys.version,
        "torch": torch.__version__,
        "input_paths": {"embeddings": str(args.embeddings.resolve()), "overlap": str(args.overlap_parquet.resolve())},
        "input_sha256": {} if args.skip_input_file_hashes else {
            "embeddings": sha256_file(args.embeddings), "overlap": sha256_file(args.overlap_parquet)
        },
        "config": config,
        "config_sha256": config_hash,
        "split_sha256": hash_array(fold_assignment),
        "finite_mask_sha256": mask_hashes,
        "outputs_sha256": {"fold_scores.csv": sha256_file(score_path), "oof_predictions.csv": sha256_file(prediction_path)},
        "sanity_checks": {
            "n_overlap_rows": n_rows, "fold_sizes": np.bincount(fold_assignment).tolist(),
            "all_embeddings_finite": True, "oof_once_per_variant": True, "variant_masks_identical": True,
        },
    }
    (args.out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {score_path}, {prediction_path}, and provenance.json")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
