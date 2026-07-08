"""
Metadata-only baseline — predict physics targets from raw instrument metadata
on the `overlap` subset, with no learned image embeddings.

CLI:

    python predict_instrument_to_physics.py \
        --labels-source outputs/embeddings/base.h5 \
        --out           outputs/predictions/predict_instrument_baseline_v2.csv \
        [--device cuda] [--max-epochs 200] [--seeds 0,1,2,3,4] \
        [--preprocess {none,v2}] [--include-coords]

The labels-source H5 is any `prepare_combined.py` output — every checkpoint's
H5 carries the same `/overlap/labels/*` arrays, so the choice doesn't affect
results. We just need a place to read the labels from.

Two preprocessing modes:

    --preprocess none  → original behavior. Stack raw HSC+Legacy instrument
                         columns from HSC_INSTRUMENT_TARGETS / LEGACY_INSTRUMENT_TARGETS,
                         feed to MLP with only nan_to_num cleanup. Targets are
                         z-scored on the train fold by `_train_eval`.

    --preprocess v2   → audit-driven preprocessing (see audit_baseline_features.py
                        and the plan in /home/pablomer/.claude/plans/):
                        - drop HSC `*_variance_value` (70-80% NaN, removing them
                          gets us to 100% complete data with no imputation)
                        - drop `legacy_psf_fwhm_*` (numerically identical to
                          `legacy_PSFSIZE_*`)
                        - drop `legacy_ra/dec` (duplicate of hsc_ra/dec)
                        - log10 PSFDEPTH/GALDEPTH (heavy-tailed counts, not mags)
                        - log1p NOBS_* (Poisson-like)
                        - -2.5*log10 MW_TRANSMISSION_* → Galactic extinction in mag
                        - leave PSF FWHM, PSFSIZE, hsc_a_*, EBV, RA/DEC as-is
                        - z-score every column on the train fold
                        - for `provabgs_avg_sfr` and `provabgs_z_mw` (raw, heavy-
                          tailed): log10 internally for training; report R² in
                          original linear units (exp10 the predictions)

Output CSV — one row per (target, seed) — uses the same schema as
predict_combined.py plus a `seed` column so multi-seed runs can be merged:
    checkpoint, task_family, target, latent_variant, score, score_metric,
    n_train, n_test, seed
with fixed values:
    checkpoint     = "instrument-baseline" (or "instrument-baseline-v2")
    task_family    = "physics_provabgs"
    latent_variant = "combined_instrument"
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).parent))
from predict_combined import (
    BATCH_SIZE,
    CkptH5,
    HSC_INSTRUMENT_TARGETS,
    LEGACY_INSTRUMENT_TARGETS,
    LitProbe,
    MIN_LABELS,
    PHYSICS_TARGETS,
    TRAIN_FRAC,
    _filter_finite,
    _physics_label,
    _train_eval,
)

CHECKPOINT_TAG_V1 = "instrument-baseline"
CHECKPOINT_TAG_V2 = "instrument-baseline-v2"
LATENT_VARIANT = "combined_instrument"
TASK_FAMILY = "physics_provabgs"
SUBSET = "overlap"

# ---------------------------------------------------------------------------
# v2 preprocessing tables (derived from outputs/diagnostics/baseline_feature_stats.csv)
# ---------------------------------------------------------------------------

# Columns dropped from the v1 input set.
V2_DROP_HSC = {f"{b}_variance_value" for b in ("g", "r", "i", "z")}     # 70–80% NaN
V2_DROP_LEGACY = {f"psf_fwhm_{b}" for b in ("g", "r", "i", "z")}        # duplicate of PSFSIZE_*

# Per-column transforms applied before z-scoring. Keys are the prefixed H5 keys.
def _build_v2_transform_map() -> dict[str, str]:
    m: dict[str, str] = {}
    for b in ("G", "R", "I", "Z"):
        m[f"legacy_PSFDEPTH_{b}"] = "log10"
        m[f"legacy_GALDEPTH_{b}"] = "log10"
        m[f"legacy_NOBS_{b}"] = "log1p"
        m[f"legacy_MW_TRANSMISSION_{b}"] = "neg_log10"
    return m

V2_TRANSFORMS = _build_v2_transform_map()

# Targets that should be trained in log10-space and exponentiated back for R².
LOG_TARGETS = {"provabgs_avg_sfr", "provabgs_z_mw"}

# Coordinate columns we *can* add (HSC RA/DEC; legacy duplicates dropped).
HSC_COORD_COLS = ["hsc_ra", "hsc_dec"]


def _apply_transform(x: np.ndarray, kind: str) -> np.ndarray:
    if kind == "log10":
        return np.log10(np.maximum(x, 1e-30))
    if kind == "log1p":
        return np.log1p(np.maximum(x, 0.0))
    if kind == "neg_log10":
        return -2.5 * np.log10(np.maximum(x, 1e-30))
    raise ValueError(f"Unknown transform: {kind}")


# ---------------------------------------------------------------------------
# Feature matrix construction
# ---------------------------------------------------------------------------

def _v1_columns() -> list[str]:
    cols: list[str] = []
    cols += [f"hsc_{c}" for c in HSC_INSTRUMENT_TARGETS]
    cols += [f"legacy_{c}" for c in LEGACY_INSTRUMENT_TARGETS]
    return cols


def _v2_columns(include_coords: bool) -> list[str]:
    cols: list[str] = []
    cols += [f"hsc_{c}" for c in HSC_INSTRUMENT_TARGETS if c not in V2_DROP_HSC]
    cols += [f"legacy_{c}" for c in LEGACY_INSTRUMENT_TARGETS if c not in V2_DROP_LEGACY]
    if include_coords:
        cols += HSC_COORD_COLS
    return cols


def build_instrument_matrix(h5: CkptH5, columns: list[str]) -> tuple[np.ndarray, list[str]]:
    """Stack the requested label columns into a (N, F) feature matrix."""
    chunks: list[np.ndarray] = []
    used: list[str] = []
    missing: list[str] = []
    for key in columns:
        v = h5.label(SUBSET, key)
        if v is None:
            missing.append(key)
            continue
        chunks.append(np.asarray(v, dtype=np.float32).reshape(-1, 1))
        used.append(key)
    if missing:
        print(f"  warn: {len(missing)} instrument columns missing from H5: "
              f"{missing[:5]}{' ...' if len(missing) > 5 else ''}")
    if not chunks:
        raise RuntimeError(f"No instrument columns found under /{SUBSET}/labels/")
    X = np.concatenate(chunks, axis=1)
    return X, used


# ---------------------------------------------------------------------------
# v2 training loop — duplicates _train_eval to add per-feature standardization
# (fit on train fold) and optional log-target wrapping. Kept separate so
# predict_combined.py / latent probes stay byte-identical.
# ---------------------------------------------------------------------------

def _train_eval_baseline_v2(
    X_raw: np.ndarray,
    y: np.ndarray,
    columns: list[str],
    transforms: dict[str, str],
    target_log: bool,
    max_epochs: int,
    seed: int,
    device: str,
):
    """Train an MLP probe with v2 preprocessing fit on the train fold.

    Returns [(score, "r2", n_train, n_test)] in the same shape as _train_eval.
    """
    pl.seed_everything(seed, workers=True)

    rng = np.random.default_rng(seed)
    idx = np.arange(X_raw.shape[0])
    rng.shuffle(idx)
    n_tr = int(TRAIN_FRAC * len(idx))
    tr, va = idx[:n_tr], idx[n_tr:]

    # --- per-column transforms (deterministic, no statistics) -----------
    X = X_raw.astype(np.float64).copy()
    for j, col in enumerate(columns):
        if col in transforms:
            X[:, j] = _apply_transform(X[:, j], transforms[col])

    # --- per-column z-score using train-fold mean/std --------------------
    mu_x = X[tr].mean(axis=0)
    sd_x = X[tr].std(axis=0)
    sd_x = np.where(sd_x < 1e-8, 1.0, sd_x)
    X_norm = (X - mu_x) / sd_x

    # --- target standardization (in log space if target_log) -------------
    y_for_train = y.astype(np.float64).copy()
    if target_log:
        y_for_train = np.log10(np.maximum(y_for_train, 1e-30))
    mu_y = y_for_train[tr].mean()
    sd_y = y_for_train[tr].std()
    if sd_y < 1e-8:
        sd_y = 1.0
    y_norm = (y_for_train - mu_y) / sd_y

    X_t = torch.tensor(np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0),
                       dtype=torch.float32)
    y_t = torch.tensor(y_norm.reshape(-1, 1), dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_t[tr], y_t[tr]),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(TensorDataset(X_t[va], y_t[va]),
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    use_gpu = (device == "cuda" and torch.cuda.is_available())
    probe = LitProbe(in_dim=X_norm.shape[1], out_dim=1, mode="regression")
    ckpt = ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1, save_last=False)
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        callbacks=[ckpt, EarlyStopping(monitor="val/loss", mode="min", patience=5)],
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=50,
        logger=False,
    )
    trainer.fit(probe, train_loader, val_loader)
    best_path = ckpt.best_model_path
    best = LitProbe.load_from_checkpoint(best_path) if best_path else probe
    best.to("cuda" if use_gpu else "cpu")
    best.eval()

    preds = []
    with torch.no_grad():
        for xb, _yb in val_loader:
            xb = xb.to("cuda" if use_gpu else "cpu")
            preds.append(best(xb).cpu().numpy())
    preds = np.concatenate(preds, axis=0).ravel()

    # Un-standardize → maybe exp10 → R² in *original* (linear) y units.
    preds = preds * sd_y + mu_y
    if target_log:
        preds = np.power(10.0, preds)
    yt = y[va]
    valid = np.isfinite(yt) & np.isfinite(preds)
    if valid.sum() < 2 or np.std(yt[valid]) < 1e-6:
        score = float("nan")
    else:
        score = float(r2_score(yt[valid], preds[valid]))
    return [(score, "r2", len(tr), len(va))]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels-source", required=True, type=Path,
                   help="Any prepare_combined.py H5 (labels are identical across checkpoints).")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--seeds", default="0,1,2,3,4",
                   help="Comma-separated list of seeds. Each seed gets its own train/val split + init.")
    p.add_argument("--preprocess", choices=("none", "v2"), default="v2",
                   help="`none` reproduces v1 raw-features behavior; `v2` applies the audit-"
                        "driven log/std preprocessing and the log-space training for "
                        "provabgs_avg_sfr / provabgs_z_mw.")
    p.add_argument("--include-coords", action="store_true",
                   help="(v2 only) Append hsc_ra and hsc_dec to the feature matrix.")
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading labels from {args.labels_source}")
    print(f"Preprocess mode: {args.preprocess}"
          + ("  +coords" if (args.preprocess == "v2" and args.include_coords) else ""))
    h5 = CkptH5(args.labels_source)
    if not h5.has_subset(SUBSET):
        raise RuntimeError(f"Labels source has no /{SUBSET} group")

    if args.preprocess == "none":
        columns = _v1_columns()
        if args.include_coords:
            print("  warn: --include-coords ignored under --preprocess none")
        ckpt_tag = CHECKPOINT_TAG_V1
    else:
        columns = _v2_columns(include_coords=args.include_coords)
        ckpt_tag = CHECKPOINT_TAG_V2

    X, used_cols = build_instrument_matrix(h5, columns)
    print(f"  feature matrix: shape={X.shape}, n_features={len(used_cols)}")
    if args.preprocess == "v2":
        n_logged = sum(1 for c in used_cols if c in V2_TRANSFORMS)
        print(f"  v2 transforms applied to {n_logged} of {len(used_cols)} input columns")

    rows: list[dict] = []
    for target in PHYSICS_TARGETS:
        y_full = _physics_label(h5, SUBSET, target)
        if y_full is None:
            print(f"  [skip] {target}: label not present in H5")
            continue
        Xf, yf = _filter_finite(X, np.asarray(y_full, dtype=np.float32))
        if len(yf) < MIN_LABELS:
            print(f"  [skip] {target}: only {len(yf)} finite labels (< {MIN_LABELS})")
            continue
        target_log = (args.preprocess == "v2") and (target in LOG_TARGETS)
        for seed in seeds:
            tag = "log10→linear" if target_log else "linear"
            print(f"  [physics] {target}  ({tag})  seed={seed}  (n={len(yf)})")
            if args.preprocess == "v2":
                results = _train_eval_baseline_v2(
                    X_raw=Xf,
                    y=yf,
                    columns=used_cols,
                    transforms=V2_TRANSFORMS,
                    target_log=target_log,
                    max_epochs=args.max_epochs,
                    seed=seed,
                    device=args.device,
                )
            else:
                results = _train_eval(Xf, yf.reshape(-1, 1), mode="regression",
                                      max_epochs=args.max_epochs, seed=seed,
                                      device=args.device)
            for score, metric, n_tr, n_te in results:
                rows.append(dict(
                    checkpoint=ckpt_tag,
                    task_family=TASK_FAMILY,
                    target=target,
                    latent_variant=LATENT_VARIANT,
                    score=score,
                    score_metric=metric,
                    n_train=n_tr,
                    n_test=n_te,
                    seed=seed,
                ))

    h5.close()

    if not rows:
        print("No probe results — writing empty CSV with header only.")
        df = pd.DataFrame(columns=["checkpoint", "task_family", "target", "latent_variant",
                                   "score", "score_metric", "n_train", "n_test", "seed"])
    else:
        df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}  ({len(df)} rows, {len(seeds)} seeds × "
          f"{df['target'].nunique() if len(df) else 0} targets)")

    if len(df):
        print("\nSummary (mean ± std across seeds):")
        agg = df.groupby("target")["score"].agg(["mean", "std", "count"]).reindex(PHYSICS_TARGETS).dropna(how="all")
        for tgt, row in agg.iterrows():
            print(f"  {tgt:24s}  R² = {row['mean']:+.4f} ± {row['std']:.4f}  "
                  f"(n_seeds={int(row['count'])})")


if __name__ == "__main__":
    main()
