"""
Stage 2: Downstream prediction for the HSC-only evaluation.

Consumes: downstream_hsc_new_{suffix}.h5 (from prepare_hsc_downstream.py)

Three prediction tasks:
  - instrument_hsc:       predict HSC instrument properties  using hsc_encoder2
  - physics_hsc_provabgs: predict PROVABGS physical props    using hsc_encoder1 (DESI cross-match subset)
  - morphology_hsc:       predict SDSS i-band shape moments  using hsc_encoder1

Output: predict_hsc_downstream_{suffix}.csv + per-objective PNG plots
"""

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl

# ROCm workaround
if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "preferred_blas_library"):
    torch.backends.cuda.preferred_blas_library("hipblas")

from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Embedding keys and task definitions
# ---------------------------------------------------------------------------

KEY1 = "hsc_encoder1"   # physics / galaxy encoder
KEY2 = "hsc_encoder2"   # instrument encoder
EMBEDDING_VARIANTS = ("real", "untrained", "random")

instrument_hsc = [
    # Galactic dust extinction (all 5 HSC bands; a_y tests generalisation past model input)
    "a_g", "a_r", "a_i", "a_z", "a_y",
    # Sky noise variance (4 model bands)
    "g_variance_value", "r_variance_value", "i_variance_value", "z_variance_value",
    # PSF FWHM in arcsec (4 model bands, from image_psf_fwhm list column)
    "psf_fwhm_g", "psf_fwhm_r", "psf_fwhm_i", "psf_fwhm_z",
]

physics_hsc_provabgs = [
    # DESI BGS cross-match; nullable — only available for a subset of galaxies
    "desi_z",               # spectroscopic redshift
    "provabgs_logmstar",    # log stellar mass
    "provabgs_tage_mw",     # mass-weighted age [Gyr]
    "provabgs_avg_sfr",     # average SFR [M_sun/yr]
    "provabgs_z_mw",        # mass-weighted metallicity
]

morphology_hsc = [
    # SDSS adaptive moments in i-band (arcsec^2); proxy for size and ellipticity
    "i_sdssshape_shape11",
    "i_sdssshape_shape22",
    "i_sdssshape_shape12",
]

# (objective, task_name, h5_stem, default_embedding_for_untrained/random, target_list)
TASKS = [
    ("hsc_new", "instrument_hsc",       "hsc_new", 2, instrument_hsc),
    ("hsc_new", "physics_hsc_provabgs", "hsc_new", 1, physics_hsc_provabgs),
    ("hsc_new", "morphology_hsc",       "hsc_new", 1, morphology_hsc),
]

MLP_HIDDEN  = (256, 128)
BATCH_SIZE  = 64
TRAIN_FRAC  = 0.9


# ---------------------------------------------------------------------------
# H5 loading
# ---------------------------------------------------------------------------

def load_h5_variants(path: Path):
    """Return dict: variant -> (emb1, emb2, metadata, param_names)."""
    with h5py.File(path, "r") as f:
        raw_cols      = f.attrs.get("label_columns", [])
        label_columns = [c.decode() if isinstance(c, bytes) else c for c in raw_cols]
        if not label_columns:
            label_columns = [k[len("labels/"):] for k in f.keys() if k.startswith("labels/")]

        meta_list, param_names = [], []
        for col in label_columns:
            key = "labels/" + col
            if key not in f:
                continue
            arr = np.array(f[key][:])
            if arr.dtype.kind not in "fiu":
                try:
                    arr = arr.astype(np.float64)
                except Exception:
                    continue
            if arr.ndim == 1:
                meta_list.append(arr)
                param_names.append(col)
            elif arr.ndim == 2:
                for j in range(arr.shape[1]):
                    meta_list.append(arr[:, j].astype(np.float64))
                    param_names.append(f"{col}_{j}")

        if not meta_list:
            raise ValueError(f"No numeric label columns in {path}")
        n        = meta_list[0].shape[0]
        metadata = np.stack(meta_list, axis=1)   # (N, n_cols)

        out = {}
        for variant in EMBEDDING_VARIANTS:
            suf = "" if variant == "real" else f"_{variant}"
            k1, k2 = KEY1 + suf, KEY2 + suf
            if k1 not in f or k2 not in f:
                raise ValueError(f"Missing {k1} or {k2} in {path}")
            emb1 = np.array(f[k1][:])
            emb2 = np.array(f[k2][:])
            if emb1.shape[0] != n or emb2.shape[0] != n:
                raise ValueError(f"Length mismatch in {path}: embeddings vs labels")
            out[variant] = (emb1, emb2, metadata.copy(), list(param_names))
    return out


# ---------------------------------------------------------------------------
# MLP training infrastructure
# ---------------------------------------------------------------------------

def _standardize_with_stats(data, mean, std):
    std = np.where(std == 0, 1.0, std)
    return (np.asarray(data, dtype=np.float64) - mean) / (std + 1e-8)


class MLPRegressor(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=None, dropout=0.2):
        super().__init__()
        hidden = hidden or MLP_HIDDEN
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LitRegressor(pl.LightningModule):
    def __init__(self, in_dim, out_dim, use_embedding=1):
        super().__init__()
        self.save_hyperparameters()
        self.model   = MLPRegressor(in_dim=in_dim, out_dim=out_dim)
        self.loss_fn = nn.SmoothL1Loss(beta=1.0)

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, stage):
        emb1, emb2, y = batch
        x    = emb1 if self.hparams.use_embedding == 1 else emb2
        y_hat = torch.nan_to_num(self(x), nan=0.0)
        loss  = self.loss_fn(y_hat, y)
        self.log(f"{stage}/loss", loss, on_epoch=True, prog_bar=True, batch_size=x.size(0))
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def configure_optimizers(self):
        opt   = AdamW(self.parameters(), lr=1e-3, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
        return {"optimizer": opt, "lr_scheduler": sched}

    def on_before_optimizer_step(self, optimizer):
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)


def evaluate_per_target(model, loader, param_names, device, use_embedding=1):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for emb1, emb2, y in loader:
            x     = emb1 if use_embedding == 1 else emb2
            y_hat = model(x.to(device))
            all_preds.append(y_hat.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    all_preds   = np.concatenate(all_preds,   axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    results = []
    for i in range(all_targets.shape[1]):
        y_true, y_pred = all_targets[:, i], all_preds[:, i]
        valid  = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_t, y_p = y_true[valid], y_pred[valid]
        if len(y_t) == 0:
            r2, mae, rmse = np.nan, np.nan, np.nan
        else:
            r2   = np.nan if np.std(y_t) < 1e-6 else r2_score(y_t, y_p)
            mae  = mean_absolute_error(y_t, y_p)
            rmse = np.sqrt(mean_squared_error(y_t, y_p))
        name = param_names[i] if i < len(param_names) else f"target_{i}"
        results.append({"target": name, "r2": r2, "mae": mae, "rmse": rmse, "n_valid": len(y_t)})
    return results


def evaluate_mean_baseline(meta, param_names, tr_idx, va_idx):
    mean_tr  = meta[tr_idx].mean(axis=0)
    meta_val = meta[va_idx]
    results  = []
    for i in range(meta_val.shape[1]):
        y_true = meta_val[:, i]
        y_pred = np.full_like(y_true, mean_tr[i])
        valid  = ~np.isnan(y_true)
        y_t, y_p = y_true[valid], y_pred[valid]
        if len(y_t) == 0:
            r2, mae, rmse = np.nan, np.nan, np.nan
        else:
            r2   = np.nan if np.std(y_t) < 1e-6 else r2_score(y_t, y_p)
            mae  = mean_absolute_error(y_t, y_p)
            rmse = np.sqrt(mean_squared_error(y_t, y_p))
        name = param_names[i] if i < len(param_names) else f"target_{i}"
        results.append({"target": name, "r2": r2, "mae": mae, "rmse": rmse, "n_valid": len(y_t)})
    return results


def train_and_eval_one(emb, meta, param_names, tr_idx, va_idx, use_embedding, use_gpu):
    mean   = meta[tr_idx].mean(axis=0)
    std    = meta[tr_idx].std(axis=0)
    meta_s = _standardize_with_stats(meta, mean, std)

    e1 = torch.tensor(emb[0], dtype=torch.float32)
    e2 = torch.tensor(emb[1], dtype=torch.float32)
    m  = torch.tensor(meta_s, dtype=torch.float32)

    train_ds = TensorDataset(e1[tr_idx], e2[tr_idx], m[tr_idx])
    val_ds   = TensorDataset(e1[va_idx], e2[va_idx], m[va_idx])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = LitRegressor(in_dim=emb[0].shape[1], out_dim=meta_s.shape[1], use_embedding=use_embedding)
    ckpt  = ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1, save_last=False)
    trainer = Trainer(
        max_epochs=100,
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        callbacks=[
            ckpt,
            EarlyStopping(monitor="val/loss", mode="min", patience=20),
            LearningRateMonitor(logging_interval="epoch"),
        ],
        log_every_n_steps=20,
        enable_progress_bar=True,
    )
    trainer.fit(model, train_loader, val_loader)
    best   = LitRegressor.load_from_checkpoint(ckpt.best_model_path) if ckpt.best_model_path else model
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    return evaluate_per_target(best.to(device), val_loader, param_names, device, use_embedding)


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

def run_task(task_name, default_use_embedding, target_list, h5_path, seed, use_gpu):
    data = load_h5_variants(h5_path)
    _, _, metadata, param_names = data["real"]

    col_set = set(target_list)
    indices = [i for i, n in enumerate(param_names) if n in col_set]
    missing = sorted(col_set - {param_names[i] for i in indices})
    if missing:
        print(f"  Warning {task_name}: targets not found in H5, skipping: {missing}")
    if not indices:
        raise ValueError(f"No requested target columns found in H5 for task '{task_name}'")

    param_names = [param_names[i] for i in indices]
    meta        = metadata[:, indices].astype(np.float64)
    finite      = np.isfinite(meta).all(axis=1)
    n_finite    = int(finite.sum())
    print(f"  {n_finite}/{len(finite)} rows have complete labels for {task_name}")
    if n_finite == 0:
        print(f"  WARNING: skipping task '{task_name}' — no rows with finite labels (sparse cross-match expected in small samples)")
        return None, param_names
    meta = meta[finite]

    rng    = np.random.default_rng(seed)
    idx    = np.arange(meta.shape[0])
    rng.shuffle(idx)
    n_train = int(TRAIN_FRAC * len(idx))
    tr_idx, va_idx = idx[:n_train], idx[n_train:]

    out = {}
    for variant in EMBEDDING_VARIANTS:
        emb1, emb2, _, _ = data[variant]
        emb = (
            np.nan_to_num(emb1[finite], nan=0.0, posinf=0.0, neginf=0.0),
            np.nan_to_num(emb2[finite], nan=0.0, posinf=0.0, neginf=0.0),
        )
        if variant == "real":
            print(f"  Training physics MLP (encoder1)...")
            out["real_physics"]    = train_and_eval_one(emb, meta, param_names, tr_idx, va_idx, 1, use_gpu)
            print(f"  Training instrument MLP (encoder2)...")
            out["real_instrument"] = train_and_eval_one(emb, meta, param_names, tr_idx, va_idx, 2, use_gpu)
        else:
            print(f"  Training {variant} MLP...")
            out[variant] = train_and_eval_one(emb, meta, param_names, tr_idx, va_idx,
                                              default_use_embedding, use_gpu)

    out["mean"] = evaluate_mean_baseline(meta, param_names, tr_idx, va_idx)
    return out, param_names


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_results_csv(all_results, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "objective", "task", "target",
        "r2_physics", "r2_instrument", "r2_untrained", "r2_mean",
        "mae_physics", "mae_instrument", "mae_untrained", "mae_mean",
    ]
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for objective, task_name, results_per_variant, param_names in all_results:
            for t in param_names:
                row = {"objective": objective, "task": task_name, "target": t}
                for key, short in [
                    ("real_physics",    "physics"),
                    ("real_instrument", "instrument"),
                    ("untrained",       "untrained"),
                    ("mean",            "mean"),
                ]:
                    rec = next((x for x in results_per_variant.get(key, []) if x["target"] == t), None)
                    if rec is None:
                        row[f"r2_{short}"]  = ""
                        row[f"mae_{short}"] = ""
                    else:
                        row[f"r2_{short}"]  = "" if (rec["r2"]  is None or (isinstance(rec["r2"],  float) and np.isnan(rec["r2"])))  else rec["r2"]
                        row[f"mae_{short}"] = "" if (rec["mae"] is None or (isinstance(rec["mae"], float) and np.isnan(rec["mae"]))) else rec["mae"]
                w.writerow(row)
    print(f"Results CSV saved: {output_path}")


def plot_objective(df_obj, objective, out_path: Path):
    df_plot = df_obj.sort_values(["task", "target"]).reset_index(drop=True)
    if len(df_plot) == 0:
        return

    x     = np.arange(len(df_plot))
    width = 0.25

    vals_phys = pd.to_numeric(df_plot["r2_physics"],    errors="coerce").to_numpy()
    vals_inst = pd.to_numeric(df_plot["r2_instrument"], errors="coerce").to_numpy()
    vals_untr = pd.to_numeric(df_plot["r2_untrained"],  errors="coerce").to_numpy()

    fig, ax = plt.subplots(figsize=(max(12, len(df_plot) * 0.6), 6))
    ax.bar(x - width, np.nan_to_num(vals_phys, nan=0.0), width, label="Physics (enc1)",    color="#8AC3EE")
    ax.bar(x,         np.nan_to_num(vals_inst, nan=0.0), width, label="Instrument (enc2)", color="#E5254E")
    ax.bar(x + width, np.nan_to_num(vals_untr, nan=0.0), width, label="Untrained",         color="#B19221")

    labels = [f"{t}\n({task})" for t, task in zip(df_plot["target"], df_plot["task"])]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("R²")
    ax.set_title(f"HSC Downstream — {objective}")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="upper left")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Predict HSC downstream objectives (flow matching model).")
    p.add_argument("--suffix",     required=True,
                   help="Suffix matching the prepare stage (used to find downstream_hsc_new_{suffix}.h5)")
    p.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--seed",       type=int,  default=42)
    p.add_argument("--no-gpu",     action="store_true")
    args = p.parse_args()

    output_dir = args.output_dir
    h5_path    = output_dir / f"downstream_hsc_new_{args.suffix}.h5"
    if not h5_path.exists():
        raise FileNotFoundError(
            f"H5 not found: {h5_path}\n"
            "Run prepare_hsc_downstream.py first."
        )

    use_gpu = (not args.no_gpu) and torch.cuda.is_available()
    print(f"GPU enabled: {use_gpu}")

    all_results = []
    for objective, task_name, _h5_stem, default_use_embedding, targets in TASKS:
        if not targets:
            continue
        print(f"\n--- {objective} :: {task_name} ({len(targets)} targets) ---")
        results_per_variant, param_names = run_task(
            task_name, default_use_embedding, targets, h5_path, args.seed, use_gpu
        )
        if results_per_variant is None:
            continue
        all_results.append((objective, task_name, results_per_variant, param_names))

    csv_path = output_dir / f"predict_hsc_downstream_{args.suffix}.csv"
    if not all_results:
        print("\nNo tasks produced results (all tasks skipped — likely insufficient data). CSV not written.")
        return
    save_results_csv(all_results, csv_path)

    df = pd.read_csv(csv_path)
    for objective in sorted(df["objective"].unique()):
        df_obj    = df[df["objective"] == objective].copy()
        plot_path = output_dir / f"predict_hsc_downstream_{args.suffix}_{objective}.png"
        plot_objective(df_obj, objective, plot_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
