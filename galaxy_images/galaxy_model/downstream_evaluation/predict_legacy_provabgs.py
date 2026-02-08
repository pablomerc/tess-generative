"""
Downstream evaluation for Legacy ProvaBGS: predict galaxy parameters from embeddings.
Uses H5 from prepare_legacy_provabgs.py (6 embedding types + labels/labels_repeated).

Run a single experiment per run: legacy, legacy_hsc, or hsc.
  python predict_legacy_provabgs.py --experiment legacy --single
  python predict_legacy_provabgs.py --experiment legacy_hsc --single
  python predict_legacy_provabgs.py --experiment hsc --single

Run on all generated latent sets (default: 3 trained H5s from prepare_legacy_provabgs); saves plot + CSV per (dataset, experiment) to r2_comparisons_legacy/.
  python predict_legacy_provabgs.py
  python predict_legacy_provabgs.py --datasets a.h5 b.h5 c.h5
"""
import argparse
import csv
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

_here = Path(__file__).resolve().parent

# Default paths (Legacy downstream H5s)
DATASET_PATH = _here / "downstream_legacy_provabgs.h5"
DATASET_PATH_UNTRAINED = _here / "downstream_legacy_provabgs_untrained.h5"

# Default list of trained H5s from prepare_legacy_provabgs.py (3 model configs)
DEFAULT_DATASET_LIST = [
    _here / "downstream_legacy_provabgs_zdim16_geom_neighbors.h5",
    _here / "downstream_legacy_provabgs_zdim16_nogeom_neighbors.h5",
    _here / "downstream_legacy_provabgs_zdim16_geom_old_dataloader.h5",
]
R2_COMPARISONS_LEGACY_DIR = _here / "r2_comparisons_legacy_DEC"

# One experiment per run: which embedding pair to use (same keys as HSC)
EXPERIMENTS = {
    "legacy": ("legacy_encoder1", "legacy_encoder2"),
    "legacy_hsc": ("hsc_legacy_encoder1", "hsc_legacy_encoder2"),
    "hsc": ("hsc_encoder1", "hsc_encoder2"),
}

# MLP architecture for downstream head (used in filenames)
# MLP_HIDDEN = (512, 256, 128)
MLP_HIDDEN = (256, 128)
MLP_SUFFIX = "-".join(map(str, MLP_HIDDEN))  # "512-256-128"

# Default target columns for Legacy ProvaBGS (None = all numeric in H5)
# DEFAULT_TARGETS = [
#     "LOG_MSTAR", "Z_HP", "MAG_R", "sSFR", "TAGE_MW", "LOG_Z_MW", "desi_Z",
#     "legacysurvey_SHAPE_R", "tok_shape_r", "TSNR2_BGS", "PROVABGS_W_FIBASSIGN",
#     "desi_FLUX_IVAR_G", "desi_FLUX_IVAR_R", "desi_FLUX_IVAR_Z",
# ]
DEFAULT_TARGETS = ["desi_Z", "LOG_MSTAR", "TAGE_MW", "LOG_Z_MW", "sSFR", "hsc_g_extendedness_value", 'DEC']


def load_legacy_provabgs_h5(path, key1, key2):
    """
    Load two embedding arrays and numeric labels from legacy ProvaBGS H5.
    Returns embeddings_1, embeddings_2, metadata (numeric cols only), param_names.
    Uses labels/ (N rows) when embedding length equals num_examples; labels_repeated/ (2N) only if length == 2*num_examples.
    """
    with h5py.File(path, 'r') as f:
        emb1 = np.array(f[key1][:])
        emb2 = np.array(f[key2][:])
        n1, n2 = emb1.shape[0], emb2.shape[0]
        if n1 != n2:
            raise ValueError(f"Embedding length mismatch: {key1} has {n1}, {key2} has {n2}")
        num_examples = int(f.attrs.get('num_examples', 0))
        use_repeated = (num_examples > 0 and n1 == num_examples * 2)
        label_prefix = 'labels_repeated/' if use_repeated else 'labels/'
        raw_cols = f.attrs.get('label_columns', [])
        label_columns = []
        for c in (raw_cols if isinstance(raw_cols, (list, tuple)) else list(raw_cols)):
            label_columns.append(c.decode('utf-8') if isinstance(c, bytes) else c)
        if not label_columns:
            label_columns = [k[len(label_prefix):] for k in f.keys() if k.startswith(label_prefix)]
        meta_list = []
        param_names = []
        for col in label_columns:
            key = label_prefix + col
            if key not in f:
                continue
            arr = np.array(f[key][:])
            if arr.dtype.kind in 'fiu':
                meta_list.append(arr.astype(np.float64))
                param_names.append(col)
            elif arr.dtype.kind in 'SU':
                continue
            else:
                try:
                    meta_list.append(arr.astype(np.float64))
                    param_names.append(col)
                except Exception:
                    continue
        if not meta_list:
            raise ValueError("No numeric label columns found in H5")
        metadata = np.stack(meta_list, axis=1)
    return emb1, emb2, metadata, param_names


def check_gpu():
    """Check GPU availability and return device info."""
    if not torch.cuda.is_available():
        return False, "CUDA not available"
    try:
        torch.zeros(1).cuda()
        torch.cuda.synchronize()
        return True, f"GPU: {torch.cuda.get_device_name(0)}"
    except RuntimeError as e:
        return False, str(e)


def _standardize_with_stats(data, mean, std):
    """Standardize data using precomputed mean and std."""
    std = np.where(std == 0, 1.0, std)
    return (np.asarray(data, dtype=np.float64) - mean) / (std + 1e-8)


class MLPRegressor(nn.Module):
    """MLP regressor for predicting galaxy parameters from embeddings."""
    def __init__(self, in_dim, out_dim, hidden=None, dropout=0.2):
        if hidden is None:
            hidden = MLP_HIDDEN
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [
                nn.Linear(prev, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LitRegressor(pl.LightningModule):
    """PyTorch Lightning module for regression training."""
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden=None,
        dropout=0.2,
        lr=1e-3,
        weight_decay=1e-2,
        use_embedding=1,
    ):
        if hidden is None:
            hidden = MLP_HIDDEN
        super().__init__()
        if use_embedding not in (1, 2):
            raise ValueError("use_embedding must be 1 or 2")
        self.save_hyperparameters()
        self.model = MLPRegressor(in_dim=in_dim, out_dim=out_dim, hidden=hidden, dropout=dropout)
        self.loss_fn = nn.SmoothL1Loss(beta=1.0)

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage: str):
        emb1, emb2, y = batch
        x = emb1 if self.hparams.use_embedding == 1 else emb2

        y_hat = self(x)

        # Check for NaN in predictions
        if torch.isnan(y_hat).any():
            print(f"WARNING: NaN detected in predictions during {stage} step")
            y_hat = torch.nan_to_num(y_hat, nan=0.0)

        loss = self.loss_fn(y_hat, y)

        # Check for NaN loss
        if torch.isnan(loss):
            print(f"WARNING: NaN loss detected during {stage} step")
            # Use a fallback loss
            loss = torch.mean((y_hat - y) ** 2)
            if torch.isnan(loss):
                loss = torch.tensor(1e6, device=loss.device, requires_grad=True)

        mse = torch.mean((y_hat - y) ** 2)

        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=x.size(0))
        self.log(f"{stage}/mse", mse, on_step=False, on_epoch=True, batch_size=x.size(0))
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def configure_optimizers(self):
        opt = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
        return {"optimizer": opt, "lr_scheduler": sched}

    def on_before_optimizer_step(self, optimizer):
        # Clip gradients to prevent exploding gradients that cause NaN
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)


def evaluate_per_target(model, dataloader, param_names, device):
    """Evaluate model and compute per-target metrics."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            emb1, emb2, y = batch
            x = emb1 if model.hparams.use_embedding == 1 else emb2
            x = x.to(device)
            y = y.to(device)

            y_hat = model(x)
            all_preds.append(y_hat.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Check for NaN in predictions
    nan_preds = np.isnan(all_preds).any(axis=0)
    nan_targets = np.isnan(all_targets).any(axis=0)

    if nan_preds.any():
        print(f"WARNING: NaN values found in predictions for {nan_preds.sum()} targets")
    if nan_targets.any():
        print(f"WARNING: NaN values found in targets for {nan_targets.sum()} targets")

    # Compute metrics for each target
    results = []
    for i in range(all_targets.shape[1]):
        y_true = all_targets[:, i]
        y_pred = all_preds[:, i]

        # Remove NaN values for this target
        valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[valid_mask]
        y_pred_clean = y_pred[valid_mask]

        if len(y_true_clean) == 0:
            # All values are NaN
            r2 = np.nan
            mae = np.nan
            rmse = np.nan
        else:
            try:
                # 3) Don't report R² for near-constant targets
                if np.std(y_true_clean) < 1e-6:
                    r2 = np.nan  # Near-constant target, R² is not meaningful
                else:
                    r2 = r2_score(y_true_clean, y_pred_clean)
                mae = mean_absolute_error(y_true_clean, y_pred_clean)
                rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
            except Exception as e:
                print(f"Warning: Error computing metrics for target {i}: {e}")
                r2 = np.nan
                mae = np.nan
                rmse = np.nan

        param_name = param_names[i] if i < len(param_names) else f"target_{i}"
        results.append({
            'target': param_name,
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'n_valid': len(y_true_clean)
        })

    return results


SERIES = ["trained_emb1", "trained_emb2", "untrained_emb1", "untrained_emb2"]
SERIES_LABELS = ["Physics (trained)", "Instrument (trained)", "Physics (untrained)", "Instrument (untrained)"]


def save_results_csv(results, param_names, series, filepath):
    """Save R² (and MAE) per target and per series to a CSV file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["target"] + [f"r2_{k}" for k in series] + [f"mae_{k}" for k in series]
    with open(filepath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for t in param_names:
            row = {"target": t}
            for k in series:
                r = next((x for x in results[k] if x["target"] == t), None)
                if r is not None:
                    row[f"r2_{k}"] = r["r2"] if not np.isnan(r["r2"]) else ""
                    row[f"mae_{k}"] = r["mae"] if not np.isnan(r["mae"]) else ""
                else:
                    row[f"r2_{k}"] = row[f"mae_{k}"] = ""
            w.writerow(row)
    print(f"Results CSV saved: {filepath}")


def train_and_eval(use_embedding, train_loader, val_loader, param_names, emb_dim, out_dim, use_gpu, precision):
    """Train and evaluate a single model with specified embedding. Returns list of per-target results."""
    model = LitRegressor(
        in_dim=emb_dim,
        out_dim=out_dim,
        hidden=MLP_HIDDEN,
        dropout=0.2,
        lr=1e-3,
        weight_decay=1e-2,
        use_embedding=use_embedding,
    )
    ckpt = ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1, filename=f"emb{use_embedding}-{{epoch:02d}}-{{val_loss:.4f}}")
    trainer = Trainer(
        max_epochs=100,
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        precision=precision,
        callbacks=[ckpt, EarlyStopping(monitor="val/loss", mode="min", patience=20), LearningRateMonitor(logging_interval="epoch")],
        log_every_n_steps=20,
    )
    trainer.fit(model, train_loader, val_loader)
    best = LitRegressor.load_from_checkpoint(ckpt.best_model_path) if ckpt.best_model_path else model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return evaluate_per_target(best.to(device), val_loader, param_names, device)


def run_one_experiment(dataset_path, experiment, dataset_untrained_path, target_columns, seed, use_gpu, precision):
    """Run training and evaluation for one (dataset, experiment). Returns (results dict, param_names list)."""
    key1, key2 = EXPERIMENTS[experiment]
    emb1_tr, emb2_tr, meta_tr, param_names = load_legacy_provabgs_h5(str(dataset_path), key1, key2)

    if target_columns is not None:
        col_set = set(target_columns)
        indices = [i for i, n in enumerate(param_names) if n in col_set]
        if not indices:
            raise ValueError("No requested target columns found in H5.")
        meta_tr = meta_tr[:, indices]
        param_names = [param_names[i] for i in indices]

    finite = np.isfinite(meta_tr).all(axis=1)
    emb1_tr = np.nan_to_num(emb1_tr[finite], nan=0.0, posinf=0.0, neginf=0.0)
    emb2_tr = np.nan_to_num(emb2_tr[finite], nan=0.0, posinf=0.0, neginf=0.0)
    meta_tr = meta_tr[finite]

    n = len(meta_tr)
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(0.9 * n)
    tr_idx, va_idx = idx[:n_train], idx[n_train:]
    mean = meta_tr[tr_idx].mean(axis=0)
    std = meta_tr[tr_idx].std(axis=0)
    meta_tr = _standardize_with_stats(meta_tr, mean, std)

    def make_loaders(emb1, emb2, meta, batch_size=64):
        e1 = torch.tensor(emb1, dtype=torch.float32)
        e2 = torch.tensor(emb2, dtype=torch.float32)
        m = torch.tensor(meta, dtype=torch.float32)
        train_ds = TensorDataset(e1[tr_idx], e2[tr_idx], m[tr_idx])
        val_ds = TensorDataset(e1[va_idx], e2[va_idx], m[va_idx])
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        return train_loader, val_loader

    train_loader, val_loader = make_loaders(emb1_tr, emb2_tr, meta_tr)
    emb_dim_1 = emb1_tr.shape[1]
    emb_dim_2 = emb2_tr.shape[1]
    out_dim = meta_tr.shape[1]

    emb1_u, emb2_u, meta_u, pn_u = load_legacy_provabgs_h5(str(dataset_untrained_path), key1, key2)
    col_idx_u = [pn_u.index(n) for n in param_names]
    meta_u = meta_u[finite][:, col_idx_u]
    meta_u = _standardize_with_stats(meta_u, mean, std)
    train_loader_u, val_loader_u = make_loaders(emb1_u[finite], emb2_u[finite], meta_u)

    results = {}
    for (label, ld_tr, ld_va), emb_num, emb_dim in [
        (("trained", train_loader, val_loader), 1, emb_dim_1),
        (("trained", train_loader, val_loader), 2, emb_dim_2),
        (("untrained", train_loader_u, val_loader_u), 1, emb_dim_1),
        (("untrained", train_loader_u, val_loader_u), 2, emb_dim_2),
    ]:
        key = f"{label}_emb{emb_num}"
        results[key] = train_and_eval(emb_num, ld_tr, ld_va, param_names, emb_dim, out_dim, use_gpu, precision)
    return results, param_names


def print_results_table(results, param_names, experiment, dataset_stem=""):
    """Print R² comparison table to stdout."""
    fmt = lambda v: "    NaN" if np.isnan(v) else f"{v:>14.4f}"
    print("\n" + "=" * 100)
    print(f"R² — {dataset_stem} — {experiment}" if dataset_stem else f"R² — {experiment}")
    print("=" * 100)
    print(f"{'Target':<28} " + " ".join(f"{s:>14}" for s in SERIES_LABELS))
    print("-" * 100)
    for t in param_names:
        r2s = [next((r["r2"] for r in results[k] if r["target"] == t), np.nan) for k in SERIES]
        print(f"{t:<28} " + " ".join(fmt(r) for r in r2s))
    avgs = [np.nanmean([next((r["r2"] for r in results[k] if r["target"] == t), np.nan) for t in param_names]) for k in SERIES]
    print("-" * 100)
    print(f"{'Average':<28} " + " ".join(fmt(a) for a in avgs))
    print("=" * 100)


def save_plot(results, param_names, experiment, filepath, dataset_stem=""):
    """Save R² bar chart to filepath."""
    plot_data = []
    for t in param_names:
        r2s = [next((r["r2"] for r in results[k] if r["target"] == t), np.nan) for k in SERIES]
        if any(not np.isnan(v) for v in r2s):
            plot_data.append({"target": t, "r2s": r2s})
    if not plot_data:
        return
    targets = [d["target"] for d in plot_data]
    n_t = len(targets)
    x = np.arange(n_t)
    width = 0.2
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    colors = ["#2E86AB", "#A23B72", "#5DA271", "#E07A5F"]
    fig, ax = plt.subplots(figsize=(max(12, n_t * 0.5), 8))
    for i, (k, lbl) in enumerate(zip(SERIES, SERIES_LABELS)):
        vals = [d["r2s"][i] if not np.isnan(d["r2s"][i]) else 0 for d in plot_data]
        ax.bar(x + offsets[i], vals, width, label=lbl, color=colors[i], alpha=0.8)
    ax.set_xlabel("Target")
    ax.set_ylabel("R²")
    title = f"Legacy ProvaBGS — {dataset_stem} — {experiment}" if dataset_stem else f"Legacy ProvaBGS — {experiment}"
    ax.set_title(title + ": Trained vs Untrained (Physics & Instrument)")
    ax.set_xticks(x)
    ax.set_xticklabels(targets, rotation=45, ha="right", fontsize=9)
    ax.legend()
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {filepath}")
    plt.close()


def main():
    p = argparse.ArgumentParser(description="Legacy ProvaBGS downstream prediction (single or batch over datasets)")
    p.add_argument("--experiment", type=str, choices=list(EXPERIMENTS), default="legacy_hsc",
                   help="Embedding set (single-run mode): legacy, legacy_hsc, or hsc")
    p.add_argument("--dataset", type=str, default=str(DATASET_PATH), help="Path to trained-embeddings H5 (single-run mode)")
    p.add_argument("--datasets", type=str, nargs="*", default=None,
                   help="List of .h5 paths; run all experiments on each. Default: run on the 3 H5s from prepare_legacy_provabgs. Pass --dataset and --experiment with --single for single-run instead.")
    p.add_argument("--single", action="store_true",
                   help="Single-run mode: use --dataset and --experiment only (no batch over datasets).")
    p.add_argument("--dataset-untrained", type=str, default=str(DATASET_PATH_UNTRAINED), help="Path to untrained-embeddings H5")
    p.add_argument("--out-dir", type=str, default=str(R2_COMPARISONS_LEGACY_DIR),
                   help="Output directory for plots and CSVs (default: r2_comparisons_legacy)")
    p.add_argument("--targets", type=str, nargs="*", default=None,
                   help="Target columns (default: use DEFAULT_TARGETS); empty = all numeric")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.targets is not None:
        target_columns = None if len(args.targets) == 0 else args.targets
    else:
        target_columns = DEFAULT_TARGETS

    gpu_ok, gpu_msg = check_gpu()
    print(f"GPU: {gpu_msg}")
    use_gpu = gpu_ok
    precision = "16-mixed" if use_gpu else "32-true"
    out_dir = Path(args.out_dir)

    # Batch mode: list of .h5 datasets, run all experiments on each (default unless --single)
    if not args.single:
        dataset_list = [Path(p) for p in args.datasets] if (args.datasets and len(args.datasets) > 0) else [Path(p) for p in DEFAULT_DATASET_LIST]
        out_dir.mkdir(parents=True, exist_ok=True)
        for dataset_path in dataset_list:
            if not dataset_path.exists():
                print(f"Skip (not found): {dataset_path}")
                continue
            stem = dataset_path.stem
            print(f"\n{'='*60}\nDataset: {dataset_path}\n{'='*60}")
            for experiment in EXPERIMENTS:
                print(f"\n--- Experiment: {experiment} ---")
                try:
                    results, param_names = run_one_experiment(
                        dataset_path, experiment, args.dataset_untrained,
                        target_columns, args.seed, use_gpu, precision,
                    )
                    print_results_table(results, param_names, experiment, stem)
                    save_plot(results, param_names, experiment, out_dir / f"{stem}_{experiment}_{MLP_SUFFIX}.png", stem)
                    save_results_csv(results, param_names, SERIES, out_dir / f"{stem}_{experiment}_{MLP_SUFFIX}.csv")
                except Exception as e:
                    print(f"  Error: {e}")
                    raise
        return

    # Single-run mode
    key1, key2 = EXPERIMENTS[args.experiment]
    print(f"Experiment: {args.experiment} -> {key1}, {key2}")
    print(f"Dataset (trained):   {args.dataset}")
    print(f"Dataset (untrained): {args.dataset_untrained}")
    emb1_tr, emb2_tr, meta_tr, param_names = load_legacy_provabgs_h5(args.dataset, key1, key2)
    print(f"  Loaded: emb1 {emb1_tr.shape}, emb2 {emb2_tr.shape}, labels {meta_tr.shape}, {len(param_names)} targets")
    if target_columns is not None:
        col_set = set(target_columns)
        indices = [i for i, n in enumerate(param_names) if n in col_set]
        missing = col_set - {param_names[i] for i in indices}
        if missing:
            print(f"  Warning: targets not in H5 (skipped): {missing}")
        if not indices:
            raise ValueError("No requested target columns found in H5.")
        meta_tr = meta_tr[:, indices]
        param_names = [param_names[i] for i in indices]
    print(f"  Using {len(param_names)} targets: {param_names}")

    results, param_names = run_one_experiment(
        Path(args.dataset), args.experiment, args.dataset_untrained,
        target_columns, args.seed, use_gpu, precision,
    )
    print_results_table(results, param_names, args.experiment)
    stem = Path(args.dataset).stem
    save_plot(results, param_names, args.experiment, out_dir / f"{stem}_{args.experiment}_{MLP_SUFFIX}.png", stem)
    save_results_csv(results, param_names, SERIES, out_dir / f"{stem}_{args.experiment}_{MLP_SUFFIX}.csv")


if __name__ == "__main__":
    main()
