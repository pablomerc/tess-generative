"""
predict_vs_true.py

Trains models on physics properties and creates prediction vs true value plots.
Compares 3 models:
  1. Our model (physics latents / encoder1)
  2. Our model (instrument latents / encoder2)
  3. Untrained ResNet-18

Saves predictions to HDF5 and creates scatter plots.

Run from galaxy_model/ or downstream_evaluation/:
  python downstream_evaluation/final/predict_vs_true.py --suffix zdim16_nogeom_neighbors

#python predict_vs_true.py --suffix zdim16_nogeom_neighbors
"""
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
_src = _here.parent.parent  # galaxy_model
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

import argparse
import h5py
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Target lists and embedding keys (must match prepare_all.py)
import prepare_all as pa

# Which embedding pair to use
KEY1 = "hsc_legacy_encoder1"
KEY2 = "hsc_legacy_encoder2"
EMBEDDING_VARIANTS = ("real", "untrained")

# Physics tasks only
PHYSICS_TASKS = [
    ("physics_mmu", "mmu", pa.physics_mmu),
    ("physics_provabgs", "legacy_provabgs", pa.physics_provabgs),
]

MLP_HIDDEN = (256, 128)
BATCH_SIZE = 64
TRAIN_FRAC = 0.9


def load_h5_variants(path, key1, key2, verbose=False):
    """
    Load real and untrained embedding pairs + labels from an H5 produced by prepare_all.py.
    Returns dict with keys "real", "untrained"; each value is (emb1, emb2, metadata, param_names).
    """
    with h5py.File(path, "r") as f:
        raw_cols = f.attrs.get("label_columns", [])
        label_columns = [c.decode("utf-8") if isinstance(c, bytes) else c for c in (raw_cols if isinstance(raw_cols, (list, tuple)) else list(raw_cols))]
        if not label_columns:
            label_columns = [k.replace("labels/", "") for k in f.keys() if k.startswith("labels/")]

        meta_list = []
        param_names = []
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
            else:
                continue

        if not meta_list:
            raise ValueError(f"No numeric label columns in {path}")
        n = meta_list[0].shape[0]
        metadata = np.stack(meta_list, axis=1)

        out = {}
        for variant in EMBEDDING_VARIANTS:
            suf = "" if variant == "real" else "_untrained"
            k1, k2 = key1 + suf, key2 + suf
            if k1 not in f or k2 not in f:
                raise ValueError(f"Missing {k1} or {k2} in {path}")
            emb1 = np.array(f[k1][:])
            emb2 = np.array(f[k2][:])
            if emb1.shape[0] != n or emb2.shape[0] != n:
                raise ValueError(f"Length mismatch: {k1} {emb1.shape[0]} vs metadata {n}")
            out[variant] = (emb1, emb2, metadata.copy(), list(param_names))
    return out


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
    def __init__(self, in_dim, out_dim, use_embedding=1, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = MLPRegressor(in_dim=in_dim, out_dim=out_dim, **kwargs)
        self.loss_fn = nn.SmoothL1Loss(beta=1.0)

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, stage):
        emb1, emb2, y = batch
        x = emb1 if self.hparams.use_embedding == 1 else emb2
        y_hat = torch.nan_to_num(self(x), nan=0.0)
        loss = self.loss_fn(y_hat, y)
        self.log(f"{stage}/loss", loss, on_epoch=True, prog_bar=True, batch_size=x.size(0))
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def configure_optimizers(self):
        opt = AdamW(self.parameters(), lr=1e-3, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
        return {"optimizer": opt, "lr_scheduler": sched}

    def on_before_optimizer_step(self, optimizer):
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)


def get_predictions(model, loader, device, use_embedding=1):
    """Get predictions and targets from model."""
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for emb1, emb2, y in loader:
            x = emb1 if use_embedding == 1 else emb2
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            all_preds.append(y_hat.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    return np.concatenate(all_preds, axis=0), np.concatenate(all_targets, axis=0)


def train_and_get_predictions(emb, meta_tr, meta_val, tr_idx, va_idx, use_embedding, use_gpu, seed):
    """Train model and return predictions on validation set."""
    mean = meta_tr[tr_idx].mean(axis=0)
    std = meta_tr[tr_idx].std(axis=0)
    meta_s_tr = _standardize_with_stats(meta_tr, mean, std)
    meta_s_val = _standardize_with_stats(meta_val, mean, std)

    emb1, emb2 = emb[0], emb[1]
    e1 = torch.tensor(emb1, dtype=torch.float32)
    e2 = torch.tensor(emb2, dtype=torch.float32)
    m_tr = torch.tensor(meta_s_tr, dtype=torch.float32)
    m_val = torch.tensor(meta_s_val, dtype=torch.float32)

    train_ds = TensorDataset(e1[tr_idx], e2[tr_idx], m_tr[tr_idx])
    val_ds = TensorDataset(e1[va_idx], e2[va_idx], m_val[va_idx])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    emb_dim = emb1.shape[1]
    out_dim = meta_s_tr.shape[1]

    model = LitRegressor(in_dim=emb_dim, out_dim=out_dim, use_embedding=use_embedding)
    ckpt = ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1, save_last=False)
    trainer = Trainer(
        max_epochs=100,
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        callbacks=[ckpt, EarlyStopping(monitor="val/loss", mode="min", patience=20), LearningRateMonitor(logging_interval="epoch")],
        log_every_n_steps=20,
        enable_progress_bar=False,  # Less verbose
    )
    trainer.fit(model, train_loader, val_loader)
    best = LitRegressor.load_from_checkpoint(ckpt.best_model_path) if ckpt.best_model_path else model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preds, targets = get_predictions(best.to(device), val_loader, device, use_embedding)

    # Unstandardize predictions and targets
    preds_unstd = preds * std + mean
    targets_unstd = targets * std + mean

    return preds_unstd, targets_unstd, mean, std


def run_task(task_name, h5_stem, target_list, h5_path, seed, use_gpu):
    """Run prediction task and return predictions for all models."""
    data = load_h5_variants(h5_path, KEY1, KEY2, verbose=False)

    # Use real variant to get param_names and build target index
    _, _, metadata, param_names = data["real"]
    col_set = set(target_list)
    indices = [i for i, n in enumerate(param_names) if n in col_set]
    missing = col_set - {param_names[i] for i in indices}
    if missing:
        print(f"  Warning {task_name}: targets not in H5 (skipped): {missing}")
    if not indices:
        raise ValueError(f"No requested target columns for {task_name} in H5.")
    param_names = [param_names[i] for i in indices]
    meta = metadata[:, indices].astype(np.float64)

    finite = np.isfinite(meta).all(axis=1)
    n = np.sum(finite)
    if n == 0:
        raise ValueError(f"No finite targets for {task_name}")
    meta = meta[finite]
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(TRAIN_FRAC * n)
    tr_idx, va_idx = idx[:n_train], idx[n_train:]

    results = {}
    for variant in EMBEDDING_VARIANTS:
        emb1, emb2, _, _ = data[variant]
        emb1 = np.nan_to_num(emb1[finite], nan=0.0, posinf=0.0, neginf=0.0)
        emb2 = np.nan_to_num(emb2[finite], nan=0.0, posinf=0.0, neginf=0.0)
        emb = (emb1, emb2)

        if variant == "real":
            # Train with both encoder1 (physics) and encoder2 (instrument)
            print(f"    Training physics latents model...")
            preds_physics, targets_physics, mean, std = train_and_get_predictions(
                emb, meta, meta, tr_idx, va_idx, use_embedding=1, use_gpu=use_gpu, seed=seed
            )
            results["physics"] = {
                "predictions": preds_physics,
                "targets": targets_physics,
                "mean": mean,
                "std": std,
            }

            print(f"    Training instrument latents model...")
            preds_instrument, targets_instrument, _, _ = train_and_get_predictions(
                emb, meta, meta, tr_idx, va_idx, use_embedding=2, use_gpu=use_gpu, seed=seed
            )
            results["instrument"] = {
                "predictions": preds_instrument,
                "targets": targets_instrument,
            }
        else:  # untrained
            print(f"    Training untrained model...")
            preds_untrained, targets_untrained, _, _ = train_and_get_predictions(
                emb, meta, meta, tr_idx, va_idx, use_embedding=1, use_gpu=use_gpu, seed=seed
            )
            results["untrained"] = {
                "predictions": preds_untrained,
                "targets": targets_untrained,
            }

    return results, param_names, va_idx


def save_predictions_h5(all_results, output_path):
    """Save all predictions to HDF5 file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        for task_name, results_dict, param_names, _ in all_results:
            grp = f.create_group(task_name)
            grp.attrs["param_names"] = [n.encode("utf-8") if isinstance(n, str) else n for n in param_names]

            for model_name in ["physics", "instrument", "untrained"]:
                if model_name in results_dict:
                    model_grp = grp.create_group(model_name)
                    model_grp.create_dataset("predictions", data=results_dict[model_name]["predictions"])
                    model_grp.create_dataset("targets", data=results_dict[model_name]["targets"])
                    if "mean" in results_dict[model_name]:
                        model_grp.create_dataset("mean", data=results_dict[model_name]["mean"])
                        model_grp.create_dataset("std", data=results_dict[model_name]["std"])

    print(f"Predictions saved to: {output_path}")


def make_plots(all_results, output_path, suffix):
    """Create prediction vs true scatter plots for all physics properties."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect all tasks and targets
    all_plots = []
    for task_name, results_dict, param_names, _ in all_results:
        for target_idx, target_name in enumerate(param_names):
            # Get predictions and targets for all 3 models
            preds_physics = results_dict["physics"]["predictions"][:, target_idx]
            targets_physics = results_dict["physics"]["targets"][:, target_idx]

            preds_instrument = results_dict["instrument"]["predictions"][:, target_idx]
            targets_instrument = results_dict["instrument"]["targets"][:, target_idx]

            preds_untrained = results_dict["untrained"]["predictions"][:, target_idx]
            targets_untrained = results_dict["untrained"]["targets"][:, target_idx]

            # Filter out NaN values
            valid_physics = ~(np.isnan(preds_physics) | np.isnan(targets_physics))
            valid_instrument = ~(np.isnan(preds_instrument) | np.isnan(targets_instrument))
            valid_untrained = ~(np.isnan(preds_untrained) | np.isnan(targets_untrained))

            all_plots.append({
                "task": task_name,
                "target": target_name,
                "preds_physics": preds_physics[valid_physics],
                "targets_physics": targets_physics[valid_physics],
                "preds_instrument": preds_instrument[valid_instrument],
                "targets_instrument": targets_instrument[valid_instrument],
                "preds_untrained": preds_untrained[valid_untrained],
                "targets_untrained": targets_untrained[valid_untrained],
            })

    n_plots = len(all_plots)
    if n_plots == 0:
        print("No plots to create.")
        return

    # Calculate grid dimensions
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.3)

    colors = ["#8AC3EE", "#E5254E", "#B19221"]  # Physics, Instrument, Untrained
    labels = ["Physics Latents", "Instrument Latents", "Untrained ResNet-18"]
    alpha = 0.5

    for idx, plot_data in enumerate(all_plots):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])

        # Plot all three models
        ax.scatter(plot_data["targets_physics"], plot_data["preds_physics"],
                   c=colors[0], label=labels[0], alpha=alpha, s=10, edgecolors='none')
        ax.scatter(plot_data["targets_instrument"], plot_data["preds_instrument"],
                   c=colors[1], label=labels[1], alpha=alpha, s=10, edgecolors='none')
        ax.scatter(plot_data["targets_untrained"], plot_data["preds_untrained"],
                   c=colors[2], label=labels[2], alpha=alpha, s=10, edgecolors='none')

        # Perfect prediction line
        all_targets = np.concatenate([
            plot_data["targets_physics"],
            plot_data["targets_instrument"],
            plot_data["targets_untrained"]
        ])
        if len(all_targets) > 0:
            min_val = np.nanmin(all_targets)
            max_val = np.nanmax(all_targets)
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, alpha=0.5, label='Perfect')

        # Calculate and display R²
        r2_physics = r2_score(plot_data["targets_physics"], plot_data["preds_physics"]) if len(plot_data["targets_physics"]) > 0 else np.nan
        r2_instrument = r2_score(plot_data["targets_instrument"], plot_data["preds_instrument"]) if len(plot_data["targets_instrument"]) > 0 else np.nan
        r2_untrained = r2_score(plot_data["targets_untrained"], plot_data["preds_untrained"]) if len(plot_data["targets_untrained"]) > 0 else np.nan

        title = f"{plot_data['target']}\nR²: P={r2_physics:.3f}, I={r2_instrument:.3f}, U={r2_untrained:.3f}"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("True Value", fontsize=9)
        ax.set_ylabel("Predicted Value", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='upper left')

    plt.suptitle(f"Prediction vs True: Physics Properties ({suffix})", fontsize=14, y=0.995)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {output_path}")
    plt.close()


def main():
    p = argparse.ArgumentParser(description="Predict physics properties and create prediction vs true plots")
    p.add_argument("--suffix", required=True, help="Suffix used in prepare_all (e.g. zdim16_nogeom_neighbors)")
    p.add_argument("--output-dir", type=Path, default=_here, help="Directory containing the H5s and where to write outputs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-gpu", action="store_true", help="Disable GPU")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    h5_paths = {
        "mmu": output_dir / f"downstream_mmu_{args.suffix}.h5",
        "legacy_provabgs": output_dir / f"downstream_legacy_provabgs_{args.suffix}.h5",
    }
    for k, path in h5_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Run prepare_all first. Missing: {path}")

    use_gpu = not args.no_gpu and torch.cuda.is_available()
    print(f"GPU: {'yes' if use_gpu else 'no'}")

    all_results = []
    for task_name, h5_stem, target_list in PHYSICS_TASKS:
        if not target_list:
            continue
        h5_path = h5_paths["mmu" if h5_stem == "mmu" else "legacy_provabgs"]
        print(f"\n--- Task: {task_name} ({len(target_list)} targets) ---")
        try:
            results_dict, param_names, va_idx = run_task(task_name, h5_stem, target_list, h5_path, args.seed, use_gpu)
            all_results.append((task_name, results_dict, param_names, va_idx))
            print(f"  Completed: {len(param_names)} targets")
        except Exception as e:
            print(f"  Error: {e}")
            raise

    # Save predictions to HDF5
    h5_path = output_dir / f"predict_vs_true_{args.suffix}.h5"
    save_predictions_h5(all_results, h5_path)

    # Create plots
    plot_path = output_dir / f"predict_vs_true_{args.suffix}_plot.png"
    make_plots(all_results, plot_path, args.suffix)

    print("\nDone.")


if __name__ == "__main__":
    main()
