"""
Downstream evaluation script for predicting galaxy parameters from embeddings.
Indices 0-8191 are HSC and indices 8192-16383 are Legacy.
"""
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

# Configuration
dataset_path = '/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy_images/galaxy_model/downstream_evaluation/downstream_dataset.h5'
sanity_check = True  # Set to True for overfitting sanity check (256 samples, no regularization)
# Sample range for normal mode (inclusive start, exclusive end)
sample_start_idx = 0  # Start index (inclusive)
sample_end_idx = 8192  # End index (exclusive), use None to use all samples


def check_gpu_availability():
    """Check GPU availability and return device info."""
    if not torch.cuda.is_available():
        return False, "CUDA not available"

    try:
        # Try to get GPU count
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            return False, "No CUDA devices found"

        # Try to get current device
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)

        # Try a simple operation to see if GPU is actually usable
        test_tensor = torch.zeros(1).cuda()
        del test_tensor
        torch.cuda.synchronize()

        return True, f"GPU available: {device_name} (device {current_device}/{gpu_count-1})"
    except RuntimeError as e:
        return False, f"GPU error: {str(e)}"
    except Exception as e:
        return False, f"Unexpected error checking GPU: {str(e)}"


def standarize_data(data):
    """Standardize data by subtracting mean and dividing by std."""
    data = np.asarray(data)
    mean = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0)
    # Handle columns with zero or NaN std
    std = np.where(std == 0, 1.0, std)
    std = np.where(np.isnan(std), 1.0, std)
    return (data - mean) / (std + 1e-8)


class MLPRegressor(nn.Module):
    """MLP regressor for predicting galaxy parameters from embeddings."""
    def __init__(self, in_dim, out_dim=62, hidden=(512, 256, 128), dropout=0.2):
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
        out_dim=62,
        hidden=(512, 256, 128),
        dropout=0.2,
        lr=3e-4,
        weight_decay=1e-2,
        loss_name="huber",   # "mse" or "huber"
        use_embedding=1,     # 1 or 2
    ):
        super().__init__()
        if use_embedding not in (1, 2):
            raise ValueError("use_embedding must be 1 or 2")

        self.save_hyperparameters()

        self.model = MLPRegressor(in_dim=in_dim, out_dim=out_dim, hidden=hidden, dropout=dropout)
        self.loss_fn = nn.MSELoss() if loss_name == "mse" else nn.SmoothL1Loss(beta=1.0)

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


def train_and_evaluate_model(
    use_embedding,
    train_loader,
    val_loader,
    param_names,
    emb_dim,
    out_dim,
    sanity_check,
    gpu_available,
    use_gpu,
    precision
):
    """Train and evaluate a single model with specified embedding."""
    emb_name = "Physics" if use_embedding == 1 else "Instrument"
    print(f"\n{'='*80}")
    print(f"Training model with {emb_name} embeddings (embedding {use_embedding})")
    print(f"{'='*80}")

    # Create model
    if sanity_check:
        model = LitRegressor(
            in_dim=emb_dim,
            out_dim=out_dim,
            hidden=(512, 256, 128),
            dropout=0.0,  # No dropout for overfitting
            lr=1e-3,
            weight_decay=0.0,  # No weight decay for overfitting
            loss_name="mse",  # Use MSE for cleaner loss
            use_embedding=use_embedding,
        )
    else:
        model = LitRegressor(
            in_dim=emb_dim,
            out_dim=out_dim,
            hidden=(512, 256, 128),
            dropout=0.2,
            lr=1e-3,  # Same LR as sanity check mode
            weight_decay=1e-2,
            loss_name="huber",
            use_embedding=use_embedding,
        )

    # Setup callbacks
    if sanity_check:
        # For sanity check, monitor train loss and don't use early stopping
        ckpt = ModelCheckpoint(
            monitor="train/loss",
            mode="min",
            save_top_k=1,
            filename=f"sanity-emb{use_embedding}-{{epoch:02d}}-{{train_loss:.6f}}"
        )
        callbacks = [ckpt, LearningRateMonitor(logging_interval="epoch")]
        max_epochs = 100
    else:
        ckpt = ModelCheckpoint(
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            filename=f"emb{use_embedding}-{{epoch:02d}}-{{val_loss:.4f}}"
        )
        early = EarlyStopping(monitor="val/loss", mode="min", patience=10)
        callbacks = [ckpt, early, LearningRateMonitor(logging_interval="epoch")]
        max_epochs = 100

    # Create trainer
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        precision=precision,
        callbacks=callbacks,
        log_every_n_steps=20,
    )

    # Train
    emb_name = "Physics" if use_embedding == 1 else "Instrument"
    print(f"Starting training for {emb_name} embeddings...")
    try:
        trainer.fit(model, train_loader, val_loader)
    except RuntimeError as e:
        if "CUDA" in str(e) or "busy" in str(e).lower():
            print(f"\nERROR: CUDA device is busy or unavailable: {e}")
            raise
        else:
            raise

    print(f"Best checkpoint: {ckpt.best_model_path}")

    if sanity_check:
        # Get final train loss for sanity check
        logged_metrics = trainer.callback_metrics
        final_train_loss = logged_metrics.get('train/loss_epoch', None)
        if final_train_loss is not None:
            final_train_loss = final_train_loss.item()
            print(f"Final train loss: {final_train_loss:.6f}")

    # Load best model and evaluate
    print(f"Evaluating {emb_name} embeddings on validation set...")
    if ckpt.best_model_path:
        best_model = LitRegressor.load_from_checkpoint(ckpt.best_model_path)
    else:
        best_model = model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model = best_model.to(device)

    # Evaluate per-target metrics
    results = evaluate_per_target(best_model, val_loader, param_names, device)

    return results


def main():
    # Load data
    print("Loading dataset...")
    with h5py.File(dataset_path, 'r') as f:
        embeddings_1 = f['embeddings_1'][:]  # Physics embeddings
        embeddings_2 = f['embeddings_2'][:]  # Instrument embeddings
        metadata = f['metadata'][:]          # Labels
        param_names = [name.decode('utf-8') for name in f['param_names'][:]]  # Parameter names

    # Check for NaN/Inf in embeddings
    print("Checking data quality...")
    emb1_nan = np.isnan(embeddings_1).any() or np.isinf(embeddings_1).any()
    emb2_nan = np.isnan(embeddings_2).any() or np.isinf(embeddings_2).any()

    if emb1_nan:
        print(f"WARNING: NaN/Inf found in physics embeddings. Count: {np.isnan(embeddings_1).sum() + np.isinf(embeddings_1).sum()}")
        # Replace NaN/Inf with 0
        embeddings_1 = np.nan_to_num(embeddings_1, nan=0.0, posinf=0.0, neginf=0.0)
    if emb2_nan:
        print(f"WARNING: NaN/Inf found in instrument embeddings. Count: {np.isnan(embeddings_2).sum() + np.isinf(embeddings_2).sum()}")
        embeddings_2 = np.nan_to_num(embeddings_2, nan=0.0, posinf=0.0, neginf=0.0)

    # 1) Drop non-finite metadata rows before standardizing
    finite_rows = np.isfinite(metadata).all(axis=1)
    n_dropped = (~finite_rows).sum()
    if n_dropped > 0:
        print(f"Dropping rows with any non-finite label: {n_dropped}")
        embeddings_1 = embeddings_1[finite_rows]
        embeddings_2 = embeddings_2[finite_rows]
        metadata = metadata[finite_rows]

    # Apply sample range selection for normal mode
    if not sanity_check and sample_end_idx is not None:
        print(f"Selecting samples {sample_start_idx} to {sample_end_idx} (normal mode)")
        embeddings_1 = embeddings_1[sample_start_idx:sample_end_idx]
        embeddings_2 = embeddings_2[sample_start_idx:sample_end_idx]
        metadata = metadata[sample_start_idx:sample_end_idx]

    # 2) Split indices first, then standardize using train stats only (avoid leakage)
    print("Splitting data and standardizing using train stats only...")
    n = len(metadata)
    idx = np.arange(n)
    rng = np.random.default_rng(42)
    rng.shuffle(idx)

    n_train = int(0.9 * n)
    tr_idx, va_idx = idx[:n_train], idx[n_train:]

    # Compute mean/std on train only
    mean = metadata[tr_idx].mean(axis=0)
    std = metadata[tr_idx].std(axis=0)
    std = np.where(std == 0, 1.0, std)

    # Apply to all data (train + val)
    metadata = (metadata - mean) / (std + 1e-8)

    # Create dataset and dataloaders
    print("Creating dataset and dataloaders...")
    # Convert to tensors
    emb1_tensor = torch.tensor(embeddings_1, dtype=torch.float32)
    emb2_tensor = torch.tensor(embeddings_2, dtype=torch.float32)
    meta_tensor = torch.tensor(metadata, dtype=torch.float32)

    if sanity_check:
        print("="*80)
        print("SANITY CHECK MODE: Overfitting to first 256 examples")
        print("="*80)
        # Use only first n_sanity examples
        n_sanity = 8192
        emb1_tensor = emb1_tensor[:n_sanity]
        emb2_tensor = emb2_tensor[:n_sanity]
        meta_tensor = meta_tensor[:n_sanity]
        # Use same data for train and val (overfitting test)
        train_ds = TensorDataset(emb1_tensor, emb2_tensor, meta_tensor)
        val_ds = TensorDataset(emb1_tensor, emb2_tensor, meta_tensor)
        print(f"Using {len(train_ds)} samples for both train and validation")
    else:
        # Use pre-computed split indices
        train_ds = TensorDataset(
            emb1_tensor[tr_idx],
            emb2_tensor[tr_idx],
            meta_tensor[tr_idx]
        )
        val_ds = TensorDataset(
            emb1_tensor[va_idx],
            emb2_tensor[va_idx],
            meta_tensor[va_idx]
        )
        print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # Infer dimensions from a batch
    emb1, emb2, y = next(iter(train_loader))

    if sanity_check:
        print(f"\nData shape check:")
        print(f"  Physics embeddings: {emb1.shape}")
        print(f"  Instrument embeddings: {emb2.shape}")
        print(f"  targets (y): {y.shape}")
        print(f"  Target stats: min={y.min().item():.4f}, max={y.max().item():.4f}, mean={y.mean().item():.4f}, std={y.std().item():.4f}")

    # Check GPU availability
    print("Checking GPU availability...")
    gpu_available, gpu_info = check_gpu_availability()
    print(f"GPU status: {gpu_info}")

    if not gpu_available:
        print("WARNING: GPU not available or busy. Falling back to CPU training (will be slower).")
        print("You may want to:")
        print("  1. Check GPU usage with: nvidia-smi")
        print("  2. Kill other processes using the GPU")
        print("  3. Wait for other jobs to finish")
        use_gpu = False
        precision = "32-true"  # CPU doesn't support mixed precision
    else:
        use_gpu = True
        precision = "16-mixed"  # good on V100

    # Get dimensions
    emb1_dim = emb1.shape[1]
    emb2_dim = emb2.shape[1]
    out_dim = y.shape[1]

    # Train both models
    all_results = {}

    for emb_num in [1, 2]:
        emb_dim = emb1_dim if emb_num == 1 else emb2_dim
        results = train_and_evaluate_model(
            use_embedding=emb_num,
            train_loader=train_loader,
            val_loader=val_loader,
            param_names=param_names,
            emb_dim=emb_dim,
            out_dim=out_dim,
            sanity_check=sanity_check,
            gpu_available=gpu_available,
            use_gpu=use_gpu,
            precision=precision
        )
        all_results[emb_num] = results

    # Print comparison table
    print("\n" + "="*100)
    print("COMPARISON: R² Scores for Physics vs Instrument Embeddings")
    print("="*100)
    print(f"{'Target':<30} {'R² (Physics)':>14} {'R² (Instrument)':>16} {'Difference':>12} {'Better':>10}")
    print("-"*100)

    def format_metric(val):
        """Format metric value, handling NaN."""
        if np.isnan(val):
            return "        NaN"
        return f"{val:>12.4f}"

    # Get all unique targets
    all_targets = set()
    for results in all_results.values():
        for r in results:
            all_targets.add(r['target'])
    all_targets = sorted(list(all_targets))

    comparison_data = []
    for target in all_targets:
        r2_1 = next((r['r2'] for r in all_results[1] if r['target'] == target), np.nan)
        r2_2 = next((r['r2'] for r in all_results[2] if r['target'] == target), np.nan)

        # Calculate difference (Instrument - Physics)
        if not np.isnan(r2_1) and not np.isnan(r2_2):
            diff = r2_2 - r2_1
            better = "Instrument" if diff > 0 else "Physics" if diff < 0 else "Equal"
        else:
            diff = np.nan
            better = "N/A"

        comparison_data.append({
            'target': target,
            'r2_1': r2_1,
            'r2_2': r2_2,
            'diff': diff,
            'better': better
        })

        print(f"{target:<30} {format_metric(r2_1)} {format_metric(r2_2)} {format_metric(diff)} {better:>10}")

    # Summary statistics
    valid_r2_1 = [d['r2_1'] for d in comparison_data if not np.isnan(d['r2_1'])]
    valid_r2_2 = [d['r2_2'] for d in comparison_data if not np.isnan(d['r2_2'])]
    valid_diff = [d['diff'] for d in comparison_data if not np.isnan(d['diff'])]

    avg_r2_1 = np.mean(valid_r2_1) if valid_r2_1 else np.nan
    avg_r2_2 = np.mean(valid_r2_2) if valid_r2_2 else np.nan
    avg_diff = np.mean(valid_diff) if valid_diff else np.nan

    print("-"*100)
    print(f"{'Average (valid only)':<30} {format_metric(avg_r2_1)} {format_metric(avg_r2_2)} {format_metric(avg_diff)}")

    # Count wins
    physics_wins = sum(1 for d in comparison_data if d['better'] == 'Physics')
    instrument_wins = sum(1 for d in comparison_data if d['better'] == 'Instrument')
    ties = sum(1 for d in comparison_data if d['better'] == 'Equal')

    print(f"{'Wins/Ties':<30} {'Physics: ' + str(physics_wins):>14} {'Instrument: ' + str(instrument_wins):>16} {'Ties: ' + str(ties):>12}")
    print("="*100)

    # Create bar chart
    print("\nGenerating bar chart comparison...")

    # Prepare data for plotting (only valid targets)
    plot_data = [d for d in comparison_data if not (np.isnan(d['r2_1']) and np.isnan(d['r2_2']))]

    if len(plot_data) > 0:
        targets = [d['target'] for d in plot_data]
        r2_1_vals = [d['r2_1'] if not np.isnan(d['r2_1']) else 0 for d in plot_data]
        r2_2_vals = [d['r2_2'] if not np.isnan(d['r2_2']) else 0 for d in plot_data]

        # Create figure
        fig, ax = plt.subplots(figsize=(max(12, len(targets) * 0.5), 8))

        x = np.arange(len(targets))
        width = 0.35  # Width of bars

        # Create bars
        bars1 = ax.bar(x - width/2, r2_1_vals, width, label='Physics Embeddings', color='#2E86AB', alpha=0.8)
        bars2 = ax.bar(x + width/2, r2_2_vals, width, label='Instrument Embeddings', color='#A23B72', alpha=0.8)

        # Customize plot
        ax.set_xlabel('Target Parameters', fontsize=12)
        ax.set_ylabel('R² Score', fontsize=12)
        ax.set_title('R² Score Comparison: Physics vs Instrument Embeddings', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(targets, rotation=45, ha='right', fontsize=9)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Set y-axis limits
        all_vals = [v for v in r2_1_vals + r2_2_vals if v != 0]
        if all_vals:
            y_min = min(all_vals)
            y_max = max(all_vals)
            y_range = y_max - y_min
            ax.set_ylim(bottom=min(0, y_min - y_range * 0.1), top=y_max + y_range * 0.1)
        else:
            ax.set_ylim(bottom=0, top=1)

        # Add value labels on bars
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                if height != 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom', fontsize=7, rotation=90)

        # Only add labels if there aren't too many targets (to avoid clutter)
        if len(targets) <= 30:
            add_value_labels(bars1)
            add_value_labels(bars2)

        plt.tight_layout()

        # Save figure
        plot_filename = 'r2_comparison_plot.png'
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"Bar chart saved to: {plot_filename}")

        # Optionally show plot (comment out if running headless)
        # plt.show()
        plt.close()
    else:
        print("No valid data to plot (all R² scores are NaN)")


if __name__ == "__main__":
    main()
