"""
Cross-prediction experiment: predict instrument properties from the other survey's image.

Directions:
  hsc_to_legacy  -- HSC image  -> 16 Legacy instrument properties
  legacy_to_hsc  -- Legacy image -> 7 HSC instrument properties (3 variance + 4 derived PSF FWHM)

Usage:
  python cross_predict.py --direction hsc_to_legacy --epochs 50
  python cross_predict.py --direction legacy_to_hsc --epochs 50
  python cross_predict.py --direction hsc_to_legacy --epochs 3 --quick-test
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import r2_score

print("cross_predict: loading imports...", flush=True)

_script_dir = Path(__file__).resolve().parent
_model_dir = _script_dir.parent
_project_root = _model_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import timm
import h5py
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from galaxy_images.galaxy_model.neighbors import (
    preprocess_raw_image,
    NORM_DICT,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NEIGHBORS_HDF5 = "/data/vision/billf/scratch/pablomer/data/neighbours_v2.h5"
BASELINE_CSV = _model_dir / "downstream_evaluation" / "final" / "predict_all_zdim16_nogeom_neighbors_table.csv"

LEGACY_TARGETS = [
    "legacy_PSFSIZE_G", "legacy_PSFSIZE_I", "legacy_PSFSIZE_R", "legacy_PSFSIZE_Z",
    "legacy_PSFDEPTH_G", "legacy_PSFDEPTH_R", "legacy_PSFDEPTH_I", "legacy_PSFDEPTH_Z",
    "legacy_GALDEPTH_G", "legacy_GALDEPTH_R", "legacy_GALDEPTH_I", "legacy_GALDEPTH_Z",
    "legacy_NOBS_G", "legacy_NOBS_R", "legacy_NOBS_I", "legacy_NOBS_Z",
]

HSC_TARGETS_DIRECT = [
    "hsc_i_variance_value",
    "hsc_r_variance_value",
    "hsc_z_variance_value",
]
HSC_TARGETS_DERIVED = [
    "hsc_g_psf_fwhm",
    "hsc_i_psf_fwhm",
    "hsc_r_psf_fwhm",
    "hsc_z_psf_fwhm",
]

DISCORD_WEBHOOK = (
    "https://discord.com/api/webhooks/1488692651334177071/"
    "8b8KvACfQIVYCNY3ovee04BixCEWiqbqp1iQk4z9sXHlgR29kMkGIjl1pahV5uEPSbxe"
)

SEED = 42
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
VAL_FRAC = 0.1
WANDB_PROJECT = "galaxy-cross-predict"


# ---------------------------------------------------------------------------
# PSF FWHM derivation (same formula as downstream_evaluation/final/predict_all.py)
# ---------------------------------------------------------------------------

def compute_hsc_psf_seeing(shape11, shape22):
    pixel_scale_hsc = 0.168
    return 2.355 * np.sqrt((shape11 + shape22) / 2) * pixel_scale_hsc


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class CrossPredictDataset(Dataset):
    """
    Lean HDF5 dataset for cross-prediction. Reads only the needed image
    (HSC or Legacy) and skips metadata entirely. Targets are precomputed
    via vectorized column reads at init time.
    """

    def __init__(self, hdf5_path, direction, crop_size=48, norm_dict=NORM_DICT,
                 target_mean=None, target_std=None):
        self.hdf5_path = hdf5_path
        self.direction = direction
        self.crop_size = crop_size
        self.norm_dict = norm_dict
        self.file = None

        if direction == "hsc_to_legacy":
            self.image_key = "images_hsc"
            self.survey = "hsc"
            self.target_keys = LEGACY_TARGETS
            self.derived_keys = []
        else:
            self.image_key = "images_legacy"
            self.survey = "legacy"
            self.target_keys = HSC_TARGETS_DIRECT
            self.derived_keys = HSC_TARGETS_DERIVED

        self._init_indices_and_targets(target_mean, target_std)

    def _init_indices_and_targets(self, target_mean, target_std):
        print(f"  Loading targets for {self.direction}...", flush=True)
        with h5py.File(self.hdf5_path, 'r') as f:
            sources = f['source_type'][:]
            self.indexes_mmu = np.where(sources == 0)[0]
            n = len(self.indexes_mmu)

            n_direct = len(self.target_keys)
            n_derived = len(self.derived_keys)
            n_total = n_direct + n_derived
            self.all_targets = np.zeros((n, n_total), dtype=np.float32)

            for j, key in enumerate(self.target_keys):
                self.all_targets[:, j] = f[key][:][self.indexes_mmu].astype(np.float32)
                print(f"    read {key}", flush=True)

            if self.derived_keys:
                for k, band in enumerate(("g", "i", "r", "z")):
                    s11 = f[f"hsc_{band}_sdssshape_psf_shape11"][:][self.indexes_mmu].astype(np.float64)
                    s22 = f[f"hsc_{band}_sdssshape_psf_shape22"][:][self.indexes_mmu].astype(np.float64)
                    self.all_targets[:, n_direct + k] = compute_hsc_psf_seeing(s11, s22).astype(np.float32)
                    print(f"    derived hsc_{band}_psf_fwhm", flush=True)

        valid_mask = np.isfinite(self.all_targets).all(axis=1)
        bad_count = (~valid_mask).sum()
        if bad_count > 0:
            print(f"  Filtering {bad_count}/{n} samples with non-finite targets.", flush=True)
            self.indexes_mmu = self.indexes_mmu[valid_mask]
            self.all_targets = self.all_targets[valid_mask]
            n = len(self.indexes_mmu)
            print(f"  {n} valid samples remain.", flush=True)

        if target_mean is not None and target_std is not None:
            self.target_mean = target_mean
            self.target_std = target_std
        else:
            self.target_mean = np.mean(self.all_targets, axis=0)
            self.target_std = np.std(self.all_targets, axis=0)
            self.target_std[self.target_std < 1e-8] = 1.0

        self.all_targets_z = (self.all_targets - self.target_mean) / self.target_std
        self.all_target_names = self.target_keys + self.derived_keys
        print(f"  {n_total} targets, {n} samples ready.", flush=True)

    def _open_file(self):
        if self.file is None:
            self.file = h5py.File(self.hdf5_path, 'r', libver='latest', swmr=True)

    def __len__(self):
        return len(self.indexes_mmu)

    def __getitem__(self, idx):
        self._open_file()
        index_mmu = self.indexes_mmu[idx]
        img_raw = self.file[self.image_key][index_mmu]
        img = preprocess_raw_image(img_raw, self.survey, self.crop_size, self.norm_dict)
        img = img[:4]
        targets = torch.from_numpy(self.all_targets_z[idx]).float()
        return img, targets


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CrossPredictResNet(nn.Module):
    def __init__(self, in_channels=4, num_targets=16):
        super().__init__()
        self.backbone = timm.create_model("resnet18", pretrained=False, num_classes=0)

        if in_channels != 3:
            old_conv = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                in_channels, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )

        self.head = nn.Linear(512, num_targets)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        preds = model(imgs)
        loss = criterion(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    total_loss = 0.0
    n_batches = 0
    criterion = nn.MSELoss()
    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        preds = model(imgs)
        total_loss += criterion(preds, targets).item()
        n_batches += 1
        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
    preds_np = np.concatenate(all_preds, axis=0)
    targets_np = np.concatenate(all_targets, axis=0)
    avg_loss = total_loss / max(n_batches, 1)
    r2_per_target = []
    for j in range(preds_np.shape[1]):
        r2_per_target.append(r2_score(targets_np[:, j], preds_np[:, j]))
    return avg_loss, r2_per_target


# ---------------------------------------------------------------------------
# Comparison plot
# ---------------------------------------------------------------------------

def make_comparison_plot(direction, target_names, r2_cross, output_path):
    """Bar chart comparing cross-predict R2 vs model instrument-latent R2."""
    df_baseline = pd.read_csv(BASELINE_CSV)

    if direction == "hsc_to_legacy":
        group_filter = "instrument (legacy)"
        title = "HSC Image -> Legacy Properties"
    else:
        group_filter = "instrument (hsc)"
        title = "Legacy Image -> HSC Properties"

    baseline_r2 = {}
    for _, row in df_baseline.iterrows():
        if row["group"] == group_filter:
            baseline_r2[row["target"]] = row["Instrument latents"]

    targets_plot = []
    r2_model_vals = []
    r2_cross_vals = []
    for i, name in enumerate(target_names):
        if name in baseline_r2:
            targets_plot.append(name)
            r2_model_vals.append(baseline_r2[name])
            r2_cross_vals.append(r2_cross[i])

    n = len(targets_plot)
    if n == 0:
        print("  No matching targets for comparison plot.", flush=True)
        return

    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, n * 0.8), 6))
    ax.bar(x - width / 2, r2_model_vals, width, label="Model (Instrument latents)", color="#2E86AB")
    ax.bar(x + width / 2, r2_cross_vals, width, label="Cross-predict ResNet", color="#CC546D")

    ax.set_ylabel(r"R$^2$ Score")
    ax.set_title(title)
    ax.set_xticks(x)
    short_labels = [t.replace("legacy_", "").replace("hsc_", "").replace("_value", "") for t in targets_plot]
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.set_ylim(-0.1, 1.05)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Comparison plot saved: {output_path}", flush=True)


# ---------------------------------------------------------------------------
# Discord notification
# ---------------------------------------------------------------------------

def send_discord_notification(message, image_path=None):
    cmd = ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}"]
    cmd += ["-F", f"content={message}"]
    if image_path and Path(image_path).exists():
        cmd += ["-F", f"file=@{image_path}"]
    cmd.append(DISCORD_WEBHOOK)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        print(f"  Discord response code: {result.stdout.strip()}", flush=True)
    except Exception as e:
        print(f"  Discord notification failed: {e}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cross-predict instrument properties")
    parser.add_argument("--direction", type=str, required=True,
                        choices=["hsc_to_legacy", "legacy_to_hsc"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--quick-test", action="store_true",
                        help="Limit dataset to 1024 samples for quick testing")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--no-discord", action="store_true")
    parser.add_argument("--output-dir", type=str, default=str(_script_dir))
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # --- W&B ---
    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb
        wandb.init(
            project=WANDB_PROJECT,
            name=f"cross-predict-{args.direction}-ep{args.epochs}",
            config=vars(args),
        )

    # --- Data ---
    print("Loading CrossPredictDataset...", flush=True)
    full_cross_ds = CrossPredictDataset(
        hdf5_path=NEIGHBORS_HDF5, direction=args.direction,
        crop_size=48, norm_dict=NORM_DICT,
    )
    n_total = len(full_cross_ds)

    if args.quick_test:
        n_use = min(1024, n_total)
        subset_indices = list(range(n_use))
    else:
        subset_indices = list(range(n_total))
    n_use = len(subset_indices)

    n_val = int(n_use * VAL_FRAC)
    n_train = n_use - n_val
    generator = torch.Generator().manual_seed(SEED)
    splits = torch.utils.data.random_split(subset_indices, [n_train, n_val], generator=generator)
    train_indices = [subset_indices[i] for i in splits[0].indices]
    val_indices = [subset_indices[i] for i in splits[1].indices]

    train_ds = Subset(full_cross_ds, train_indices)
    val_ds = Subset(full_cross_ds, val_indices)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}", flush=True)

    # --- Model ---
    num_targets = len(full_cross_ds.all_target_names)
    model = CrossPredictResNet(in_channels=4, num_targets=num_targets).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_mb = n_params * 4 / 1024 / 1024  # float32
    print(f"Model: ResNet18 -> {num_targets} targets", flush=True)
    print(f"  Total params:     {n_params:,}", flush=True)
    print(f"  Trainable params: {n_trainable:,}", flush=True)
    print(f"  Model size:       {model_mb:.1f} MB (float32)", flush=True)
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB", flush=True)

    if use_wandb:
        wandb.config.update({
            "total_params": n_params,
            "trainable_params": n_trainable,
            "model_size_mb": round(model_mb, 1),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "num_targets": num_targets,
            "target_names": full_cross_ds.all_target_names,
        })

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    # --- Training loop ---
    best_val_loss = float("inf")
    best_r2 = None
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_r2 = evaluate(model, val_loader, device)
        scheduler.step()

        avg_r2 = np.mean(val_r2)
        lr_now = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.5f}  "
            f"val_loss={val_loss:.5f}  avg_r2={avg_r2:.4f}  lr={lr_now:.2e}",
            flush=True,
        )

        if use_wandb:
            log_dict = {
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/avg_r2": avg_r2,
                "lr": lr_now,
            }
            for j, name in enumerate(full_cross_ds.all_target_names):
                log_dict[f"val_r2/{name}"] = val_r2[j]
            wandb.log(log_dict, step=epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_r2 = list(val_r2)

    # --- Final summary ---
    print("\n=== Final R² (best validation epoch) ===", flush=True)
    target_names = full_cross_ds.all_target_names
    for j, name in enumerate(target_names):
        print(f"  {name:40s}  R²={best_r2[j]:.4f}", flush=True)
    print(f"  {'AVERAGE':40s}  R²={np.mean(best_r2):.4f}", flush=True)

    # Save R² CSV
    out_dir = Path(args.output_dir)
    r2_csv_path = out_dir / f"r2_results_{args.direction}.csv"
    r2_df = pd.DataFrame({"target": target_names, "r2_cross_predict": best_r2})
    r2_df.to_csv(r2_csv_path, index=False)
    print(f"R² results saved: {r2_csv_path}", flush=True)

    # Comparison plot
    plot_path = out_dir / f"comparison_{args.direction}.png"
    make_comparison_plot(args.direction, target_names, best_r2, str(plot_path))

    # Discord notification
    if not args.no_discord:
        avg_r2_str = f"{np.mean(best_r2):.3f}"
        direction_label = "HSC->Legacy" if args.direction == "hsc_to_legacy" else "Legacy->HSC"
        job_id = os.environ.get("SLURM_JOB_ID", "local")
        msg = (
            f"**Cross-predict done** ({direction_label}) | "
            f"Job `{job_id}` | {args.epochs} epochs | "
            f"Avg R²={avg_r2_str}"
        )
        send_discord_notification(msg, str(plot_path))

    if use_wandb:
        if plot_path.exists():
            wandb.log({"comparison_plot": wandb.Image(str(plot_path))})
        wandb.finish()

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
