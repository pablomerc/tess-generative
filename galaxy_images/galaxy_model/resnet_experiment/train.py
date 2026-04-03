"""
Train ResNet-18 to predict galaxy ellipticity (SHAPE_E1, SHAPE_E2) from
4-channel HSC images.

Input:  resnet_data.h5  (produced by prepare_data.py)
Output: resnet_best.pth       — best checkpoint (lowest val loss)
        split_indices.json    — train/test indices for reproducibility
        outputs/train_val_loss.png
        outputs/test_scatter_e1.png
        outputs/test_scatter_e2.png

Architecture:
  Standard torchvision ResNet-18 with:
  - conv1 modified: 3 → 4 input channels (HSC G,R,I,Z)
  - fc modified: 512 → 2 outputs (e1, e2)
  48x48 spatial input flows through the network normally; global average pooling
  before fc still produces a 512-dim vector.

Training:
  Loss: MSE(predicted_e1e2, true_e1e2)
  Optimizer: Adam, lr=1e-4
  Schedule: ReduceLROnPlateau (patience=5, factor=0.5)
  Split: 80% train / 20% test, seed=42
  Val check: done on 10% of the training set each epoch
  Best model: tracked by val loss

Usage:
  python train.py
  python train.py --epochs 100 --batch-size 256
"""

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import torchvision.models as tv_models

_here = Path(__file__).resolve().parent

DATA_H5     = _here / "resnet_data.h5"
BEST_CKPT   = _here / "resnet_best.pth"
SPLIT_JSON  = _here / "split_indices.json"
OUTPUT_DIR  = _here / "outputs"

# On-the-fly normalization constants (from data.py / NORM_DICT['hsc'])
NORM_MEAN = 0.022
NORM_STD  = 0.05

# Train/test split
TRAIN_FRAC = 0.80
SEED       = 42

# Training defaults (can be overridden via CLI)
DEFAULT_EPOCHS      = 80
DEFAULT_BATCH_SIZE  = 256
DEFAULT_LR          = 1e-4
DEFAULT_NUM_WORKERS = 4


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HSCEllipticityDataset(Dataset):
    """
    Loads resnet_data.h5 entirely into RAM (images + labels).
    Returns (image, label) where:
      image  : (4, 48, 48) float32 tensor, normalized
      label  : (2,)         float32 tensor [SHAPE_E1, SHAPE_E2]
    """

    def __init__(self, h5_path: str):
        print(f"  Loading all data into RAM …", end=" ", flush=True)
        with h5py.File(str(h5_path), "r") as f:
            self._images = f["hsc_images"][:]   # (N, 4, 48, 48) float32
            self._e1     = f["shape_e1"][:]     # (N,)
            self._e2     = f["shape_e2"][:]     # (N,)
        self._images = (self._images - NORM_MEAN) / NORM_STD
        print(f"{len(self._images):,} images, "
              f"{self._images.nbytes / 1e9:.1f} GB loaded.")

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        img   = torch.from_numpy(self._images[idx])
        label = torch.tensor([self._e1[idx], self._e2[idx]], dtype=torch.float32)
        return img, label


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_resnet18() -> nn.Module:
    """
    ResNet-18 modified for 4-channel 48x48 input, 2-output regression.

    Changes from standard ResNet-18:
    1. conv1: 3 → 4 input channels (same kernel/stride/padding)
    2. fc:    512 → 2 outputs (direct regression, no softmax)

    Why this works at 48x48:
    Standard ResNet-18 has:
      conv1(stride=2) → 24x24
      maxpool(stride=2) → 12x12
      layer1 → 12x12   (no downsampling)
      layer2 → 6x6
      layer3 → 3x3
      layer4 → 2x2
      global avg pool → 512-dim
    All spatial sizes are non-zero and the global average pool handles any size.
    """
    model = tv_models.resnet18(weights=None)
    # 1. Replace first conv: 3 → 4 channels
    model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # 2. Replace FC: 512 → 2
    model.fc = nn.Linear(512, 2)
    return model


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def r2_score(pred: torch.Tensor, true: torch.Tensor) -> float:
    """R² for a single output column."""
    ss_res = ((true - pred) ** 2).sum().item()
    ss_tot = ((true - true.mean()) ** 2).sum().item()
    return 1.0 - ss_res / (ss_tot + 1e-12)


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    n_batches  = 0
    all_pred   = []
    all_true   = []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        pred = model(imgs)
        total_loss += criterion(pred, labels).item()
        n_batches  += 1
        all_pred.append(pred.cpu())
        all_true.append(labels.cpu())
    avg_loss = total_loss / max(n_batches, 1)
    all_pred = torch.cat(all_pred, dim=0)
    all_true = torch.cat(all_true, dim=0)
    return avg_loss, all_pred, all_true


def make_scatter_plots(pred_np, true_np, output_dir: Path, tag: str):
    """Scatter plots of predicted vs true e1 and e2, one figure each."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    for i, name in enumerate(["e1", "e2"]):
        p = pred_np[:, i]
        t = true_np[:, i]
        r2 = 1.0 - np.sum((t - p) ** 2) / (np.sum((t - t.mean()) ** 2) + 1e-12)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(t, p, s=2, alpha=0.3, rasterized=True)
        lim = max(abs(t).max(), abs(p).max()) * 1.05
        ax.plot([-lim, lim], [-lim, lim], "r--", lw=1, label="y=x")
        ax.set_xlabel(f"True SHAPE_{name.upper()}")
        ax.set_ylabel(f"Predicted SHAPE_{name.upper()}")
        ax.set_title(f"{tag} — SHAPE_{name.upper()}  R²={r2:.4f}")
        ax.legend(fontsize=8)
        plt.tight_layout()
        out_path = output_dir / f"{tag}_scatter_{name}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out_path}  (R²={r2:.4f})")


def make_loss_plot(train_losses, val_losses, output_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, label="train loss")
    ax.plot(epochs, val_losses,   label="val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("ResNet-18 Ellipticity Training")
    ax.legend()
    ax.set_yscale("log")
    plt.tight_layout()
    out_path = output_dir / "train_val_loss.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading dataset from {DATA_H5} …")
    dataset = HSCEllipticityDataset(DATA_H5)
    N = len(dataset)
    print(f"  Total examples: {N:,}")

    # --- Train / test split (80/20) -----------------------------------------
    n_test  = int(N * (1.0 - TRAIN_FRAC))
    n_train = N - n_test

    gen = torch.Generator().manual_seed(SEED)
    train_ds, test_ds = random_split(dataset, [n_train, n_test], generator=gen)

    # Save indices for Stage 2 reproducibility
    train_indices = sorted(train_ds.indices)
    test_indices  = sorted(test_ds.indices)
    with open(SPLIT_JSON, "w") as f:
        json.dump({"train": train_indices, "test": test_indices, "seed": SEED}, f)
    print(f"  Train: {len(train_indices):,}  Test: {len(test_indices):,}")
    print(f"  Split indices saved to {SPLIT_JSON}")

    # --- DataLoaders ----------------------------------------------------------
    # Hold out 10% of train as validation
    n_val   = int(len(train_ds) * 0.10)
    n_tr    = len(train_ds) - n_val
    gen2    = torch.Generator().manual_seed(SEED + 1)
    tr_ds, val_ds = random_split(train_ds, [n_tr, n_val], generator=gen2)

    loader_kwargs = dict(
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        pin_memory  = True,
    )
    train_loader = DataLoader(tr_ds,  shuffle=True,  drop_last=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds, shuffle=False, drop_last=False, **loader_kwargs)

    # --- Model ----------------------------------------------------------------
    print("Building ResNet-18 (4-channel input, 2 outputs) …")
    model = build_resnet18().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5, min_lr=1e-6
    )

    # --- Training loop --------------------------------------------------------
    best_val_loss   = float("inf")
    train_losses    = []
    val_losses_list = []

    print(f"\nStarting training: {args.epochs} epochs …")
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0     = time.time()
        ep_loss = 0.0
        n_steps = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            pred = model(imgs)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            n_steps += 1

        train_loss = ep_loss / max(n_steps, 1)
        val_loss, _, _ = evaluate(model, val_loader, device, criterion)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses_list.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(),
                 "val_loss": val_loss, "norm_mean": NORM_MEAN, "norm_std": NORM_STD},
                BEST_CKPT,
            )
            ckpt_tag = " ← best"
        else:
            ckpt_tag = ""

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch {epoch:3d}/{args.epochs}  "
            f"train={train_loss:.5f}  val={val_loss:.5f}  "
            f"lr={lr_now:.2e}  {elapsed:.1f}s{ckpt_tag}"
        )

    # --- Plots ----------------------------------------------------------------
    print("\nSaving training curves …")
    make_loss_plot(train_losses, val_losses_list, OUTPUT_DIR)

    # --- Final evaluation on test set -----------------------------------------
    print("\nLoading best checkpoint for test evaluation …")
    ckpt = torch.load(BEST_CKPT, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"  Best epoch: {ckpt['epoch']}  val_loss: {ckpt['val_loss']:.5f}")

    print("Evaluating on test set …")
    test_loss, pred_t, true_t = evaluate(model, test_loader, device, criterion)
    pred_np = pred_t.numpy()
    true_np = true_t.numpy()

    r2_e1 = 1.0 - np.sum((true_np[:,0] - pred_np[:,0])**2) / (np.sum((true_np[:,0] - true_np[:,0].mean())**2) + 1e-12)
    r2_e2 = 1.0 - np.sum((true_np[:,1] - pred_np[:,1])**2) / (np.sum((true_np[:,1] - true_np[:,1].mean())**2) + 1e-12)

    print(f"\n  Test MSE: {test_loss:.5f}")
    print(f"  R²  e1:   {r2_e1:.4f}")
    print(f"  R²  e2:   {r2_e2:.4f}")

    make_scatter_plots(pred_np, true_np, OUTPUT_DIR, tag="test")

    print(f"\nDone.  Best checkpoint: {BEST_CKPT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet-18 for galaxy ellipticity")
    parser.add_argument("--epochs",      type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size",  type=int,   default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr",          type=float, default=DEFAULT_LR)
    parser.add_argument("--num-workers", type=int,   default=DEFAULT_NUM_WORKERS)
    args = parser.parse_args()
    main(args)
