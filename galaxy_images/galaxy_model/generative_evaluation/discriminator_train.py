"""Step 2: Train per-survey discriminators on pre-generated data.

Loads the .npz files produced by discriminator_generate_dataset.py
and trains a lightweight CNN to distinguish real from generated images,
separately for HSC and Legacy.

Usage:
  python discriminator_train.py --n-samples 1000
  python discriminator_train.py --n-samples 10000

Reads from disc_data_{tag}/ and writes to figures_discriminator_{tag}/.

An ideal generative model yields discriminator accuracy ~50% (chance).
"""

import sys
from pathlib import Path
import argparse

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import os
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")

import matplotlib
matplotlib.use("Agg")

import json
import torch
torch.backends.cuda.preferred_blas_library("hipblas")
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support, accuracy_score

# ======================== CONFIG ========================

DISC_LR = 1e-4
DISC_EPOCHS = 30
DISC_BATCH_SIZE = 64
TRAIN_FRAC = 0.7
VAL_FRAC = 0.15

IN_CHANNELS = 4
CHANNEL_NAMES = ['g', 'r', 'i', 'z']


def _make_tag(n):
    if n >= 1000:
        return f"{n // 1000}k"
    return str(n)


# ======================== DISCRIMINATOR ========================

class PatchDiscriminator(nn.Module):
    """Lightweight PatchGAN-style discriminator for 48x48 x 4-channel images."""

    def __init__(self, in_channels=4, base_filters=32):
        super().__init__()
        def _block(in_c, out_c, stride=2, norm=True):
            layers = [nn.Conv2d(in_c, out_c, 4, stride, 1, bias=not norm)]
            if norm:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        nf = base_filters
        self.features = nn.Sequential(
            _block(in_channels, nf, stride=2, norm=False),  # 48 -> 24
            _block(nf, nf * 2, stride=2),                   # 24 -> 12
            _block(nf * 2, nf * 4, stride=2),               # 12 -> 6
            _block(nf * 4, nf * 8, stride=2),               # 6  -> 3
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(nf * 8, 1),
        )

    def forward(self, x):
        return self.head(self.features(x)).squeeze(-1)


# ======================== DATASET ========================

class RealVsGenDataset(Dataset):
    """Binary classification dataset from stacked numpy arrays."""

    def __init__(self, real: np.ndarray, gen: np.ndarray):
        n = min(len(real), len(gen))
        self.images = np.concatenate([real[:n], gen[:n]], axis=0)
        self.labels = np.concatenate([np.ones(n), np.zeros(n)]).astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.from_numpy(self.images[idx]).float(), self.labels[idx]


# ======================== TRAINING ========================

def train_discriminator(real, gen, device):
    """Train a discriminator and return (model, test_metrics, test_preds, test_labels, history)."""
    dataset = RealVsGenDataset(real, gen)
    n_total = len(dataset)
    n_train = int(n_total * TRAIN_FRAC)
    n_val = int(n_total * VAL_FRAC)
    n_test = n_total - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=DISC_BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=DISC_BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=DISC_BATCH_SIZE, shuffle=False,
                             num_workers=4, pin_memory=True)

    disc = PatchDiscriminator(in_channels=IN_CHANNELS).to(device)
    opt = torch.optim.Adam(disc.parameters(), lr=DISC_LR, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=DISC_EPOCHS)
    criterion = nn.BCEWithLogitsLoss()

    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')
    best_state = None

    for epoch in range(DISC_EPOCHS):
        disc.train()
        train_losses = []
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = disc(imgs)
            loss = criterion(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        disc.eval()
        val_losses, val_preds, val_labels = [], [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = disc(imgs)
                val_losses.append(criterion(logits, labels).item())
                val_preds.append(torch.sigmoid(logits).cpu())
                val_labels.append(labels.cpu())

        scheduler.step()

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        val_preds_cat = torch.cat(val_preds).numpy()
        val_labels_cat = torch.cat(val_labels).numpy()
        val_acc = accuracy_score(val_labels_cat, (val_preds_cat > 0.5).astype(int))

        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        history['val_acc'].append(float(val_acc))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in disc.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{DISC_EPOCHS}  "
                  f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")

    disc.load_state_dict(best_state)
    disc.to(device).eval()

    test_preds, test_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = disc(imgs)
            test_preds.append(torch.sigmoid(logits).cpu())
            test_labels.append(labels.cpu())

    test_preds = torch.cat(test_preds).numpy()
    test_labels = torch.cat(test_labels).numpy()

    test_acc = accuracy_score(test_labels, (test_preds > 0.5).astype(int))
    test_auc = roc_auc_score(test_labels, test_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        test_labels, (test_preds > 0.5).astype(int), average='binary', pos_label=1)

    metrics = {
        'accuracy': float(test_acc),
        'auc_roc': float(test_auc),
        'precision_real': float(prec),
        'recall_real': float(rec),
        'f1_real': float(f1),
        'n_train': int(n_train),
        'n_val': int(n_val),
        'n_test': int(n_test),
    }
    return disc, metrics, test_preds, test_labels, history


# ======================== PLOTTING ========================

def _make_rgb(chw):
    rgb = chw[:3].copy()
    for c in range(3):
        lo, hi = np.percentile(rgb[c], (2, 98))
        if hi > lo:
            rgb[c] = (rgb[c] - lo) / (hi - lo)
        else:
            rgb[c] = 0.0
    return np.clip(rgb.transpose(1, 2, 0), 0, 1)


def plot_roc_curves(metrics_hsc, preds_hsc, labels_hsc,
                    metrics_leg, preds_leg, labels_leg, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, preds, labels, survey, metrics in [
        (axes[0], preds_hsc, labels_hsc, 'HSC', metrics_hsc),
        (axes[1], preds_leg, labels_leg, 'Legacy', metrics_leg),
    ]:
        fpr, tpr, _ = roc_curve(labels, preds)
        ax.plot(fpr, tpr, lw=2, label=f"AUC = {metrics['auc_roc']:.3f}")
        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{survey} — Real vs Generated\n'
                     f"Acc={metrics['accuracy']:.3f}  |  "
                     f"Prec={metrics['precision_real']:.3f}  Rec={metrics['recall_real']:.3f}")
        ax.legend(loc='lower right', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    plt.tight_layout()
    path = output_dir / 'discriminator_roc_curves.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path.name}")


def plot_training_history(history_hsc, history_leg, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for col, (history, survey) in enumerate(
            [(history_hsc, 'HSC'), (history_leg, 'Legacy')]):
        epochs = range(1, len(history['train_loss']) + 1)

        axes[0, col].plot(epochs, history['train_loss'], label='Train', lw=1.5)
        axes[0, col].plot(epochs, history['val_loss'], label='Val', lw=1.5)
        axes[0, col].set_ylabel('BCE Loss')
        axes[0, col].set_title(f'{survey} — Loss')
        axes[0, col].legend()
        axes[0, col].grid(True, alpha=0.3)

        axes[1, col].plot(epochs, history['val_acc'], color='green', lw=1.5)
        axes[1, col].axhline(0.5, color='gray', ls='--', alpha=0.5, label='Chance')
        axes[1, col].set_ylabel('Validation Accuracy')
        axes[1, col].set_xlabel('Epoch')
        axes[1, col].set_title(f'{survey} — Accuracy')
        axes[1, col].legend()
        axes[1, col].grid(True, alpha=0.3)
        axes[1, col].set_ylim(0.4, 1.0)

    plt.tight_layout()
    path = output_dir / 'discriminator_training_history.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path.name}")


def plot_example_panels(real, gen, survey, output_dir, n_show=8):
    """Side-by-side panels of real vs generated images (RGB composite)."""
    n_show = min(n_show, len(real), len(gen))
    fig, axes = plt.subplots(2, n_show, figsize=(2.2 * n_show, 5))

    for j in range(n_show):
        axes[0, j].imshow(_make_rgb(real[j]), origin='lower')
        axes[0, j].set_xticks([])
        axes[0, j].set_yticks([])
        if j == 0:
            axes[0, j].set_ylabel('Real', fontsize=12, fontweight='bold')

        axes[1, j].imshow(_make_rgb(gen[j]), origin='lower')
        axes[1, j].set_xticks([])
        axes[1, j].set_yticks([])
        if j == 0:
            axes[1, j].set_ylabel('Generated', fontsize=12, fontweight='bold')

    fig.suptitle(f'{survey.upper()} — Real vs Generated samples', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = output_dir / f'discriminator_examples_{survey}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path.name}")


# ======================== MAIN ========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-samples', type=int, default=1000,
                        help='Must match the --n-samples used in discriminator_generate_dataset.py')
    args = parser.parse_args()

    tag = _make_tag(args.n_samples)
    data_dir = _script_dir / f'disc_data_{tag}'
    output_dir = _script_dir / f'figures_discriminator_{tag}'
    output_dir.mkdir(exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Tag: {tag}  |  data_dir: {data_dir}  |  output_dir: {output_dir}")

    all_metrics = {}
    all_preds = {}
    all_labels = {}
    all_histories = {}

    for survey in ['hsc', 'legacy']:
        npz_path = data_dir / f'disc_dataset_{survey}.npz'
        if not npz_path.exists():
            print(f"WARNING: {npz_path} not found — skipping {survey.upper()}.")
            print(f"  Run discriminator_generate_dataset.py --n-samples {args.n_samples} first.")
            continue

        print(f"\nLoading {npz_path.name} …")
        data = np.load(npz_path)
        real, gen = data['real'], data['gen']
        print(f"  real: {real.shape}  gen: {gen.shape}")

        plot_example_panels(real, gen, survey, output_dir)

        print(f"\n{'='*60}")
        print(f"Training discriminator: real {survey.upper()} vs generated {survey.upper()}")
        print(f"  {len(real)} real + {len(gen)} generated = {len(real)+len(gen)} total")
        print(f"{'='*60}")

        disc, metrics, preds, labels, history = train_discriminator(real, gen, device)

        all_metrics[survey] = metrics
        all_preds[survey] = preds
        all_labels[survey] = labels
        all_histories[survey] = history

        print(f"\n  {survey.upper()} test results:")
        print(f"    Accuracy:  {metrics['accuracy']:.4f}")
        print(f"    AUC-ROC:   {metrics['auc_roc']:.4f}")
        print(f"    Precision: {metrics['precision_real']:.4f}")
        print(f"    Recall:    {metrics['recall_real']:.4f}")
        print(f"    F1:        {metrics['f1_real']:.4f}")

    if 'hsc' in all_metrics and 'legacy' in all_metrics:
        plot_roc_curves(
            all_metrics['hsc'], all_preds['hsc'], all_labels['hsc'],
            all_metrics['legacy'], all_preds['legacy'], all_labels['legacy'],
            output_dir)
        plot_training_history(all_histories['hsc'], all_histories['legacy'], output_dir)

    metrics_path = output_dir / 'discriminator_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    print(f"\n{'='*60}")
    print("DISCRIMINATOR EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Survey':<10} {'Accuracy':>10} {'AUC-ROC':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 62)
    for survey in ['hsc', 'legacy']:
        if survey in all_metrics:
            m = all_metrics[survey]
            print(f"{survey.upper():<10} {m['accuracy']:>10.4f} {m['auc_roc']:>10.4f} "
                  f"{m['precision_real']:>10.4f} {m['recall_real']:>10.4f} {m['f1_real']:>10.4f}")
    print(f"\nAccuracy ~0.50 = indistinguishable (ideal)")
    print(f"Accuracy ~1.00 = easily distinguishable (poor generation)")
    print(f"\nAll figures saved to {output_dir}")


if __name__ == '__main__':
    main()
