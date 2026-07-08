"""
Stage 3 — train MLP probes across (task_family × target × latent_variant) for one
checkpoint's H5 (produced by prepare_combined.py) and write a long-form CSV.

CLI shape (locked):

    python predict_combined.py \
        --embeddings outputs/embeddings/<ckpt_name>.h5 \
        --out        outputs/predictions/predict_<ckpt_name>.csv \
        [--device cuda] [--max-epochs 50] [--seed 0]

CSV columns:
    checkpoint, task_family, target, latent_variant, score, score_metric,
    n_train, n_test

Tasks (allowed latent variants per task family):

    physics_provabgs    overlap                  hsc_e1 / legacy_e1 / combined_e1   R²
    morphology_gz10     overlap+hsc_only+legacy_only (where label exists)
                        hsc_e1 / legacy_e1 / combined_e1                            accuracy + macro-F1
    instrument_hsc      overlap.hsc + hsc_only   hsc_e2                             R²
    instrument_legacy   overlap.legacy + legacy_only  legacy_e2                     R²
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ROCm workaround (harmless on H100/CUDA).
try:
    import torch
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "preferred_blas_library"):
        torch.backends.cuda.preferred_blas_library("hipblas")
except Exception:
    pass

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score, accuracy_score, f1_score


MLP_HIDDEN = 256
TRAIN_FRAC = 0.9
BATCH_SIZE = 64
MIN_LABELS = 50

PHYSICS_TARGETS = [
    "desi_z",
    "provabgs_logmstar",
    "provabgs_tage_mw",
    "provabgs_z_mw",
    "ssfr",
    "legacy_shape_e1", "legacy_shape_e2",
    "hsc_shape_e1_i",  "hsc_shape_e2_i",
]

HSC_INSTRUMENT_TARGETS = [
    "a_g", "a_r", "a_i", "a_z", "a_y",
    "psf_fwhm_g", "psf_fwhm_r", "psf_fwhm_i", "psf_fwhm_z",
]

LEGACY_INSTRUMENT_TARGETS = [
    "PSFSIZE_G", "PSFSIZE_R", "PSFSIZE_I", "PSFSIZE_Z",
    "PSFDEPTH_G", "PSFDEPTH_R", "PSFDEPTH_I", "PSFDEPTH_Z",
    "GALDEPTH_G", "GALDEPTH_R", "GALDEPTH_I", "GALDEPTH_Z",
    "MW_TRANSMISSION_G", "MW_TRANSMISSION_R", "MW_TRANSMISSION_I", "MW_TRANSMISSION_Z",
    "EBV",
    "NOBS_G", "NOBS_R", "NOBS_I", "NOBS_Z",
]


# ---------------------------------------------------------------------------
# H5 loading
# ---------------------------------------------------------------------------

class CkptH5:
    """Thin reader that knows the prepare_combined.py layout."""

    def __init__(self, path: Path):
        self.path = path
        self.f = h5py.File(path, "r")
        self.checkpoint_name = self.f.attrs.get("checkpoint_name", path.stem)
        if isinstance(self.checkpoint_name, bytes):
            self.checkpoint_name = self.checkpoint_name.decode()
        self.has_e2 = bool(self.f.attrs.get("has_e2", True))
        self.architecture = self.f.attrs.get("architecture", "?")
        if isinstance(self.architecture, bytes):
            self.architecture = self.architecture.decode()

    def close(self):
        self.f.close()

    def has_subset(self, name: str) -> bool:
        return name in self.f

    def latents(self, subset: str, key: str) -> np.ndarray:
        return np.asarray(self.f[f"{subset}/{key}"][:])

    def label(self, subset: str, col: str) -> np.ndarray | None:
        ds = f"{subset}/labels/{col}"
        return np.asarray(self.f[ds][:]) if ds in self.f else None

    def label_keys(self, subset: str) -> list[str]:
        if subset not in self.f or "labels" not in self.f[subset]:
            return []
        return list(self.f[subset]["labels"].keys())


# ---------------------------------------------------------------------------
# MLP / Lightning
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Two-layer MLP per the AION paper text. `arch` switches between the
    strict reading (`aion`: Linear→GELU→Linear) and a regularized variant
    (`aion_reg`: adds LayerNorm + Dropout)."""

    def __init__(self, in_dim: int, out_dim: int, arch: str = "aion",
                 hidden: int = MLP_HIDDEN, dropout: float = 0.2):
        super().__init__()
        if arch == "aion":
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, out_dim),
            )
        elif arch == "aion_reg":
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, out_dim),
            )
        else:
            raise ValueError(f"unknown mlp arch: {arch!r}")

    def forward(self, x):
        return self.net(x)


class LitProbe(pl.LightningModule):
    def __init__(self, in_dim: int, out_dim: int, lr: float = 1e-3, weight_decay: float = 1e-2,
                 mode: str = "regression", n_classes: int | None = None,
                 mlp_arch: str = "aion"):
        super().__init__()
        self.save_hyperparameters()
        self.model = MLP(in_dim=in_dim, out_dim=out_dim, arch=mlp_arch)
        if mode == "regression":
            # element-wise so we can mask NaN targets and average over valid entries.
            self.loss_fn = nn.SmoothL1Loss(beta=1.0, reduction="none")
        else:
            self.loss_fn = nn.CrossEntropyLoss()
        self.mode = mode

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, stage: str):
        if self.mode == "regression":
            x, y, mask = batch
            y_hat = self(x)
            elem = self.loss_fn(y_hat, y) * mask
            denom = mask.sum().clamp(min=1.0)
            loss = elem.sum() / denom
        else:
            x, y = batch
            y_hat = self(x)
            loss = self.loss_fn(y_hat, y.long())
        self.log(f"{stage}/loss", loss, on_epoch=True, prog_bar=False, batch_size=x.size(0))
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def configure_optimizers(self):
        opt = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(self.trainer.max_epochs, 1))
        return {"optimizer": opt, "lr_scheduler": sched}

    def on_before_optimizer_step(self, optimizer):
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)


def _parse_best_epoch(best_path: str) -> int:
    """Lightning encodes `epoch=N` in the best-checkpoint filename."""
    if not best_path:
        return -1
    import re
    m = re.search(r"epoch=(\d+)", best_path)
    return int(m.group(1)) if m else -1


def _train_classification(X: np.ndarray, y: np.ndarray, max_epochs: int, seed: int, device: str,
                          n_classes: int, mlp_arch: str):
    """Single-target classification probe (for morphology_gz10).
    Returns list of (score, metric, n_train, n_test, conv_dict) tuples."""
    pl.seed_everything(seed, workers=True)
    rng = np.random.default_rng(seed)
    idx = np.arange(X.shape[0])
    rng.shuffle(idx)
    n_tr = int(TRAIN_FRAC * len(idx))
    tr, va = idx[:n_tr], idx[n_tr:]

    X_t = torch.tensor(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_t[tr], y_t[tr]), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(TensorDataset(X_t[va], y_t[va]), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    use_gpu = (device == "cuda" and torch.cuda.is_available())
    probe = LitProbe(in_dim=X.shape[1], out_dim=int(n_classes),
                     mode="classification", n_classes=n_classes, mlp_arch=mlp_arch)
    ckpt = ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1, save_last=False)
    es = EarlyStopping(monitor="val/loss", mode="min", patience=5)
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        callbacks=[ckpt, es],
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

    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to("cuda" if use_gpu else "cpu")
            preds.append(best(xb).cpu().numpy())
            targets.append(yb.numpy())
    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    pred_cls = preds.argmax(axis=1)
    acc = accuracy_score(targets, pred_cls)
    f1 = f1_score(targets, pred_cls, average="macro", zero_division=0)

    conv = dict(
        best_epoch=_parse_best_epoch(best_path),
        epochs_run=int(trainer.current_epoch + 1),
        max_epochs=int(max_epochs),
        early_stopped=bool(es.stopped_epoch > 0),
    )
    return [
        (acc, "accuracy", len(tr), len(va), conv),
        (f1,  "f1_macro", len(tr), len(va), conv),
    ]


def _train_family(X: np.ndarray, Y_dict: dict, target_names: list[str],
                  max_epochs: int, seed: int, device: str, mlp_arch: str):
    """Multi-target masked regression probe — one MLP for the whole task family.

    Returns a list of per-target dicts:
        target, score, score_metric, n_train, n_test,
        best_epoch, epochs_run, max_epochs, early_stopped
    """
    pl.seed_everything(seed, workers=True)

    K = len(target_names)
    N = X.shape[0]
    Y = np.full((N, K), np.nan, dtype=np.float32)
    for j, t in enumerate(target_names):
        Y[:, j] = np.asarray(Y_dict[t], dtype=np.float32)

    valid_mask_full = np.isfinite(Y)
    keep = valid_mask_full.any(axis=1)
    X_keep = X[keep]
    Y_keep = Y[keep]
    valid_mask = valid_mask_full[keep]
    n_total = X_keep.shape[0]
    if n_total < MIN_LABELS:
        return []

    rng = np.random.default_rng(seed)
    idx = np.arange(n_total)
    rng.shuffle(idx)
    n_tr = int(TRAIN_FRAC * n_total)
    tr, va = idx[:n_tr], idx[n_tr:]

    # Per-column standardization on finite training entries only.
    mu = np.zeros(K, dtype=np.float32)
    sd = np.ones(K, dtype=np.float32)
    for j in range(K):
        v = Y_keep[tr, j]
        m = valid_mask[tr, j]
        if m.sum() >= 2:
            mu[j] = float(v[m].mean())
            sj = float(v[m].std())
            sd[j] = sj if sj >= 1e-8 else 1.0

    Y_norm = (Y_keep - mu[None, :]) / sd[None, :]
    Y_norm = np.where(valid_mask, Y_norm, 0.0).astype(np.float32)

    X_t = torch.tensor(np.nan_to_num(X_keep, nan=0.0, posinf=0.0, neginf=0.0), dtype=torch.float32)
    Y_t = torch.tensor(Y_norm, dtype=torch.float32)
    M_t = torch.tensor(valid_mask.astype(np.float32), dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_t[tr], Y_t[tr], M_t[tr]),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(TensorDataset(X_t[va], Y_t[va], M_t[va]),
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    use_gpu = (device == "cuda" and torch.cuda.is_available())
    probe = LitProbe(in_dim=X_keep.shape[1], out_dim=K,
                     mode="regression", mlp_arch=mlp_arch)
    ckpt = ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1, save_last=False)
    es = EarlyStopping(monitor="val/loss", mode="min", patience=5)
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        callbacks=[ckpt, es],
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

    preds_chunks, mask_chunks = [], []
    with torch.no_grad():
        for xb, _yb, mb in val_loader:
            xb = xb.to("cuda" if use_gpu else "cpu")
            preds_chunks.append(best(xb).cpu().numpy())
            mask_chunks.append(mb.numpy())
    preds = np.concatenate(preds_chunks, axis=0)
    masks_va = np.concatenate(mask_chunks, axis=0)
    preds = preds * sd[None, :] + mu[None, :]
    targets_real = Y_keep[va]

    epochs_run = int(trainer.current_epoch + 1)
    early_stopped = bool(es.stopped_epoch > 0)
    best_epoch = _parse_best_epoch(best_path)

    out = []
    for j, t in enumerate(target_names):
        m = masks_va[:, j].astype(bool)
        n_test_j = int(m.sum())
        n_train_j = int(valid_mask[tr, j].sum())
        if n_test_j < 2 or n_train_j < MIN_LABELS:
            score = float("nan")
        else:
            yt = targets_real[m, j]
            yp = preds[m, j]
            score = float("nan") if np.std(yt) < 1e-6 else float(r2_score(yt, yp))
        out.append(dict(
            target=t, score=score, score_metric="r2",
            n_train=n_train_j, n_test=n_test_j,
            best_epoch=best_epoch, epochs_run=epochs_run,
            max_epochs=int(max_epochs), early_stopped=early_stopped,
        ))
    return out


# ---------------------------------------------------------------------------
# Probe iteration
# ---------------------------------------------------------------------------

def _physics_label(h5: CkptH5, subset: str, target: str) -> np.ndarray | None:
    """Return the requested physics target from `subset`. Computes ssfr and
    HSC i-band ellipticity on the fly; reads Legacy SHAPE_E1/E2 directly."""
    if target == "ssfr":
        for prefix in ("hsc_", "legacy_"):
            sf = h5.label(subset, f"{prefix}provabgs_avg_sfr")
            ms = h5.label(subset, f"{prefix}provabgs_logmstar")
            if sf is not None and ms is not None:
                ratio = sf / np.power(10.0, ms)
                with np.errstate(invalid="ignore", divide="ignore"):
                    return np.where(ratio > 0, np.log10(ratio), np.nan).astype(np.float32)
        return None

    if target == "legacy_shape_e1":
        return h5.label(subset, "legacy_SHAPE_E1")
    if target == "legacy_shape_e2":
        return h5.label(subset, "legacy_SHAPE_E2")

    if target in ("hsc_shape_e1_i", "hsc_shape_e2_i"):
        S11 = h5.label(subset, "hsc_i_sdssshape_shape11")
        S22 = h5.label(subset, "hsc_i_sdssshape_shape22")
        if S11 is None or S22 is None:
            return None
        with np.errstate(invalid="ignore", divide="ignore"):
            trace = S11 + S22
            if target == "hsc_shape_e1_i":
                return np.where(trace > 0, (S11 - S22) / trace, np.nan).astype(np.float32)
            S12 = h5.label(subset, "hsc_i_sdssshape_shape12")
            if S12 is None:
                return None
            return np.where(trace > 0, 2.0 * S12 / trace, np.nan).astype(np.float32)

    # Try both prefixes (overlap has both, hsc_only has only hsc_, legacy_only has only legacy_).
    for prefix in ("hsc_", "legacy_"):
        v = h5.label(subset, f"{prefix}{target}")
        if v is not None:
            return v
    return None


def _gz10_label(h5: CkptH5, subset: str) -> np.ndarray | None:
    for prefix in ("hsc_", "legacy_"):
        v = h5.label(subset, f"{prefix}gz10_label")
        if v is not None:
            return v
    return None


def _stack_subset_latents(h5: CkptH5, subsets_keys: list[tuple[str, str]]) -> np.ndarray | None:
    """Concatenate latents across (subset, key) pairs."""
    chunks = []
    for s, k in subsets_keys:
        if not h5.has_subset(s):
            return None
        if f"{s}/{k}" not in h5.f:
            return None
        chunks.append(h5.latents(s, k))
    if not chunks:
        return None
    return np.concatenate(chunks, axis=0)


def _stack_subset_labels(h5: CkptH5, subsets_keys: list[tuple[str, str | None]],
                        label_resolver) -> np.ndarray | None:
    """Concatenate labels using a resolver fn (subset -> 1-d array)."""
    chunks = []
    for s, _k in subsets_keys:
        v = label_resolver(h5, s)
        if v is None:
            return None
        chunks.append(v)
    return np.concatenate(chunks, axis=0)


def _physics_resolver(target: str):
    return lambda h5, subset: _physics_label(h5, subset, target)


def _hsc_label_resolver(target: str):
    return lambda h5, subset: h5.label(subset, f"hsc_{target}")


def _legacy_label_resolver(target: str):
    return lambda h5, subset: h5.label(subset, f"legacy_{target}")


def _filter_finite(X: np.ndarray, y: np.ndarray):
    mask = np.isfinite(y)
    if mask.ndim > 1:
        mask = mask.all(axis=1)
    return X[mask], y[mask]


def _run_family(h5, rows, ck, task_family, targets, variants, label_resolver_factory):
    """One multi-target MLP per (variant, task_family). Per-target R² extracted
    from the same probe."""
    for variant, subset_keys in variants:
        X = _stack_subset_latents(h5, subset_keys)
        if X is None:
            continue
        Y_dict, target_names = {}, []
        for target in targets:
            y = _stack_subset_labels(h5, subset_keys, label_resolver_factory(target))
            if y is None:
                continue
            Y_dict[target] = y
            target_names.append(target)
        if not target_names:
            continue
        tag = "physics" if task_family == "physics_provabgs" else \
              "hsc-inst" if task_family == "instrument_hsc" else \
              "legacy-inst" if task_family == "instrument_legacy" else task_family
        print(f"  [{tag}] family probe :: {variant}  ({len(target_names)} targets, n={X.shape[0]})")
        results = _train_family(X, Y_dict, target_names,
                                max_epochs=h5._max_epochs, seed=h5._seed,
                                device=h5._device, mlp_arch=h5._mlp_arch)
        for r in results:
            rows.append(dict(
                checkpoint=ck, task_family=task_family,
                target=r["target"], latent_variant=variant,
                score=r["score"], score_metric=r["score_metric"],
                n_train=r["n_train"], n_test=r["n_test"],
                best_epoch=r["best_epoch"], epochs_run=r["epochs_run"],
                max_epochs=r["max_epochs"], early_stopped=r["early_stopped"],
            ))


def run_all(h5: CkptH5, max_epochs: int, seed: int, device: str, mlp_arch: str) -> list[dict]:
    """Three multi-target regression probes per latent variant
    (physics / instrument_hsc / instrument_legacy), plus the existing
    single-target morphology classifier.
    """
    rows: list[dict] = []
    ck = h5.checkpoint_name
    has_e2 = h5.has_e2

    h5._max_epochs, h5._seed, h5._device, h5._mlp_arch = max_epochs, seed, device, mlp_arch

    combined_variants = [("combined_e1", [("overlap", "combined_e1")])]
    if has_e2:
        combined_variants.append(("combined_e2", [("overlap", "combined_e2")]))

    # ---- 1. physics_provabgs ------------------------------------------------
    _run_family(h5, rows, ck, "physics_provabgs",
                PHYSICS_TARGETS, combined_variants, _physics_resolver)

    # ---- 2. morphology_gz10 (single-target classifier; unchanged spirit) ----
    for variant, subset_keys in combined_variants:
        X = _stack_subset_latents(h5, subset_keys)
        if X is None:
            continue
        y = _stack_subset_labels(h5, subset_keys, lambda h, s: _gz10_label(h, s))
        if y is None:
            continue
        Xf, yf = _filter_finite(X, y)
        if len(yf) < MIN_LABELS:
            continue
        classes = np.unique(yf.astype(np.int64))
        if classes.size < 2:
            continue
        class_to_idx = {int(c): i for i, c in enumerate(classes)}
        yf_idx = np.array([class_to_idx[int(v)] for v in yf], dtype=np.int64)
        print(f"  [morph] gz10_label :: {variant}  (n={len(yf)}, k={classes.size})")
        results = _train_classification(Xf, yf_idx, max_epochs=max_epochs, seed=seed,
                                        device=device, n_classes=int(classes.size),
                                        mlp_arch=mlp_arch)
        for score, metric, n_tr, n_te, conv in results:
            rows.append(dict(checkpoint=ck, task_family="morphology_gz10",
                             target="gz10_label", latent_variant=variant,
                             score=score, score_metric=metric,
                             n_train=n_tr, n_test=n_te,
                             best_epoch=conv["best_epoch"],
                             epochs_run=conv["epochs_run"],
                             max_epochs=conv["max_epochs"],
                             early_stopped=conv["early_stopped"]))

    # ---- 3. instrument_hsc --------------------------------------------------
    _run_family(h5, rows, ck, "instrument_hsc",
                HSC_INSTRUMENT_TARGETS, combined_variants, _hsc_label_resolver)

    # ---- 4. instrument_legacy -----------------------------------------------
    _run_family(h5, rows, ck, "instrument_legacy",
                LEGACY_INSTRUMENT_TARGETS, combined_variants, _legacy_label_resolver)

    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embeddings", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mlp-arch", choices=["aion", "aion_reg"], default="aion",
                   help="aion: Linear→GELU→Linear (hidden=256). aion_reg: + LN + Dropout(0.2).")
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading embeddings from {args.embeddings}")
    h5 = CkptH5(args.embeddings)
    print(f"  checkpoint_name={h5.checkpoint_name}  arch={h5.architecture}  has_e2={h5.has_e2}")
    print(f"  mlp_arch={args.mlp_arch}")

    rows = run_all(h5, max_epochs=args.max_epochs, seed=args.seed,
                   device=args.device, mlp_arch=args.mlp_arch)
    h5.close()

    columns = ["checkpoint", "task_family", "target", "latent_variant",
               "score", "score_metric", "n_train", "n_test",
               "best_epoch", "epochs_run", "max_epochs", "early_stopped"]
    if not rows:
        print("No probe results — writing empty CSV with header only.")
        df = pd.DataFrame(columns=columns)
    else:
        df = pd.DataFrame(rows)[columns]
    df.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
