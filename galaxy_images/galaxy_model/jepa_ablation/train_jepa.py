"""
Unified training entry for the JEPA ablation arms (see PLAN.md).

    --arm hsc-resnet   JEPAHSCModule            on HSC48PoolDataset (367k pool)
    --arm hsc-vit      IJEPAViTModule           on HSC48PoolDataset (367k pool)
    --arm cf           JEPACounterfactualModule on NeighborsRAM48Dataset (anchors)
    --arm cross        JEPACrossSurveyModule    on AnchorPairs48Dataset (anchors)

Cloned from contrastive_baseline/train_contrastive_engaging.py (env hygiene,
wandb, checkpoint/resume/DONE contract are identical) with two deliberate
deviations, both PLAN.md decisions:
  * step-based cosine LR + warmup instead of the baseline's T_max quirk
    (schedules live in jepa_base.configure_optimizers);
  * survey-colored UMAP panels use a FIXED set of anchor pairs (set_umap_pairs)
    instead of harvesting val batches — required for the HSC-only arms, whose
    batches never carry Legacy images.
"""

import argparse
import os
import sys
from pathlib import Path

# Avoid matplotlib / numba trying to write to non-writable $HOME on cluster nodes.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
# Single-threaded numpy/BLAS per process so DataLoader workers don't oversubscribe.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import pytorch_lightning as pl
import torch
torch.set_num_threads(1)

# hipBLASLt workaround — guarded so it is a harmless no-op on Engaging (NVIDIA).
try:
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "preferred_blas_library"):
        if os.environ.get("TORCH_BLAS_PREFER_HIPBLASLT") == "0":
            torch.backends.cuda.preferred_blas_library("hipblas")
except Exception:
    pass

from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

_current_path = Path(__file__).resolve()
_project_root = _current_path.parents[3]  # .../tess-generative
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from galaxy_images.galaxy_model.jepa_ablation.datasets import (  # noqa: E402
    DEFAULT_DATA_DIR,
    AllStamps48Dataset,
    AnchorPairs48Dataset,
    HSC48PoolDataset,
    load_umap_pairs,
)

# Shared hyperparameters — matched to the contrastive 2x2 sweep (PLAN.md).
BATCH_SIZE = 64
LR = 1e-4
WEIGHT_DECAY = 1e-4
WARMUP_STEPS = 2000
EMA_BASE_MOMENTUM = 0.996
EMBEDDING_DIM = 64
PROJECTION_DIM = 32
PROJECTION_HIDDEN_DIM = 64
PREDICTOR_HIDDEN_DIM = 64
IMAGE_SIZE = 48
MAX_NEIGHBORS = 5
VAL_FRACTION = 0.05
SEED = 42

ARMS = ("hsc-resnet", "hsc-vit", "cf", "cross", "cf-vit", "ae-hsc",
        "cf-token", "xjepa", "multi-vit")


class VerboseModelCheckpoint(ModelCheckpoint):
    def _save_checkpoint(self, trainer, filepath):
        super()._save_checkpoint(trainer, filepath)
        if trainer.is_global_zero:
            print(f"[checkpoint] saved: {filepath}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--encoder-pool", choices=["avg", "conv1x1"], default="conv1x1",
                   help="ResNet arms only (ignored for hsc-vit).")
    p.add_argument("--max-steps", type=int, default=100_000)
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--wandb-project", default="galaxy-jepa-engaging")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--max-neighbors", type=int, default=MAX_NEIGHBORS)
    p.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    p.add_argument("--val-check-interval", type=int, default=1000)
    p.add_argument("--enable-umap", type=int, choices=[0, 1], default=1)
    p.add_argument("--umap-pairs", type=int, default=512,
                   help="Fixed anchor pairs for the survey-colored UMAP panel (0 = off).")
    p.add_argument("--load-into-ram", type=int, choices=[0, 1], default=1,
                   help="0 = memmap (smoke tests only; Ceph random reads are slow).")
    # Smoke-test knobs.
    p.add_argument("--limit-val-batches", type=int, default=None)
    p.add_argument("--limit-train-batches", type=int, default=None)
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def build_dataset_and_collate(args):
    load_ram = bool(args.load_into_ram)
    if args.arm in ("hsc-resnet", "hsc-vit", "ae-hsc"):
        return HSC48PoolDataset(args.data_dir, IMAGE_SIZE, load_into_ram=load_ram), None
    if args.arm == "multi-vit":
        return AllStamps48Dataset(args.data_dir, IMAGE_SIZE, load_into_ram=load_ram), None
    if args.arm == "cross":
        return AnchorPairs48Dataset(args.data_dir, IMAGE_SIZE, load_into_ram=load_ram), None
    # cf / cf-vit / cf-token / xjepa: the contrastive baseline's dataset + collate.
    from galaxy_images.galaxy_model.contrastive_baseline.neighbors_ram48 import (
        NeighborsRAM48Dataset,
    )
    from galaxy_images.galaxy_model.neighbors import collate_neighbors
    ds = NeighborsRAM48Dataset(
        data_dir=args.data_dir, crop_size=IMAGE_SIZE,
        max_neighbors=args.max_neighbors, random_neighbors=False,
        load_into_ram=load_ram,
    )
    return ds, collate_neighbors


def build_model(args):
    common = dict(
        in_channels=4,
        image_size=IMAGE_SIZE,
        lr=args.lr,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        ema_base_momentum=EMA_BASE_MOMENTUM,
        enable_umap_logging=bool(args.enable_umap),
        umap_n_neighbors=15,
        umap_min_dist=0.1,
    )
    resnet_common = dict(
        embedding_dim=EMBEDDING_DIM,
        projection_dim=PROJECTION_DIM,
        projection_hidden_dim=PROJECTION_HIDDEN_DIM,
        predictor_hidden_dim=PREDICTOR_HIDDEN_DIM,
        encoder_pool=args.encoder_pool,
        **common,
    )
    if args.arm == "hsc-resnet":
        from galaxy_images.galaxy_model.jepa_ablation.jepa_modules import JEPAHSCModule
        return JEPAHSCModule(**resnet_common)
    if args.arm == "cf":
        from galaxy_images.galaxy_model.jepa_ablation.jepa_modules import JEPACounterfactualModule
        return JEPACounterfactualModule(lambda_galaxy=1.0, lambda_instrument=1.0, **resnet_common)
    if args.arm == "cf-vit":
        from galaxy_images.galaxy_model.jepa_ablation.jepa_modules import JEPACounterfactualModule
        return JEPACounterfactualModule(
            encoder_arch="vit", lambda_galaxy=1.0, lambda_instrument=1.0, **resnet_common
        )
    if args.arm == "cross":
        from galaxy_images.galaxy_model.jepa_ablation.jepa_modules import JEPACrossSurveyModule
        return JEPACrossSurveyModule(**resnet_common)
    if args.arm == "ae-hsc":
        from galaxy_images.galaxy_model.jepa_ablation.autoencoder_baseline import AEHSCModule
        return AEHSCModule(
            in_channels=4, image_size=IMAGE_SIZE, embedding_dim=EMBEDDING_DIM,
            encoder_pool=args.encoder_pool, lr=args.lr, weight_decay=WEIGHT_DECAY,
            warmup_steps=WARMUP_STEPS, enable_umap_logging=bool(args.enable_umap),
            umap_n_neighbors=15, umap_min_dist=0.1,
        )
    if args.arm == "cf-token":
        from galaxy_images.galaxy_model.jepa_ablation.jepa_round3 import CFTokenJEPAModule
        return CFTokenJEPAModule(lambda_galaxy=1.0, lambda_instrument=1.0, **common)
    if args.arm == "xjepa":
        from galaxy_images.galaxy_model.jepa_ablation.jepa_round3 import XJEPAModule
        return XJEPAModule(lambda_mask=0.5, lambda_galaxy=1.0, lambda_instrument=1.0, **common)
    # hsc-vit and multi-vit share IJEPAViTModule; only the dataset differs.
    from galaxy_images.galaxy_model.jepa_ablation.jepa_vit import IJEPAViTModule
    return IJEPAViTModule(**common)


def main():
    args = parse_args()
    pl.seed_everything(SEED, workers=True)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ckpt] run directory: {ckpt_dir}", flush=True)
    print(f"[cfg] arm={args.arm} encoder_pool={args.encoder_pool} "
          f"max_steps={args.max_steps} batch_size={args.batch_size} lr={args.lr}", flush=True)

    # ---- data ----------------------------------------------------------------
    dataset, collate = build_dataset_and_collate(args)
    total = len(dataset)
    val_size = max(1, int(total * args.val_fraction))
    train_size = total - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )
    print(f"[data] total={total:,}  train={train_size:,}  val={val_size:,}", flush=True)

    persistent = args.num_workers > 0
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate,
        persistent_workers=persistent, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate,
        persistent_workers=persistent, pin_memory=True,
    )

    # ---- model ---------------------------------------------------------------
    model = build_model(args)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {type(model).__name__}  trainable_params={n_params:,}", flush=True)

    if args.umap_pairs > 0 and args.enable_umap:
        hsc_pairs, legacy_pairs = load_umap_pairs(args.data_dir, n=args.umap_pairs, seed=SEED)
        model.set_umap_pairs(hsc_pairs, legacy_pairs)
        print(f"[umap] fixed pair set: {hsc_pairs.shape[0]} anchors per survey", flush=True)

    # ---- logging -------------------------------------------------------------
    logger = False
    if not args.no_wandb:
        logger = WandbLogger(
            project=args.wandb_project,
            name=args.run_name,
            log_model=False,
            config={
                "arm": args.arm,
                "encoder_pool": args.encoder_pool if args.arm != "hsc-vit" else "vit",
                "batch_size": args.batch_size,
                "max_neighbors": args.max_neighbors,
                "embedding_dim": EMBEDDING_DIM if args.arm != "hsc-vit" else 192,
                "projection_dim": PROJECTION_DIM,
                "lr": args.lr,
                "warmup_steps": WARMUP_STEPS,
                "ema_base_momentum": EMA_BASE_MOMENTUM,
                "lr_schedule": "cosine+warmup (fixed, PLAN.md decision 4)",
                "max_steps": args.max_steps,
                "data_dir": args.data_dir,
                "precision": "bf16-mixed",
                "cluster": "engaging",
            },
        )

    best_ckpt = VerboseModelCheckpoint(
        dirpath=str(ckpt_dir), monitor="val/loss", mode="min", save_top_k=1,
        filename="best-epoch={epoch:02d}-step={step}", auto_insert_metric_name=False,
    )
    periodic_ckpt = VerboseModelCheckpoint(
        dirpath=str(ckpt_dir), every_n_train_steps=500, save_top_k=1,
        filename="latest-step={step}", save_last=True, auto_insert_metric_name=False,
    )

    trainer = pl.Trainer(
        max_steps=args.max_steps,
        logger=logger,
        accelerator="auto",
        devices=1,
        precision="bf16-mixed",
        log_every_n_steps=10,
        val_check_interval=args.val_check_interval,
        check_val_every_n_epoch=None,
        callbacks=[best_ckpt, periodic_ckpt],
        num_sanity_val_steps=0,
        default_root_dir=str(ckpt_dir),
        limit_val_batches=args.limit_val_batches if args.limit_val_batches is not None else 1.0,
        limit_train_batches=args.limit_train_batches if args.limit_train_batches is not None else 1.0,
    )

    last_ckpt = ckpt_dir / "last.ckpt"
    resume_path = str(last_ckpt) if last_ckpt.exists() else None
    if resume_path:
        print(f"[resume] resuming from {resume_path}", flush=True)

    trainer.fit(model, train_loader, val_loader, ckpt_path=resume_path)

    if trainer.global_step >= args.max_steps:
        (ckpt_dir / "DONE").write_text(f"global_step={trainer.global_step}\n")
        print(f"[done] reached {trainer.global_step} steps; wrote DONE marker.", flush=True)
    else:
        print(f"[incomplete] stopped at global_step={trainer.global_step} "
              f"(< {args.max_steps}); a successor job should resume.", flush=True)


if __name__ == "__main__":
    main()
