"""
Trainer for every post-review contrastive ablation arm.

Three independent axes, all defaulting to the published `contrastive-spatial-conv1x1`
configuration so that any single flag isolates one variable:

  --holdout-ids-txt PATH     exclude the n=5,469 downstream-eval galaxies (DWNVAL)
  --galaxy-negatives MODE     mixed | same_survey | cross_survey
  --instrument-negatives MODE mixed | same_survey
  --projection-head 0|1       SimCLR head (default 1) vs InfoNCE on the representation

With all defaults and no holdout this reproduces the published arm; the loss path is
literally delegated to the unmodified baseline class in that case (see
dual_encoder_contrastive_variants.py), so "default == baseline" is enforced by code,
not by comment. `test_negative_variants.py` asserts it numerically.

The DWNVAL axis — why it exists
-------------------------------
Rebuttal audit (`contrastive_ablation/ANALYSIS.md`) established that neither the
paper's `base` flow-matching checkpoint nor any of the four contrastive runs
excluded the n=5,469 downstream-overlap galaxies from SSL training. For the FM
side we could bound the effect from existing checkpoints (`base-5NB` vs
`base-5NB-DWNVAL`: mean ΔR² ≈ ±0.005, i.e. noise). The contrastive side has no
such pair, and InfoNCE is the objective where identity memorisation is most
plausible — its galaxy branch is literally trained to match the two views of one
galaxy. This script produces the missing arm.

Relationship to the existing runs
---------------------------------
This is `contrastive_baseline/train_contrastive_engaging.py` with the axes above
bolted on. Everything else is deliberately identical to
`contrastive-spatial-conv1x1` so every comparison is a clean A/B:

  * same model class or a subclass of it that delegates to it when unrestricted (so
    the repo-wide LR-schedule quirk documented in CLAUDE.md applies identically to
    every arm and cannot confound any comparison)
  * same hyperparameters (embedding 64, projection 32, T=0.1, lr 1e-4, wd 1e-4,
    batch 64, K=5, bf16-mixed, 1 GPU, seed 42, val_fraction 0.05, 100k steps)
  * same anchor-survey parity (`idx % 2`), preserved because exclusion goes
    through `torch.utils.data.Subset`, which passes ORIGINAL indices through to
    the underlying dataset's `__getitem__`
  * same exclusion semantics as the FM DWNVAL runs: the holdout positions are
    resolved by importing `data_factory._resolve_downstream_holdout_positions`
    rather than reimplementing the `object_id_legacy` matching (single source of
    truth, including the `b'...'` byte-repr normalisation)
  * same exclusion/split ORDER as `data_factory.build_neighbors_dataloaders`:
    Subset(kept) first, then `random_split(kept, [train, val], seed 42)`

Known, accepted deviation: dropping 5,466 of 103,741 anchors (5.3%) shortens the
epoch from 1,539 to 1,458 steps (-5.3%), which shifts the phase of the `T_max=-1`
cosine LR alternation documented in CLAUDE.md. Both arms still receive ~50%
effective updates over 100k steps (68.6 vs 65.0 epochs), so the effect is far
below the contamination signal being measured. Not worth tiling around — but if a
reviewer asks, this is the one thing that is not held exactly fixed.

Cluster portability
-------------------
No cluster-specific paths. Pass `--data-dir` and `--holdout-ids-txt` explicitly.

  Engaging : --loader ram48     --data-dir /orcd/pool/007/pablomer/efficient_neighs_48
  AMD      : --loader efficient --data-dir /work1/jeroenaudenaert/pablomer/data/neighbors_efficient

Both loaders apply the same preprocessing to the same catalog rows; `ram48` has
it baked in at build time (float16 round-trip), `efficient` recomputes it from
the 160x160 store. See ANALYSIS.md §1.

Resume: writes `<ckpt-dir>/last.ckpt` continually and resumes from it if present,
so the run survives a wall-limit kill under a self-chaining SLURM launcher.
"""

import argparse
import os
import sys
from pathlib import Path

# Avoid matplotlib / numba trying to write to non-writable $HOME on cluster nodes.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
# Single-threaded numpy/BLAS per process so DataLoader workers don't oversubscribe cores.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import pytorch_lightning as pl
import torch
torch.set_num_threads(1)

# hipBLASLt workaround — guarded, so it is a fix on AMD MI210 and a no-op on NVIDIA.
try:
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "preferred_blas_library"):
        if os.environ.get("TORCH_BLAS_PREFER_HIPBLASLT") == "0":
            torch.backends.cuda.preferred_blas_library("hipblas")
except Exception:
    pass

from torch.utils.data import DataLoader, Subset, random_split
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

_current_path = Path(__file__).resolve()
_project_root = _current_path.parents[3]  # .../tess-generative
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from galaxy_images.galaxy_model.neighbors import collate_neighbors
from galaxy_images.galaxy_model.contrastive_ablation.dual_encoder_contrastive_variants import (
    ContrastiveNegativeAblationModule,
)

# Frozen to match contrastive-spatial-conv1x1 exactly. Do not "improve" these:
# the whole value of this run is that it differs from that one in ONE respect.
BATCH_SIZE = 64
LR = 1e-4
WEIGHT_DECAY = 1e-4
EMBEDDING_DIM = 64
PROJECTION_DIM = 32
PROJECTION_HIDDEN_DIM = 64
IMAGE_SIZE = 48
MAX_NEIGHBORS = 5
TEMPERATURE = 0.1
VAL_FRACTION = 0.05
SEED = 42


class VerboseModelCheckpoint(ModelCheckpoint):
    def _save_checkpoint(self, trainer, filepath):
        super()._save_checkpoint(trainer, filepath)
        if trainer.is_global_zero:
            print(f"[checkpoint] saved: {filepath}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--encoder-pool", choices=["avg", "conv1x1"], default="conv1x1",
                   help="conv1x1 mirrors Ours' Conv2d(512->16,1x1) -> 4 tokens x 16 dims.")
    p.add_argument("--random-neighbors", type=int, choices=[0, 1], default=0)
    p.add_argument("--max-steps", type=int, default=100_000)
    p.add_argument("--data-dir", required=True,
                   help="Neighbors store. ram48: efficient_neighs_48; efficient: neighbors_efficient.")
    p.add_argument("--loader", choices=["ram48", "efficient"], required=True)
    p.add_argument("--holdout-ids-txt", default=None,
                   help="One legacy object_id per line; anchors matching these are excluded. "
                        "A tracked copy ships at contrastive_ablation/holdout_legacy_ids.txt. "
                        "Omit for a run comparable to the published (non-holdout) arms, but "
                        "then --no-holdout must be passed explicitly.")
    p.add_argument("--no-holdout", action="store_true",
                   help="Acknowledge that no holdout is applied. Required when "
                        "--holdout-ids-txt is omitted, so a forgotten flag cannot silently "
                        "produce a contaminated run that looks like a DWNVAL one.")
    # Negative-sampling / projection-head axes (see dual_encoder_contrastive_variants.py).
    p.add_argument("--galaxy-negatives", choices=["mixed", "same_survey", "cross_survey"],
                   default="mixed",
                   help="Which in-batch items may act as negatives for the galaxy branch. "
                        "mixed = published behaviour (~half the negatives differ from the "
                        "positive by survey, so survey identity partly solves the task).")
    p.add_argument("--instrument-negatives", choices=["mixed", "same_survey"], default="mixed",
                   help="Which neighbours may act as negatives for the instrument branch. "
                        "mixed = published behaviour (pool spans both surveys, so a survey "
                        "classifier scores well); same_survey forces within-survey seeing/"
                        "depth discrimination.")
    p.add_argument("--projection-head", type=int, choices=[0, 1], default=1,
                   help="1 = SimCLR MLP head (published); 0 = InfoNCE directly on the "
                        "encoder output that the probes consume.")
    p.add_argument("--min-holdout-matches", type=int, default=5000,
                   help="Fail fast if fewer anchors than this matched (guards against an "
                        "object_id encoding mismatch silently training a non-holdout run).")
    p.add_argument("--ckpt-dir", required=True, help="Run checkpoint directory (on pool/work, not scratch).")
    p.add_argument("--run-name", required=True, help="WandB run name / tag.")
    p.add_argument("--wandb-project", default="galaxy-contrastive-engaging")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--max-neighbors", type=int, default=MAX_NEIGHBORS)
    p.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    p.add_argument("--val-check-interval", type=int, default=1000)
    p.add_argument("--enable-umap", type=int, choices=[0, 1], default=1)
    # Smoke-test knobs.
    p.add_argument("--limit-val-batches", type=int, default=None)
    p.add_argument("--limit-train-batches", type=int, default=None)
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def build_dataset(args):
    if args.loader == "ram48":
        from galaxy_images.galaxy_model.contrastive_baseline.neighbors_ram48 import (
            NeighborsRAM48Dataset,
        )
        return NeighborsRAM48Dataset(
            data_dir=args.data_dir,
            crop_size=IMAGE_SIZE,
            max_neighbors=args.max_neighbors,
            random_neighbors=bool(args.random_neighbors),
            load_into_ram=True,
        )
    from galaxy_images.galaxy_model.neighbors_efficient import NeighborsEfficientDataset
    return NeighborsEfficientDataset(
        data_dir=args.data_dir,
        crop_size=IMAGE_SIZE,
        max_neighbors=args.max_neighbors,
        random_neighbors=bool(args.random_neighbors),
    )


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
    print(f"[cfg] encoder_pool={args.encoder_pool} random_neighbors={bool(args.random_neighbors)} "
          f"loader={args.loader} max_steps={args.max_steps} batch_size={args.batch_size}", flush=True)
    print(f"[cfg] galaxy_negatives={args.galaxy_negatives} "
          f"instrument_negatives={args.instrument_negatives} "
          f"projection_head={bool(args.projection_head)} "
          f"holdout={'yes' if args.holdout_ids_txt else 'NO'}", flush=True)

    # ---- data ----------------------------------------------------------------
    dataset = build_dataset(args)
    total = len(dataset)

    # Exclude the downstream-eval galaxies. Import the FM pipeline's resolver so the
    # anchor matching is bit-identical to how the *-DWNVAL flow-matching runs did it.
    if args.holdout_ids_txt:
        from galaxy_images.galaxy_model.data_factory import (
            _resolve_downstream_holdout_positions,
        )
        holdout_positions = _resolve_downstream_holdout_positions(
            holdout_txt=args.holdout_ids_txt,
            efficient_data_dir=args.data_dir,
        )
        if len(holdout_positions) < args.min_holdout_matches:
            raise SystemExit(
                f"[FATAL] only {len(holdout_positions):,} anchors matched the holdout list "
                f"(expected >= {args.min_holdout_matches:,}). This would silently train a "
                f"NON-holdout run. Check object_id_legacy encoding in "
                f"{Path(args.data_dir) / 'catalog.parquet'} against {args.holdout_ids_txt}, "
                f"then re-run contrastive_ablation/preflight_holdout.py."
            )
    else:
        if not args.no_holdout:
            raise SystemExit(
                "[FATAL] no --holdout-ids-txt given. Pass --no-holdout to confirm this arm "
                "is deliberately trained on all anchors (comparable to the published runs)."
            )
        holdout_positions = []
        print("[downstream-holdout] DISABLED — training on all anchors, matching the "
              "published contrastive arms.", flush=True)

    excluded = set(holdout_positions)
    if excluded:
        kept_positions = [i for i in range(total) if i not in excluded]
        kept_dataset = Subset(dataset, kept_positions)
    else:
        # Keep the un-Subset dataset so the split is bit-identical to the published arms.
        kept_positions = list(range(total))
        kept_dataset = dataset

    # Parity audit: the anchor survey is fixed by dataset-index parity (CLAUDE.md), so an
    # exclusion that happened to be parity-skewed would change the HSC/Legacy training mix
    # as well as the holdout. Log it; the holdout is id-based so it should be ~50/50.
    n_even_excl = sum(1 for i in holdout_positions if i % 2 == 0)
    n_even_kept = sum(1 for i in kept_positions if i % 2 == 0)
    print(
        f"[data] total anchors={total:,} | excluded={len(excluded):,} "
        f"(HSC-anchor {n_even_excl:,} / Legacy-anchor {len(excluded) - n_even_excl:,}) | "
        f"kept={len(kept_dataset):,} "
        f"(HSC-anchor {n_even_kept:,} / Legacy-anchor {len(kept_positions) - n_even_kept:,})",
        flush=True,
    )

    kept_size = len(kept_dataset)
    val_size = max(1, int(kept_size * args.val_fraction))
    train_size = kept_size - val_size
    train_ds, val_ds = random_split(
        kept_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )
    steps_per_epoch = train_size // args.batch_size
    print(f"[data] kept={kept_size:,}  train={train_size:,}  val={val_size:,}  "
          f"steps/epoch={steps_per_epoch:,}", flush=True)

    persistent = args.num_workers > 0
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_neighbors,
        persistent_workers=persistent, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_neighbors,
        persistent_workers=persistent, pin_memory=True,
    )

    # ---- model ---------------------------------------------------------------
    # Subclass of the published module; with mixed/mixed negatives and the head on, its
    # loss path delegates straight to the baseline implementation.
    model = ContrastiveNegativeAblationModule(
        galaxy_negatives=args.galaxy_negatives,
        instrument_negatives=args.instrument_negatives,
        use_projection_head=bool(args.projection_head),
        in_channels=4,
        embedding_dim=EMBEDDING_DIM,
        projection_dim=PROJECTION_DIM,
        projection_hidden_dim=PROJECTION_HIDDEN_DIM,
        pretrained_encoder=False,
        encoder_pool=args.encoder_pool,
        image_size=IMAGE_SIZE,
        temperature_galaxy=TEMPERATURE,
        temperature_instrument=TEMPERATURE,
        lambda_galaxy=1.0,
        lambda_instrument=1.0,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        enable_umap_logging=bool(args.enable_umap),
        num_umap_batches=8,
        umap_n_neighbors=15,
        umap_min_dist=0.1,
        use_random_instrument_positives=False,
    )

    # ---- logging -------------------------------------------------------------
    logger = False
    if not args.no_wandb:
        logger = WandbLogger(
            project=args.wandb_project,
            name=args.run_name,
            log_model=False,
            config={
                "encoder_pool": args.encoder_pool,
                "random_neighbors": bool(args.random_neighbors),
                "downstream_holdout": bool(args.holdout_ids_txt),
                "holdout_ids_txt": args.holdout_ids_txt or "",
                "galaxy_negatives": args.galaxy_negatives,
                "instrument_negatives": args.instrument_negatives,
                "projection_head": bool(args.projection_head),
                "n_anchors_total": total,
                "n_anchors_excluded": len(excluded),
                "n_anchors_kept": kept_size,
                "batch_size": args.batch_size,
                "max_neighbors": args.max_neighbors,
                "embedding_dim": EMBEDDING_DIM,
                "projection_dim": PROJECTION_DIM,
                "lr": LR,
                "max_steps": args.max_steps,
                "data_dir": args.data_dir,
                "loader": args.loader,
                "precision": "bf16-mixed",
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

    # Mark completion so the self-chaining SLURM launcher stops resubmitting.
    if trainer.global_step >= args.max_steps:
        (ckpt_dir / "DONE").write_text(
            f"global_step={trainer.global_step}\n"
            f"n_anchors_excluded={len(excluded)}\n"
            f"n_anchors_kept={kept_size}\n"
            f"holdout_ids_txt={args.holdout_ids_txt or '(none)'}\n"
            f"encoder_pool={args.encoder_pool}\n"
            f"galaxy_negatives={args.galaxy_negatives}\n"
            f"instrument_negatives={args.instrument_negatives}\n"
            f"projection_head={bool(args.projection_head)}\n"
            f"loader={args.loader}\ndata_dir={args.data_dir}\n"
        )
        print(f"[done] reached {trainer.global_step} steps; wrote DONE marker.", flush=True)
    else:
        print(f"[incomplete] stopped at global_step={trainer.global_step} "
              f"(< {args.max_steps}); a successor job should resume.", flush=True)


if __name__ == "__main__":
    main()
