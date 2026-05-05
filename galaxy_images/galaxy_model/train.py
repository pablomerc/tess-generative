from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import List, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from galaxy_images.galaxy_model.config import ExperimentConfig, load_experiment_config
from galaxy_images.galaxy_model.data_factory import build_neighbors_dataloaders
from galaxy_images.galaxy_model.lens_val_callback import LensValidationCallback
from galaxy_images.galaxy_model.variants import (
    filter_supported_model_kwargs,
    get_variant,
)


def is_h100_gpu() -> bool:
    try:
        if not torch.cuda.is_available():
            return False
    except Exception:
        return False

    try:
        for i in range(torch.cuda.device_count()):
            if "h100" in torch.cuda.get_device_name(i).lower():
                return True
    except Exception:
        return False
    return False


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified trainer for galaxy model variants.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file (.json recommended; .yaml/.yml also supported if PyYAML is installed).",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config values with key=value, e.g. --set trainer.num_steps=20000",
    )
    return parser.parse_args(argv)


def _build_run_dir(config: ExperimentConfig, variant_name: str) -> Path:
    base = Path(config.run.output_dir) / variant_name
    today = date.today().strftime("%Y-%m-%d")
    candidate = base / today
    if candidate.exists():
        n = 2
        while (base / f"{today}_{n}").exists():
            n += 1
        candidate = base / f"{today}_{n}"
    candidate.mkdir(parents=True, exist_ok=True)
    (candidate / "checkpoints").mkdir(parents=True, exist_ok=True)
    return candidate


def _build_model(config: ExperimentConfig):
    variant = get_variant(config.run.variant)
    model_kwargs = asdict(config.model)
    model_kwargs["channel_mult"] = tuple(model_kwargs["channel_mult"])
    model_kwargs.update(variant.model_overrides)
    model_kwargs = filter_supported_model_kwargs(variant.model_cls, model_kwargs)
    model = variant.model_cls(**model_kwargs)
    return model, variant


def _effective_training_settings(config: ExperimentConfig) -> tuple[int, str, bool]:
    runtime_batch_size = config.data.batch_size
    runtime_precision = config.trainer.precision
    h100 = is_h100_gpu()
    if h100 and config.trainer.auto_adjust_for_h100:
        runtime_batch_size = config.trainer.h100_batch_size
        runtime_precision = config.trainer.h100_precision
    return runtime_batch_size, runtime_precision, h100


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    config = load_experiment_config(args.config, args.overrides)

    if config.trainer.seed is not None:
        pl.seed_everything(config.trainer.seed, workers=True)

    runtime_batch_size, runtime_precision, h100 = _effective_training_settings(config)
    model, variant = _build_model(config)
    train_loader, val_loader, lens_loader = build_neighbors_dataloaders(config, runtime_batch_size)

    resume_ckpt = config.run.resume_from
    if resume_ckpt is not None:
        # Reuse the existing run directory (checkpoints live in <run_dir>/checkpoints/)
        run_dir = Path(resume_ckpt).parent.parent
    else:
        run_dir = _build_run_dir(config, variant.name)

    if config.wandb.enabled:
        wandb_logger = WandbLogger(
            project=config.wandb.project,
            name=config.wandb.name,
            log_model=config.wandb.log_model,
            resume="allow" if resume_ckpt is not None else "never",
            config={
                **asdict(config),
                "runtime": {
                    "h100_detected": h100,
                    "batch_size": runtime_batch_size,
                    "precision": runtime_precision,
                    "variant": variant.name,
                },
            },
        )
    else:
        wandb_logger = None

    # Best checkpoint goes to a shared dir (one subfolder per run-name) so it's
    # easy to compare runs side-by-side. Periodic latest checkpoints stay in the
    # per-run output dir.
    if config.run.shared_checkpoint_dir:
        run_subdir_name = config.wandb.name or variant.name
        best_ckpt_dir = Path(config.run.shared_checkpoint_dir) / run_subdir_name
        best_ckpt_dir.mkdir(parents=True, exist_ok=True)
    else:
        best_ckpt_dir = run_dir / "checkpoints"
    print(f"Best-checkpoint dir: {best_ckpt_dir}")

    best_checkpoint = ModelCheckpoint(
        dirpath=best_ckpt_dir,
        monitor=config.trainer.monitor_metric,
        mode="min",
        save_top_k=1,
        filename="best-epoch={epoch:02d}-step={step}",
        auto_insert_metric_name=False,
    )
    periodic_checkpoint = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        every_n_train_steps=config.trainer.checkpoint_every_n_train_steps,
        save_top_k=1,
        filename="latest-step={step}",
        save_last=False,
    )

    callbacks = [best_checkpoint, periodic_checkpoint]
    if lens_loader is not None:
        figures_dir = run_dir / "figures" / "lens_val"
        callbacks.append(
            LensValidationCallback(
                lens_loader=lens_loader,
                every_n_validations=config.lens_val.every_n_validations,
                num_integration_steps=config.lens_val.num_integration_steps,
                num_samples_per_cond=config.lens_val.num_samples_per_cond,
                figures_dir=figures_dir,
                run_name=config.wandb.name or variant.name,
            )
        )
        print(
            f"[lens-val] callback enabled: every {config.lens_val.every_n_validations} validations, "
            f"{len(config.lens_val.lens_indices_zero_based)} lenses"
        )

    n_devices = config.trainer.devices
    max_steps = config.trainer.num_steps
    if isinstance(n_devices, int) and n_devices > 1 and config.trainer.scale_steps_by_devices:
        max_steps = max(1, int(max_steps / n_devices))

    strategy = config.trainer.strategy
    if isinstance(n_devices, int) and n_devices <= 1:
        strategy = "auto"

    trainer = pl.Trainer(
        max_steps=max_steps,
        logger=wandb_logger,
        accelerator=config.trainer.accelerator,
        devices=n_devices,
        strategy=strategy,
        log_every_n_steps=config.trainer.log_every_n_steps,
        precision=runtime_precision,
        val_check_interval=config.trainer.val_check_interval,
        check_val_every_n_epoch=None,
        callbacks=callbacks,
        num_sanity_val_steps=config.trainer.num_sanity_val_steps,
    )

    print(f"Variant: {variant.name}")
    print(f"Description: {variant.description}")
    print(f"Data mode: {config.data.mode}")
    print(f"Batch size: {runtime_batch_size}")
    print(f"Precision: {runtime_precision}")
    print(f"H100 detected: {h100}")
    print(f"Run dir: {run_dir}")

    trainer.fit(model, train_loader, val_loader, ckpt_path=resume_ckpt)


if __name__ == "__main__":
    main()
