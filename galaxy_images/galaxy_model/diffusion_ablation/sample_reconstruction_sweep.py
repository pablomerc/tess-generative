#!/usr/bin/env python3
"""Generate matched validation reconstructions across DDIM step/eta settings.

The selected validation examples and initial Gaussian noise are fixed across all
settings. Outputs include raw tensors, a JSON manifest with per-setting MSEs,
one grid per setting, and a combined comparison grid.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from galaxy_images.galaxy_model.config import load_experiment_config
from galaxy_images.galaxy_model.data_factory import build_neighbors_dataloaders
from galaxy_images.galaxy_model.diffusion_ablation.double_train_ddpm_neighbors import (
    ConditionalDDPMModule,
)

ABLATION_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = Path(
    "/home/pablomer/orcd/pool/checkpoints_new/diffusion_ablation/best/"
    "ddpm-eps-ram48-h200-75k/best-epoch=72-step=28000_copy.ckpt"
)
DEFAULT_CONFIG = ABLATION_DIR / "configs" / "neighbors_ddpm_eps.json"


def _unpack_batch(batch):
    if len(batch) == 5:
        target, samegal, sameins, masks, metadata = batch
    else:
        target, samegal, sameins, metadata = batch
        batch_size, n_neighbors, _, _, _ = sameins.shape
        masks = torch.ones((batch_size, n_neighbors), dtype=torch.bool)
    return target, samegal, sameins, masks, metadata


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value) and value.numel() == 1:
        return value.item()
    return value


def _setting_key(num_steps: int, eta: float) -> str:
    return f"steps={num_steps}_eta={eta:g}"


def _setting_slug(num_steps: int, eta: float) -> str:
    return f"steps-{num_steps}_eta-{eta:g}".replace(".", "p")


def _target_scaled_rgb(image: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    """Scale the first three channels using the target's per-channel range."""
    image = image[:3].float()
    target = target[:3].float()
    vmin = target.amin(dim=(1, 2), keepdim=True)
    vmax = target.amax(dim=(1, 2), keepdim=True)
    scaled = ((image - vmin) / (vmax - vmin).clamp_min(1e-8)).clamp(0, 1)
    return scaled.permute(1, 2, 0).numpy()


def _plot_setting(
    output_path: Path,
    target: torch.Tensor,
    samegal: torch.Tensor,
    sameins: torch.Tensor,
    reconstruction: torch.Tensor,
    metadata: list[dict[str, Any]],
    title: str,
) -> None:
    n = target.shape[0]
    fig, axes = plt.subplots(n, 4, figsize=(8, 2 * n), squeeze=False)
    column_titles = ["Same galaxy", "Same instrument", "Target", "Reconstruction"]
    for col, column_title in enumerate(column_titles):
        axes[0, col].set_title(column_title, fontsize=10)

    for row in range(n):
        panels = (samegal[row], sameins[row, 0], target[row], reconstruction[row])
        for col, panel in enumerate(panels):
            axes[row, col].imshow(_target_scaled_rgb(panel, target[row]))
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(
            f"{metadata[row]['anchor_survey']}\nidx={metadata[row].get('idx', '?')}",
            fontsize=8,
        )

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_combined(
    output_path: Path,
    target: torch.Tensor,
    samegal: torch.Tensor,
    sameins: torch.Tensor,
    reconstructions: dict[str, torch.Tensor],
    settings: list[tuple[int, float]],
    metadata: list[dict[str, Any]],
) -> None:
    n = target.shape[0]
    n_cols = 3 + len(settings)
    fig, axes = plt.subplots(
        n,
        n_cols,
        figsize=(2 * n_cols, 2 * n),
        squeeze=False,
    )
    titles = ["Same galaxy", "Same instrument", "Target"] + [
        f"{steps} steps\nη={eta:g}" for steps, eta in settings
    ]
    for col, title in enumerate(titles):
        axes[0, col].set_title(title, fontsize=9)

    for row in range(n):
        fixed_panels = (samegal[row], sameins[row, 0], target[row])
        for col, panel in enumerate(fixed_panels):
            axes[row, col].imshow(_target_scaled_rgb(panel, target[row]))
            axes[row, col].axis("off")
        for setting_idx, (steps, eta) in enumerate(settings):
            reconstruction = reconstructions[_setting_key(steps, eta)][row]
            axes[row, 3 + setting_idx].imshow(
                _target_scaled_rgb(reconstruction, target[row])
            )
            axes[row, 3 + setting_idx].axis("off")
        axes[row, 0].set_ylabel(
            f"{metadata[row]['anchor_survey']}\nidx={metadata[row].get('idx', '?')}",
            fontsize=8,
        )

    fig.suptitle("DDIM reconstruction sweep (matched validation examples and initial noise)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _mse_summary(
    target: torch.Tensor,
    reconstruction: torch.Tensor,
    metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    per_image = ((reconstruction.float() - target.float()) ** 2).mean(
        dim=(1, 2, 3)
    )
    result: dict[str, Any] = {
        "overall": float(per_image.mean()),
        "per_image": [float(value) for value in per_image],
    }
    for survey in ("hsc", "legacy"):
        indices = [
            idx for idx, item in enumerate(metadata) if item["anchor_survey"] == survey
        ]
        result[survey] = (
            float(per_image[indices].mean()) if indices else math.nan
        )
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument(
        "--num-steps",
        type=int,
        nargs="+",
        default=[25, 50, 100, 250],
        help="DDIM inference-step counts.",
    )
    parser.add_argument(
        "--eta",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 1.0],
        help="DDIM eta values.",
    )
    parser.add_argument("--noise-seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: diffusion_ablation/results/reconstruction_sweep/<checkpoint-stem>",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty output directory.",
    )
    args = parser.parse_args(argv)

    if args.num_images < 1:
        parser.error("--num-images must be >= 1")
    if any(steps < 1 or steps > 1000 for steps in args.num_steps):
        parser.error("--num-steps values must be in [1, 1000]")
    if any(eta < 0 for eta in args.eta):
        parser.error("--eta values must be >= 0")
    if not args.checkpoint.exists():
        parser.error(f"checkpoint not found: {args.checkpoint}")

    output_dir = args.output_dir or (
        ABLATION_DIR / "results" / "reconstruction_sweep" / args.checkpoint.stem
    )
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        parser.error(
            f"output directory is not empty: {output_dir}; use --overwrite or a new path"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    config = load_experiment_config(args.config)
    # Avoid extra worker processes for this small, one-batch extraction. The RAM48
    # dataset still loads once into the job's RAM.
    config.data.num_workers = 0
    _, val_loader, _ = build_neighbors_dataloaders(
        config, batch_size=args.num_images
    )
    batch = next(iter(val_loader))
    target, samegal, sameins, masks, metadata = _unpack_batch(batch)
    target = target[: args.num_images]
    samegal = samegal[: args.num_images]
    sameins = sameins[: args.num_images]
    masks = masks[: args.num_images]
    metadata = metadata[: args.num_images]

    val_subset_indices = getattr(val_loader.dataset, "indices", None)
    dataset_indices = (
        [int(value) for value in val_subset_indices[: args.num_images]]
        if val_subset_indices is not None
        else list(range(args.num_images))
    )

    print(f"[sweep] loading checkpoint: {args.checkpoint}", flush=True)
    model = ConditionalDDPMModule.load_from_checkpoint(
        str(args.checkpoint), map_location="cpu"
    )
    model.eval().to(device)
    torch.set_grad_enabled(False)

    initial_generator = torch.Generator(device=device).manual_seed(args.noise_seed)
    initial_noise = torch.randn(
        args.num_images,
        model.in_channels,
        model.image_size,
        model.image_size,
        device=device,
        generator=initial_generator,
    )

    target_device = target.to(device)
    samegal_device = samegal.to(device)
    sameins_device = sameins.to(device)
    masks_device = masks.to(device)

    settings = [
        (num_steps, eta) for num_steps in args.num_steps for eta in args.eta
    ]
    reconstructions: dict[str, torch.Tensor] = {}
    metrics: dict[str, Any] = {}
    for setting_index, (num_steps, eta) in enumerate(settings, start=1):
        key = _setting_key(num_steps, eta)
        print(
            f"[sweep] {setting_index}/{len(settings)}: "
            f"num_steps={num_steps}, eta={eta:g}",
            flush=True,
        )
        # Recreate this generator for every setting so stochastic sweeps begin
        # from the same random stream as well as the same initial x_noise.
        step_generator = torch.Generator(device=device).manual_seed(
            args.noise_seed + 10_000
        )
        reconstruction = model.sample(
            samegal_device,
            sameins_device,
            masks=masks_device,
            num_steps=num_steps,
            x_noise=initial_noise,
            eta=eta,
            generator=step_generator,
        )
        reconstruction_cpu = reconstruction.detach().float().cpu()
        reconstructions[key] = reconstruction_cpu
        metrics[key] = _mse_summary(target, reconstruction_cpu, metadata)
        _plot_setting(
            output_dir / f"{_setting_slug(num_steps, eta)}.png",
            target,
            samegal,
            sameins,
            reconstruction_cpu,
            metadata,
            title=f"DDPM reconstruction: {num_steps} DDIM steps, η={eta:g}",
        )

    _plot_combined(
        output_dir / "combined_sweep.png",
        target,
        samegal,
        sameins,
        reconstructions,
        settings,
        metadata,
    )

    tensor_artifact = {
        "target": target.half(),
        "samegal": samegal.half(),
        "sameins": sameins.half(),
        "masks": masks,
        "initial_noise": initial_noise.detach().half().cpu(),
        "reconstructions": {
            key: value.half() for key, value in reconstructions.items()
        },
    }
    torch.save(tensor_artifact, output_dir / "reconstruction_sweep.pt")

    manifest = {
        "checkpoint": str(args.checkpoint.resolve()),
        "config": str(args.config.resolve()),
        "output_dir": str(output_dir.resolve()),
        "num_images": args.num_images,
        "num_steps": args.num_steps,
        "eta": args.eta,
        "noise_seed": args.noise_seed,
        "split_seed": config.trainer.seed,
        "val_ratio": config.data.val_ratio,
        "dataset_indices": dataset_indices,
        "catalog_indices": [
            _json_scalar(item.get("idx")) for item in metadata
        ],
        "metadata": [
            {key: _json_scalar(value) for key, value in item.items()}
            for item in metadata
        ],
        "metrics": metrics,
    }
    with (output_dir / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=True)

    print(f"[sweep] wrote outputs to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
