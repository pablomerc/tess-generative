"""
Lens validation callback.

Runs the model on a small fixed set of gravitational lens galaxies that are
explicitly held out of the training set, every N validation rounds.

Logs:
    lens_val/loss       - flow-matching velocity MSE on the lens batch
    lens_val/mse_48     - reconstruction MSE (centered 48x48 crop)
    lens_val/sample_grid (image)

Designed to work with both ConditionalFlowMatchingModule and
HierarchicalGlobalInstrumentFlowMatchingModule (both expose .compute_loss
and .sample with the same signature).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import wandb


def _normalize_for_vis(img: torch.Tensor) -> np.ndarray:
    img = img.clone().detach().float()
    img = img - img.min()
    if img.max() > 0:
        img = img / img.max()
    return img.cpu().permute(1, 2, 0).numpy()


def _row_scale_rgb(x_chw: torch.Tensor, vmin: torch.Tensor, vmax: torch.Tensor) -> np.ndarray:
    x = x_chw[:3]
    vmin_t = torch.as_tensor(vmin, device=x.device, dtype=x.dtype).view(3, 1, 1)
    vmax_t = torch.as_tensor(vmax, device=x.device, dtype=x.dtype).view(3, 1, 1)
    y = (x - vmin_t) / (vmax_t - vmin_t + 1e-8)
    return y.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()


class LensValidationCallback(pl.Callback):
    def __init__(
        self,
        lens_loader,
        every_n_validations: int = 5,
        num_integration_steps: int = 100,
        num_samples_per_cond: int = 5,
        figures_dir: Optional[str | Path] = None,
        run_name: Optional[str] = None,
    ):
        super().__init__()
        self.lens_loader = lens_loader
        self.every_n_validations = max(1, int(every_n_validations))
        self.num_integration_steps = num_integration_steps
        self.num_samples_per_cond = num_samples_per_cond
        self.figures_dir = Path(figures_dir) if figures_dir else None
        self.run_name = run_name or "lens_val"
        self._val_round = 0

    def _move_batch(self, batch, device):
        target, samegal, sameins, masks, metadata = batch
        return (
            target.to(device, non_blocking=True),
            samegal.to(device, non_blocking=True),
            sameins.to(device, non_blocking=True),
            masks.to(device, non_blocking=True) if masks is not None else None,
            metadata,
        )

    @torch.no_grad()
    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if not trainer.is_global_zero:
            return
        self._val_round += 1
        if self._val_round % self.every_n_validations != 0:
            return

        device = pl_module.device
        was_training = pl_module.training
        pl_module.eval()

        try:
            losses = []
            mse_48 = []
            mse_32 = []
            first_batch_for_plot = None

            for batch in self.lens_loader:
                batch = self._move_batch(batch, device)
                target, samegal, sameins, masks, metadata = batch

                # Velocity loss (re-uses the model's own loss path)
                loss = pl_module.compute_loss((target, samegal, sameins, masks, metadata))
                losses.append(float(loss.detach().cpu()))

                # Reconstruction MSE via sampling
                samples = pl_module.sample(
                    samegal,
                    sameins,
                    masks=masks,
                    num_steps=self.num_integration_steps,
                )
                diff = target - samples
                _, _, h, w = diff.shape
                for size, store in [(48, mse_48), (32, mse_32)]:
                    sx = (w - size) // 2
                    sy = (h - size) // 2
                    crop = diff[:, :, sy:sy + size, sx:sx + size]
                    store.append(float((crop ** 2).mean().detach().cpu()))

                if first_batch_for_plot is None:
                    first_batch_for_plot = (target.cpu(), samegal.cpu(), sameins.cpu(), masks.cpu() if masks is not None else None)

            mean_loss = float(np.mean(losses)) if losses else float("nan")
            mean_mse_48 = float(np.mean(mse_48)) if mse_48 else float("nan")
            mean_mse_32 = float(np.mean(mse_32)) if mse_32 else float("nan")

            log_dict = {
                "lens_val/loss": mean_loss,
                "lens_val/mse_48": mean_mse_48,
                "lens_val/mse_32": mean_mse_32,
                "lens_val/round": self._val_round,
                "global_step": trainer.global_step,
            }
            if trainer.logger and hasattr(trainer.logger, "experiment") and hasattr(trainer.logger.experiment, "log"):
                trainer.logger.experiment.log(log_dict)
            else:
                print(f"[lens-val] {log_dict}")

            # Sample grid
            if first_batch_for_plot is not None:
                self._save_grid(
                    pl_module=pl_module,
                    device=device,
                    target=first_batch_for_plot[0],
                    samegal=first_batch_for_plot[1],
                    sameins=first_batch_for_plot[2],
                    masks=first_batch_for_plot[3],
                    trainer=trainer,
                )
        finally:
            if was_training:
                pl_module.train()

    @torch.no_grad()
    def _save_grid(self, pl_module, device, target, samegal, sameins, masks, trainer):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = target.shape[0]
        ns = self.num_samples_per_cond
        ncols = 3 + ns + 1
        fig, axes = plt.subplots(n, ncols, figsize=(2 * ncols, 2 * n), squeeze=False)
        col_titles = ["SameGal", "SameIns[0]", "Target"] + [f"S{j+1}" for j in range(ns)] + ["Mean"]
        for j, t in enumerate(col_titles):
            axes[0, j].set_title(t, fontsize=10)

        for i in range(n):
            sg = samegal[i:i+1].to(device)
            si = sameins[i:i+1].to(device)
            mk = masks[i:i+1].to(device) if masks is not None else None
            sg_rep = sg.repeat(ns, 1, 1, 1)
            si_rep = si.repeat(ns, 1, 1, 1, 1)
            mk_rep = mk.repeat(ns, 1) if mk is not None else None
            samples = pl_module.sample(sg_rep, si_rep, masks=mk_rep, num_steps=self.num_integration_steps)
            mean_s = samples.mean(dim=0, keepdim=True)

            tgt = target[i:i+1].to(device)
            vmin = tgt[0, :3].amin(dim=(1, 2))
            vmax = tgt[0, :3].amax(dim=(1, 2))

            axes[i, 0].imshow(_row_scale_rgb(sg[0, :3], vmin, vmax)); axes[i, 0].axis("off")
            axes[i, 1].imshow(_row_scale_rgb(si[0, 0, :3], vmin, vmax)); axes[i, 1].axis("off")
            axes[i, 2].imshow(_row_scale_rgb(tgt[0, :3], vmin, vmax)); axes[i, 2].axis("off")
            for j in range(ns):
                axes[i, 3 + j].imshow(_row_scale_rgb(samples[j, :3], vmin, vmax)); axes[i, 3 + j].axis("off")
            axes[i, -1].imshow(_row_scale_rgb(mean_s[0, :3], vmin, vmax)); axes[i, -1].axis("off")

        plt.tight_layout()

        if trainer.logger and hasattr(trainer.logger, "experiment") and hasattr(trainer.logger.experiment, "log"):
            trainer.logger.experiment.log({
                "lens_val/sample_grid": wandb.Image(fig),
                "global_step": trainer.global_step,
            })

        if self.figures_dir is not None:
            self.figures_dir.mkdir(parents=True, exist_ok=True)
            out_path = self.figures_dir / f"lens_val_step{trainer.global_step}.png"
            fig.savefig(out_path, dpi=120, bbox_inches="tight")
            print(f"[lens-val] saved figure {out_path}")
        plt.close(fig)
