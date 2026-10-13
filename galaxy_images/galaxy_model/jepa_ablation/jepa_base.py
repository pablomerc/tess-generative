"""
Shared machinery for the JEPA ablation modules (see PLAN.md).

Every arm is a BYOL/JEPA-style predict-in-embedding-space objective:

    online (context) encoder -> projector -> predictor  ==>  EMA target encoder -> projector

with stop-gradient on the target side and no negatives. Collapse is prevented by
the EMA + predictor asymmetry; we log embedding-std diagnostics to verify.

Conventions shared with the contrastive baseline:
  * the EMA *target* encoders are exposed as `encoder_galaxy` / `encoder_instrument`
    (properties defined in the concrete subclasses), so prepare_combined.py,
    encode_latents.py and the UMAP tooling duck-type on them unchanged;
  * downstream eval always uses raw encoder outputs, never the projection heads.

Schedules (PLAN.md decision #4): step-based cosine LR with linear warmup — NOT the
baseline's `CosineAnnealingLR(T_max=max_epochs)` quirk (see CLAUDE.md "LR schedule").
EMA momentum follows the BYOL cosine ramp tau_base -> 1.0 over max_steps.
"""
from __future__ import annotations

import copy
import math

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


class JEPABase(pl.LightningModule):
    """Base class: EMA updates, loss, schedules, diagnostics, survey-colored UMAP."""

    # ------------------------------------------------------------------ EMA --
    @staticmethod
    def _clone_frozen(module: torch.nn.Module) -> torch.nn.Module:
        target = copy.deepcopy(module)
        for p in target.parameters():
            p.requires_grad_(False)
        return target

    def _ema_pairs(self):
        """Return [(online_module, target_module), ...]; defined per arm."""
        raise NotImplementedError

    def _max_steps(self) -> int:
        ms = int(self.trainer.max_steps or 0)
        if ms <= 0:
            raise ValueError(
                "JEPA modules require trainer.max_steps > 0 (step-based cosine "
                "LR + EMA schedules; see CLAUDE.md 'LR schedule')."
            )
        return ms

    def _current_tau(self) -> float:
        max_steps = self._max_steps()
        step = min(int(self.global_step), max_steps)
        tau_base = float(self.hparams.ema_base_momentum)
        # BYOL ramp: tau_base at step 0, 1.0 at max_steps.
        return 1.0 - (1.0 - tau_base) * 0.5 * (1.0 + math.cos(math.pi * step / max_steps))

    @torch.no_grad()
    def _ema_update(self):
        tau = self._current_tau()
        for online, target in self._ema_pairs():
            for p_o, p_t in zip(online.parameters(), target.parameters()):
                p_t.mul_(tau).add_(p_o.detach(), alpha=1.0 - tau)
            # Buffers (BatchNorm running stats) are copied, not EMA'd.
            for b_o, b_t in zip(online.buffers(), target.buffers()):
                b_t.copy_(b_o)
        self.log("train/ema_tau", tau, on_step=True, on_epoch=False)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self._ema_update()

    # ----------------------------------------------------------------- loss --
    @staticmethod
    def _cosine_loss(pred: torch.Tensor, target: torch.Tensor):
        """BYOL loss 2 - 2*cos on L2-normalized vectors. Returns (loss, mean_cos)."""
        p = F.normalize(pred, dim=-1)
        t = F.normalize(target.detach(), dim=-1)
        cos = (p * t).sum(dim=-1)
        return (2.0 - 2.0 * cos).mean(), cos.detach().mean()

    @staticmethod
    def _embedding_std(z: torch.Tensor) -> torch.Tensor:
        """Mean per-dim std of L2-normalized embeddings — the collapse diagnostic.

        A healthy 32-d space sits near 1/sqrt(32) ~ 0.18; collapse drives it to ~0.
        """
        z = F.normalize(z.detach().float(), dim=-1)
        return z.std(dim=0).mean()

    # ------------------------------------------------------------- optimizer --
    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = AdamW(params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

        max_steps = self._max_steps()
        warmup = min(int(self.hparams.warmup_steps), max(1, max_steps // 2))

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return (step + 1) / warmup
            t = (step - warmup) / max(1, max_steps - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))

        scheduler = LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    # ----------------------------------------------------------------- UMAP --
    # Fixed (hsc, legacy) anchor pairs set by the train entry via set_umap_pairs();
    # plain attributes, not checkpointed. If unset, UMAP logging is a no-op.
    _umap_hsc: torch.Tensor | None = None
    _umap_legacy: torch.Tensor | None = None

    def set_umap_pairs(self, hsc: torch.Tensor, legacy: torch.Tensor):
        self._umap_hsc = hsc
        self._umap_legacy = legacy

    def _umap_panels(self):
        """[(title, encoder_module), ...] — one UMAP panel per encoder."""
        return [("Encoder (EMA) UMAP", self.encoder_galaxy)]

    @torch.no_grad()
    def _encode_chunks(self, encoder, imgs: torch.Tensor, chunk: int = 256) -> torch.Tensor:
        outs = []
        for i in range(0, imgs.shape[0], chunk):
            x = imgs[i : i + chunk].to(self.device)
            outs.append(encoder(x).flatten(start_dim=1).float().cpu())
        return torch.cat(outs, dim=0)

    @torch.no_grad()
    def on_validation_epoch_end(self):
        if not getattr(self.hparams, "enable_umap_logging", True):
            return
        if self._umap_hsc is None or self._umap_legacy is None:
            return
        if not self.logger or not hasattr(self.logger, "experiment"):
            return
        try:
            import umap
            import matplotlib.pyplot as plt
            import wandb
        except Exception as e:
            if not hasattr(self, "_umap_import_warned"):
                print(f"[UMAP] Skipping UMAP logging; import failed: {e}")
                self._umap_import_warned = True
            return

        reducer_kwargs = {
            "n_neighbors": int(self.hparams.umap_n_neighbors),
            "min_dist": float(self.hparams.umap_min_dist),
            "n_components": 2,
            "metric": "euclidean",
            "random_state": 42,
            "n_jobs": 1,
        }
        panels = self._umap_panels()
        fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5), squeeze=False)
        n_hsc = self._umap_hsc.shape[0]
        for ax, (title, encoder) in zip(axes[0], panels):
            z_h = self._encode_chunks(encoder, self._umap_hsc)
            z_l = self._encode_chunks(encoder, self._umap_legacy)
            emb = umap.UMAP(**reducer_kwargs).fit_transform(
                torch.cat([z_h, z_l], dim=0).numpy()
            )
            ax.scatter(emb[:n_hsc, 0], emb[:n_hsc, 1], s=6, alpha=0.6, label="HSC")
            ax.scatter(emb[n_hsc:, 0], emb[n_hsc:, 1], s=6, alpha=0.6, label="Legacy")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend()
        plt.tight_layout()
        self.logger.experiment.log(
            {"val/latent_umap": wandb.Image(fig), "global_step": self.global_step}
        )
        plt.close(fig)
