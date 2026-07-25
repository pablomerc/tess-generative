"""
Negative-sampling and projection-head ablations for the contrastive baseline.

Motivation (see ANALYSIS.md §2-3). The published contrastive arms beat Ours on
instrument properties by a wide margin (Galaxy Depth 0.881 vs 0.682). Part of that
may be *real* — sky-neighbour positives share the anchor's PSF/depth, so the
instrument InfoNCE is near-supervised metric learning on the downstream instrument
targets. But part of it may be an artefact of WHAT ENDS UP IN THE NEGATIVE POOL:

  * The anchor survey alternates by dataset-index parity, so in any batch about
    half the items are HSC-anchored and half Legacy-anchored.
  * Galaxy branch: anchor i's negatives are the other items' `samegal` embeddings,
    which are a 50/50 mixture of "same survey as anchor i's positive" and "the
    other survey". Half the discrimination problem is therefore solvable by reading
    off survey identity rather than galaxy identity.
  * Instrument branch: anchor i's negatives are other anchors' neighbours, again a
    50/50 survey mixture. Separating HSC-from-Legacy is trivially easy compared to
    separating two HSC patches with different seeing, so a large slice of the
    instrument InfoNCE objective may be satisfied by a survey classifier.

If the instrument advantage survives when negatives are restricted to the anchor's
own survey, it is genuine within-survey observing-condition sensitivity. If it
collapses, then a large part of the published contrastive instrument score was
cross-survey discrimination, which is exactly the "sensitive to the composition of
a training batch" objection about contrastive objectives.

Three independent knobs, all defaulting to the published behaviour so this class
reproduces `DualEncoderContrastiveModule` bit-for-bit when left alone:

  galaxy_negatives:
    "mixed"        (default) published behaviour: every other item is a negative
    "same_survey"  keep only negatives drawn from the same survey as the anchor's
                   positive -> survey identity carries no signal (HARDER)
    "cross_survey" keep only negatives from the other survey (EASIER; included to
                   bracket the effect rather than because we expect to use it)

  instrument_negatives:
    "mixed"        (default) published behaviour: the candidate pool is all valid
                   neighbours from all items, both surveys
    "same_survey"  restrict candidates to neighbours whose owner shares the
                   anchor's survey -> must discriminate seeing/depth WITHIN a
                   survey (HARDER)

  use_projection_head:
    True  (default) SimCLR-style: InfoNCE on a 2-layer MLP head, and downstream
          probes read the pre-head encoder output (standard, and what the published
          arms did)
    False InfoNCE directly on the encoder output -- the representation the probes
          actually consume. Tests whether the head is doing the work: with a head,
          the encoder is free to keep information the loss would otherwise discard,
          which is normally an ADVANTAGE for contrastive on downstream probes.

Note on negative COUNT. Restricting negatives to one survey roughly halves the
number of candidates per anchor, and InfoNCE quality depends on negative count.
To separate "harder negatives" from "fewer negatives", run the restricted variants
at double batch size (--batch-size 128) as well; that restores the count while
keeping the restriction. Both are cheap.

Eval compatibility: this subclass adds no new encoder attributes, so
`prepare_combined.py` still duck-types it as contrastive and extracts
`encoder_galaxy` / `encoder_instrument` exactly as for the published arms. Point it
at this file with:
  --module contrastive_ablation/dual_encoder_contrastive_variants.py \
  --model-class ContrastiveNegativeAblationModule
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from galaxy_images.galaxy_model.contrastive_baseline.dual_encoder_contrastive import (
    DualEncoderContrastiveModule,
)

GALAXY_NEG_MODES = ("mixed", "same_survey", "cross_survey")
INSTRUMENT_NEG_MODES = ("mixed", "same_survey")


class ContrastiveNegativeAblationModule(DualEncoderContrastiveModule):
    def __init__(
        self,
        galaxy_negatives: str = "mixed",
        instrument_negatives: str = "mixed",
        use_projection_head: bool = True,
        **base_kwargs,
    ):
        if galaxy_negatives not in GALAXY_NEG_MODES:
            raise ValueError(f"galaxy_negatives must be one of {GALAXY_NEG_MODES}, "
                             f"got {galaxy_negatives!r}")
        if instrument_negatives not in INSTRUMENT_NEG_MODES:
            raise ValueError(f"instrument_negatives must be one of {INSTRUMENT_NEG_MODES}, "
                             f"got {instrument_negatives!r}")
        super().__init__(**base_kwargs)

        # Record on hparams so the settings land in the checkpoint and are passed back
        # by load_from_checkpoint (the parent's save_hyperparameters() only sees its own
        # signature). Do this before any use of self.hparams below.
        self.hparams.galaxy_negatives = galaxy_negatives
        self.hparams.instrument_negatives = instrument_negatives
        self.hparams.use_projection_head = use_projection_head

        if not use_projection_head:
            # InfoNCE directly on the encoder output. Identity keeps every call site
            # (including the parent's) working unchanged and adds no parameters.
            self.head_galaxy = nn.Identity()
            self.head_instrument = nn.Identity()

        print(
            f"[ContrastiveNegativeAblationModule] galaxy_negatives={galaxy_negatives} "
            f"instrument_negatives={instrument_negatives} "
            f"use_projection_head={use_projection_head}",
            flush=True,
        )

    # ---- helpers -------------------------------------------------------------

    @staticmethod
    def _is_hsc(metadata, device) -> torch.Tensor:
        return torch.tensor(
            [m.get("anchor_survey", "hsc") == "hsc" for m in metadata],
            dtype=torch.bool, device=device,
        )

    def _masked_clip_loss(self, anchors, positives, temperature, keep_mask):
        """Symmetric InfoNCE with the diagonal as positives and `keep_mask` gating negatives.

        keep_mask: (B, B) bool, True = this (anchor, candidate) pair may appear in the
        denominator. The diagonal must be True. Both directions are masked, so the
        A->P and P->A halves see the same restricted candidate set.
        """
        a = F.normalize(anchors, dim=1)
        p = F.normalize(positives, dim=1)
        logits = (a @ p.T) / temperature
        neg_inf = torch.finfo(logits.dtype).min
        logits_fwd = logits.masked_fill(~keep_mask, neg_inf)
        # Reverse direction: candidate m against all anchors -> transpose the mask too.
        logits_rev = logits.T.masked_fill(~keep_mask.T, neg_inf)

        labels = torch.arange(logits.size(0), device=logits.device)
        loss = 0.5 * (F.cross_entropy(logits_fwd, labels)
                      + F.cross_entropy(logits_rev, labels))
        acc = 0.5 * ((logits_fwd.argmax(dim=1) == labels).float().mean()
                     + (logits_rev.argmax(dim=1) == labels).float().mean())
        return loss, acc

    def _multi_positive_infonce_masked(
        self, anchors, positive_pool, positive_owner, temperature, candidate_mask,
    ):
        """Parent's multi-positive InfoNCE, plus a (B, M) gate on the candidate pool.

        Positives are always allowed regardless of the gate (they are same-survey by
        construction, so this only matters defensively).
        """
        if positive_pool.size(0) == 0:
            zero = anchors.new_tensor(0.0)
            return zero, zero

        a = F.normalize(anchors, dim=1)
        p = F.normalize(positive_pool, dim=1)
        logits = (a @ p.T) / temperature  # (B, M)

        owner = positive_owner.unsqueeze(0)                                    # (1, M)
        anchor_ids = torch.arange(a.size(0), device=a.device).unsqueeze(1)      # (B, 1)
        pos_mask = (owner == anchor_ids)                                        # (B, M)

        allowed = candidate_mask | pos_mask
        neg_inf = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(~allowed, neg_inf)

        log_denom = torch.logsumexp(logits, dim=1)
        log_num = torch.logsumexp(logits.masked_fill(~pos_mask, neg_inf), dim=1)

        valid_anchor = pos_mask.any(dim=1)
        if not valid_anchor.any():
            zero = anchors.new_tensor(0.0)
            return zero, zero

        loss = -(log_num[valid_anchor] - log_denom[valid_anchor]).mean()
        pred_owner = positive_owner[logits.argmax(dim=1)]
        acc = (pred_owner[valid_anchor]
               == torch.arange(a.size(0), device=a.device)[valid_anchor]).float().mean()
        return loss, acc

    # ---- objective -----------------------------------------------------------

    def _compute_losses(self, batch):
        # Delegate untouched to the published implementation when nothing is restricted,
        # so the default configuration is provably the baseline (no re-derivation risk).
        if (self.hparams.galaxy_negatives == "mixed"
                and self.hparams.instrument_negatives == "mixed"):
            return super()._compute_losses(batch)

        targets, samegals, sameins, masks, metadata = batch
        B = targets.size(0)
        is_hsc = self._is_hsc(metadata, targets.device)

        # ---- galaxy branch ---------------------------------------------------
        z_t_g = self.head_galaxy(self.encoder_galaxy(targets))
        z_sg_g = self.head_galaxy(self.encoder_galaxy(samegals))

        if self.hparams.galaxy_negatives == "mixed":
            keep = torch.ones(B, B, dtype=torch.bool, device=targets.device)
        else:
            # samegal_j lives in the survey opposite to anchor j, so candidate j shares
            # the anchor i's positive survey exactly when anchor j shares anchor i's survey.
            same_anchor_survey = is_hsc.unsqueeze(1) == is_hsc.unsqueeze(0)   # (B, B)
            keep = same_anchor_survey if self.hparams.galaxy_negatives == "same_survey" \
                else ~same_anchor_survey
            # The positive pair must always survive.
            keep = keep | torch.eye(B, dtype=torch.bool, device=targets.device)
        loss_galaxy, acc_galaxy = self._masked_clip_loss(
            z_t_g, z_sg_g, self.hparams.temperature_galaxy, keep,
        )

        # ---- instrument branch ----------------------------------------------
        z_t_i = self.head_instrument(self.encoder_instrument(targets))

        _B, K, C, H, W = sameins.shape
        z_si_flat = self.head_instrument(self.encoder_instrument(sameins.view(B * K, C, H, W)))
        masks_flat = masks.view(B * K).bool()
        z_si_valid = z_si_flat[masks_flat]
        owner = (
            torch.arange(B, device=targets.device)
            .unsqueeze(1).expand(B, K).reshape(B * K)[masks_flat]
        )

        if self.hparams.instrument_negatives == "mixed":
            candidate_mask = torch.ones(B, owner.numel(), dtype=torch.bool,
                                        device=targets.device)
        else:
            # Candidate m belongs to owner[m], whose neighbours come from that owner's
            # survey. Allow only candidates whose survey matches this anchor's.
            owner_is_hsc = is_hsc[owner]                                      # (M,)
            candidate_mask = is_hsc.unsqueeze(1) == owner_is_hsc.unsqueeze(0)  # (B, M)

        loss_instrument, acc_instrument = self._multi_positive_infonce_masked(
            z_t_i, z_si_valid, owner, self.hparams.temperature_instrument, candidate_mask,
        )

        loss = (self.hparams.lambda_galaxy * loss_galaxy
                + self.hparams.lambda_instrument * loss_instrument)
        return loss, {
            "loss": loss,
            "loss_galaxy": loss_galaxy.detach(),
            "loss_instrument": loss_instrument.detach(),
            "acc_galaxy": acc_galaxy.detach(),
            "acc_instrument": acc_instrument.detach(),
        }
