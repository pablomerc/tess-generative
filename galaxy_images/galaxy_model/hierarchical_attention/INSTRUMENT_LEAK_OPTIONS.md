# Hierarchical Attention — Instrument Leak: Logic Behind Each Proposed Option

## Recap of the diagnosis (one paragraph)

In the **baseline** (`ConditionalFlowMatchingModule`), `encoder_2` produces **spatial tokens** that get **concatenated with `encoder_1`'s tokens** and routed through UNet cross-attention as one big set. The UNet can attend to any token at any spatial query position. Since `encoder_2`'s inputs (k same-instrument neighbors) are guaranteed to share the anchor's survey, the gradient pushes `encoder_2` to encode survey-specific spatial information (PSF, noise, color). `encoder_1`'s input is the same-galaxy image from the *opposite* survey, so it has no gradient pressure to encode the anchor's survey — it ends up encoding galaxy-only structure. **Disentanglement is emergent**, purely from the asymmetric input role.

In the **hierarchical variant** (`HierarchicalGlobalInstrumentFlowMatchingModule`), `encoder_2` is mean-pooled into a single global vector and injected only into the **time embedding** (AdaGN-style). It can no longer carry **spatially-resolved** survey signal. Cross-attention now sees only `encoder_1`'s spatial tokens. Because the UNet still needs spatial instrument cues to denoise correctly, gradient flow forces `encoder_1` to absorb survey identity — visible as HSC/Legacy separation in the physics latent UMAP.

Each option below tackles this in a different way. Logic, mechanism, expected outcome, cost, and risk are spelled out for each.

---

## Option 1 — Spatial instrument variant (recommended)

### What it changes

Add a new variant (e.g. `neighbors_hier_spatial_ins`) in which `encoder_2`:
- keeps `mean_pool=False` (spatial output, like the baseline);
- emits per-neighbor spatial tokens that are **concatenated with `encoder_1`'s spatial tokens at the matched UNet cross-attn level**;
- the global `instrument_global_proj` pathway is removed (or kept as a cheap auxiliary).

In other words: bring the baseline's "concat-and-attend" instrument routing back into the hierarchical model, while keeping the multi-level + RoPE + global_vec hierarchical features that you wanted from the new architecture.

### Logic / why it should work

The diagnosis says the leak is caused by removing encoder_2's spatial pathway. The most direct way to test that — and the most direct fix if it's right — is to put the spatial pathway back. If the leak goes away, the hypothesis is confirmed in the same run that fixes the model. You don't need a separate ablation.

This option also keeps the "physics vs instrument" separation **emergent from input asymmetry**, which is the same mechanism the baseline relies on. You're not introducing a new loss term, new hyperparameter, or new failure mode — you're restoring a property the baseline already proved works.

### Mechanism, concretely

For `bn_36x16` config, encoder_1 emits one spatial level at 6×6=36 tokens of dim 16, plus a 64-d global vec. Encoder_2 with `mean_pool=False` on the same backbone emits a 6×6=36-token map per neighbor. With k=5 neighbors that's 5×36=180 tokens. Concatenated with encoder_1's 36 tokens = 216 tokens, all dim 16. Padding mask zeros out invalid neighbors. Routed to the matched cross-attn level via the existing `level_map` machinery in `DualGlobalConditionedUNet`.

If you want to be conservative on compute, restrict the spatial instrument tokens to one level (the deepest), or downsample neighbors from 5 to 2-3.

### Expected outcome

- Physics-latent UMAP (combined-spatial and global panels) should overlap HSC/Legacy rather than splitting.
- `val/loss`, `val/mse_hsc`, `val/mse_legacy` should be the same or better than the current hierarchical run — restoring the spatial instrument path gives the UNet more, not less, to work with.
- `val/loss` of the new variant should land near the baseline's, since the wiring is now structurally similar to baseline at the cross-attn level.

### Cost / risk

- **Cost:** one new variant + one new SLURM run, ~1 day of training to be comparable to your current hierarchical checkpoint at the same step count. Code change is small (~50 lines + config + SLURM).
- **Risk:** a bit more cross-attention compute (216 vs 36 tokens at the deepest level). Memory should still fit fine. Negligible architectural risk — this is exactly the routing the baseline already runs successfully.
- **What if it doesn't work?** Then encoder_2's spatial pathway alone doesn't explain the leak, and the second global pathway (encoder_1's `global_vec` → `class_embedding`) is also contributing. Move to Option 3 to isolate that, or stack Option 2 on top.

---

## Option 2 — Add an instrument-invariance loss on encoder_1

### What it changes

Add an explicit term to the training objective that pulls encoder_1's outputs for `(HSC view, Legacy view)` of the same galaxy together. Concrete forms:
- **InfoNCE**: positives = same-galaxy cross-survey pair, negatives = other galaxies in the batch. Operates on a flattened encoder_1 embedding (e.g. `spatial_flat` or `global_vec`).
- **Sinkhorn / Geometric**: divergence between the distribution of HSC vs Legacy encoder_1 embeddings in the batch (the existing `geom_loss_fn` already supports this — `lambda_geometric` was set to 0).
- **Adversarial / DANN**: a small MLP head on encoder_1 that tries to predict survey, with a gradient-reversal layer so encoder_1 learns to remove the signal.

### Logic / why it should work

The baseline's instrument-invariance is **emergent** from input asymmetry — there's no explicit loss enforcing it. In the hierarchical variant, the architecture has changed enough that the emergence breaks down. Option 2 doesn't try to fix the architecture; instead it adds an **explicit constraint** that does what the architecture used to do for free.

This is appealing because:
- It's architecture-agnostic. The same loss would also disentangle the single-encoder ablation (which currently has no mechanism for disentanglement at all), and it would harden the baseline against future architecture changes.
- It directly targets the symptom (survey-separation in encoder_1) rather than the upstream cause.

### Mechanism, concretely

The dataloader already produces (anchor, same-galaxy) pairs where one is HSC and one is Legacy (50/50 anchor survey). For each batch, run encoder_1 on both anchor and same-galaxy images, then:
- For InfoNCE: project both through a small head, L2-normalize, compute symmetric cross-entropy with positives along the diagonal. Add `lambda_invariance * info_nce_loss` to the total.
- For Sinkhorn: simply turn `lambda_geometric` back on (it's already wired up — `compute_loss` lines 494-498 in `double_train_fm_neighbors_hier_global_ins.py`, but currently 0).

### Expected outcome

- Strong reduction (likely elimination) of HSC/Legacy separation in the encoder_1 UMAP.
- Some risk to generative quality if the loss is too strong — the encoder may sacrifice physics features that happened to correlate with survey (e.g., flux scale, SNR).
- Need to tune `lambda_invariance` — probably start small (0.01-0.1) and observe both UMAP and val/mse.

### Cost / risk

- **Cost:** 30-100 lines in the model module, no new variant required. New hyperparameter to tune.
- **Risk:** **higher than Option 1.** Invariance losses can be fiddly: too weak → no effect, too strong → degrades reconstruction. They also pull all of `encoder_1`'s features toward instrument-invariance even when survey-correlated structure was useful. The Sinkhorn variant is already implemented, so trying that first costs almost nothing — but the existing baseline never needed it, which is suspicious: why pay a hyperparameter cost for something the architecture should give you for free?
- **What if it doesn't work?** Either the loss weight is wrong (tune it), or the leak is too entrenched in the architecture (combine with Option 1).

---

## Option 3 — Diagnostic ablation: drop the encoder_1 `global_vec` → `class_embedding` pathway

### What it changes

In `DualGlobalConditionedUNet.forward` (lines 77-78), the encoder_1's `global_vec` is added to the time embedding via `self.unet.class_embedding(class_labels)`. Disable that line (or pass `class_labels=None`) in a variant. Encoder_2's instrument global path stays intact.

### Logic / why it should run

The current architecture gives encoder_1 **two** routes to the UNet: spatial cross-attention (per-token) AND a global modulation pathway (via `class_embedding`). The second route is a clean extra channel for any global property of the input image — including instrument identity. It's plausible that this global pathway is doing a meaningful share of the leak, separate from the missing encoder_2 spatial path.

This option doesn't fix the leak; it **isolates** a contribution. If disabling `class_embedding(global_vec)` alone substantially reduces UMAP separation, then the global pathway is a major culprit and Option 1 should additionally remove or constrain it. If it doesn't change the UMAP at all, the leak is dominated by gradient pressure on encoder_1's spatial tokens and Option 1 alone should suffice.

### Mechanism, concretely

One-line change in `DualGlobalConditionedUNet.forward`: gate the class-embedding addition on a config flag. Add a new variant entry. Train and compare UMAPs at matched step counts.

### Expected outcome

- Likely partial reduction in the "Physics - Global" panel (since you literally removed one path that was carrying global signal).
- May or may not affect "Physics - Combined Spatial" — that depends on whether spatial tokens were independently absorbing survey signal or were just downstream of the same backbone features that fed `global_vec`.
- Likely small hit to `val/mse` since you removed conditioning capacity.

### Cost / risk

- **Cost:** smallest of all three. ~10 lines + config + SLURM. One run.
- **Risk:** doesn't actually fix the model on its own. If you go this route first you'll likely still need Option 1 afterwards.
- **What if it changes nothing?** Then the global pathway isn't the issue, and you've still narrowed the diagnosis. If it changes a lot, you've found a secondary fix to fold into Option 1.

---

## Option 4 — Just the diagnosis, no fix

### What it changes

Nothing — finalize the plan as a write-up of the root cause and stop.

### Logic

If the hierarchical model is a side experiment and you don't need to fix it right now (e.g. you're moving on to a different architecture), spending compute on a fix is wasted. The diagnosis itself is the deliverable: future work knows that any architecture which removes the spatial instrument pathway must add an explicit invariance constraint.

### Cost / risk

- **Cost:** zero compute, zero code.
- **Risk:** the model stays broken. The single_encoder_ablation also stays broken (though that's already known/expected).

---

## How to choose between them

| If you want to...                                                        | Pick      |
|--------------------------------------------------------------------------|-----------|
| ...fix the model with one experiment, low risk, modest compute           | Option 1  |
| ...build a property that hardens *all* variants (incl. single-encoder)   | Option 2  |
| ...understand which of the two architecture changes contributes more    | Option 3  |
| ...stop here and use the diagnosis as the deliverable                    | Option 4  |
| ...be thorough — run Option 3 then Option 1                              | 3 then 1  |
| ...maximally robust fix — run Option 1 and stack Option 2 on top         | 1 + 2     |

My recommendation remains **Option 1**: it's the most direct test of the hypothesis, the most direct fix, requires the fewest new hyperparameters, and the result is informative either way. Option 2 is the right move if Option 1 fails or if you want a property that generalizes to other architectures. Option 3 is a cheap "before/after" diagnostic if you want to fully decompose the architectural contributions.
