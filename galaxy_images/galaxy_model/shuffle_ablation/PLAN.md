# Shuffle-Conditioning Ablation — Plan (Reviewer SYJm, Q2)

**Status: IMPLEMENTED** (`run_shuffle_ablation.py`, `metrics.py`,
`make_tables.py`, `make_figures.py`, `run_engaging.slurm`). Defaults below are
recommendations; overridable decision points are collected in §7.
**Target: NeurIPS rebuttal (~2026-07-30). Runs on Engaging, no retraining.**
All harness/metric claims in §2–§4 were verified against the code on 2026-07-24
(file:line cites throughout).

---

## 1. What the reviewer asked (Q2, review_1_SYJm)

> Replace either z_phy or z_ins with a shuffled/random latent and examine how the
> generated output changes. If the disentanglement is effective, corrupting z_phy
> should mainly affect galaxy morphology/source identity, while corrupting z_ins
> should mainly affect instrument-specific statistics.

Interpretation (settled): **inference-time intervention on the trained paper
checkpoint** — no retraining. The reviewer's premise (x_t carries target info at
large t, so the FM loss doesn't force conditioning use there) is correct for the
*training objective*, but at sampling the ODE starts from pure noise and the anchor
is never encoded, so all content must flow through the latents. What this ablation
adds beyond existing results is the **causal division of labor at the decoder**
(probes only show information *content* in the latents, not decoder *usage*).

What the paper already contains (cite these in the response):
- Fig. 7 SNR traversal + artifact removal = existing *interventions on z_ins*
  (fixed noise, swapped instrument conditioning).
- App. A.5 random-vs-spatial-neighbor retraining = existing *training-time*
  corruption of the instrument pathway (Galaxy Depth R² 0.68→0.20).
- Reconstruction MSE + zero-shot ellipticity transfer = aggregate decoder reliance.

The genuinely new piece: **systematic, quantitative, paired corruption of each
latent separately — especially the z_phy direction.**

---

## 2. Core experimental design

### 2.1 Conditions (locked)

For each anchor i in the fixed eval set, generate with **identical noise x_0**
across all conditions (paired design):

| # | Condition     | samegal (→ z_phy)        | sameins (→ z_ins)          |
|---|---------------|--------------------------|----------------------------|
| C0| intact        | own                      | own                        |
| C1| shuffle-phy   | donor π(i)'s samegal     | own                        |
| C2| shuffle-ins   | own                      | donor π(i)'s sameins+masks |
| C3| shuffle-both  | donor π(i)'s samegal     | donor π(i)'s sameins+masks |

C3 is the cheap "everything follows the donor" reference row.
Random-Gaussian-latent variants (reviewer's "or random") are **deferred to an
optional phase 2** (§7 Q-A): they need a small `forward()` override (§4.4) and
are likely off-manifold for the cross-attention KV, so shuffled is the primary,
fairest intervention. If asked in the rebuttal thread, phase 2 is ~1 extra day.

### 2.2 The permutation π — within-survey, no fixed points (locked)

- Group the 256 anchors by survey: **123 hsc / 133 legacy** (verified against
  the manifest; survey == dataset-index parity, `neighbors_ram48.py:69-100`,
  parity==anchor_survey checked for all 256 entries).
- π = **cyclic shift by 1 within each survey group** (deterministic, zero fixed
  points, no RNG). Groups come from the manifest `anchor_surveys` list — no
  loader changes needed.
- **Within-survey only**: shuffling across surveys would turn the intervention
  into a survey swap and conflate this ablation with the cross-instrument
  transfer results already in the paper (Fig. 6/7). Within-survey keeps the
  latent *plausible* and changes only its *identity/field*. It also keeps
  samegal/sameins survey roles consistent automatically (samegal is always the
  opposite survey, sameins always the anchor survey).

Key payoff of the donor design — *positive* controls, not just damage metrics,
because every donor galaxy also has ground truth in the anchor survey:
- **C1 (shuffle-phy):** generated image should match the *donor's* anchor-survey
  image x_1[π(i)], not the own target x_1[i] → "identity follows z_phy".
- **C2 (shuffle-ins):** generated background noise σ should track the *donor
  field's* σ, not the own field's σ → "observing conditions follow z_ins".

### 2.3 Eval set + noise protocol (locked — inherited from diffusion ablation)

Reuse the diffusion-ablation protocol verbatim so all numbers are mutually
comparable:
- n=256 anchors = the **first 4 val batches of 64**, seed-42 split
  (`pl.seed_everything(42)` → `build_neighbors_dataloaders`, val_ratio=0.05,
  val loader unshuffled; `eval_recon_mse.py:364-369`, `data_factory.py:279-297`).
- Identity guarded by `diffusion_ablation/results/recon_eval_manifest.json`
  (dataset_indices, catalog_idxs, anchor_surveys, split_seed=42,
  noise_seed=1234; `_load_or_create_manifest`, `eval_recon_mse.py:133-184`).
- **Noise is positional**: one generator per batch, seeded
  `noise_seed + batch_idx` (`eval_recon_mse.py:227-236`). Batch boundaries
  (64) are part of the protocol identity ⇒ **permute conditioning tensors
  only; never re-order anchors or noise.** Keep batch_size=64 configs or the
  manifest guard refuses.
- Extra posterior samples (M>1, §7 Q-C): sample m=0 uses the exact existing
  seeding (stays comparable with the diffusion-ablation CSVs); samples m≥1 use
  `noise_seed + 20_000*m + batch_idx`.
- Data: ram48 store `/orcd/pool/007/pablomer/efficient_neighs_48` via
  `diffusion_ablation/configs/neighbors_fm_control.json` (identical data block
  to the ddpm config; loads ~16–17 GB into RAM → keep the 64G slurm `--mem`).
- Do NOT use `data.save_heldout_validation` with ram48 (4-tuple items break the
  exporter, `data_factory.py:165`) — the manifest replaces it.
- Note for the response: internally consistent across conditions, but the split
  is not byte-identical to the paper's original heldout file (AMD-cluster only),
  so absolute MSE may differ slightly from the paper's 0.081/0.197.

### 2.4 Checkpoint (locked: paper base only)

- Paper model (baseline `neighbors_all_attn`, best-epoch=228-step=87000):
  `galaxy_images/galaxy_model/checkpoints/base/snapshot.ckpt`
  (sha256-verified copy of the /work1 original; provenance in
  `checkpoints/base/source.txt`). Loads via
  `ConditionalFlowMatchingModule.load_from_checkpoint(path, map_location='cpu')`
  (`eval_recon_mse.py:187-195`); image_size=48 / in_channels=4 ⇒ shape-
  compatible with the ram48 eval data.
- **Bonus (verified): the paper checkpoint has never been run through this
  harness** — `eval_recon_mse.slurm` always passed `--skip-paper`, and no CSV
  has an fm-paper row. Our C0 condition therefore doubles as the missing
  paper-model baseline row that diffusion_ablation/PROGRESS.md lists as TODO,
  and gives the W1 response an FM-paper reference point for free.
- Optional extension (§7 Q-D): rerun the same script with the DDPM/FM-control
  finals for the W1 tie-in — the harness pieces are shared by construction.

---

## 3. Metrics

Per condition × anchor survey (hsc-anchored / legacy-anchored). **All metrics
computed in the model's normalized space for BOTH generated and real images**
(verified correct: `model.sample()` returns normalized-space images, dataset
targets are stored in the same space, and the σ estimators are shift-invariant
and scale-covariant — see §4.3; never compare absolute σ against
`hsc_noise_metrics.h5`, which is in raw flux units; for sky pixels
σ_flux ≈ 0.315·σ_norm if ever needed).

**Identity / morphology (should move under C1, not C2):**
1. `MSE_own`   = MSE(gen_i, x_1[i]) — full 48×48 + center-32 crop, per-image,
   aggregated mean±sem by survey exactly as `eval_recon_mse.py:255-266`.
2. `MSE_donor` = MSE(gen_i, x_1[π(i)]) — C1 prediction: MSE_donor ≪ MSE_own and
   ≈ C0's MSE_own level ("identity follows z_phy"). Also record C0's MSE_donor
   as the unrelated-pair reference level. Donor targets are in the eval set —
   free.

**Instrument statistics (should move under C2, not C1):**
3. Background noise σ: `corner_sky_rms`, `adjacent_diff_sigma`,
   `masked_mad_sigma` (input: (N, bands, 48, 48) numpy). Headline analysis:
   per-anchor scatter/regression of σ(gen_i) against σ(own anchor) vs
   σ(donor anchor) under C2 — generated σ should track the **donor**.
4. Per-band radial power spectral density + pixel-lag autocorrelation
   (`radial_average`, `apply_window`, `compute_radial_power_spectrum`,
   `compute_radial_correlation_direct`; per-band 2D float64 input). Report
   high-k power (noise floor) per condition.

**Dropped for this ablation (§7 Q-B):** zero-shot ellipticity R² — the frozen
ResNet weights (`resnet_experiment/resnet_best.pth` + `resnet_data.h5`) are not
on Engaging. σ-tracking + MSE_donor + high-k PSD carry the 2×2 signature
without it. Revisit only if the reviewer pushes back.

**Qualitative figure (rebuttal/appendix):**
5. Grid per example: `own target | C0 | C1 | C2 | donor target`, a handful of
   galaxies per direction, shared noise. This makes the result legible at a
   glance.

Expected-outcome table (what "disentanglement works" predicts):

| Metric                          | C1 shuffle-phy      | C2 shuffle-ins        |
|---------------------------------|---------------------|-----------------------|
| MSE_own                         | ↑↑ (≈ C3 level)     | ≈ C0 (small ↑)        |
| MSE_donor                       | ↓ (≈ C0's MSE_own)  | ≈ unrelated-pair level|
| σ(gen) tracks…                  | own field           | **donor** field       |
| high-k PSD                      | ≈ C0                | shifts toward donor   |

Failure modes this exposes: decoder ignoring z_ins (C2 ≈ C0 on all metrics incl.
σ-tracking), z_phy leaking instrument info (C1 moves σ/PSD), decoder ignoring
both (C1 ≈ C0 on identity — already excluded by existing MSE, but shown
directly here).

---

## 4. Implementation

New files under `galaxy_images/galaxy_model/shuffle_ablation/`. Run everything
as `python -m galaxy_images.galaxy_model.shuffle_ablation.<module>` from the
repo root (the diffusion_ablation module prepends the repo to sys.path and
expects that layout).

### 4.1 `run_shuffle_ablation.py` (core — zero model-code changes)

**Import, don't copy, from `diffusion_ablation.eval_recon_mse`** (verified
module-level functions): `_collect_eval_set` (:88), `_load_or_create_manifest`
(:133), `_load_model` (:187), `_unpack_batch` (:78), `_aggregate_rows` (:266),
`_sem` (:69), `_ping` (:23). Only `main()` is orchestration.

Flow:
1. Build loaders from `configs/neighbors_fm_control.json`, verify against the
   existing manifest (guards the eval-set identity).
2. Materialize the 256-anchor eval set in RAM (x_1 + samegal ≈ 9.4 MB each,
   sameins ≤ 47 MB — trivial). **Re-pad sameins to a common k across the set**
   (collate pads per-batch to the batch max, `neighbors.py:162`, so k can
   differ between the 4 batches; donors cross batch boundaries).
3. Compute π = within-survey cyclic shift from the manifest `anchor_surveys`.
4. Per batch (original order, 4×64) and per condition, generate:

```python
gen_c0 = model.sample(samegal,      sameins,      masks=masks,     x_noise=noise)
gen_c1 = model.sample(samegal[pi],  sameins,      masks=masks,     x_noise=noise)
gen_c2 = model.sample(samegal,      sameins[pi],  masks=masks[pi], x_noise=noise)
gen_c3 = model.sample(samegal[pi],  sameins[pi],  masks=masks[pi], x_noise=noise)
```

   (signature verified `double_train_fm_neighbors.py:643-649`; num_steps=250.)
   Verified safe: `forward()` has no cross-sample state — masks only zero the
   padded neighbor tokens (`:380-387`), encoders run per-image, conditioning is
   re-encoded every Euler step with no caching (`:690-695`).
   **Two hard rules:** (a) masks must ALWAYS travel with their sameins donor —
   and must be passed explicitly, since `sample()` silently substitutes
   all-ones when `masks=None` (`:672-674`), which would make padded zeros
   attended as real neighbors; (b) noise pairing is positional — permute
   conditioning only.
5. Outputs: generations per condition (`.npz`, ~38 MB per condition per M),
   per-anchor tidy CSV (`anchor_id, catalog_idx, survey, condition, m,
   mse_own, mse_donor, sigma_corner, sigma_adjdiff, sigma_mad, ...`), and
   `results/shuffle_manifest.json` (split seed, noise seeds, π, num_steps,
   checkpoint sha).

### 4.2 `metrics.py`

**Copy (not import) — verdicts verified:**
- The 4 PSD/autocorr functions from
  `generative_evaluation/power_autocorrelation_analysis_v3_heldout.py:49-90`
  — pure numpy/scipy, but their module runs an ROCm-specific
  `preferred_blas_library("hipblas")` call and drags in torch/geomloss/
  matplotlib config at import time, and the dir has no `__init__.py`.
- The 3 σ estimators + `CORNER = 8` from
  `latent_traversal/compute_hsc_noise_metrics.py:33,56-102` (~40 lines pure
  numpy; importing would pull a `discord_notify`/requests side-import).

### 4.3 `make_figures.py` + `make_tables.py`

§3.5 qualitative grid; condition × survey summary table (mean ± sem, paired
deltas C1−C0, C2−C0); σ-tracking scatter (σ_gen vs σ_own and σ_donor under C0
vs C2); high-k PSD overlay per condition.

### 4.4 (Phase 2, only if Q-A revisited) `z`-override hooks

~20-line backward-compatible extension to `ConditionalFlowMatchingModule.forward`
(+ threading through `sample`): optional `z_phy_override` / `z_ins_override`
kwargs that bypass `encoder_1` / `encoder_2` (+ `ins_proj`/mask logic) and drop
into the token concat at `double_train_fm_neighbors.py:386`. Random draws
moment-matched per token-dim to the eval set's real latent statistics.

### 4.5 `run_engaging.slurm`

Engaging conventions (per `downstream_evaluation/engaging/run_variant.slurm`):
`--partition=mit_normal_gpu --gres=gpu:1 --cpus-per-task=8 --mem=64G
--time=04:00:00`; `conda activate torchenv`; `--chdir` to repo root +
`PYTHONPATH=$REPO_ROOT`; Discord webhook ping on start/finish/failure (reuse
the `_ping`/`_ping_file` helpers + EXIT trap pattern from
`run_instrument_to_physics.slurm:22-37`; take the webhook from env).

**Compute (locked M=8):** 256 anchors × 4 conditions × M=8 × 250 Euler steps at
batch 64 on one H100 ≈ 1–2 h wall-clock; metrics are CPU-trivial. M=1 for a
smoke run first.

---

## 5. Reviewer response — draft skeleton (→ `reviews/claude-suggestions/`, Q2)

> **Hard rule:** never write into `reviews/` directly — those are the official
> review + official response files. All drafts go to
> `reviews/claude-suggestions/response_1_SYJm_q2.md`; Pablo merges them himself.

> We thank the reviewer for this suggestion and have run the requested ablation.
> Two clarifications first: (i) at sampling time the ODE is integrated from pure
> Gaussian noise and the anchor is never encoded, so all target content must
> flow through the conditioning latents — the interpolant x_t only contains
> target information during training; (ii) Fig. 7 already contains
> interventions on z_ins (SNR traversal with fixed noise; artifact removal with
> never-seen-in-training HSC+HSC conditioning). The new ablation adds the
> systematic, quantitative version, including the z_phy direction.
>
> Protocol: for 256 held-out galaxies we generate with identical initial noise
> under intact conditioning, shuffled z_phy (conditioning image of a different
> galaxy, same survey), and shuffled z_ins (sky-neighbors of a different field,
> same survey). Because each "donor" galaxy also has ground truth in the target
> survey, we can test not only that corruption *hurts*, but that generations
> *follow the donor*:
>
> | | intact | shuffled z_phy | shuffled z_ins |
> |---|---|---|---|
> | MSE vs own target | [X] | [X] | [X] |
> | MSE vs donor target | [X: unrelated-pair level] | [X] | – |
> | background σ correlates with | own field r=[X] | own field r=[X] | **donor field r=[X]** |
> | high-k power vs own | [X] | [X] | [X] |
>
> Shuffling z_phy makes the generation match the *donor* galaxy (MSE vs donor
> [X] < MSE vs own [X]) while leaving noise statistics unchanged; shuffling
> z_ins leaves source identity intact (MSE vs own ≈ intact) while the generated
> background noise now tracks the donor field's observing conditions
> (r=[X] vs r=[X]). We will include the full table and a qualitative figure in
> the appendix. We also note App. A.5 already reports the training-time analog
> (random vs spatial conditioning neighbors), which collapses instrument-property
> recovery (Galaxy Depth R² 0.68→0.20).

[Fill [X] after the run; trim to rebuttal length limits.]

---

## 6. Execution checklist

1. [x] `run_shuffle_ablation.py` + `metrics.py` (+ tests on 1 batch, M=1,
       num_steps=25 smoke).
2. [ ] Smoke run on GPU (SLURM, Discord ping) — verify C0 MSE ≈ FM-control
       levels and manifest guard passes.
3. [ ] Full run: M=8, 250 steps, 4 conditions. (~1–2 h.)
4. [x] `make_tables.py` / `make_figures.py`; sanity-check the expected-outcome
       table in §3 after the run.
5. [ ] Fill §5 numbers → write `reviews/claude-suggestions/response_1_SYJm_q2.md`
       (never edit `reviews/response_1_SYJm.md` directly); Pablo merges.
6. [ ] Optional: rerun with DDPM/FM-control finals for W1 (same command,
       different `--checkpoint`).

---

## 7. Decision log (defaults locked; flag to override)

- **Q-A — Conditions:** LOCKED core C0–C3. Random-Gaussian variants deferred to
  phase 2 (§4.4) — off-manifold caveat; add only if the reviewer insists.
- **Q-B — Ellipticity metric:** LOCKED skip (weights absent on Engaging).
  Alternative if desired: copy `resnet_best.pth` + `resnet_data.h5` from the
  AMD/CSAIL side and add R² vs own/donor shapes.
- **Q-C — Posterior samples:** LOCKED M=8 (m=0 reproduces the existing
  shared-noise protocol exactly; m≥1 seeded `noise_seed+20_000*m+batch_idx`).
  **Run as two jobs of M=4** (`M_LIST=0,1,2,3` and `M_LIST=4,5,6,7`, separate
  `OUT_DIR`s): measured throughput is ~60 batch-steps/min on an L40S
  (from `recon-mse-75k`: 4800 batch-steps in ~80 min), and a monolithic M=8 run
  is 32,000 batch-steps ⇒ ~9 h on L40S / ~4–5 h on H200 against the **6 h
  partition wall**. Splitting halves each job (~2.5 h on H200) and means a
  single completed job already yields a usable M=4 result. Seeds depend only on
  (m, batch_idx), so the split is numerically identical to the monolithic run
  (asserted in-code). The per-anchor CSV is also re-flushed after every
  (batch, m) block, so a wall-clock kill loses no computed work.
- **Q-D — Checkpoints:** LOCKED paper `base` only for the rebuttal table; the
  run itself also produces the long-missing fm-paper C0 row for the W1/diffusion
  comparison. DDPM/FM-control finals as an optional follow-up run.
