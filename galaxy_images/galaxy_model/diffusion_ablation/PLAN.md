# Diffusion vs Flow Matching Ablation — Implementation Plan

**Purpose:** answer reviewer SYJm's W1 ("the paper does not sufficiently justify the choice of the
flow-matching objective … could also be learned using a diffusion objective or a conditional VAE").
The conditional-VAE half is addressed separately in [`VAE_DISCUSSION.md`](VAE_DISCUSSION.md) — short
version: it is a different beast, we should rebut it in text, not implement it.

**Success criterion:** *comparable is enough.* The paper never claims flow matching is essential —
the contribution is the counterfactual dual-encoder objective, and FM is presented as the chosen
instantiation in a single sentence (tex line 209: "We model this conditional distribution using flow
matching…"). There is currently **zero** justification text anywhere in the paper (the word
"diffusion" never appears except in the `diffusers` citation). So the rebuttal has two prongs:

1. **Conceptual:** linear-path FM (a.k.a. rectified flow) and DDPM-style diffusion are members of
   the same family of Gaussian-path denoising objectives (stochastic interpolants); they differ in
   interpolant schedule, network parameterization (velocity vs ε), and loss weighting — not in what
   distribution they can represent. We chose FM for its simplicity (no noise schedule to tune,
   uniform-t unweighted MSE) and its deterministic few-step ODE sampler (which the fixed-x₀ SNR
   traversal in Fig. 7 relies on). Cite: Lipman et al. 2023 (already cited), Liu et al. 2022
   (rectified flow), Albergo & Vanden-Eijnden 2023 (stochastic interpolants), Kingma & Gao 2023
   (diffusion objectives as weighted ELBOs), Esser et al. 2024 (SD3: rectified flow ≥ diffusion at
   scale).
2. **Empirical (this plan):** retrain the *identical* architecture with a standard diffusion
   objective and show comparable downstream R², reconstruction MSE, and latent-space structure.

---

## 1. Why the swap is small (current code audit)

The baseline (`neighbors_all_attn` → `ConditionalFlowMatchingModule` in
`double_train_fm_neighbors.py`) is pure rectified-flow/OT-CFM:

| Piece | Code | Detail |
|---|---|---|
| t sampling | `double_train_fm_neighbors.py:414` | `t ~ U[0,1)`, per-sample |
| Path | `:416-417` | `x_t = (1-t)·x_0 + t·x_1`, x_0 = N(0,I) noise, x_1 = data; no σ_min |
| Target | `:419` | constant velocity `x_1 − x_0` |
| Loss | `:430-447` | plain unweighted MSE, mean over pixels/batch |
| Timestep into UNet | `:339` | `timesteps = t * 1000` → **floats in [0,1000)** into the stock diffusers sinusoidal embedding |
| Sampler | `sample()`, `:642-697` | fixed-grid forward Euler, 250 steps (config `num_integration_steps`) |
| Aux losses | `:483` | `lambda_geometric = 0.0` in the trained baseline → generative loss only |

Everything else — both ResNet18 encoders, the `UNet2DConditionModel` decoder, cross-attention
conditioning, neighbor masking, data pipeline, optimizer (AdamW 1e-4 + cosine), validation MSE/UMAP
plumbing — is objective-agnostic and inherited untouched.

**So yes: it is "just" (a) a ~15-line change to the corruption + regression target, (b) the timestep
convention, and (c) a new ~30-line sampler.** Three touch points, cleanly overridable in a subclass,
because `training_step`/`validation_step` call `self.compute_loss()` and `compute_mse`/the sample
grid call `self.sample()` polymorphically.

A lucky architectural accident: since FM already feeds the UNet `t*1000 ∈ [0,1000)`, DDPM's integer
timesteps 0–999 exercise the *exact same* embedding frequency band. The time-conditioning is
unusually well matched for this ablation — no retuning needed.

---

## 2. The diffusion menu

### Option A — DDPM with ε-prediction (RECOMMENDED primary arm)

The canonical "diffusion objective" (Ho et al. 2020). Variance-preserving forward process
`x_t = √ᾱ_t·x_1 + √(1−ᾱ_t)·ε`, network regresses ε, uniform discrete t, "simple" unweighted MSE
loss. Implemented entirely with `diffusers.DDPMScheduler` (already in `torchenv`: diffusers 0.38.0,
verified — cosine schedule, v-prediction, DDIM all work).

- **Why this one:** when a reviewer writes "a diffusion objective", this is what they mean. It is
  the most standard, most defensible instantiation, and the maximal-contrast comparison (different
  path, different parameterization, different time discretization) while sharing the unweighted-MSE
  / uniform-t structure with our FM loss — so the comparison isolates *the objective*, nothing else.
- Schedule: **cosine (`squaredcos_cap_v2`)**, T=1000. (The linear β schedule was tuned for [-1,1]
  natural images; cosine is the standard modern default for small images — Nichol & Dhariwal 2021.)
- **Sampling:** DDIM, η=0 (deterministic), **250 steps** — matched to FM's 250 Euler steps, and
  deterministic-in-x₀ just like the FM ODE, so the Fig. 7 SNR-traversal demo (fix x₀, vary
  conditioning) carries over unchanged. Ancestral DDPM-1000 sampling can be reported as a secondary
  reference if we want it.

### Option B — v-prediction (recommended second arm, if budget allows)

Same VP forward process, but the network regresses `v = √ᾱ·ε − √(1−ᾱ)·x` (Salimans & Ho 2022).
One-line target change via `scheduler.get_velocity()`. This is the modern best-practice diffusion
parameterization and, notably, v-prediction on a cosine schedule is mathematically the closest
diffusion relative of flow matching (FM with a trigonometric interpolant *is* v-prediction). Having
both A and B brackets the space: A = classic diffusion, B = diffusion at its best / nearest
neighbor of FM. Cost: one extra config file + one extra training run, zero extra code.

### Option C — EDM (Karras et al. 2022) — NOT recommended

Best-in-class sample quality, but introduces many extra design decisions (σ_data preconditioning,
log-normal σ sampling, tuned λ(σ) weighting, Heun sampler). Every knob weakens the
"everything held fixed except the objective" claim, and our z-scored heavy-tailed astronomical data
would require nontrivial preconditioning choices. Wrong tool for a rebuttal ablation.

### Option D — VE/NCSN score matching — no

Subsumed by EDM, obsolete as a standalone baseline.

**Recommendation: A, plus B if the rebuttal window allows a third run. Not C or D.**

---

## 3. Code changes

### 3.1 New files (all in `diffusion_ablation/`)

```
diffusion_ablation/
├── PLAN.md, VAE_DISCUSSION.md            # these docs
├── __init__.py
├── double_train_ddpm_neighbors.py        # ConditionalDDPMModule(ConditionalFlowMatchingModule)
├── configs/
│   ├── neighbors_ddpm_eps.json           # variant "neighbors_all_attn_ddpm"
│   ├── neighbors_ddpm_vpred.json         # (optional arm B)
│   └── neighbors_fm_control.json         # (only if FM control retrain is needed, see §4)
├── train_ddpm.slurm                      # cluster-appropriate, see §4
└── eval_recon_mse.py                     # standalone matched recon-MSE eval (§5.2)
```

### 3.2 The module: `ConditionalDDPMModule(ConditionalFlowMatchingModule)`

Overrides exactly two methods + constructor; inherits forward, encoders, validation, UMAP,
optimizer. Sketch:

```python
class ConditionalDDPMModule(ConditionalFlowMatchingModule):
    def __init__(self, prediction_type="epsilon", num_train_timesteps=1000,
                 beta_schedule="squaredcos_cap_v2", **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()          # subclass args must be captured for load_from_checkpoint
        self.train_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps, beta_schedule=beta_schedule,
            prediction_type=prediction_type, clip_sample=False)
        self.inference_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps, beta_schedule=beta_schedule,
            prediction_type=prediction_type, clip_sample=False)

    def compute_loss(self, batch):
        x_1, cond_samegal, cond_sameins, masks, metadata = ...   # same unpack as parent :402-407
        noise = torch.randn_like(x_1)
        t = torch.randint(0, T, (B,), device=x_1.device)
        x_t = self.train_scheduler.add_noise(x_1, noise, t)
        target = noise if eps else self.train_scheduler.get_velocity(x_1, noise, t)
        pred = self(x_t, t.float() / 1000.0, cond_samegal, cond_sameins, masks)  # see note below
        # then identical per-example MSE + per-survey hsc/legacy logging as parent :430-452

    @torch.no_grad()
    def sample(self, cond_samegal, cond_sameins, masks=None, num_steps=None, x_noise=None):
        sched = self.inference_scheduler
        sched.set_timesteps(num_steps or self.num_integration_steps)   # 250, same knob as FM
        x = x_noise if x_noise is not None else torch.randn(...)
        for t in sched.timesteps:
            pred = self(x, t.expand(B).float() / 1000.0, cond_samegal, cond_sameins, masks)
            x = sched.step(pred, t, x, eta=0.0).prev_sample
        return x
```

**Implementation notes / gotchas (all verified against the code):**

- **Timestep trick:** the inherited `forward` multiplies t by 1000 (`:339`). Passing
  `t_int.float()/1000` reproduces the integer timestep inside forward with zero changes to the
  parent — both objectives then use identical time embedding logic. (Float round-trip error is
  ~1e-13, irrelevant to a sinusoidal embedding.)
- **`clip_sample=False` is critical.** Our images are z-scored with heavy positive tails
  (arcsinh-compressed, per-survey normalized — NOT in [-1,1]). The diffusers default
  `clip_sample=True` clamps predicted x₀ to [-1,1] during `step()` and would silently corrupt every
  reconstruction.
- Keep the exact `sample()` signature `(cond_samegal, cond_sameins, masks, num_steps, x_noise)` —
  `compute_mse` (`:714`), the validation sample grid (`:824`), and every downstream generative
  script call it that way.
- We deliberately do **not** touch `double_train_fm_neighbors.py` (it produced the paper
  checkpoints; keep it frozen). The ~40 lines of unpack/logging boilerplate duplicated into the
  subclass's `compute_loss` is the price; the Sinkhorn branch is dropped (λ_geom = 0 in the
  baseline).
- **Checkpoint selection must be objective-neutral:** `val/loss` is ε-MSE in one arm and
  velocity-MSE in the other — not comparable, and "best val/loss" would bias selection. Compare
  **fixed-step final checkpoints (75k)** across arms; additionally set
  `trainer.monitor_metric = "val/mse"` (sampled reconstruction MSE, objective-neutral, already
  config-exposed) so the "best" checkpoint is also meaningful.

### 3.3 Edits to existing files (small, additive)

| File | Edit |
|---|---|
| `variants.py` | register `neighbors_all_attn_ddpm` (+`_vpred`) → `VariantSpec(model_cls=ConditionalDDPMModule, model_overrides={"all_attention": True, "prediction_type": ...})` |
| `config.py` (`ModelConfig`) | add `prediction_type`, `num_train_timesteps`, `beta_schedule` fields (unknown JSON keys raise `TypeError` in `from_dict`, and `filter_supported_model_kwargs` silently drops them for the FM class — safe) |
| `downstream_evaluation/engaging/variants.json` | add `diffusion-base` entry (module `diffusion_ablation/double_train_ddpm_neighbors.py`, class `ConditionalDDPMModule`); `prepare_combined.py` duck-types on `encoder_1`/`encoder_2` so the base branch is hit automatically — zero code edits |
| *(only if training on Engaging)* `config.py` + `data_factory.py` | whitelist a `ram48` data mode and add the `NeighborsRAM48Dataset` branch (~20 lines; the dataset is already a verified drop-in for the FM 4-tuple + `collate_neighbors` contract) |

### 3.4 Smoke test before submitting anything

```bash
python -m galaxy_images.galaxy_model.train \
  --config galaxy_images/galaxy_model/diffusion_ablation/configs/neighbors_ddpm_eps.json \
  --set trainer.devices=1 --set trainer.num_steps=200 \
  --set data.batch_size=8 --set wandb.enabled=false \
  --set model.num_integration_steps=10
```
Checks: loss decreases, one validation pass runs end-to-end (sampler + MSE + UMAP grid), checkpoint
saves and reloads via `ConditionalDDPMModule.load_from_checkpoint`.

---

## 4. Training setup — the main design decision

Matched budget in all cases (= paper, tex line 789 / Table 4): same UNet + encoders, effective
batch 256, AdamW lr 1e-4 + cosine, bf16-mixed, **75k steps**, seed 42, max_neighbors=5.
One seed per arm (standard for a rebuttal; the spatial-vs-random retrain ablation in the appendix
sets the precedent for single-run ablations).

### Option A — AMD cluster (NOT chosen; kept as fallback)

**Correction (Pablo, 2026-07-23):** the AMD cluster has **AMD GPUs only** — MI210 (`mi2101x` 1×/12h,
`mi2104x` 4×/24h), MI250 (`mi2508x` 8×/12h), and MI300X/MI325X/MI350X 8-GPU nodes
(`mi3008x`/`mi3258x`/`mi3508x`, 12h batch). The repo CLAUDE.md's "plus an H100 partition" note is
wrong (now fixed). Provenance resolved (Pablo, 2026-07-23): the paper checkpoint was trained on a
*third* cluster with NVIDIA GPUs (neither AMD nor Engaging), so the paper's "4×H100, ~8h" line
(tex 789) is accurate as written — no camera-ready correction needed. This also means no current
cluster reproduces the paper environment exactly, which is one more reason the retrained FM control
(§ Option B) is the right comparison anchor.

- Pros: data native (`neighbors_efficient` mmap store, no RAM48 plumbing), existing slurm templates.
- Cons: 75k steps on 4×MI210 likely exceeds the 24h `mi2104x` wall (resume dance); the fast
  MI300X-class nodes are whole-node 8-GPU (batch-256 parity would waste half a node, 12h cap);
  hipBLASLt workaround; checkpoints must shuttle back to Engaging for all evals; can't be driven
  from an Engaging session.

### Option B — Engaging (this cluster) ← CHOSEN

- **Data:** must use the RAM48 store (`/orcd/pool/007/pablomer/efficient_neighs_48`) — the 160×160
  mmap store is Ceph random-read bound (~0.35 batch/s, measured during the contrastive work; a 75k
  FM-UNet run would never finish). RAM48 is bit-for-bit equivalent preprocessing modulo fp16
  rounding, and its per-item contract is identical to `NeighborsEfficientDataset`.
- **Partition (primary): `pg_mki_aryeh`** — node4900 = 8× H200 (141 GB), 2 TB RAM, 120 CPUs,
  **48h wall limit** (Pablo is on the access list; nodes 4701/4702 are CPU-only). This mirrors the
  paper's training layout exactly: 4 GPUs × batch 64 DDP × 75k steps ≈ 8h per arm, in a single job —
  **no self-chaining needed** (keep a resume-if-checkpoint-exists guard for robustness). Per-rank
  16 GB RAM48 copies (4 ranks = 64 GB) are trivial on 2 TB. Caveat: single shared node, often fully
  allocated — check `sinfo -p pg_mki_aryeh` before targeting; queued jobs wait behind ≤48h jobs.
- **Partition (fallback): `mit_normal_gpu`** — 1 GPU × batch 256 × 75k steps ≈ 30–35h wall per arm,
  self-chained across the 6h wall limit using the proven contrastive slurm pattern (`afterany`
  successor + resume + DONE marker; resume via newest `latest-step=*.ckpt` glob passed to
  `--set run.resume_from` since `train.py` doesn't write `last.ckpt`). QOS caps us at 2 concurrent
  GPUs → two arms in parallel, ~1.5 days wall. Write the slurm with the partition/layout as
  variables so switching is one `--export` change.
- **Confound control:** the setup differs from the paper run (RAM48 fp16 store, different node), so
  we **also retrain an FM control under identical Engaging conditions** (same config, only the
  objective differs). On 4×H200 both arms finish within a day even run back-to-back.
- Upside: everything (training + eval) stays on this cluster, no checkpoint shuttling; I can babysit
  end-to-end.

**Either way, all evaluation runs on Engaging** (that's where the n=5469 pipeline, the 474k anomaly
pool, and the paper-figure code live).

---

## 5. Evaluation protocol

The key structural fact: **every representation-side eval is encoder-only** (embeddings from
`encoder_1`/`encoder_2`) and needs zero objective-specific code; only generative-side evals touch
`sample()`, which §3.2 provides. Proposed scope:

### Core (goes in the rebuttal — recommended)

1. **Downstream R², n=5469, 90/10 MLP probes** — the paper's headline table.
   - `ONLY=diffusion-base bash downstream_evaluation/engaging/run_per_variant.sh` (~10 min/variant
     on 1 GPU; fully idempotent).
   - `predict_combined.py --mlp-arch aion --out .../predict_diffusion-base__aion.csv`, then overlay
     with `plot_unified_three_probes.py --extra-variant 'diffusion-base:DDPM (ε)'` — the exact
     mechanism already used for the contrastive baseline overlay, zero code edits.
   - Instrument-side bar needs the mean-pooled twin: `build_meanpool_h5.py` → probe (mirrors how
     the paper's "Ours (Instr)" = `base-meanpool` was made).
   - **Data policy (decided 2026-07-23): no downstream holdout, matching the paper.** The repo does
     have a holdout mechanism (`data.downstream_holdout_ids_txt` → filtering in
     `data_factory.py:237`, used by the four `*_dwnval` configs), but the paper's base config never
     set it — the published model saw the n=5469 galaxies (as unlabeled images only) during
     training, which is the standard SSL-probe protocol; labels can't leak, and the probe's own
     90/10 split handles probe-level generalization. Both new arms therefore train with the
     identical no-holdout policy: internal validity requires the two arms match *each other*, and
     matching the paper keeps absolute R² comparable to the published table. (The ids file
     `holdout_legacy_ids.txt` only exists on the AMD cluster anyway; if a strict-holdout version is
     ever wanted — camera-ready at most — copy it over and apply to BOTH arms.)
2. **Reconstruction MSE, matched protocol** — new `eval_recon_mse.py`: load both arms' checkpoints
   (+ FM control), rebuild the deterministic seed-42 val split, draw the **same 256 held-out
   galaxies with the same conditioning and the same x₀ seeds**, FM → 250-step Euler, DDPM →
   250-step DDIM(η=0), report MSE split by anchor survey (paper reference: 0.081 HSC / 0.197
   Legacy, tex line 518 — but those came from W&B val curves, so the rebuttal table should quote
   our matched re-eval for *both* objectives rather than mixing protocols).
   - **η robustness check (Pablo, 2026-07-23):** additionally run the DDPM arm at η=1 (ancestral-
     style stochasticity, seeded per-step noise) with everything else identical. η is inference-only,
     so this is one extra pass, no retraining. Reported as a within-diffusion side note ("conclusion
     is sampler-independent"), NOT in the headline FM-vs-DDPM row, which stays deterministic-vs-
     deterministic since FM has no stochastic sampler implemented. `eval_recon_mse.py` gets an
     `--eta` flag.
3. **UMAP disentanglement figure** — side-by-side physics/instrument UMAPs
   (`visualization_scripts/regenerate_umap_base.py --checkpoint …`; needs its hardcoded
   `ConditionalFlowMatchingModule` import loosened, one line). Qualitative but it's the paper's
   Fig. 2 and the visual proof that disentanglement survives the objective swap.

### Optional extensions (nice-to-have, in rough priority order)

4. **Posterior calibration Z-scores + PSD/autocorrelation** (appendix Fig. 8/9 protocol, 32
   samples/anchor). This is the one axis where FM vs diffusion might *differ interestingly*
   (the paper found ~15% variance underestimation for FM — diffusion's ancestral sampler might
   calibrate differently). Decoder-dependent but `sample()` covers it; scripts currently live on
   the AMD side with hardcoded checkpoint constants (`generative_evaluation/*_v3_heldout.py`).
5. **Anomaly top-12** (474k pool): rerun `encode_{ours,instrument}_hscdown.py --checkpoint … --suffix
   hscdown_474k_diffusion` → NSF fit → top-12 panel. ~2×6h GPU jobs, qualitative payoff.
6. **Similarity search / lens retrieval**: `neighbor_search/make_latents_all.py` with the new
   checkpoint. Qualitative, anecdotal — skip unless the rebuttal needs a visual.

My take: 1–3 fully answer W1 ("comparable R², comparable reconstructions, same disentanglement
structure"); 4 adds a genuinely informative scientific wrinkle if time permits; 5–6 add little
argumentative value for their cost.

### Rebuttal table sketch

| Objective | z | log M★ | e₂ | … | Gal. Depth (instr) | Recon MSE (HSC / Legacy) |
|---|---|---|---|---|---|---|
| Flow matching (paper) | | | | | | |
| DDPM ε-pred | | | | | | |
| *(v-pred, optional)* | | | | | | |

---

## 6. Design decisions — LOCKED (Pablo, 2026-07-23)

1. **Arms: DDPM ε-prediction only.** (v-pred deferred; can be added for camera-ready — zero extra
   code, one extra run.)
2. **Training cluster: Engaging.** RAM48 store; primary partition `pg_mki_aryeh` (4×H200 DDP,
   ~8h/arm, single 48h job, no chaining), fallback `mit_normal_gpu` (1 GPU, 6h self-chaining,
   ~1.5 days for both arms under the 2-GPU QOS cap). **Plus an FM control retrained under
   identical Engaging conditions** so the objective comparison is airtight.
3. **Eval scope: core** — downstream R² (n=5469), matched 256-galaxy recon MSE, side-by-side UMAPs.
4. **Deadline: ~1 week.** Timeline: implement + smoke test (day 1), submit both training chains
   (day 1 evening), training (days 2–3), evals (day 3–4), rebuttal table + text (day 4–5), buffer
   (days 6–7).

Fixed defaults: cosine β schedule (`squaredcos_cap_v2`), T=1000, DDIM η=0 @ 250 steps,
batch 256 / lr 1e-4 / 75k steps / seed 42 matched to the paper, fixed-75k-step checkpoint
comparison, `monitor_metric: val/mse`.

---

## 7. Relation to the reviewer's other points

- **Q2 (decoder-reliance / shuffled-latent diagnostic)** is a separate, cheap, inference-only
  experiment on the *existing* FM checkpoint — it needs only `sample()` with permuted
  `cond_samegal` / `cond_sameins` batches. Worth doing regardless; the diffusion module inherits
  the same interface so the diagnostic can be run for both objectives for free once implemented.
  Not part of this plan, but the code here doesn't block it.
- **Q1 (attention-based conditioning)** is pure prose — explain the variable-length token
  concatenation + cross-attention (forward `:341-393`).
