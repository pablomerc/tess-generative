# JEPA Ablation — Plan

> **See [`REPORT.md`](REPORT.md) for the single comprehensive write-up** (design, results, bugs, caveats, next steps). This file is one slice of it.


Created 2026-07-31. Decisions locked with Pablo the same day (see §Decisions).
**Status (2026-07-31): implemented & smoke-testing.** All files in §Files exist;
CPU sanity checks passed (losses finite, grads only on online params, masks
well-formed, duck-typed properties verified). `encode_latents.py` class sniff
extended for JEPA checkpoints. Checkpoints symlink into the mki_aryeh group
allocation (`/orcd/data/mki_aryeh/001/pablomer/jepa_ablation/`) because POOL is
at ~93%. wandb: project `galaxy-jepa-engaging` (online via ~/.netrc).

Quick reference:
```bash
sbatch smoke_test.slurm                      # end-to-end smoke (30-step arms + eval surfaces)
bash submit_all_jepa.sh                      # the 4 real runs (100k steps each)
sbatch --export=ALL,ARM=cf --job-name=jepa-eval-cf eval_jepa.slurm   # re-run one eval
```
Convention follows `diffusion_ablation/PLAN.md`: this file is the source of truth
for design + decisions; update it as things change.

## Goal

Test JEPA-style (predict-in-embedding-space, no negatives) objectives against the
published contrastive dual-encoder 2×2 arms, on the same data, encoder, and
evaluation pipeline. Deliverables per arm: downstream R² (n=5469, overlaid on the
Ours/AION/RandInit bars), the anomaly-detection figure pipeline, and survey-colored
UMAPs (in-training on wandb + paper-style).

## Run matrix (4 training runs)

| arm | objective | encoder(s) | data | notes |
|---|---|---|---|---|
| `jepa-hsc-resnet` | image-level JEPA: masked input → predict clean-image EMA embedding | 1× ResNet18 (conv1x1 pool, 64-d) | 366,706 HSC stamps (`source_type ∈ {0,1}`) | the comparable exp-1 arm |
| `jepa-hsc-vit` | true I-JEPA: latent multi-block masking | ViT (p4, 12×12 tokens, 192-d) | same 367k HSC pool | exploratory bonus arm; architecture confound acknowledged |
| `jepa-cf` | counterfactual JEPA: samegal → anchor (physics), neighbors → anchor (instrument) | 2× ResNet18 (physics + instrument towers) | 103,741 anchor triplets, spatial K=5 | no masking — the views are the corruption |
| `jepa-x` | cross-survey JEPA: HSC ↔ Legacy of same galaxy, both directions | 1× ResNet18 | 103,741 anchor pairs | direction-specific predictors |

## Decisions (2026-07-31, with Pablo)

1. **Exp-1 architecture: BOTH arms.** ResNet18 image-level JEPA (comparable) +
   small ViT true I-JEPA (bonus). Rationale: ResNet's final map at 48×48 is 2×2 —
   latent patch masking is meaningless there; true I-JEPA needs a ViT.
2. **Exp-1 pool: 367k** (`source_type ∈ {0,1}`, = the paper anomaly pool with
   valid HSC stamps). Exps 2–3 are constrained to the 103,741 anchors anyway
   (only anchors have guaranteed valid cross-survey pairs + neighbor lists).
3. **No DWNVAL holdout** — match the published 2×2 protocol so the overlay on the
   paper bars is apples-to-apples. SSL sees no labels; contamination is mild.
4. **Same budget, fixed schedule**: batch 64, 100k steps, AdamW lr 1e-4 wd 1e-4 —
   but proper step-based cosine LR (2k warmup) instead of the baseline's
   `T_max=max_epochs` quirk, plus EMA momentum 0.996→1.0 cosine. Deviation must be
   noted in the writeup.

## Shared design (ResNet arms)

- **Encoder**: `contrastive_baseline/dual_encoder_contrastive.py::ResNetEncoder`
  reused as-is — timm resnet18, `in_channels=4`, `image_size=48`,
  `embedding_dim=64`, `encoder_pool="conv1x1"` default (matches the flagship
  `contrastive-spatial-conv1x1` arm; `avg` available by flag).
- **Projector**: `ProjectionHead` (64 → 64 → 32), same as baseline. JEPA loss
  lives in the 32-d projected space; downstream eval always uses the raw 64-d
  encoder output — identical to how the contrastive arms are evaluated.
- **Predictor(s)**: MLP `Linear(32→64) → GELU → Linear(64→32)`.
- **Targets**: EMA copy of (encoder + projector). Momentum τ: cosine 0.996 → 1.0
  over max_steps, updated after each optimizer step. Targets are stop-grad.
- **Loss**: BYOL cosine loss, `2 − 2·cos(pred, target)` on normalized vectors,
  averaged over the batch (summed across branches where there are two).
- **Optimizer/schedule**: AdamW(1e-4, wd 1e-4), linear warmup 2,000 steps then
  cosine to 0 at 100k steps (step-based, interval="step"). bf16-mixed, 1 GPU.
- **Collapse diagnostics** (logged every 10 steps): per-dim std of the normalized
  online and target embeddings (`train/std_online`, `train/std_target`), mean
  `cos(pred, target)` (`train/align`). If collapse: first fix is predictor-only
  lr ×10; second is τ start 0.99.
- **Duck-typing contract (important)**: the EMA *target* encoders are exposed as
  `encoder_galaxy` / `encoder_instrument` properties (BYOL/I-JEPA convention is
  to evaluate the EMA encoder; it's also the smoother one). Online context
  encoders live under `context_encoder_*` names. This means
  `prepare_combined.py` (`_is_contrastive` hasattr check), `encode_latents.py`,
  and the in-training UMAP code all work with **zero changes** to their access
  pattern. Single-encoder arms expose the SAME module under both names → e1 == e2
  in the embeddings H5, and the physics/instrument target families of
  `predict_combined.py` then both probe the single embedding, which is exactly
  the question for those arms.
- **Val**: 5% random split (seed 42), `val_check_interval=1000`, best ckpt on
  `val/loss`, periodic every 500 steps + `last.ckpt` + `DONE` marker — identical
  to the baseline harness.
- **wandb**: project `galaxy-jepa-engaging`, run names = arm tags. Same metric
  naming style as the contrastive runs (`train/loss`, `train/loss_galaxy`,
  `train/loss_instrument` where applicable, `val/` mirrors) + `val/latent_umap`
  panel (same UMAP params `n_neighbors=15, min_dist=0.1, seed 42`; 2 panels for
  `jepa-cf`, 1 panel for single-encoder arms; HSC vs Legacy colors).
  - `jepa-hsc-*` UMAP caveat: Legacy points in the panel are rendered by pushing
    the anchors' legacy stamps through the HSC-trained encoder — visualization
    only, never trained on. Draw the UMAP batch from anchor rows (`source_type==0`)
    so both stamps are valid.

## Per-arm specifics

### `jepa-hsc-resnet` (exp 1a)
- New `HSC48PoolDataset`: rows with `source_type ∈ {0,1}` from
  `/orcd/pool/007/pablomer/efficient_neighs_48/` — loads **only** `hsc48.bin`
  (8.6 GB) via `np.fromfile` (NOT `ascontiguousarray` — see the no-op gotcha in
  `neighbors_ram48.py`). Returns single clean images; masking happens on GPU.
- Corruption (in `training_step`): 1–4 random rectangles, sides 12–24 px, aspect
  0.5–2, resampled until total masked fraction ∈ [0.3, 0.6]; masked pixels set to
  0 (data is normalized, so 0 ≈ mean). Masking only — no flips/crops (JEPA
  philosophy; a `--flips` flag exists but defaults off).
- Online: enc(masked) → proj → pred. Target: EMA enc(clean) → proj. Cosine loss.
- 100k × 64 ≈ 17.5 epochs over 367k.

### `jepa-hsc-vit` (exp 1b)
- timm `VisionTransformer(img_size=48, patch_size=4, in_chans=4, embed_dim=192,
  depth=6, num_heads=3)` → 144 tokens. Param count logged next to ResNet18's for
  the writeup.
- I-JEPA masking: 4 target blocks, scale (0.15, 0.2), aspect (0.75, 1.5);
  context block scale (0.85, 1.0) with target tokens removed.
- Predictor: narrow ViT, dim 96, depth 4, heads 3, mask tokens + positional
  embeddings; loss = MSE on EMA target-encoder token outputs (post final LN),
  averaged over target blocks.
- Eval embedding: mean-pooled EMA target tokens (192-d), wrapped in a small
  nn.Module so `encoder_galaxy(imgs)` returns `(B, 192)`. Width differs from the
  64-d arms — the MLP probe doesn't care, but flag it in the comparison.

### `jepa-cf` (exp 2)
- Data: `NeighborsRAM48Dataset` + `collate_neighbors` **unchanged** (spatial
  neighbors, `max_neighbors=5`, anchor survey alternates by index parity).
- Physics branch: online phys-enc(`samegal`) → proj → pred_g vs EMA
  phys-enc(`target`) → proj.
- Instrument branch: online instr-enc on each valid neighbor → proj → masked mean
  over K → pred_i vs EMA instr-enc(`target`) → proj. Anchors with 0 valid
  neighbors dropped from that branch (99.6% have ≥5).
- Loss = cosine(physics) + cosine(instrument), λ = 1/1 like the baseline.

### `jepa-x` (exp 3)
- New thin `AnchorPairs48Dataset` (~30 lines): the 103,741 anchors, returns
  (hsc_img, legacy_img) — both stamps valid by construction.
- Single encoder + projector; **two direction predictors** `pred_h2l`, `pred_l2h`
  (simpler than a survey-token-conditioned predictor, same effect).
- Loss = cos(pred_h2l(proj(enc(hsc))), EMA proj(enc(legacy))) + the mirror term,
  every sample both directions.

## Round 3 (2026-08-01, brainstorm with Pablo): token-level counterfactual designs

Motivated by round-2's central lesson (token-level targets are the lever; pooled
losses destroy spatial info — cf-vit shape 0.803→0.439) and by Pablo's two goals:
(A) one model whose HSC+Legacy embeddings combine well; (B) two disentangled
latent spaces. Design rule throughout: **conditioning lives only in predictors,
never in encoders** (target survey = complement of input's, so it is inferable) —
this keeps the duck-typed eval stack untouched.

| arm | design | tests |
|---|---|---|
| `cf-token` | cf-vit's exact dual-tower structure, pooled→TOKEN targets; physics = TokenTranslator (position-aligned partner→anchor, valid because Legacy was zoomed to HSC pixel scale at build), instrument = CrossAttnPredictor over the neighbour token set (no correspondence assumed); NO masking | the controlled pooling fix inside the counterfactual family; vindicates the original "views are the corruption" intuition if shape recovers to ~0.9 |
| `xjepa` | shared ViT trunk + per-token physics(128-d)/instrument(64-d) subspace heads; L = 0.5·L_mask(trunk I-JEPA) + L_phys(cross-survey translation) + L_inst(neighbour cross-attn) | goals A+B jointly: survey-aligned physics subspace + observing-condition subspace, spatial info insured by the mask term |
| `multi-vit` | IJEPAViTModule unchanged, trained on both surveys' stamps (`AllStamps48Dataset`: 366,706 HSC + 205,232 Legacy = 571,938) | the honest goal-A baseline any fancier design must beat; makes the probe's Legacy half trained instead of zero-shot |

Success criteria (set before results): physics side ≥0.70 physics / ≥0.90 shape
with `cross`-like instrument leakage (~0.03) would be the disentanglement win;
instrument side challenging contrastive's 0.697/0.795 would be a home run;
`multi-vit` beating hsc-vit's 0.664 instr-Legacy confirms the zero-shot gap.
Known confound: cf-token/xjepa train on the 104k anchors (they need pairs), so
comparisons vs hsc-vit carry the corpus difference; the hsc-vit-on-anchors
control remains the disambiguator (not yet run — Pablo picked these 3).

Files: `jepa_round3.py` (TokenTranslator, CrossAttnPredictor, CFTokenJEPAModule,
XJEPAModule), `AllStamps48Dataset` in datasets.py, `smoke_round3.slurm` (30-step
micro-runs + prepare_combined on xjepa's asymmetric 128/64 widths + round-3
loader check; submits the real runs on pass). Sniff markers: `phys_dim`→xjepa,
`translator_depth`→cf-token — both carry lambda_galaxy, so these are checked
BEFORE the cf branch in encode_latents.py and guarded in regenerate_umap_jepa.py.

## Files to write (all in `jepa_ablation/` unless noted)

| file | contents |
|---|---|
| `datasets.py` | `HSC48PoolDataset`, `AnchorPairs48Dataset` (ram48-store readers, np.fromfile) |
| `jepa_base.py` | EMA machinery (cosine τ, post-step update), cosine loss, collapse diagnostics, UMAP val hook, shared LightningModule base |
| `jepa_modules.py` | `JEPAHSCModule`, `JEPACounterfactualModule`, `JEPACrossSurveyModule` (ResNet arms; imports `ResNetEncoder`/`ProjectionHead` from contrastive_baseline) |
| `jepa_vit.py` | `IJEPAViTModule` + masking utils + eval wrapper |
| `train_jepa.py` | single entry, `--arm {hsc-resnet,hsc-vit,cf,cross}`, env hygiene + wandb + ckpt/resume logic cloned from `train_contrastive_engaging.py`, fixed step-based schedules |
| `train_jepa.slurm` | `pg_mki_aryeh`, `gpu:h200:1`, 16 cpu, 96G, 24h; self-chain + DONE guard + attempts cap kept (cheap insurance); Discord pings; submits `eval_jepa.slurm` on DONE |
| `submit_all_jepa.sh` | submits the 4 arms |
| `eval_jepa.slurm` | runs the 3-stage downstream eval + kicks anomaly pipeline + paper UMAP |
| `regenerate_umap_jepa.py` | clone of `diffusion_ablation/regenerate_umap.py` → npz → `visualization_scripts/plot_umap_from_file.py` |
| `smoke_test.sh` | 50-step micro-run per arm + end-to-end `prepare_combined` on the smoke ckpt (catches duck-typing/shape issues before real submits) |

Small patches **outside** `jepa_ablation/`:
- `contrastive_anomalies/encode_latents.py`: extend the `hyper_parameters` class
  sniff to recognize the JEPA modules; skip the duplicate tower encode for
  single-encoder arms. (Mind the `torch.set_grad_enabled(False)` global gotcha.)
- Nothing in `prepare_combined.py` / `predict_combined.py` /
  `plot_unified_three_probes.py` — duck-typing + repeatable `--extra-variant`
  already cover us.

## Evaluation per arm (auto-chained from training)

1. `prepare_combined.py --module jepa_ablation/jepa_modules.py --model-class <cls>`
   → `outputs/embeddings/jepa-<arm>.h5`
2. `predict_combined.py --mlp-arch aion --seed 0` → long-form R² CSV
3. `plot_unified_three_probes.py --mlp-arch aion --no-cross-predict` with
   `--extra-variant contrastive-spatial-conv1x1:Contrastive` +
   `--extra-variant jepa-<arm>:<label>`; once all 4 JEPA CSVs exist, render one
   combined 5-variant figure (baseline auto-render pattern) → Discord.
4. Anomaly: `contrastive_anomalies/run_pipeline.slurm CKPT=<best> TAG=jepa-<arm>`
   over the 103,741 anchors — top-12 grids, lens ROC/recall@k, physics-vs-instrument
   Jaccard, compared against the `spatial-conv1x1` run. (Known: iforest ~chance in
   64-d; read the flow/knn columns.)
5. Paper-style UMAP via `regenerate_umap_jepa.py`.

## Compute & logistics

- Throughput estimate: ram48 gave the baseline 90–220 batch/s; JEPA adds a second
  forward + EMA update → expect well under 2 h per 100k-step run on an H200.
  All 4 submitted to `pg_mki_aryeh`; queue resolves concurrency.
- Checkpoints → `/orcd/pool/007/pablomer/checkpoints_new/jepa-<arm>/`.
- RAM: exp-1 arms load only `hsc48.bin` (8.6 GB); cf/x load both bins (17.2 GB).
  `OMP/MKL=1` for workers, `torch.set_num_threads(1)` — baseline env hygiene.
- Discord webhook pings on submit/start/finish (standing instruction), and eval
  posts the R² overlay PNG as the baseline eval already does.

## Findings during the first run (2026-07-31)

- **`best-on-val/loss` is the wrong selection rule for this family, and it bit us.**
  `hsc-vit`'s best val/loss landed at **step 2,000 of 100,000**; the first eval ran
  on that checkpoint before it was caught and cancelled. A JEPA prediction loss has
  a *learned* target, so it can fall via collapse and legitimately rise as the
  representation gets richer. `eval_jepa.slurm` now takes
  `latest-step=<MAX_STEPS>.ckpt` (final EMA weights); `best-*.ckpt` is kept as a
  diagnostic only. This is a deliberate divergence from the contrastive eval script.
- Wall clock on one H200, all four arms co-resident on node4900: `hsc-vit` 57 min,
  `cross` 70 min for 100k steps. Dataset load off Ceph is a few minutes up front
  when several jobs read concurrently (~52 GB combined).
- Added `check_collapse.py` — measures `norm_std` relative to isotropic `1/√D` plus
  PCA effective rank straight from a checkpoint, and runs inside `eval_jepa.slurm`
  ahead of the probe so a collapsed arm is labelled rather than silently reported
  as a low R².
- **SLURM snapshots the batch script at submit time**, so fixing `eval_jepa.slurm`
  did NOT fix the eval jobs the training chain had already queued — one of them ran
  the old best-ckpt logic and had to be cancelled and resubmitted. When editing a
  script that self-chaining jobs auto-submit, verify the queued job with
  `scontrol write batch_script <jobid> -` and resubmit if stale.
- Two self-inflicted bugs in that same edit, both fixed: the collapse check was
  first placed *above* the `conda activate` block (system python, no numpy), and
  `COLLAPSE_SUMMARY=$(grep …)` aborts the whole script under `set -e` + `pipefail`
  when grep matches nothing. Diagnostics must never be able to kill the eval.

## Risks / open items

- **Collapse** is the main risk for BYOL-style arms — mitigations + diagnostics
  above; smoke runs must show `std_target` well above 0 before full submits.
- ViT-arm hyperparameters at 48×48/patch-4 are judgment calls; treat 1b as
  exploratory, not headline.
- Fixed-LR-schedule deviation from the published arms: if any JEPA arm *beats*
  the contrastive arms, consider a `contrastive-spatial-conv1x1` rerun with the
  fixed schedule as a control before claiming an objective-level win.
- `jepa-hsc-*` downstream R² on Legacy-side targets probes transfer of an
  HSC-only encoder — expected to be weak; that's informative, not a bug.
