# JEPA Ablation — Full Report

**Everything about this experiment in one place.** Written 2026-08-01, covering work
done 2026-07-31 on Engaging. Self-contained: you should not need any other file or
any chat history to understand what was built, what was found, what went wrong, and
what to do next.

Companion files (all consistent with this one): `PLAN.md` = design decisions as
locked before the run · `README.md` = how to operate the code · `RESULTS.md` =
results only. This report supersedes them where they overlap.

---

## 1. What this tests, and why

The published work learns physics and instrument representations with a
flow-matching generative objective ("Ours"), benchmarked against a contrastive
dual-encoder baseline and AION-1. This ablation asks whether a **JEPA-style
objective** — predict a target *embedding* from a context embedding, no negatives,
no pixel reconstruction — does better, and in what respect.

Three questions, four training arms:

| # | question | arm | objective | data |
|---|---|---|---|---|
| 1a | Can you do JEPA on HSC with a ResNet by masking? | `hsc-resnet` | encode masked image → predict EMA embedding of the clean image | 366,706 HSC stamps |
| 1b | Or does the architecture have to change? | `hsc-vit` | true I-JEPA: latent multi-block masking on a ViT | same 366,706 |
| 2 | JEPA with counterfactual/triplet views instead of masking | `cf` | physics tower: cross-survey partner → anchor; instrument tower: mean of 5 sky-neighbours → anchor | 103,741 anchor triplets |
| 3 | JEPA across surveys, physics encoder only | `cross` | predict each survey's embedding from the other, both directions | 103,741 anchor pairs |

All four are BYOL-style: online encoder → projector → predictor, regressing onto a
stop-gradient EMA target, cosine loss `2 − 2·cos`. `hsc-vit` instead uses I-JEPA's
token MSE. Arms 1a/2/3 reuse the contrastive baseline's exact `ResNetEncoder`
(timm resnet18, 4-channel, 48×48, `conv1x1` pooling, 64-d out) so they are
dim-matched to the published contrastive arms.

---

## 2. Setup

**Data** — the prebuilt in-RAM store `/orcd/pool/007/pablomer/efficient_neighs_48`
(float16, preprocessing baked in at build time).

- HSC pool = 366,706 rows with a valid HSC stamp (`source_type ∈ {0,1}`); this is
  the same pool as the paper's anomaly figure. Train/val 348,371 / 18,335.
- Anchors = 103,741 rows (`source_type == 0`); each carries an HSC **and** a Legacy
  stamp of the same galaxy plus 20 sky-kNN neighbour indices. Train/val 98,554 / 5,187.
- Loading uses `np.fromfile` — `np.ascontiguousarray`/`np.array` on an
  already-contiguous memmap is a no-op that silently leaves data on Ceph.

**Hyperparameters** (identical across arms unless noted): batch 64, 100k steps,
AdamW lr 1e-4 wd 1e-4, bf16-mixed, 1 GPU, 5% val, seed 42, EMA momentum cosine
0.996 → 1.0, **step-based cosine LR with 2,000-step warmup**.

That LR schedule is a deliberate deviation — see §6, it is the one caveat that
gates the headline.

**Training cost** — all four ran concurrently on node4900's four H200s, so the whole
matrix finished in under 90 minutes wall clock.

| arm | wall clock | trainable params |
|---|---|---|
| `hsc-vit` | 57 min | 3.2 M (encoder 2.71 M + predictor 0.50 M) |
| `cross` | 70 min | 11.2 M |
| `hsc-resnet` | 79 min | 11.2 M |
| `cf` | 83 min | 22.4 M (two towers) |

**Tracking** — wandb project `galaxy-jepa-engaging`, same metric style as the
contrastive runs, plus `val/latent_umap` survey-coloured panels and the collapse
diagnostics below. Checkpoints live in the 200 TB `mki_aryeh` group allocation
(3.7 GB total), symlinked into `checkpoints_new/` so nothing else had to change;
POOL was at ~93% and moved by 0.2 GB.

**The integration trick.** Every module exposes its **EMA target** encoders as
`encoder_galaxy` / `encoder_instrument` properties, with single-encoder arms
aliasing one module under both names. `prepare_combined.py`, `encode_latents.py`
and the UMAP tooling all duck-type on exactly those attribute names, so the entire
existing eval stack ran unchanged. The only edit outside `jepa_ablation/` was one
JEPA branch in `contrastive_anomalies/encode_latents.py`'s class sniff.

Consequence: for single-encoder arms `e1 == e2`, so their "physics" and "instrument"
columns probe the *same* space. That is the intended reading for those arms, not a bug.

---

## 3. Results — downstream R²

n=5469 overlap sample, AION MLP probe, seed 0, 90/10 split, mean over each target
family. **All numbers from the final checkpoint** (`latest-step=100000.ckpt`), never
`best-*.ckpt` — see §5.

| run | spaces | probe input | physics | instr HSC | instr Legacy | mean shape |
|---|---|---|---|---|---|---|
| Ours (FM base) | 2 | 128-d | 0.704 | 0.559 | 0.649 | 0.929 |
| AION-1 | 1 (no e2) | — | 0.742 | — | — | 0.938 |
| Contrastive sp-conv1x1 | 2 | 128-d | 0.663 | 0.697 | 0.795 | 0.897 |
| **JEPA `hsc-vit`** | **1 (shared)** | 384-d | **0.753** | 0.572 | 0.664 | 0.919 |
| **JEPA `hsc-vit` PCA** | **1 (shared)** | **64-d** | **0.745** | 0.545 | 0.607 | 0.909 |
| JEPA `hsc-resnet` | 1 (shared) | 128-d | 0.561 | 0.273 | 0.258 | 0.638 |
| JEPA `cf` | 2 | 128-d | 0.495 | 0.352 | 0.443 | 0.803 |
| JEPA `cross` | **1 (shared)** | 128-d | 0.511 | **0.035** | **0.014** | 0.772 |

### ⚠️ How to read this table — the `e1`/`e2` labels are misleading

The eval harness assumes a two-tower model and probes physics targets on `e1` and
instrument targets on `e2`. **The single-encoder arms (`hsc-resnet`, `hsc-vit`,
`cross`) have only ONE latent space** — they alias the same EMA encoder under both
`encoder_galaxy` and `encoder_instrument`, so the harness probes the *same vectors*
twice. Verified directly in the H5: `hsc_e1` and `hsc_e2` are bit-identical, and the
probe returns the same score to 4 dp (`hsc-vit` physics 0.7525 on both, instrument
0.5717 on both).

So for those rows the correct reading is **"how much of each property is decodable
from the one shared space"**, not "how well two specialized spaces divide labour".
Only Ours, Contrastive and `cf` have two genuinely different spaces; AION has no `e2`
dataset at all, hence the blanks.

This makes `cross` **stronger**, not weaker: 0.035/0.014 is not a dedicated
instrument tower failing at its job, it is the single shared space genuinely not
carrying instrument information while still holding physics at 0.511.

### What the probe actually consumes

`combined_e*` = `concat(HSC image embedding, Legacy image embedding)` for the same
galaxy (verified by reconstruction), hence the "probe input" column: 2 × 192 = 384-d
for `hsc-vit`, 2 × 64 = 128-d for the 64-d arms.

Two consequences worth stating:

- **The instr-Legacy column for the HSC-only arms is zero-shot transfer.** Half of
  that concatenation is a Legacy stamp pushed through an encoder that never saw
  Legacy data. `hsc-vit` reaching 0.664 on Legacy instrument properties is transfer,
  not training.
- **The PCA width control overcorrects.** It reduced `combined_e1` from 384-d to
  **64-d — half the 128-d every other arm gets.** `hsc-vit` still scored 0.745 vs
  Contrastive's 0.663, i.e. it wins while handing the probe half as many features.
  The advantage does not merely survive the control; the control is biased against it.

### Finding 1 — I-JEPA on a ViT is the strongest physics representation here

0.753 physics beats Ours (0.704) and the contrastive baseline (0.663), and edges
AION-1 (0.742). From **HSC images alone** — no cross-survey pairs, no negatives, no
catalogue.

Because it was the only arm wider than 64-d, its embeddings were PCA-reduced and
re-probed with the identical MLP (the convention the paper already uses for AION's
`pca64`): **0.745**, essentially unchanged (−0.008). *The advantage is not a
wider-probe artifact* — and note the control is conservative, since it leaves the
probe 64-d against the other arms' 128-d (see "What the probe actually consumes").

### Finding 2 — masking a ResNet does not work; the architecture must change

This is the direct answer to question 1. `hsc-resnet` is matched to `hsc-vit` on
data, width, budget and schedule, differing only in architecture and where the mask
is applied — and it reaches just 0.561 physics / 0.638 shape.

The mechanism is structural: at 48×48 input a resnet18's final feature map is
**2×2**, so there is nothing meaningful to mask in latent space, and masking input
pixels is a poor substitute. The collapse diagnostics agree with the gap rather than
explaining it away (effective rank 7.9/64 vs 20.2/192).

### Finding 3 — cross-survey JEPA gives the cleanest disentanglement in the project

`cross` drives instrument R² to **0.035 (HSC) and 0.014 (Legacy)** — essentially
zero — while keeping physics at 0.511 and shape at 0.772, all from a **single
shared** embedding. Compare Ours (0.559/0.649) and Contrastive (0.697/0.795).

Predicting one survey's embedding from the other forces the model to retain only
what the two surveys share (the galaxy) and discard what is survey-specific (PSF,
depth, seeing). That is precisely the invariance the two-encoder architecture was
designed to obtain — here it falls out of the objective itself. This is the result
most worth writing up.

`cf` sits in between (0.495 physics; 0.352/0.443 instrument). Its instrument tower
legitimately retains instrument signal: it is trained on sky-neighbours that *share*
observing conditions, so unlike `cross` it is never asked to discard them.

---

## 4. Results — collapse diagnostics and anomaly detection

### Collapse: no arm collapsed

BYOL-family objectives can collapse to a constant embedding, and the training loss
looks *good* when they do, so this is measured directly from the checkpoint by
`check_collapse.py`.

| arm / tower | dim | rel_std | **centered** | eff_rank | top1_var | verdict |
|---|---|---|---|---|---|---|
| `hsc-vit` e1 | 192 | 0.79 | 0.97 | 20.2 | 0.264 | healthy |
| `hsc-resnet` e1 | 64 | 0.54 | 0.95 | 7.9 | 0.375 | healthy |
| `cf` e1 physics | 64 | 0.39 | 0.96 | 11.3 | 0.329 | healthy (offset-dominated) |
| `cf` e2 instrument | 64 | 0.73 | 0.97 | 12.9 | 0.379 | healthy |
| `cross` e1 | 64 | 0.25 | 0.98 | 16.8 | 0.158 | healthy (offset-dominated) |

All five towers sit at 0.95–0.98 centered spread, i.e. near-isotropic once a shared
mean offset is removed.

### Anomaly detection: 103,741 anchors, 74 known strong lenses

Best lens-retrieval ROC-AUC per arm. Chance = 0.500; the contrastive
`spatial-conv1x1` reference = **0.712**.

| arm | best ROC-AUC | representation / method | trained on |
|---|---|---|---|
| `hsc-resnet` | **0.806** | physics_legacy / iforest | HSC only, 367k |
| `hsc-vit` | **0.805** | physics_hsc / mahal | HSC only, 367k |
| `cf-token` | 0.753 | — | anchors, 104k |
| `multi-vit` | 0.750 | physics_hsc / mahal | both surveys, 572k |
| Contrastive sp-conv1x1 | 0.712 | instrument_hsc / knn | anchors, 104k |
| **Ours (FM base)** | 0.696 | physics_mean / flow | anchors, 104k |
| `cf-vit` | 0.693 | physics_mean / flow | anchors, 104k |
| `xjepa` | 0.684 | — | anchors, 104k |
| `cf` | 0.652 | physics_hsc / flow | anchors, 104k |
| **AE baseline (reconstruction)** | 0.620 | physics / iforest | HSC only, 367k |
| `cross` | 0.608 | physics_mean / flow | anchors, 104k |

**Round-3 rows refine the lens story once more.** `cf-token` — trained on only the
104k anchors — reaches 0.753, breaking out of the anchor-corpus band (0.61–0.71)
and closing most of the gap to the 367k arms; `multi-vit` — trained on 572k —
sits at the same 0.750. So neither corpus size nor coverage is the driver: the
lens advantage tracks **token-level JEPA geometry**, strongest when trained
single-survey on HSC (0.805). Meanwhile `xjepa` (0.684) shows that being the best
*downstream* model does not transfer to anomaly detection — its subspace heads
evidently reshape the geometry in a way the detectors don't reward.

**The AE row changes the confound picture.** It shares the 367k single-survey
corpus with the two strong arms yet retrieves lenses no better than the
anchor-trained arms — so corpus size/coverage alone does not buy ~0.80, and the
weight shifts toward the *single-survey JEPA objective* producing lens-friendly
geometry. Not fully resolved (the AE differs in objective, and its space carries
little instrument info, which lens retrieval seems to use) — the clean
disambiguator remains `hsc-vit` retrained on the 103,741 anchors — but "it's
just more data" is no longer a tenable sole explanation.

The Ours row (added 2026-08-01, matched protocol: same 104k corpus, kNN
self-excluded, standardized — `TAG=ours-fmbase`) lands *between* the contrastive
baseline and the cross-survey JEPA arms — and it slots exactly where the
training-regime split predicts: every arm trained on the 104k anchor pairs sits at
0.61–0.71, both arms trained on 367k single-survey images sit at ~0.805. That
tightens the §4 confound story rather than resolving it. Note also iforest in the
Ours spaces is mid-pack (0.55–0.68), not at-chance like the contrastive spaces and
not best-in-class like the JEPA spaces — three different geometries, three
different iforest behaviours.

Both HSC-only arms beat the contrastive reference by ~0.09. Isolation forest — which
sat at **chance** (0.445–0.492) in every contrastive space — is best or near-best
here, supporting the existing theory that its failure there was about the geometry
of those particular latents rather than a scoring bug.

**On why the two groups differ — the honest version.** It is tempting to say the
cross-survey arms are worse because they discarded instrument information, which
lens retrieval leans on. **The data does not support that**:

| arm | mean instrument R² | best lens ROC-AUC |
|---|---|---|
| `hsc-resnet` | 0.263 | 0.806 |
| `hsc-vit` | 0.636 | 0.805 |
| `cf` | 0.416 | 0.652 |
| `cross` | 0.020 | 0.608 |

`hsc-resnet` has the second-lowest instrument R² and the highest ROC-AUC. The
ordering is not monotone in instrument information.

What the split *does* coincide with is **two confounded differences** the current
runs cannot separate: (a) the objective (single-survey vs cross-survey), and (b) the
training corpus — the HSC arms saw 366,706 images, a strict superset of the 103,741
anchors the anomaly evaluation scores, while `cf`/`cross` saw only those anchors.
More data, and better coverage of the scored corpus itself.

**Disambiguating run:** retrain `hsc-resnet` (or `hsc-vit`) on only the 103,741
anchors — matched corpus, single-survey objective. Until that exists, do not
attribute the anomaly gap to disentanglement.

---

## 4b. Round 2 (2026-08-01): AE baseline, cf-vit, Ours lens row

Added at Pablo's request after the first pass: a **reconstruction baseline**
("AE baseline" — it is *not* a JEPA arm; the `jepa-ae-hsc` file tag is harness
plumbing), a ViT version of the counterfactual arm (`cf-vit`), and the Ours row
in the lens table (§4).

### AE baseline (reconstruction): same encoder/data/budget as `hsc-resnet`

ResNet18 encoder (64-d, conv1x1) + small deconv decoder, pixel MSE on the 367k
HSC pool, 100k steps.

| | physics | instr HSC | instr Legacy | shape |
|---|---|---|---|---|
| **AE baseline** | 0.684 | 0.172 | 0.165 | 0.928 |
| JEPA `hsc-resnet` (same encoder/data) | 0.561 | 0.273 | 0.258 | 0.638 |
| JEPA `hsc-vit` | 0.753 | 0.572 | 0.664 | 0.919 |
| Ours (FM base) | 0.704 | 0.559 | 0.649 | 0.929 |

(The AE's instrument numbers are probed on its single space via `combined_e1`,
the AION convention — its H5 has no e2 datasets because `prepare_combined`'s
`_is_single_encoder` branch caught the module's `.encoder` attribute first.
Equivalent semantics to the aliased single-space JEPA arms.)

**The registered prediction was wrong, informatively.** Predicted: high
instrument R² (reconstruction must keep PSF/noise), moderate physics. Observed:
the opposite — the 64-d bottleneck spends its capacity on morphology (which
dominates pixel MSE) and nearly discards instrument signal (0.17).

Three consequences:

1. **The masked-ResNet result is now properly controlled.** Plain reconstruction
   with the *identical* encoder beats masked-image JEPA by +0.12 physics and
   +0.29 shape — so `hsc-resnet`'s weakness is the input-masking JEPA objective
   on a ResNet, not the ResNet itself.
2. **JEPA-ViT's honest margin over the strongest cheap baseline** is +0.07
   physics and **+0.40 instrument** from one shared space. Retaining
   observing-condition information is the JEPA objective's distinctive signature
   vs reconstruction.
3. **A plain AE nearly matches Ours on physics** (0.684 vs 0.704) and ties on
   shape (0.928 vs 0.929) — a reference point the paper's framing has to live with.

Collapse: healthy — and the AE has the **highest effective rank of any 64-d
space here** (22.8, centered spread 0.98) while carrying the *least* instrument
information. Rank measures spread, not content; don't use eff_rank to argue a
space is informative, only that it isn't collapsed.

### cf-vit: the ViT does NOT rescue the counterfactual objective

Same counterfactual structure as `cf`, both towers swapped to mean-pooled
PatchViTs (192-d, `encoder_arch="vit"`). Trained 1h48 (heaviest arm: ~7 ViT
forwards/step). Both towers healthy (eff_rank 20.2 / 10.3, centered 0.96–0.98).

| | physics | instr HSC | instr Legacy | shape |
|---|---|---|---|---|
| `cf` (ResNet, conv1x1) | 0.495 | 0.352 | 0.443 | **0.803** |
| `cf-vit` (mean-pooled ViT) | 0.468 | 0.411 | 0.529 | **0.439** |

Physics slightly down, instrument up a bit, **shape halves**. The comparison with
`hsc-vit` (also mean-pooled ViT at eval, shape 0.919) isolates the cause: I-JEPA's
*token-level* prediction target forces spatial information to survive pooling,
whereas `cf-vit`'s loss only ever sees pooled projections, so shape structure has
nothing protecting it (same failure family as the baseline's avg-vs-conv1x1
pooling axis).

**Refined conclusion for the whole study: the round-one "ViT vs ResNet" gap was
never about the backbone per se — it is the token-level latent-masking objective
that matters.** An encoder swap without that objective (`cf-vit`) does not
reproduce the gain. If the counterfactual idea is to benefit from ViTs, the
natural next design is a token-level cross-view objective (predict the anchor's
*tokens* from the partner's tokens — a cross-survey I-JEPA), not a pooled BYOL
loss on a ViT.

## 4c. Round 3 (2026-08-01): token-level counterfactual designs

Goals set with Pablo: (A) one model whose HSC+Legacy embeddings combine well;
(B) two disentangled latent spaces. Three arms; success criteria pre-registered
in PLAN.md §Round 3. Full cross-probe table (leakage = the off-diagonal probes):

| arm | phys@e1 | shape@e1 | iHSC@e2 | iLEG@e2 | iHSC@e1 | iLEG@e1 | phys@e2 |
|---|---|---|---|---|---|---|---|
| `cf` (pooled) | 0.495 | 0.803 | 0.352 | 0.443 | **0.021** | **0.009** | 0.474 |
| `cf-vit` (pooled) | 0.468 | 0.439 | 0.411 | 0.529 | 0.178 | 0.329 | 0.273 |
| **`cf-token`** | 0.736 | **0.897** | 0.579 | 0.669 | 0.632 | 0.707 | 0.517 |
| **`xjepa`** | 0.725 | **0.897** | **0.704** | **0.788** | 0.678 | 0.782 | 0.683 |
| `multi-vit` | 0.664 | 0.706 | 0.637 | 0.759 | (single space) | | |
| Contrastive | 0.663 | 0.897 | 0.697 | 0.795 | 0.030 | 0.009 | 0.409 |
| Ours | 0.704 | 0.929 | 0.559 | 0.649 | 0.094 | 0.177 | 0.524 |

**Goal A: achieved.** `xjepa` is the best single model in the project when both
of its outputs are used: physics 0.725 / shape 0.897 (near hsc-vit) AND
instrument 0.704/0.788 (at the contrastive baseline's level — the pre-registered
"home run" bar of 0.697/0.795, met on HSC, 0.007 short on Legacy). `cf-token`'s
physics tower alone is the best generalist single space (0.736/0.897/0.63/0.71).
Naive mixing (`multi-vit`) is strictly worse than either: physics 0.664, shape
0.706 at matched budget — structured cross-survey objectives beat more data.

**Goal B: NOT achieved — and the failure is systematic, which is the finding.**
`xjepa`'s subspaces leak almost completely (instrument 0.678/0.782 decodable from
the physics subspace; physics 0.683 from the instrument one). Across all seven
dual-space arms the pattern is now clean:

- **Pooled cross-view targets disentangle** (cf 0.02, contrastive 0.03, cross
  ~0.02 leakage) **but pay in richness** (cf physics 0.495, cross 0.511).
- **Token-level targets maximize information but resist disentanglement**
  (cf-token 0.63-0.71, xjepa 0.68-0.78 leakage): at token granularity, local
  texture is inseparable from PSF/noise, so instrument information rides along.
- **Linear subspace heads on a shared trunk provide no separation pressure** —
  both slices of an entangled token space carry everything (xjepa's mask term,
  λ=0.5, actively feeds the trunk everything; the routing losses don't remove).

`cf-token` also settles the round-2 question by construction: pooled→token was
the ONLY change vs cf-vit, and shape went 0.439→0.897, physics 0.468→0.736.
The pooling explanation is proven, and the original "views are the corruption,
no masking needed" intuition holds — at token level.

**Candidate round-4 designs for goal B** (not run; in rough order of promise):
1. **Hybrid targets**: physics tower trained with BOTH a pooled cross-survey
   term (the proven disentangler) and a low-weight token term (the proven
   richness protector) — let the weights negotiate the trade-off explicitly.
2. **xjepa + separation pressure**: decorrelation penalty between pooled
   subspace summaries, and/or stop-gradient from the instrument loss into the
   trunk (so instrument texture must live in the head, not the shared tokens).
3. **Accept the dichotomy and use two models**: `cross`/`cf` for the clean
   physics space, `xjepa`/`cf-token` for the performance space. Defensible, and
   zero additional compute.

## 5. What went wrong (five bugs, all fixed)

Recorded because several are traps that will recur in this repo.

**1. `best-on-val/loss` checkpoint selection is invalid for JEPA/BYOL.** Copied from
the contrastive harness. The target side is *learned*, so the prediction loss can
fall via collapse and legitimately *rises* as representations get richer. Every arm
put its best val/loss at **step 2,000 of 100,000**; the first eval ran on an
essentially untrained model before this was caught. `eval_jepa.slurm` now takes
`latest-step=<MAX_STEPS>.ckpt`; `best-*.ckpt` is kept as a diagnostic only. Two CSVs
and two embedding files produced from step-2000 checkpoints were deleted.

**2. SLURM snapshots the batch script at submit time.** Fixing `eval_jepa.slurm` did
*not* fix the eval jobs the training chain had already queued — one kept running the
old best-ckpt logic and had to be cancelled and resubmitted. Verify with
`scontrol write batch_script <jobid> -` and resubmit anything stale. This matters for
every self-chaining job in this repo.

**3. Two self-inflicted bugs inside the fix for #1.** The new collapse check was
first placed *above* the `conda activate` block (system python, `ModuleNotFoundError:
numpy`), and `COLLAPSE_SUMMARY=$(grep …)` aborts the whole script under `set -e` +
`pipefail` when grep matches nothing. Diagnostics must never be able to kill the
eval; every failure path is now swallowed.

**4. The collapse verdict rule was wrong and falsely flagged `cross` as "weak".** It
judged on the std of L2-normalized vectors, which is depressed by a large shared
*mean offset* even when the space is perfectly well spread around it — `cross` reads
0.25 uncentered but 0.98 centered, with healthy effective rank 16.8. A probe's first
linear layer absorbs an offset harmlessly. The script now reports both and judges
primarily on effective rank. **Lesson: a spread statistic computed without centering
measures the offset, not the spread.**

**5. `_on_exit() { rc=$?; [[ $rc -ne 0 ]] && _ping …; }` makes successful jobs report
FAILED.** When the condition is false the `[[ ]] &&` returns 1, and as the last
command in an EXIT trap that becomes the script's exit status. Fixed here with an
explicit `if` + `exit $rc`. **The same bug is still present in
`contrastive_baseline/eval_contrastive_engaging.slurm`** — left alone as out of
scope, but worth fixing.

Also worth knowing: a chained successor came back as `PENDING` with reason
`user_env_retrieval_failed_requeued_held` after being cancelled — SLURM had requeued
and held it. Check for held jobs when tidying up a chain.

**6 (round 2). The checkpoint loader rejected the AE baseline.**
`load_jepa_checkpoint` demanded `ema_base_momentum` in hparams before consulting
the arm marker, and the AE legitimately has no EMA — so its eval-time collapse
check and paper UMAP silently skipped (by design the eval survives diagnostic
failures; a fix-up job backfilled both). When a harness grows a new arm, walk
every consumer of the checkpoint, not just the training path.

One further correction, made after the results were first reported: the claim that
anomaly performance tracked residual instrument R² monotonically was wrong (§4).

---

## 6. Caveats on the headline

Ranked by how much they could change the conclusion.

1. **LR schedule differs from the published contrastive arms.** The JEPA runs use a
   fixed step-based cosine + warmup; the published arms carry the repo-wide
   `T_max=max_epochs` quirk that leaves them at LR 0 for roughly half their steps.
   Since `hsc-vit` beat contrastive, this difference is load-bearing. **The control
   is rerunning `contrastive-spatial-conv1x1` on the fixed schedule** — this is the
   single most important follow-up and was deliberately deferred at planning time.
2. **`hsc-vit` differs in architecture as well as width.** PCA-64 controls width; it
   does *not* control ViT-vs-ResNet. A 64-d ViT or a 192-d ResNet would separate them.
3. **The anomaly gap is confounded** between objective and training-corpus size (§4).
4. **No DWNVAL holdout**, matching the published protocol so the overlay bars are
   apples-to-apples. SSL sees no labels, so contamination is mild.
5. **`hsc-*` arms never saw Legacy images**; their Legacy-side numbers measure
   transfer, and their UMAP Legacy points are pushed through an HSC-trained encoder
   (visualization only).
6. Single run per arm, seed 42, one probe seed. No error bars.

## 7. Recommended next steps

1. **Matched-schedule contrastive control** (§6.1). Gates the headline claim.
2. **`hsc-*` retrained on the 103,741 anchors only** — disambiguates the anomaly
   result (§4) and doubles as a data-scaling point for the ViT.
3. **Width/architecture separation** for `hsc-vit` (§6.2).
4. If `cross`'s disentanglement is to be a paper claim, add seeds for error bars —
   instrument R² of 0.014–0.035 is a strong statement resting on one run.

---

## 8. Reproducing / operating

```bash
cd galaxy_images/galaxy_model/jepa_ablation
sbatch smoke_test.slurm      # 30-step micro-run of all 4 arms + every eval surface
bash   submit_all_jepa.sh    # the 4 real runs; each auto-chains its own eval
sbatch --export=ALL,ARM=cf --job-name=jepa-eval-cf eval_jepa.slurm        # re-run one eval
sbatch --export=ALL,ARM=hsc-vit,NCOMP=64 width_control.slurm              # PCA width control
sbatch recheck_collapse.slurm                                             # collapse re-check, all arms
```

Training self-chains (successor queued before training starts, `DONE` marker stops
the chain, `MAX_ATTEMPTS=10`), resumes from `last.ckpt`, and on completion submits
the eval, which runs collapse check → R² probe → overlay plots → paper UMAP →
anomaly pipeline. Discord pings at every transition.

**Watching for collapse during training:** `train/std_target` and `train/std_online`
(mean per-dim std of normalized projections; healthy ≈ `1/√D`), with `train/align`.
Rising align + falling std = collapsing. Fixes are hparams only: predictor lr ×10,
then `ema_base_momentum` 0.99.

### Files

| file | role |
|---|---|
| `datasets.py` | `HSC48PoolDataset` (367k), `AnchorPairs48Dataset` (anchors), `load_umap_pairs` |
| `jepa_base.py` | EMA machinery, cosine loss, schedules, diagnostics, wandb UMAP hook |
| `jepa_modules.py` | `JEPAHSCModule`, `JEPACounterfactualModule`, `JEPACrossSurveyModule` |
| `jepa_vit.py` | `PatchViT`, `ViTPredictor`, `IJEPAViTModule` |
| `train_jepa.py` | single entry for all arms (`--arm`) |
| `check_collapse.py` | collapse measurement from a checkpoint |
| `pca_reduce_embeddings.py` | width control (PCA an embeddings H5) |
| `regenerate_umap_jepa.py` | checkpoint → npz → paper-style UMAP |
| `*.slurm`, `submit_all_jepa.sh` | orchestration |

### Artifacts

- R² CSVs — `downstream_evaluation/engaging/outputs/predictions/predict_jepa-*__aion.csv` (5, incl. `pca64`)
- Combined figure — `.../plots_publication/unified_three_probes_jepa_all_vs_ours.{png,pdf}`
- UMAPs — `jepa_ablation/results/umap/` (4 arms)
- Anomaly — `/orcd/pool/007/pablomer/outputs/contrastive_anomalies/*jepa-*` (4 summaries)
- Collapse — `checkpoints_new/jepa-<arm>/collapse_check.txt` (4)
- Checkpoints — `/orcd/data/mki_aryeh/001/pablomer/jepa_ablation/` via `checkpoints_new/jepa-<arm>` symlinks
- wandb — project `galaxy-jepa-engaging`
