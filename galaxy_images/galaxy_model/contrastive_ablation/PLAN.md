# Contrastive ablation — experiment plan (rebuttal JcH2 Q1)

Answers to Pablo's questions, then the run queue split by cluster. Background and
evidence: `ANALYSIS.md`. Status legend: **DONE** · **READY** (code committed, needs a
GPU) · **TODO** (needs code).

---

## 1. Latent-size matching — what to do

**Don't retrain the contrastive model smaller. Mean-pool its tokens post-hoc and
report both dim-matched levels.** — **DONE**, numbers below.

Why not retrain at `embedding_dim=16`: the contrastive encoder's output *is* its
InfoNCE input, so shrinking it handicaps the objective itself. Ours' 4×16 tokens feed
cross-attention and are never squeezed through a 16-D contrastive bottleneck, so a
16-D contrastive run would be a harsher setup than Ours, not a fairer one.

What is fair: apply the *same* pooling operation to the *same* token shape. Both
`base` and `contrastive-spatial-conv1x1` emit 4 tokens × 16 dims, so
`ENG/build_meanpool_h5.py` (which pools each survey's half independently and
reconcatenates) applies identically to both. That gives two honest rows:

| level | Ours | contrastive | note |
|---|---|---|---|
| pooled | 16-D/image → **32-D** probe input | 16-D/image → **32-D** | matches the paper's published instrument latent |
| flat | 64-D/image → **128-D** | 64-D/image → **128-D** | matches the paper's published physics latent |

Result (`predict_contrastive-spatial-conv1x1-meanpool__aion.csv`, seed 0):

| property | Ours 32d | Con 32d | Ours 128d | Con 128d |
|---|---|---|---|---|
| Galaxy Depth | 0.682 | **0.784** | 0.738 | **0.874** |
| PSF Depth | 0.667 | **0.778** | 0.729 | **0.869** |
| # Observations | 0.501 | **0.613** | 0.543 | **0.742** |
| PSF size (Legacy) | 0.556 | **0.624** | 0.625 | **0.763** |
| E(B−V) | 0.551 | **0.681** | 0.616 | **0.740** |
| PSF FWHM (HSC) | 0.438 | **0.540** | 0.503 | **0.658** |
| *leak:* z from instrument | 0.535 | **0.364** | 0.654 | **0.568** |
| *leak:* ellipticity from instrument | 0.371 | **0.143** | 0.663 | **0.462** |
| z from physics | **0.743** | 0.686 | **0.801** | 0.740 |
| log M\* from physics | **0.654** | 0.584 | **0.755** | 0.675 |
| ellipticity (legacy) from physics | 0.734 | **0.864** | **0.926** | 0.906 |

**Read:** the dim mismatch was real but explains only about a quarter of the
instrument gap (Galaxy Depth +0.136 → +0.102 after matching). Contrastive genuinely
wins instrument properties, and at matched dims it also *leaks less* physics into the
instrument latent. Ours keeps its edge on spectroscopic physics at both levels. One
artefact to be careful about: pooling costs Ours a lot of ellipticity (0.926 → 0.734)
because shape needs spatial tokens, so **for the physics panel quote the flat 128-D
row**; the pooled row is the honest instrument comparison.

## 2. Negatives in the galaxy encoder — mixed or single-survey? — **READY** (exp B)

Currently **mixed, and it matters**. Anchors alternate survey by index parity, so for
anchor *i* the negatives (other items' `samegal`) are ~50% in the same survey as
*i*'s positive and ~50% in the other. Half the discrimination problem is therefore
solvable by reading survey identity instead of galaxy identity — an easier task than
it looks, which could mean *less* pressure to encode fine physics.
`--galaxy-negatives same_survey` keeps only negatives sharing the positive's survey,
so survey identity carries zero signal. Expect physics R² to go **up** if the mixed
pool was letting it cheat.

## 3. Negatives in the instrument encoder — same survey or both? — **READY** (exp C)

Currently **both**, and this is my top suspect for the "too good to be true" feeling.
The candidate pool is every item's neighbours, spanning both surveys, and separating
HSC from Legacy is trivially easier than separating two HSC patches with different
seeing. So a large part of the instrument InfoNCE may be satisfied by a survey
classifier — which would still score well on depth/PSF/E(B−V) because those differ
systematically between surveys. `--instrument-negatives same_survey` forces
within-survey discrimination. **If the instrument advantage collapses, that is the
single most informative result in this whole set**, and it directly supports the
paper's (currently commented-out) point that contrastive objectives are sensitive to
batch composition.

Caveat built into the design: restricting negatives also *halves* their count, and
InfoNCE quality depends on count. So each restricted arm should also be run at
`--batch-size 128` to separate "harder negatives" from "fewer negatives".

## 4. Drop SimCLR and put the loss on the representation? — **READY** (exp D), and yes, relevant

Not out of date — it is the standard SimCLR ablation, and here it has teeth. We probe
the *pre-head* encoder output (correct, and what every arm did), which means the head
lets the encoder keep information the loss would otherwise discard. That is an
advantage for contrastive on downstream probes, and a reviewer could reasonably ask
whether it is doing the work. `--projection-head 0` runs InfoNCE directly on the
64-D representation. Expect downstream R² to **drop**; if it doesn't, the head is
irrelevant here and the comparison is robust to that choice.

## 5. DWNVAL + 5NB — **READY** (exp A)

Committed and validated. One clarification: **the contrastive arms already use K=5
neighbours** (`MAX_NEIGHBORS=5`), so "5NB" is already satisfied — no separate run
needed. What matters is the *comparison partner*: `base` (the paper's Ours) has no
DWNVAL twin, but `base-5NB` / `base-5NB-DWNVAL` do. So quote the contamination delta
against that pair, giving a clean 2×2 of {FM, contrastive} × {all-data, holdout}.
Already measured for FM: mean ΔR² −0.005 (5NB) / +0.005 (15NB) — noise.

## 6. Five folds with error bars — **DONE** (exp E)

Job 18809342, 56 min on `mit_normal`. Full tables: `RESULT_folds.md` (mean ± std per
cell) and `RESULT_paired_{flat128,pooled32}.md`. Typical fold-to-fold std is **0.020**
(90th pct 0.032), so unpaired differences below ~0.04 are not resolvable.

**Read the differences paired**, not by combining the two variants' std's: every fold
tests the same galaxies for both models, so that noise is common and cancels. It makes a
large difference — the redshift gap is 0.032 ± 0.008 unpaired but **0.032 ± 0.002**
paired. `paired_fold_test.py` does this.

Dim-matched flat (128-D), Ours − contrastive, ± SE over folds:

| cell | Ours − Contr | verdict |
|---|---|---|
| Redshift z | **+0.032 ± 0.002** | Ours wins |
| log M\* | **+0.048 ± 0.004** | Ours wins |
| sSFR | +0.027 ± 0.010 | marginal |
| Ellipticity (Legacy-measured) | **+0.013 ± 0.001** | ~tied |
| Ellipticity (HSC-measured) | **+0.040 ± 0.002** | Ours wins |
| Galaxy Depth | **−0.100 ± 0.002** | contrastive wins |
| E(B−V) | **−0.179 ± 0.009** | contrastive wins |
| *leak:* z from instrument | **+0.133 ± 0.014** | contrastive cleaner |
| *leak:* depth from physics | **+0.178 ± 0.003** | contrastive cleaner (Ours 0.170, Contr −0.008) |

At matched pooled 32-D the instrument gap shrinks but survives (Galaxy Depth
−0.062 ± 0.007, E(B−V) −0.140 ± 0.013), and the physics gap becomes marginal
(z +0.024 ± 0.010).

### The headline result: "collapse to lowest resolution", quantified

> (Ours − Contr on **HSC** shapes) − (Ours − Contr on **Legacy** shapes)
> = **+0.026 ± 0.003**, t = +10.6

Contrastive's shape deficit is **3× larger on the higher-resolution survey's shapes**
than on the lower-resolution one's. As a difference of differences this cannot be
explained by either model being better overall, and it is a direct quantitative
confirmation of the intro's claim (tex:177-178) that aligning views "collapses toward
the lowest-resolution observation and discards fine-grained features visible in
higher-quality instruments" — previously asserted, never measured. **This belongs in
the rebuttal.**

Caveat to state if we quote absolute numbers: 5-fold trains each probe on 80% of the
sample versus 90% for the published single split, so absolute R² sits slightly lower
(base redshift 0.784 ± 0.005 vs the published 0.801). That is a training-set-size
effect, not a bad draw, and it applies identically to every variant, so all
*comparisons* are unaffected. Quote the published 90/10 numbers in tables and these
folds as the uncertainty on the differences.

### Original rationale

`ENG/predict_combined.py` now takes `--n-folds 5 --fold {0..4}`: a disjoint partition
so every galaxy is tested exactly once, rather than 5 correlated random 90/10 draws.
Default (`--n-folds 0`) is byte-unchanged, so published numbers are unaffected. With
n_test ≈ 550 per fold, several of the gaps we are quoting (0.740 vs 0.801) need this
before they go in a rebuttal table.

---

## Run queue

### On AMD (training — 1 GPU each, ~100k steps)

All four use one launcher; defaults reproduce the published arm so each override
isolates one variable. Read `RUN_INSTRUCTIONS.md` first — **run the preflight**, and
check its throughput benchmark before committing an allocation.

| # | arm | command (from repo root on AMD) | why |
|---|---|---|---|
| A | `spatial-conv1x1-DWNVAL` | `sbatch galaxy_images/galaxy_model/contrastive_ablation/train_ablation_amd.slurm` | contamination hygiene (§5) |
| C | `instrneg-samesurvey-DWNVAL` | `sbatch --export=ALL,RUN_TAG=instrneg-samesurvey-DWNVAL,INSTRUMENT_NEGATIVES=same_survey …/train_ablation_amd.slurm` | is the instrument win a survey classifier? (§3) — **highest value** |
| B | `galneg-samesurvey-DWNVAL` | `sbatch --export=ALL,RUN_TAG=galneg-samesurvey-DWNVAL,GALAXY_NEGATIVES=same_survey …/train_ablation_amd.slurm` | harder galaxy task → better physics? (§2) |
| D | `nohead-DWNVAL` | `sbatch --export=ALL,RUN_TAG=nohead-DWNVAL,PROJECTION_HEAD=0 …/train_ablation_amd.slurm` | is the SimCLR head doing the work? (§4) |

Optional follow-ups, only if B or C move the numbers:
`--export=…,BATCH_SIZE=128` on that arm (negative-count control, §3), and a
compute-matched arm at `MAX_STEPS=350000` (Ours saw ~22M images, contrastive ~6.4M).

**All four arms hold out the downstream-eval galaxies** (`HOLDOUT=1`, the default), so
the whole set is internally consistent and no arm can be dismissed on contamination
grounds. That makes **A the reference arm for B/C/D** — each mechanism arm differs from
A in exactly one axis. The published `contrastive-spatial-conv1x1` stays in the picture
as the A-vs-published pair that measures contamination itself. So when evaluating B/C/D,
set the comparison reference accordingly:

```bash
sbatch --export=ALL,RUN_TAG=instrneg-samesurvey-DWNVAL,REF_VAR=contrastive-spatial-conv1x1-DWNVAL \
  galaxy_images/galaxy_model/contrastive_ablation/eval_ablation_engaging.slurm
```

Hand back per arm: `best-*.ckpt` + `DONE` → `/orcd/pool/007/pablomer/checkpoints_new/contrastive-<RUN_TAG>/`.

### On Engaging (everything else — needs the downstream stores, which AMD lacks)

| # | task | command | status |
|---|---|---|---|
| — | dim-fair pooled instrument comparison (§1) | `build_meanpool_h5.py` + probe | **DONE** |
| E | 5-fold error bars (4 variants x 5 folds) | `sbatch …/folds_engaging.slurm` then `paired_fold_test.py` | **DONE** (job 18809342, §6) |
| — | eval each returned checkpoint | `sbatch --export=ALL,RUN_TAG=<tag> …/eval_ablation_engaging.slurm` | **READY** |
| F | cross-survey retrieval, physics space, Ours vs contrastive | reviewer asked for it | **TODO** |
| G | NSF-flow outlier detection on contrastive physics latents | reviewer asked for it | **TODO** |
| H | frozen-embedding decoder: same UNet on frozen contrastive vs Ours tokens → counterfactual MSE / PSD / pipeline-transfer R² | the paper's actual differentiator | **TODO** (1–2 H100-days) |

F/G/H are the reviewer's other two axes plus the decisive one; H is the experiment
that tests the claim contrastive *cannot* satisfy (no pixel-level information
guarantee). Start from `contrastive_baseline/embedding_fm_decoder/`.

Expectation-setting on F: InfoNCE trains cross-survey matching directly, so expect
contrastive to win or tie retrieval. Report it anyway and pair it with G/H.

---

## Suggested rebuttal shape

Concede the R² asymmetry is reproducible with a contrastive loss on our architecture
and triplets — that supports our claim that the *routing* creates disentanglement.
Then show the costs we can now quantify: lower R² on every spectroscopic property,
larger degradation on HSC-resolution shapes than Legacy ones (collapse to lowest
resolution, measured), the instrument advantage's dependence on cross-survey
negatives (exp C, if it lands), and no reconstruction/counterfactual capability at
all (exp H). Present off-diagonal cells against the random-ResNet floor, not zero —
the paper already argues this (tex:416, 433–435).
