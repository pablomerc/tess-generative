# JEPA Ablation — Results (first full pass, 2026-07-31)

> **See [`REPORT.md`](REPORT.md) for the single comprehensive write-up** (design, results, bugs, caveats, next steps). This file is one slice of it.


All four arms trained to 100k steps and evaluated through the standard n=5469
pipeline. Design in `PLAN.md`, operations in `README.md`. **Every number below is
from the FINAL checkpoint (`latest-step=100000.ckpt`), not `best-*.ckpt`** — see
README for why best-on-val/loss is invalid for this objective family.

## Training

| arm | wall clock (1× H200) | trainable params | data |
|---|---|---|---|
| `hsc-vit` | 57 min | 3.2 M | 366,706 HSC stamps |
| `cross` | 70 min | 11.2 M | 103,741 anchor pairs |
| `hsc-resnet` | 79 min | 11.2 M | 366,706 HSC stamps |
| `cf` | 83 min | 22.4 M | 103,741 anchor triplets |

All four ran concurrently on node4900's four H200s — the whole matrix in <90 min.

## Downstream R² (n=5469, AION MLP probe, seed 0, mean per family)

| run | physics (e1) | instr HSC (e2) | instr Legacy (e2) | mean shape |
|---|---|---|---|---|
| Ours (FM base) | 0.704 | 0.559 | 0.649 | 0.929 |
| AION-1 | 0.742 | — | — | 0.938 |
| Contrastive sp-conv1x1 | 0.663 | 0.697 | 0.795 | 0.897 |
| **JEPA hsc-vit (192-d)** | **0.753** | 0.572 | 0.664 | 0.919 |
| **JEPA hsc-vit PCA (64-d)** | **0.745** | 0.545 | 0.607 | 0.909 |
| JEPA hsc-resnet (64-d) | 0.561 | 0.273 | 0.258 | 0.638 |
| JEPA cf (64-d) | 0.495 | 0.352 | 0.443 | 0.803 |
| JEPA cross (64-d) | 0.511 | **0.035** | **0.014** | 0.772 |

⚠️ **The `e1`/`e2` labels mislead for the single-encoder arms.** `hsc-resnet`,
`hsc-vit` and `cross` have ONE latent space; they alias the same EMA encoder under
both names, so the harness probes identical vectors twice (verified: `hsc_e1` and
`hsc_e2` are bit-identical, probe scores agree to 4 dp). Read those rows as "how
much of each property is decodable from the one shared space". Only Ours,
Contrastive and `cf` have two genuine spaces; AION has no `e2` at all.

Also: `combined_e*` = `concat(HSC emb, Legacy emb)`, so the probe sees 384-d for
`hsc-vit` and 128-d for the 64-d arms — and the PCA control cut `hsc-vit` to 64-d,
*half* what the others get. See `REPORT.md` §3 for both points in full.

## Three findings

**1. I-JEPA on a ViT is the strongest physics representation in the table, and it
is not a width artifact.** 0.753 at its native 192-d beats Ours (0.704) and the
contrastive baseline (0.663), and edges AION-1 (0.742). Because it is the only arm
wider than 64-d, its embeddings were PCA-reduced to 64-d and re-probed with the
identical MLP (the convention the paper already uses for AION's `pca64`): **0.745**,
essentially unchanged. The advantage survives matched width. Notably this is from
HSC images alone — no cross-survey pairs, no negatives, no catalog.

**2. Masking on a ResNet does not work; the architecture has to change.** The
directly comparable `hsc-resnet` arm — same data, same 64-d width, same schedule,
masking applied to the input instead of to latent patches — reaches only 0.561
physics and 0.638 shape. The collapse diagnostics agree with the R² gap rather
than explaining it away: `hsc-resnet` has effective rank 7.9 of 64 dims, whereas
`hsc-vit` has 20.2 of 192 with a better-conditioned spread. This answers the
original design question — a ResNet's final 2×2 feature map leaves nothing
meaningful to mask in latent space, and input masking is a poor substitute.

**3. Cross-survey JEPA produces the cleanest disentanglement anywhere in this
project.** `cross` drives instrument R² to **0.035 (HSC) and 0.014 (Legacy)** —
essentially zero — while keeping physics at 0.511 and shape at 0.772, all from a
*single shared* embedding. Compare Ours (0.559/0.649) and Contrastive
(0.697/0.795). Predicting one survey's embedding from the other's forces the model
to keep only what the two surveys share (the galaxy) and discard everything
survey-specific (PSF, depth, seeing) — exactly the invariance the two-encoder
architecture was designed to obtain, here falling out of the objective itself.
This is the result most worth writing up.

`cf` sits in between (0.495 physics; 0.352/0.443 instrument). Its instrument tower
retains real instrument signal, which makes sense — it is trained on sky-neighbours
that share observing conditions, so unlike `cross` it is never asked to discard them.

## Collapse diagnostics

**No arm collapsed** — all five towers `healthy` under the corrected verdict logic.
`check_collapse.py` measures this from the checkpoint, independent of the loss.

| arm / tower | dim | rel_std | **centered** | eff_rank | top1_var | verdict |
|---|---|---|---|---|---|---|
| `hsc-vit` e1 | 192 | 0.79 | 0.97 | 20.2 | 0.264 | healthy |
| `hsc-resnet` e1 | 64 | 0.54 | 0.95 | 7.9 | 0.375 | healthy |
| `cf` e1 (physics) | 64 | 0.39 | 0.96 | 11.3 | 0.329 | healthy [offset-dominated] |
| `cf` e2 (instrument) | 64 | 0.73 | 0.97 | 12.9 | 0.379 | healthy |
| `cross` e1 | 64 | 0.25 | 0.98 | 16.8 | 0.158 | healthy [offset-dominated] |

⚠️ **The first verdict rule was wrong and flagged `cross` "weak".** It judged on
`rel_std` alone, which is computed on L2-normalized vectors and is therefore
depressed by a large *shared mean offset* even when the space is perfectly well
spread around that offset. The centered column settles it: every arm sits at
0.95–0.98, i.e. near-isotropic once the offset is removed, and a probe's first
linear layer absorbs an offset harmlessly. The script now reports both and judges
primarily on effective rank. Lesson worth keeping: a spread statistic computed
without centering measures the offset, not the spread.

## Anomaly detection (103,741 anchors, 74 known strong lenses)

Best lens-retrieval ROC-AUC per arm (chance = 0.500; contrastive `spatial-conv1x1`
reference = **0.712**):

| arm | best ROC-AUC | representation / method | trained on |
|---|---|---|---|
| `hsc-resnet` | **0.806** | physics_legacy / iforest | HSC only |
| `hsc-vit` | **0.805** | physics_hsc / mahal | HSC only |
| Contrastive sp-conv1x1 | 0.712 | instrument_hsc / knn | anchors |
| Ours (FM base, `ours-fmbase`, added 2026-08-01) | 0.696 | physics_mean / flow | anchors |
| `cf` | 0.652 | physics_hsc / flow | anchors + neighbours |
| `cross` | 0.608 | physics_mean / flow | anchors |

Two things stand out. **Both HSC-only arms beat the contrastive reference by ~0.09
ROC-AUC**, and isolation forest — which sat at chance (0.445–0.492) in every
contrastive space — is the best or near-best detector here. That supports the
existing suspicion that iforest's failure in the contrastive spaces was about the
geometry of those particular latents rather than a bug in the scoring code.

**The split is by training regime, NOT by residual instrument information.** The two
single-survey arms sit at ~0.805; the two cross-survey arms drop to 0.652 and 0.608.

⚠️ An earlier draft of this file claimed the ordering tracked residual instrument R²
monotonically. **That was wrong** — check it against the numbers:

| arm | mean instrument R² | best lens ROC-AUC |
|---|---|---|
| `hsc-resnet` | 0.263 | 0.806 |
| `hsc-vit` | 0.636 | 0.805 |
| `cf` | 0.416 | 0.652 |
| `cross` | 0.020 | 0.608 |

`hsc-resnet` has the *second-lowest* instrument R² and the *highest* ROC-AUC, so the
instrument-invariance explanation does not survive contact with the data.

What the split does coincide with is **at least two confounded differences**, and the
current runs cannot separate them:

1. **Objective**: single-survey masked prediction vs cross-survey prediction.
2. **Training corpus**: the HSC arms saw 366,706 images (`source_type` 0+1), which is
   a strict superset of the 103,741 anchors the anomaly evaluation scores; `cf` and
   `cross` only ever saw those 103,741 anchors. More data, and better coverage of the
   scored corpus itself.

So the honest statement is: cross-survey-trained arms are markedly worse at lens
retrieval here, and we do not yet know whether that is the invariance or the 3.5×
smaller/narrower training set. **The disambiguating run is `hsc-resnet` or `hsc-vit`
retrained on only the 103,741 anchors** — matched corpus, single-survey objective.
Until then, do not attribute the anomaly gap to disentanglement.

For the single-encoder arms the `physics_*` and `instrument_*` rows are
bit-identical, the expected signature of the duck-typed alias and a useful check
that it resolves as documented.

## Caveats

- **LR schedule differs from the published contrastive arms** (fixed step-based
  cosine + warmup vs the repo's `T_max=max_epochs` quirk). Since `hsc-vit` does beat
  the contrastive arms, the matched control — rerunning `contrastive-spatial-conv1x1`
  with the fixed schedule — should be run before claiming an objective-level win.
  This is the single most important follow-up.
- `hsc-vit` differs from the 64-d arms in architecture as well as width. PCA-64
  controls the width; it does **not** control ViT-vs-ResNet. A 64-d ViT or a
  192-d ResNet would separate those.
- No DWNVAL holdout, matching the published protocol (PLAN.md decision 3).
- `hsc-*` arms never saw Legacy images; their Legacy-side numbers measure transfer.

## Artifacts

- R² CSVs: `downstream_evaluation/engaging/outputs/predictions/predict_jepa-*__aion.csv`
- Combined figure: `.../plots_publication/unified_three_probes_jepa_all_vs_ours.{png,pdf}`
- UMAPs: `jepa_ablation/results/umap/`
- Anomaly: `/orcd/pool/007/pablomer/outputs/contrastive_anomalies/*jepa-*`
- Collapse: `checkpoints_new/jepa-<arm>/collapse_check.txt`
- wandb: project `galaxy-jepa-engaging`
