# Contrastive-vs-Generative Ablation: Audit & Experiment Plan

*Analysis for the NeurIPS rebuttal (reviewer JcH2, Q1). Written 2026-07-24. All
file references are relative to the repo root unless absolute. `GM/` =
`galaxy_images/galaxy_model/`, `ENG/` = `GM/downstream_evaluation/engaging/`,
`CB/` = `GM/contrastive_baseline/`, `tex:` = line in
`tess-gen-disentanglement_neurips2026_paper/disentanglement_neurips2026.tex`.*

---

## TL;DR — answers to the six questions

| # | Question | Verdict |
|---|----------|---------|
| 1 | Implemented correctly? | **Yes — no bugs found.** Same data rows, same K=5 spatial-neighbor triplets, same 95/5 split seed, identical eval pipeline. Three *deliberate* deviations exist (anchor routing, compute budget, projection heads) — documented below with their bias direction. |
| 2 | Is "spatial-conv1x1" the closest to Ours? | **Yes.** Spatial neighbors = same triplets; conv1x1 head = `Conv2d(512→16, 1×1)` → 2×2×16 = 4 tokens × 16 dims, exactly Ours' encoder head. The `avg` variant has a different head (mean-pool + `Linear(512→64)`). |
| 3 | Instrument latent size matched? | **Per-image yes (64-D both), but the published comparison is not:** the `Ours_Instr` column uses the 32-D *mean-pooled* e2 (`base-meanpool`) while contrastive instrument is probed at 128-D flat — a 4× probe-input advantage for contrastive. Ours' own flat e2 scores higher (GalDepth 0.738 vs 0.682) though still below contrastive (0.881). |
| 4 | Eval set excluded from SSL training? | **No — for either model** (only the `*-DWNVAL` FM variants exclude it). Measured effect via `base-5NB` vs `base-5NB-DWNVAL` (and 15NB pair): mean ΔR² = −0.005 / +0.005 → **contamination is noise-level**, and the protocol is matched between models. |
| 5 | Neighbor supervision unfair? | No implementation unfairness. The instrument InfoNCE **is** near-direct metric learning on observing conditions (positives share the anchor's ≤3′ patch → same PSF/depth/EBV) — that's inherent to "same triplets + contrastive loss", i.e. exactly what the reviewer asked for. The `random-neighbors` control proves the mechanism (instrument R² collapses 0.88→0.29). |
| 6 | Too good to be true? | **No — and the premise is half wrong.** Only *one* off-diagonal is ~0 (phys→instr, the textbook outcome of cross-survey-positive InfoNCE). The other off-diagonal leaks as much as Ours: contrastive's *instrument* latent predicts redshift at **R²=0.546**. And contrastive pays on-diagonal: it loses to Ours on **every** spectroscopic physics property and degrades more on HSC-resolution shapes than Legacy ones — the first quantitative sign of the paper's "collapse to lowest resolution" claim. |

---

## 1. What was actually compared

### "Ours" (`base` in variants.json)
- Checkpoint: `/orcd/pool/007/pablomer/checkpoints_new/base/snapshot.ckpt` = AMD run
  `neighbors_all_attn/2026-04-05`, **best-val at step 87,000** (epoch 228), global batch
  64×4 GPUs ≈ **22.3 M images seen** (`source.txt` in the ckpt dir).
- Both encoders: ResNet18 (timm, no pretrain, 4-ch) → `Conv2d(512→16, 1×1)` → **2×2 = 4
  tokens × 16 dims** (`GM/double_train_fm_neighbors.py:105,124-127`;
  `GM/configs/neighbors_efficient.json:30` `cross_attention_dim: 16`).
- Training routing: **the anchor is never encoded** (tex:148). encoder_1 sees `samegal`
  (cross-survey pair), encoder_2 sees each of the K=5 sky neighbors; all tokens
  concatenated into UNet cross-attention; flow-matching MSE on the anchor.
  `lambda_geometric = 0.0` → **no explicit invariance loss anywhere**.
- Split: 95/5 `random_split`, seed 42, over 103,741 anchors (`GM/data_factory.py:279-280`).

### Contrastive (`CB/dual_encoder_contrastive.py`, 2×2 matrix on Engaging)
- Runs: `{spatial,random}-{avg,conv1x1}` = neighbors ∈ {real sky-neighbors, uniform-random} ×
  head ∈ {avgpool+Linear(512→64), conv1x1(512→16)→4×16 flat}. All completed 100k steps
  (batch 64 ≈ **6.4 M images**, ~3.5× less than Ours); best-val ckpts at steps 83k–95k.
- **Galaxy branch**: symmetric CLIP InfoNCE, positive = (anchor, samegal), negatives =
  other batch items, temperature 0.1 (`dual_encoder_contrastive.py:184-202,299-303`).
  Note: **the anchor IS encoded** here (unlike Ours).
- **Instrument branch**: multi-positive InfoNCE, anchor's own ≤5 valid sky-neighbors are
  positives, all other anchors' neighbors in the batch are negatives; masks respected;
  zero-positive anchors dropped (`:204-254,352-377`). `use_random_instrument_positives=False`,
  `include_physics_pair_as_instrument_negative=False` in all four runs.
- Projection heads (MLP 64→64→32) used **in the loss only**; eval takes raw encoder
  outputs (`ENG/prepare_combined.py:172-177`) — standard SimCLR practice.
- Data: `/orcd/pool/007/pablomer/efficient_neighs_48` — **row-identical copy** of the
  efficient store (468,197 rows; catalog.parquet copied verbatim, `CB/build_ram48.py:96`),
  same 103,741 anchors, same `idx % 2` anchor-survey parity, same 95/5 `random_split`
  seed 42 → **identical train/val membership as Ours** (same lengths + same generator seed).

### Eval (identical for both)
- `ENG/prepare_combined.py` embeds the HSC and the Legacy image of each of the n=5,469
  overlap galaxies with both encoders; probe input = concat(HSC, Legacy) = **128-D for
  both Ours-e1 and contrastive-e1/e2**; `predict_combined.py --mlp-arch aion --seed 0`,
  one multi-target `Linear(in,256)→GELU→Linear(256,K)` per task family, 90/10 split,
  R² on the 10%.
- Neighbor semantics (verified on the catalog): precomputed nearest-on-sky same-survey
  sources, ascending distance, hard 0.05° (3′) cutoff; neighbor pool structurally
  disjoint from anchors (`source_type` 1/2 vs 0); same-galaxy counterpart can never be a
  neighbor; near-duplicates negligible (<0.1% closer than 1″).

---

## 2. Why the result pattern is real, not an artifact

The numbers (from `ENG/outputs/contrastive_sweep_r2_summary.csv`, provenance verified
cell-by-cell against the source prediction CSVs):

**(a) phys→instr ≈ 0 for contrastive (0.01) vs 0.07–0.28 for Ours.**
The galaxy branch's positive pair is *always cross-survey*, so InfoNCE explicitly
optimizes instrument invariance — contrastive lands **below the random-feature floor**
(untrained ResNet gets 0.02–0.23 on these targets), i.e. active erasure. Ours has no
invariance term (λ_geom=0) and sits *at* the random floor (e.g. GalDepth: Ours 0.185 vs
RandInit 0.202). Both are "disentangled up to what any image feature carries"; contrastive
is sharper because it directly optimizes exactly this metric.

**(b) instr→instr much higher for contrastive (0.67–0.88 vs Ours 0.44–0.68).**
Positives share the anchor's ≤3′ patch → same PSF/depth/extinction; same-survey negatives
come from other patches. The instrument InfoNCE is therefore *near-supervised metric
learning on the downstream instrument targets*. The **random-neighbors control proves
it**: with random same-survey neighbors, instrument R² collapses to 0.06–0.29 while
physics R² is unchanged. (Ours shows the same ablation direction in the paper, tex:936.)

**(c) BUT the other off-diagonal is not clean for anyone.** Contrastive's *instrument*
latent predicts redshift at 0.546, logM* 0.557 — the same leak magnitude as Ours
(0.535/0.491 pooled). Plausible physical mechanism: sky neighbors at ≤3′ are correlated
in redshift (large-scale structure), so neighbor-matching rewards z-correlated features.
The "perfect asymmetry" impression only holds for one of the two off-diagonals.

**(d) Contrastive pays on-diagonal.** Ours(base) beats both spatial contrastive variants
on **every** spectroscopic property: z 0.801 vs 0.760/0.740, logM* 0.755 vs 0.699/0.675,
sSFR 0.558 vs 0.509/0.482, t_age 0.257 vs 0.189/0.206. And on shapes, the contrastive
deficit is *larger for HSC-measured* (high-res) shapes than Legacy-measured ones:

| target | Ours | C-spatial-conv1x1 | gap |
|---|---|---|---|
| legacy_shape_e1 | 0.918 | 0.918 | 0.000 |
| legacy_shape_e2 | 0.934 | 0.895 | 0.039 |
| **hsc_shape_e1_i** | 0.919 | 0.882 | **0.037** |
| **hsc_shape_e2_i** | 0.946 | 0.893 | **0.053** |

This is directionally the paper's central "collapse toward the lowest-resolution
observation" claim (tex:177-178), measured for the first time.

**(e) Artifact checks all negative.** No NaNs; every summary cell traced to its source
CSV; all 4 runs hit 100k steps with best-val ckpts at 83–95k (converged, not lucky
early stops); train loss 0.32 vs ln(64)≈4.16 chance (both branches near-solved);
SSL-contamination measured negligible (§3 below); probes identical code/seed/split.
One probe caveat: "wrong-encoder" probes early-stop at epoch 2–4 (no signal from the
start — consistent with genuine information absence), while instr→phys probes hit the
50-epoch cap — so the *leak* numbers are lower bounds, which strengthens (c).

---

## 3. Fairness ledger (everything that differs, and whom it favors)

| Difference | Detail | Favors |
|---|---|---|
| Probe input dims for instrument | `Ours_Instr` column = 32-D pooled (`base-meanpool`), contrastive = 128-D flat. Ours' own flat e2: GalDepth 0.738 (vs 0.682 pooled; contrastive 0.881), z-leak 0.654 (vs 0.535) | **contrastive** (fixable in eval, no retraining) |
| Anchor routing | Contrastive encodes the anchor in training; Ours never does. Eval embeds the query image itself → matches contrastive's training condition | **contrastive** (inherent to objective) |
| Compute | Ours ≈ 22.3M images (87k×256); contrastive ≈ 6.4M (100k×64) | **Ours** (contrastive wins its cells anyway) |
| Projection heads | Contrastive discards loss-specific head at eval; Ours' encoder feeds the UNet directly, no buffer | **contrastive** (standard practice) |
| Explicit invariance | InfoNCE hard-codes cross-survey invariance; Ours relies on the generative bottleneck only (λ_geom=0) | — (this *is* the ablation) |
| SSL saw eval galaxies | True for both (only `*-DWNVAL` excludes). Measured on FM: mean ΔR² −0.005/+0.005 → negligible | neither |

Knobs not tried (`include_physics_pair_as_instrument_negative`, temperature, hard
negatives) would all push contrastive to look *better*, not worse — there is no tuning
fix for "contrastive is good at R² asymmetry". The response is (i) fix the dim
mismatch, (ii) report the cells contrastive loses, (iii) run the diagnostics where the
generative objective is the point.

---

## 4. Experiment plan

> **Superseded by `PLAN.md`**, which carries the live queue, per-cluster commands, and
> the results of the items already finished (notably §A1 below — the dim-fair
> instrument comparison is DONE, and dim-matching turns out to explain only about a
> quarter of the instrument gap). The list below is kept as the original reasoning.

### A. Zero-training fixes (≤1 day, CPU/1 GPU) — make existing numbers rebuttal-grade
1. **Dim-fair instrument comparison.** Produce mean-pooled e2 for
   `contrastive-spatial-conv1x1` (16-D/image → 32-D combined; reuse
   `ENG/build_meanpool_h5.py`) and re-run the probe; also report Ours flat e2 alongside.
   Final table shows {pooled, flat} × {Ours, contrastive} for instrument cells.
2. **Probe error bars.** Re-run `predict_combined.py` with seeds 0/1/2 (or 5-fold) for
   `base`, `base-meanpool`, `contrastive-spatial-conv1x1`. n_test ≈ 470–550 → single-seed
   R² noise is a few points; the 0.74-vs-0.80 claims need CIs.
3. **Add the high-res-retention rows** (hsc_shape_e1_i/e2_i, already in the CSVs) to the
   summary + rebuttal table.
4. **Provenance script.** Commit a `make_summary.py` here that regenerates
   `contrastive_sweep_r2_summary.csv` from the prediction CSVs (currently no generator
   exists in the repo).

### B. Reviewer's other two axes, embedding-only (2–4 days, 1 GPU)
5. **Cross-survey retrieval** (reviewer: "cross-survey retrieval"). Top-1/top-k retrieval
   of the same galaxy across surveys over the ~103k gallery, physics space, Ours vs
   contrastive-spatial-conv1x1. *Expectation management: InfoNCE trains exactly this task
   — expect contrastive to win/tie top-1. Pair it with (6)/(8) where the generative
   objective differentiates.*
6. **Outlier detection** (reviewer: "outlier detection"). NSF flow on contrastive physics
   latents (HSC-only, 64-D), identical protocol to paper app-flow; compare top-12 outlier
   panels (Ours: artifact-free mergers; AION: instrument pathologies — where does
   contrastive land?).
7. **UMAP grid** for contrastive-spatial-conv1x1 on the same 8,192 pairs as Fig 2.

### C. Decisive experiments (the actual rebuttal payload)
8. **Frozen-embedding decoder** — quantifies "contrastive embeddings are not optimized
   for pixel-level reconstruction … no guarantee that the necessary information is
   preserved" (tex:177-178). Train the identical UNet (same config, same steps)
   conditioned on **frozen** contrastive tokens (galaxy 4×16 as physics tokens,
   instrument branch as instrument tokens) vs frozen Ours tokens. Compare counterfactual
   MSE (paper: 0.081 HSC / 0.197 Legacy), PSD curves, pipeline-transfer R² (0.81
   generated vs 0.82 real). `CB/embedding_fm_decoder/` is a starting point; new code
   lives here in `contrastive_ablation/`. ~1–2 H100-days.
9. **Contrastive-DWNVAL** (hygiene, optional). Add `downstream_holdout_ids_txt` support
   to `CB/train_contrastive_engaging.py` (mirror `GM/data_factory.py:80-116,246-262`,
   ~15 lines) and retrain spatial-conv1x1 100k steps → closes the "InfoNCE can memorize
   eval identities" question with the same test we already ran for FM.
10. **Compute-matched contrastive** (optional). Extend spatial-conv1x1 to ~350k steps to
    match Ours' ~22M images. Best-val already at 83k/100k suggests it's converged, so
    likely flat — but it preempts "the baseline was undertrained" in either direction.

### Rebuttal framing (suggested)
- **Concede + reframe:** contrastive with our architecture and triplets *does* achieve R²
  asymmetry — consistent with our claim that disentanglement comes from the
  architecture-driven information routing, which the ablation shares. InfoNCE enforces
  invariance by construction and even erases below the random-feature floor.
- **The cost:** it loses information — lower R² on every spectroscopic property, larger
  degradation on HSC-resolution shapes (collapse to lowest resolution, now measured),
  equal physics leak in the instrument latent, and no reconstruction/counterfactual
  capability (quantified by C8).
- **Present the asymmetry cells against the random-ResNet floor**, not against 0 — the
  paper already argues this (tex:416, 433-435).

---

## 5. Key file inventory

- Loss/model: `CB/dual_encoder_contrastive.py` · trainer: `CB/train_contrastive_engaging.py`
  · matrix launcher: `CB/submit_all_engaging.sh` · eval: `CB/eval_contrastive_engaging.slurm`
- Data: `/orcd/pool/007/pablomer/efficient_neighs_48` (RAM48 store, 468,197 rows;
  built by `CB/build_ram48.py` from `/orcd/pool/007/pablomer/efficient_neighs`)
- Checkpoints: `/orcd/pool/007/pablomer/checkpoints_new/contrastive-{spatial,random}-{avg,conv1x1}/`
  (all DONE @100k) · Ours: `/orcd/pool/007/pablomer/checkpoints_new/base/snapshot.ckpt` (step 87k)
- Results: `ENG/outputs/contrastive_sweep_r2_summary.csv`,
  `ENG/outputs/predictions/predict_{base,base-meanpool,contrastive-*}__aion.csv`,
  plots in `ENG/outputs/plots_publication/unified_three_probes_contrastive-*_vs_ours.*`
- Contamination check: `predict_base-{5,15}NB.csv` vs `predict_base-{5,15}NB-DWNVAL.csv`
  (mean ΔR² −0.005/+0.005)
