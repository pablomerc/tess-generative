# HSC × LSST dual-encoder: decisions log

Running record of the choices behind the first cross-survey disentanglement
run that uses LSST DP2 as a view. Each entry gives the decision, the numbers
it rests on, and where in the code it is enforced. Append, don't rewrite;
if a decision is reversed, add a dated entry that says so.

Background primer: `data_exploration/DATASET_NOTES.md`. Strategy it slots
into: the four-step plan at the end of that file (COSMOS core experiment →
DEEP2-3 as domain-shift test → joint training with depth conditioning →
XMM-LSS sealed).

---

## 2026-09-26 — scope of the first run

### D1. Field: COSMOS only
DEEP2-3 is held out entirely as the domain-shift test set; it is not
trained on. COSMOS is the only field whose LSST depth is in the same regime
as HSC (i-band limiting magnitude ≳ 25.6 vs HSC Deep ~26.9; DEEP2-3 ≈ 24.5).
Object counts in the LSST store: COSMOS 34,132, DEEP2-3 31,031.
*Enforced in:* the dataset's anchor selection (step 3), not in any index or
store, so the same artefacts serve later joint-field runs.

### D2. Bands: g, r, i, z
The dual encoder is `in_channels=4`. The LSST store's channel order is
g,r,i,z,u,y, so `[:4]` is griz, matching `hsc_image.bin[:4]` and Legacy.
No architecture change. u is absent in DEEP2-3 anyway; y is deferred.
*Enforced in:* `preprocessing.preprocess_lsst(bands=("g","r","i","z"))`.

### D3. Quality cut: `n_input ≥ 5` in every one of g, r, i, z
Applied to anchors AND to neighbour slots (a near-empty stamp fed to the
instrument encoder is worse than no stamp). Counts on 2026-09-26:

| requirement | COSMOS objects kept |
|---|---|
| none | 34,132 |
| griz ≥ 5 | **33,346** (97.7%) |
| grizy ≥ 5 | 31,543 (y is what fails: 95% of drops) |

Neighbour survival under the cut: 99.9% of passing COSMOS anchors keep ≥ 5
passing neighbours among their 20; 99.0% keep ≥ 15.
Why 5: the AE baseline's anomaly rankings were dominated by `n_input ≤ 1`
pedestal-broken stamps; the LSST matrix froze its recipe *without* a cut for
comparability, so this run deliberately departs from that recipe.
A uniform 5 is impossible in DEEP2-3 (g median 2 exposures; 99% fail); if
DEEP2-3 is ever trained on, use per-band thresholds, e.g. r,i,z ≥ 3 and g ≥ 1.
*Enforced in:* the dataset at load time from `lsst_meta.parquet`
(`n_input_<band>`), as a boolean mask over rows. NOT baked into the
neighbour index or the RAM store, so the threshold can change without a rebuild.

### D4. Same-instrument neighbours for LSST come from a k-NN inside the LSST store
The parent catalogue's `neighbor_idx_hsc/_legacy` cannot be reused: all
65,163 LSST rows are parent *anchor* rows and the 364,456 neighbour-pool
rows were never crossmatched to DP2, so zero of an anchor's HSC neighbours
have an LSST stamp (measured). Instead: 20 nearest other store rows in the
same field, exact great-circle distance, self excluded, geometry only.

| field | 1st neighbour | 5th | 20th |
|---|---|---|---|
| COSMOS | 19″ | 58″ | 139″ |
| DEEP2-3 | 19″ | 58″ | 135″ |

At these separations a neighbour shares the anchor's coadd depth and PSF
(DP2 depth varies on the dither scale, degrees).
*Built by:* `data_preprocessing/build_lsst_neighbors.py` →
`data/lsst_efficient/lsst_neighbors.{npz,json}` (store-row space, arcsec,
plus `parent_row` = inverse of `lsst_row_map.npy`). Existing store files are
never modified.

### D5. Data path: RAM-resident 48 px store, mmap as fallback
Measured on the login node, anchor + 5 neighbours, griz 160 px:
LSST mmap 13.6 ms median / 25 ms p90 cold, 1.9 ms warm; parent HSC store
5.0 ms / 6.2 ms. Both are far below the ~1 s training step, so the old HDF5
problem (4.5 s per batch) cannot recur. Still, following
`contrastive_baseline/neighbors_ram48.py`, the run will preprocess once to
48 px griz float16 and hold everything in RAM so `__getitem__` is pure
indexing. Size for COSMOS: the 34,132 anchors reference 165,472 distinct
parent rows for HSC/Legacy neighbours (the neighbour pool is mostly
non-anchor rows), so

    HSC + Legacy 48 px griz float16: 6.1 GB  +  LSST: 0.6 GB  ≈ 6.7 GB

(5.8 GB with 5 neighbours; 13.5 GB in float32). Fits any node.
**The HSC/Legacy half already exists:** `data/efficient_neighs_48/`
(`hsc48.bin`, `legacy48.bin`, float16 (468197, 4, 48, 48), 8.6 GB each,
built 2026-07-25 by `contrastive_baseline/build_ram48.py`, catalogue
row-aligned with `neighbors_efficient`, neighbour indices preserved). Only
the LSST counterpart needs building: 65,163 rows × 4 × 48 × 48 float16 =
1.2 GB (COSMOS alone 0.6 GB), a few minutes of sequential reading.
Before relying on the July store, spot-check a handful of rows against a
fresh `NeighborsEfficientDataset._preprocess` of the 160 px store, in case
the HSC/Legacy preprocessing changed after 2026-07-25.
Login-node caveat: `ulimit -v` is 32 GB there, so the 120 GB parent binary
cannot be mmapped on the login node; use seek+read for smoke tests, as
`single_latent/ae_baseline/data.py` does. The 48 px stores are small enough to mmap anywhere.
*Built by:* step 2 script (to be written): `data_preprocessing/build_lsst48.py`,
mirroring `build_ram48.py` but calling `preprocess_lsst`.

### D6. Storage budget
Group quota 1.9 TB, 1.7 TB used (all Pablo), 144 GB free at 2026-09-26.
Everything in this plan is < 10 GB. The 582 GB `data/neighbours_v2.h5`
(float32 160 px HSC + Legacy + Legacy ivar; superseded by
`neighbors_efficient`, and the `.vds` files point at `neighbor_batches/`,
not at it) is the obvious reclaim if space is ever needed — but
`config.py`/`data_factory.py` and the scale-ablation configs still name it
as the HDF5-mode default, so confirm nothing in HDF5 mode is still wanted.

### Open — D7. Train/val split convention
Two conventions exist: the LSST AE baseline froze a 90/10 split by
`sha1(object_id_hsc)` (`single_latent/ae_baseline/data.py`, `val_object_ids.txt`), while
the neighbours work excludes the 5,466 downstream-holdout anchors
(`*-DWNVAL` variants). Comparability with one baseline or the other depends
on which is chosen. Decide before step 3.

### Known risk — resampling confound
Pixel parity zooms LSST by 0.84 to HSC's 8.0″ field; the bilinear resample
introduces correlated pixel noise that HSC lacks, so a model could identify
the survey from noise texture alone. Planned control: pass HSC stamps
through the same resample operation and check whether the instrument
probe's survey accuracy changes.

---

## Step order for the first run (status 2026-09-27)

1. DONE `data_preprocessing/build_lsst_neighbors.py` — neighbour index →
   `data/lsst_efficient/lsst_neighbors.{npz,json}` (commit eedaca2).
2. DONE `data_preprocessing/build_lsst48.py` — `data/lsst_efficient/lsst48.bin`,
   all 65,163 rows, griz 48 px float16, bit-for-bit vs `preprocess_lsst`
   (commit eedaca2). HSC side already existed: `data/efficient_neighs_48/`.
3. DONE `lsst/hsc_lsst_dataset.py` — `HscLsstRAM48Dataset`, same output
   contract as `NeighborsEfficientDataset`; anchor alternates HSC/LSST by
   position; D1–D3 applied as masks; HSC subset read with seek+read.
4. DONE wiring — `config.py` (`data.lsst_*` fields, `model.survey_names`) and
   `data_factory.py` (`mode == "hsc_lsst_ram48"`); holdout/lens options are
   refused for this mode until re-wired to this index space (D7).
5. DONE survey generalisation — `double_train_fm_neighbors.py` and
   `validation_pairs.py` take a `survey_names` pair (default
   `("hsc","legacy")`, byte-identical behaviour for existing runs); this run
   logs `train/loss_generative_lsst`, `val/mse_lsst`, UMAP legend "LSST".
   The DDPM ablation subclass keeps its historical keys via a fallback.
6. DONE `configs/hsc_lsst_cosmos.json`, `lsst/hsc_lsst_train.py`,
   `lsst/hsc_lsst_train.sbatch` (own launcher, mi2101x, 12 h). Full run:
   `sbatch galaxy_images/galaxy_model/lsst/hsc_lsst_train.sbatch`;
   20k steps ≈ 40 epochs ≈ 6–8 h on one MI210. `lr_schedule` left at
   "paper" for comparability with the registry checkpoints.
7. Smoke test on `-p devel` (job 440996, 2026-09-27) → then launch.
8. Probes: MLP R² on frozen embeddings, instrument targets now include
   per-object `n_input_*` and `psf_fwhm_*`; leak check = physics-latent R²
   on `n_input_i` ≈ 0; UMAP of physics latent coloured by survey and n_input.
   NOTE: with the D7 choice (no holdout exclusion) probe objects are inside
   the training set; any R² from this run is on seen galaxies.

## 2026-09-29 — training outcome and anomaly stage

Both runs completed (4 h 24 m each, one MI210): `paper` final val/loss 0.148,
best at step 12k; `cosine` 0.142, best at step 10k (callback 0.1389).

### D8. Anomaly search = the paper recipe on the dual-encoder latents
`lsst/hsc_lsst_anomaly.py` (+ `.sbatch`): encode every row through both
encoders; per representation fit NSF (zuko, 6 transforms, [64,64], 50 epochs),
IsolationForest, kNN(5) and Mahalanobis on a fit set, score all rows (higher =
more anomalous); galleries as HSC | LSST pairs; physics-vs-instrument top-1 %
overlap; Spearman of scores vs `n_input_i` / `psf_fwhm_i`; UMAPs with the
top-10 marked (per space, and joint HSC+LSST per encoder); Discord post on
success (summary + figures) and on failure (log tail). Outputs under
`/work1/.../outputs/hsc_lsst/anomaly/<tag>/`, never overwritten.
Three scopes, chosen 2026-09-29:
- `anchors` — the 33,346 training anchors, fit on the model's train split
  (jobs 444251 cosine, 444252 paper).
- `field --field F --views lsst` (Pablo's request: "each field separately,
  flow on the LSST-image latents, not HSC") — COSMOS fit on the model's train
  split and scored over all 34,132 COSMOS rows incl. the 786 low-exposure
  ones; DEEP2-3 (never trained on) fit on a seeded 80 % slice of its 31,031
  rows, 20 % held out. Galleries still show the HSC counterpart for context;
  no HSC latent is used anywhere. Four jobs: {cosine, paper} × {COSMOS, DEEP2-3}.
- `lsst_all` — every store row, fit on COSMOS train only: DEEP2-3 and
  low-exposure COSMOS as out-of-distribution probes (the disentanglement
  prediction: DEEP2-3 shifts in instrument space, not physics space). Built,
  not yet run.
Caveat carried from D7: the fit set contains the downstream-probe galaxies.

Results of the six unconditional runs (anchors ×2, field ×4), all COMPLETED
2026-09-29: instrument-space NSF score tracks `n_input_i` (Spearman −0.33 to
−0.38) three to four times more strongly than the physics-space score (−0.07
to −0.15); within COSMOS the 786 low-exposure stamps shift by +12 to +16 in
physics space but +91 to +103 in instrument space; DEEP2-3 held-out shifts
< 1 in both spaces. Top-1 % physics∩instrument overlap 49–151 of ~330
(random 3): the shared tail is depth-driven plus large bright galaxies, and
on DEEP2-3 the physics-space list is led by pedestal / no-data artefacts.

### D9. Conditional flow: anomalous *for its number of visits* (2026-09-29)
Pablo's request after the unconditional results. `--cond n_input` fits a
second spline flow p(z | c) with c = standardised log1p(n_input) in g,r,i,z
(4-d context; standardisation on the fit set) and makes it the primary score
`<key>/cnsf`; the unconditional `<key>/nsf` is still fitted so
`figures/cond_vs_uncond.png` and the Discord headline show what conditioning
removed (Spearman vs n_input before/after, top-1 % overlap of the two
rankings). Expected effect: the shallow-stamp tail leaves both rankings, the
instrument-space Spearman vs n_input drops toward 0, and what remains in
instrument space is texture/PSF/artefact-driven rather than depth-driven.
Not conditioned on PSF (kept as a diagnostic axis) — a `--cond n_input_psf`
variant is the obvious follow-up if PSF still dominates.

**First conditional result (COSMOS field, cosine, 444332):** almost no effect —
instrument-space Spearman vs `n_input_i` −0.38 → −0.36, rank correlation of
conditional vs unconditional scores 0.98, low-exposure shift +91 → +84, flow
NLL better by ~1 nat over 64 dims. Cause = coverage of the conditioning
variable: 22,789 of 34,132 COSMOS objects have 80–100 i-band exposures and
only 1,354 have < 20, so the context network sees almost no depth variation
and the 786 objects below 5 exposures are extrapolated. DEEP2-3 has 31,031
objects at 1–20 exposures and nothing above.

### D10. Conditional flow v2: pooled fields, balanced context, longer fit
Three opt-in flags (defaults keep the D9 behaviour so the running v1 jobs
stay comparable): `--fit-on both_fields` (lsst_all scope; fit set = COSMOS
train split ∪ seeded 80 % of DEEP2-3, groups still reported separately, so
the context spans 1–114 exposures with only 20–40 thin), `--balance`
(inverse-density weights on log `n_input_i` over quantile bins, clipped to
[0.2, 20], mean 1, applied to the conditional flow's NLL), `--cond-epochs`
(150 for the conditional flow vs 50 unconditional). `--cond n_input_psf` adds
per-band `psf_fwhm` to the context (missing values → fit-set median). Jobs:
smoke 444376 → 444377 cosine, 444378 paper (`*-lsstall-bothfields-cond-bal`),
444379 cosine with PSF (`*-condpsf-bal`). Success criterion: instrument-space
Spearman vs `n_input_i` near 0 within each group, DEEP2-3 and low-exposure
COSMOS no longer shifted en bloc, and what remains at the top of the
instrument ranking is texture / PSF / artefact rather than depth.

### D11. Why v1/v2 conditioning could not work as a *score*, and the calibrated score (2026-09-30)
All four single-field v1 runs and the anchors run came back with the same
null result: instrument-space Spearman vs `n_input_i` −0.38 → −0.36 (COSMOS),
−0.23 → −0.23 (DEEP2-3), −0.33 → −0.32 (anchors); rank correlation between
the conditional and unconditional score 0.97–0.98; the COSMOS low-exposure
group still +84…+102 above the fit set in instrument space. The reason is
not (only) coverage of the conditioning variable. Inside the fit set itself
the median conditional NLL runs from −8 at 5–10 visits to −65 at 80–100
visits in the 64-d instrument space (COSMOS, cosine run), and −12 → −21
across 1–20 visits in DEEP2-3. Shallow stamps give noisier tokens that fill a
much larger volume of latent space, so a *correct* p(z | c) assigns them a
lower density: −log p(z | c) measures the conditional **entropy** at that
depth, not how unusual the object is for its depth. No flow, however well
fitted or balanced, removes that trend from the raw conditional NLL.

Fix: score with the highest-density-region tail probability under the same
conditional flow, p = P[−log p(z′|c) ≥ −log p(z|c)] for z′ ~ p(·|c), estimated
with K samples per row drawn from the flow at that row's context. Under the
model p is Uniform(0, 1) for typical objects at **every** depth, so
−log10 p is comparable across depths by construction ("anomalous for its
number of visits" made literal). Implementation: `--calibrate K` on top of
`--cond`; primary score becomes `<key>/ccal` = −log10 p + 10⁻³·z, where z is
the standardised conditional NLL against the sampled NLLs (breaks ties at the
1/(K+1) floor); p and z are stored in `scores.h5` under `calib/`. K = 512 →
floor p = 1/513, top ≈ 0.2 %. New figure `depth_calibration.png`: median
score per `n_input_i` bin over fit rows for nsf / cnsf / ccal (ccal should be
flat at −log10 0.5) and the p-value histogram per group (uniform for fit
rows). The raw conditional NLL (`cnsf`) is kept and its Spearman reported.
Jobs: smoke 444452 → 444453 cosine / 444454 paper pooled
(`*-lsstall-bothfields-ccal`), 444455/444456 COSMOS field, 444457/444458
DEEP2-3 field (`*-{cosmos,deep23}-field-ccal`), all `--views lsst --cond
n_input --balance --cond-epochs 150 --calibrate 512`. Success criterion
becomes: fit-row p-values uniform in every depth bin; instrument-space
Spearman(ccal, `n_input_i`) ≈ 0 within the fit set; residual shifts of
cosmos_lowexp / deep23_held then mean "genuinely unusual for their depth",
not "shallow".

### D12. Flow memorisation → out-of-fold scoring (2026-09-30)
The calibrated runs (444474–444479, `*-ccal`) do what D11 asked: inside the
conditioning coverage the instrument-space Spearman vs `n_input_i` goes
−0.38 → −0.05 (ccal) / 0.00 (cemp) in COSMOS and −0.23 → −0.06 / −0.02 in
DEEP2-3, the in-sample median score is flat across visit bins, and the
model-based and empirical calibrated scores agree at rank correlation 0.99
within coverage. Two things they also exposed:
1. **Out-of-coverage groups pin to the floor.** In the COSMOS-only fit the
   786 `cosmos_lowexp` objects (`n_input` < 5, never in the fit set) all get
   p = 1/1025 in both spaces and are 100 % of the top 1 % — the flow is
   extrapolating in the context, not judging. The pooled fit
   (`--fit-on both_fields`, DEEP2-3 supplies 2,824 rows at 1–5 visits) is the
   only run where `cosmos_lowexp` has a real reference; COSMOS-only rankings
   are read within `cosmos_train`/`cosmos_val`.
2. **The flow memorises its fit rows.** A same-distribution held-out slice
   should be exchangeable with the fit rows, but held-out rows crowd the
   extreme tail: DEEP2-3 held (20 % of rows) is 39–42 % of the top 0.1 %
   under the 50-epoch unconditional flow and 81–84 % under the 150-epoch
   conditional flow; COSMOS val (5 %) is 42 % of the instrument-space top
   0.1 % under ccal. The galleries' top 16 were 14/16 held-out rows. This is
   in-sample vs out-of-sample NLL bias, not astrophysics.
Fix = `--folds K`: fit rows split into K seeded folds; one flow per fold
(unconditional and conditional) is fitted on the other folds and scores the
held-out fold; every non-fit row is assigned to one fold and scored by that
fold's flow. Every row is scored by a flow that never saw it, so fit /
held-out rows are exchangeable and the empirical percentile (`cemp`) compares
out-of-sample NLLs with out-of-sample NLLs. Reported NLL = mean over folds.
IForest / kNN / Mahalanobis stay in-sample (paper recipe). `--headline-only`
restricts the flows to `phys/lsst_mean` and `ins/lsst_flat` (the two
representations every gallery / UMAP / diagnostic uses) to keep 5 folds ×
(50 + 150) epochs inside the wall limit. New summary field
`top1pct_group_share` (+ `group_share`) and a Discord line with the top-1 %
composition make the memorisation check explicit: a held-out slice should sit
at its population share. Jobs: smoke 444569 → 444570 cosine / 444571 paper
pooled (`*-lsstall-bothfields-oof`), 444572/444573 COSMOS, 444574/444575
DEEP2-3 (`*-{cosmos,deep23}-field-oof`), all `--folds 5 --headline-only
--calibrate 1024 --cond-epochs 150 --balance --views lsst`.

#### D12 results (2026-09-30, six `*-oof` runs, 5 folds, `--headline-only --calibrate 1024`)
Spearman(score, `n_input_i`) inside the conditioning coverage, instrument
space `ins/lsst_flat` (cosine | paper checkpoint):

| run | nsf (uncond.) | cnsf (raw cond. NLL) | ccal (model tail-prob) | cemp (emp. percentile) | cez (emp. robust z) |
|---|---|---|---|---|---|
| COSMOS field | −0.33 \| −0.32 | −0.29 \| −0.28 | −0.23 \| −0.23 | +0.01 \| +0.01 | +0.01 \| +0.02 |
| DEEP2-3 field | −0.23 \| −0.25 | −0.27 \| −0.28 | −0.17 \| −0.21 | −0.01 \| −0.01 | −0.01 \| −0.01 |
| pooled fields | −0.78 \| −0.79 | −0.77 \| −0.78 | −0.18 \| −0.35 | +0.01 \| +0.01 | +0.02 \| +0.05 |

Physics space `phys/lsst_mean`: nsf −0.13/−0.07/−0.30 (COSMOS/DEEP2-3/pooled)
→ cez 0.00 everywhere. Memorisation is gone: the same-distribution held-out
slice is 3.6–4.5 % of the cez top 1 % in COSMOS (population 5 %), 19–22 % in
DEEP2-3 (20 %), 6–13 % pooled (9.5 %). Out of sample the **model-based**
tail probability is over-dispersed (16–39 % of fit rows fall in the model's
own 10 % tail) and keeps a residual depth trend, so the *empirical* scores
are the ones to use. `cemp` saturates at 1/(n_ref+1) per cluster, which
favoured the populous 80–100-visit COSMOS clusters at the very top (pooled
cemp top-16: 0/16 DEEP2-3); **`cez` (robust z within the exposure-pattern
cluster) is the final ranking**: pooled top-16 is 56–62 % DEEP2-3 in
instrument space (population 48 %). Group medians under cemp/cez are equal
across cosmos_train / cosmos_val / deep23_fit / deep23_held (0.29–0.30, i.e.
p ≈ 0.5), so the two fields are exchangeable under the calibrated score.
`cosmos_lowexp` (786 rows, <5 visits in some griz band) has an exposure
pattern (median g4 r7 i8 z3) that no fit object shares (DEEP2-3's shallow
rows are g1 r4 i4 z8): 345 of them sit in context clusters with zero fit
rows, so they pin to the model floor and get "cannot judge" (0) under
cemp/cez — reported, not ranked.

What the calibrated ranking surfaces (cez / cemp galleries, `*-oof-cez`,
`*-oof-cemp`): large galaxies filling the 48-px stamp, bright close pairs
and star–galaxy blends, edge-on disks, compact very red sources, faint
objects on a bright neighbour's halo, and — the case the conditioning was
built for — stamps at ~95 visits whose LSST image is far noisier than the
depth warrants (COSMOS #1 in both spaces). The 1–2-visit noise stamps that
topped the unconditional and raw-conditional rankings are gone.
Physics∩instrument top-1 % overlap under ccal 118–317 (random 3–7): the
"oversized object" signal is shared by both encoders. Re-reports from saved
latents: `--report-only <tag> --primary cez|cemp` (2 min, no GPU).
Commits: 05c5f28 (D11), f7e34c4 (folds), 1997b9c (report-only), 2f46893
(cez). Consolidated table: scratchpad `consolidate.py oof`.
