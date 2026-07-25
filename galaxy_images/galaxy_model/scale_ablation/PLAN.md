# Data-Scale Ablation — Implementation Plan

**Status:** design LOCKED by Pablo 2026-07-24; plumbing implemented and CPU-verified; GPU smoke
submitted (job 18790012). Training runs NOT yet submitted — node4900 is still held by the 150k
diffusion jobs.

**Locked:** 75k steps · downstream holdout applied · half-decade ladder **1k / 3,162 / 10,000 /
31,622** · **4 runs, no full-data retrain** (so the free `fm-control-75k` anchor is plotted as an
*unmatched* reference — see §9).

**Purpose:** answer reviewer **JcH2**, who asks the same question three times:

- **Q4:** *"Could you provide any discussion on (or list as a limitation) the amount of data needed to
  train the architecture? For example, how big of a dataset on each instrument before we could expect
  to fit a set of reasonably performant physics / instrument encoders?"*
- **Weakness (minor, line 32):** *"how big of a source instrument catalog is needed for robust
  performance, **and how much overlap with the target instrument**, as is seen with Legacy / HSC?"*
- **Limitations:** *"the amount of source data required needs to be ablated for application to smaller
  survey / datasets ('when does this technique become applicable for a new survey?')"*

JcH2 rates the paper **4 (borderline accept)** and explicitly says answering these *"wouldn't tend to
boost my score"* — the score-moving item is their **Q1** (contrastive ablation). So this ablation is a
limitation-hardening exercise, not a score chase. Scope it accordingly: one curve, one table, one
crisp paragraph. Don't gold-plate it.

---

## 1. What we are ablating — LOCKED

**The number of cross-matched (overlapping) galaxies, holding single-survey coverage at full density.**

Not "total data". The reviewer's line-32 phrasing separates catalog size from *overlap* size, and the
overlap is the genuinely scarce resource: HSC covers a few hundred deg², Legacy ~14,000, and each
survey independently contains far more galaxies than their intersection. A new survey pair is limited
by crossmatches, not by images.

This framing is not a convenience — **it is how the dataset is physically built.** Verified against
`efficient_neighs_48/catalog.parquet`:

| `source_type` | rows | role |
|---|---|---|
| 0 | 103,741 | **anchors** — cross-matched, observed in *both* surveys. The trainable set. |
| 1 | 262,965 | HSC-only neighbour pool |
| 2 | 101,491 | Legacy-only neighbour pool |
| | **468,197** | total image records |

The 364,456 pool rows exist *only* so anchors can reference them as same-instrument neighbours. They
are never trainable anchors. So the store already encodes "abundant single-survey data + limited
crossmatches", and our ablation varies exactly the scarce axis.

### Why the neighbour branch is not a confound

Per training item the loader reads **7 images from 6 distinct rows**
(`neighbors_ram48.py:74-92`, wiring at `double_train_fm_neighbors.py:341-393`):

| Image | Source | Consumes a crossmatch? |
|---|---|---|
| anchor image, survey A | anchor row | **yes** — UNet target `x_1` |
| same galaxy, survey B | *same* anchor row | **yes** — → `encoder_1` (physics) |
| 5 sky-neighbours, survey A | 5 **pool** rows | **no** — → `encoder_2` (instrument) |

The physics pathway and the reconstruction target are gated by N. The instrument encoder feeds on
single-survey images that a real survey would have in abundance. Drawing neighbours from the full pool
is therefore the *correct* implementation of the assumption, not a leak.

**Verified corollaries:**
- The sky-kNN graph is **strictly bipartite anchors→pool**: 100% of `neighbor_idx_hsc` targets are
  `source_type==1`, 100% of `neighbor_idx_legacy` targets are `source_type==2`. **Zero** anchor→anchor
  edges, zero reciprocal edges, zero self-references. Pool rows have all-`-1` neighbour lists.
- Therefore subsetting (or holding out) anchors can **never** leak an excluded galaxy back in through
  the neighbour branch. A contiguous-sky-footprint design was considered and is **unnecessary** —
  neighbours are never anchors, so there is nothing to keep "inside" the subset. Random subsetting is
  correct. (BFS over the kNN graph is also impossible, for the same reason.)
- `neighbor_dist_*` is in **degrees**, not arcmin (the store README is wrong), capped at a 0.05° =
  3′ search radius; the first 5 slots have median separation ~23″ (HSC) / ~67″ (Legacy). Neighbours are
  genuinely same-exposure-conditions.

### Reporting obligation

Because "N crossmatches" is not "N galaxies", the table **must** carry both columns. Distinct-image
counts, measured with the real k=5 + parity selection rule:

| N anchors | distinct neighbour rows | total distinct images |
|---|---|---|
| 1,000 | ~4,900 | ~6,900 |
| 5,000 | 23,328 | 33,328 |
| 10,000 | ~45,000 | ~65,000 |
| 30,000 | ~110,000 | ~170,000 |
| 98,554 (full train) | 235,545 | 443,027 |

State the assumption in the caption. A reader who sees "1k" and assumes 1k galaxies total will raise
exactly the objection this framing answers.

### Frame the asymmetry as a finding, not a caveat

Expect the **instrument** curve to stay much flatter than the **physics** curve, because it doesn't
consume crossmatches. That is an architectural property worth stating outright: *the instrument
encoder requires no cross-survey overlap; only the physics pathway is overlap-limited.* It also sets
up the natural follow-on for the limitations section, at the cost of one sentence and zero compute:
an instrument branch bootstrapped purely from single-survey images, spending crossmatches only on the
physics side.

---

## 2. The free full-data anchor point — VERIFIED

A flow-matching control trained on Engaging at **exactly 75,000 steps on the full data** already
exists and is finished:

- ckpt: `/orcd/pool/007/pablomer/checkpoints_new/diffusion_ablation/frozen/fm-control-75k-fixed.ckpt`
  (2,202,779,296 B, byte-identical to the run-dir `latest-step=step=75000.ckpt`)
- variant `neighbors_all_attn` (unmodified `ConditionalFlowMatchingModule`), tag
  `fm-control-ram48-h200-75k`, job 18669497, wall 07:26:43 on 4×H200
- full training set: 103,741 anchors, 0 excluded, seed-42 split → **train 98,554 / val 5,187**
- downstream R² **already computed** (`fm-control` in `downstream_evaluation/engaging/variants.json`)
- plain recon MSE for the frozen-75k ckpt is being produced right now by job 18784184

**Consequence: use 75k steps, not 100k.** It is the smaller budget *and* it hands us the top of the
curve for free. See §9 Q1 — the in-flight 150k continuation runs put a question mark on this.

Matched recipe to replicate verbatim: `num_steps=75000`, `devices=4`, `batch_size=64` (effective
**256**), `scale_steps_by_devices=false`, `max_neighbors=5`, `seed=42`, `lr=1e-4`, `bf16-mixed`.

---

## 3. THE CRITICAL CONFOUND: a latent LR-schedule bug that is dataset-size dependent

`double_train_fm_neighbors.py:962-965` builds:

```python
scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
```

`train.py` passes only `max_steps`, never `max_epochs`, so PyTorch Lightning sets
`trainer.max_epochs == -1`. With `T_max=-1` the LR **does not anneal** — it alternates between `1e-4`
and exactly `0.0` at every epoch boundary. Verified empirically in this exact env (pl 2.6.1,
torch 2.6.0+cu124): per-step LR `[1e-4, 1e-4, 0.0, 0.0, 1e-4, 1e-4, 0.0, 0.0, ...]`.

Two implications:

1. **~50% of all optimizer steps run at lr=0.** The paper's own base model was trained this way, so it
   is internally consistent — but nominal 75k steps is ~37.5k effective updates.
2. **The square-wave period is the epoch length, which scales with training-set size.** Full data =
   385 steps/epoch (`ceil(98554/4)//64`), so 385 on / 385 off. At N=1,000 an epoch is ~3 steps, so
   3 on / 3 off. **Naively subsetting the data changes the optimization dynamics, not just the data.**

This is fatal to a naive data-scale ablation: the small rungs would underperform partly because of a
pathological LR waveform, and we would report that as "you need more crossmatches."

### Resolution — index tiling (LOCKED)

Do **not** fix the scheduler (fixing it invalidates the free full-data anchor). Instead hold the epoch
length constant by **tiling the subset's index list to the full train length**:

```python
chosen        = <N stratified positions drawn from train_ds.indices>   # N distinct anchors
tiled         = np.resize(np.asarray(chosen), 98554)                   # cyclic repeat to full length
train_dataset = Subset(kept_dataset, tiled)                           # len == 98,554 for every arm
```

Every arm then has exactly **385 steps/epoch**, an identical LR waveform, identical effective batch,
and identical total optimizer steps. The *only* thing that varies is the number of distinct
cross-matched galaxies. The free full-data control remains directly comparable.

Why tiling rather than `RandomSampler(replacement=True)`: `train.py` builds the `Trainer` without
`use_distributed_sampler=False`, so Lightning injects its own `DistributedSampler` and would discard a
custom sampler. Tiling needs no sampler plumbing, gives each anchor a *deterministic* multiplicity
(balanced, not Poisson), and shards/shuffles normally under DDP.

**Parity is preserved.** `Subset.__getitem__(i)` forwards `self.indices[i]` to the base dataset, whose
`neighbors_ram48.py:71` sets `anchor_is_hsc = (idx % 2 == 0)`. Each anchor keeps its survey role at
every scale. (Do **not** instead filter the dataset's internal `anchor_indices` array — that renumbers
positions and flips roughly half the anchors' survey role.)

**Stratify the draw by parity** (N/2 even + N/2 odd positions). Per-survey recon MSE differs 3.4×
(HSC 0.050 vs Legacy 0.171), so an unstratified small draw could shift the survey mix and confound the
curve.

**Nest the rungs** (1k ⊂ 5k ⊂ 10k ⊂ 30k), one seed, so consecutive points share galaxies and the curve
isn't dominated by draw-to-draw variance.

---

## 4. Design — LOCKED decisions

| Decision | Value | Why |
|---|---|---|
| Scale axis | N cross-matched anchors, drawn from the **train** split only | keeps the common val set intact |
| Ladder | **1,000 / 3,162 / 10,000 / 31,622** (half-decade, 10^3 → 10^4.5) | 4 runs; even N so parity splits exactly |
| Draw | nested, parity-stratified, seed 42 | variance reduction + survey balance |
| Epoch length | tiled to 98,554 in every arm | neutralises the LR-waveform confound (§3) |
| Objective | **flow matching only** | the DDPM arm has an ε-prediction pathology (recon MSE 5–8× worse, *worsens* with more sampler steps); it would swamp the data-scale signal |
| Steps | 75,000, `devices=4`, `batch_size=64`, seed 42, `max_neighbors=5` | matched to the free anchor |
| Checkpoint for eval | **fixed 75k step**, never best-val | "best" selection is itself data-scale dependent |
| Common val set | frozen seed-42 val positions, identical in every arm | see §5 |
| `checkpoint_every_n_train_steps` | **5000** (was 1000) | cuts ~220 GB → ~44 GB of write traffic per run on a shared Ceph FS; resume loss bounded to ~30 min |
| Partition | `pg_mki_aryeh`, 4×H200 per run | no QOS limits; 7.7 h/run measured |

### The common validation set — a real trap

`random_split`'s permutation length is `kept_size`, so **any change to `kept_size` completely changes
the val set.** Measured with seed 42: `len=103741` → val head `[28083, 35471, 95522, ...]`;
`len=51870` → `[42417, 51405, 3403, ...]`. Zero relation.

Prescription: freeze the canonical split **once at full length**, then subset only the train side.
`data_factory.py:279-280` already does `random_split(kept_dataset, [train_size, val_size], seed=42)`;
we intercept *after* it. Val loader is untouched in every arm. This also keeps the 256-galaxy recon
manifest (`diffusion_ablation/results/recon_eval_manifest.json`) valid, so plain-recon MSE stays
comparable to the existing FM/DDPM table.

⚠️ **This breaks if we apply the downstream holdout** (`kept_size` drops 103,741 → ~98,275 → new val
split → `eval_recon_mse.py` hard-errors on `Manifest mismatch`). See §9 Q2.

---

## 5. Code changes — small and additive

### New files, all under `scale_ablation/`

```
scale_ablation/
├── PLAN.md                     # this doc
├── PROGRESS.md                 # running log (mirror diffusion_ablation's discipline)
├── make_subsets.py             # nested parity-stratified draws -> subsets/scale_<N>.json + manifest
├── configs/
│   ├── scale_1k.json           # clones neighbors_fm_control.json + data.train_subset_json
│   ├── scale_5k.json
│   ├── scale_10k.json
│   └── scale_30k.json
├── train_scale.slurm           # clone of diffusion_ablation/train_ddpm.slurm, CONFIG_PATH-parameterised
├── eval_scale.sh               # per-arm: downstream R2 (+meanpool twin) -> recon MSE -> counterfactual
└── plot_scaling_curve.py       # mean R2 / MSE vs N, with RandInit floor + AION reference
```

### Edits to existing files (2 files, ~20 lines total)

| File | Edit |
|---|---|
| `config.py` `DataConfig` | add `train_subset_json: Optional[str] = None` and `train_subset_tile_to: Optional[int] = None`. Unknown JSON keys raise `TypeError` in `from_dict`, so the field must exist. |
| `data_factory.py` (after `random_split`, ~:280) | if `train_subset_json` set: load positions, assert ⊆ `train_ds.indices`, `np.resize` to `train_subset_tile_to`, `train_ds = Subset(kept_dataset, tiled)`. Print distinct-N and tiled-length so the log is self-documenting. |

`variants.py` needs **no** change — the arms use the unmodified `neighbors_all_attn` variant.
`variants.json` needs no change either: the downstream eval can be driven by the direct-sbatch route
(`--export=ALL,VARIANT=…,CKPT=…,MODULE=…,MODEL_CLASS=…`), which is read-only w.r.t. the repo.

### Smoke test before any submission

```bash
python -m galaxy_images.galaxy_model.train \
  --config galaxy_images/galaxy_model/scale_ablation/configs/scale_1k.json \
  --set trainer.devices=1 --set trainer.num_steps=50 \
  --set data.batch_size=8 --set wandb.enabled=false \
  --set trainer.monitor_metric=val/loss \
  --set model.num_integration_steps=10
```

`monitor_metric=val/loss` is mandatory with wandb off: `val/mse` is logged only inside
`on_validation_epoch_end`, which early-returns when there is no logger, and
`ModelCheckpoint(monitor='val/mse')` then raises.

Checks: loss decreases; the log prints `distinct=1000, tiled=98554`; steps/epoch is 385 (not 3); one
validation pass completes; the ckpt reloads.

---

## 6. Compute budget — measured, not estimated

Throughput on 4×H200 with this exact workload: **2.61–2.76 optimizer steps/s ⇒ 1.03 h per 10k steps.**
Derived by differencing consecutive `Step N | Elapsed:` log lines, and confirmed end-to-end by two
completed runs (`sacct`: ddpm-eps-75k 07:34:53, fm-ctrl-75k 07:26:43).

> ⚠️ Do **not** use the `Speed: 6.40 steps/s` figure printed in the running 150k logs — it divides the
> absolute step counter (which starts at 75000 on resume) by elapsed-since-resume. It would
> underestimate a run by 2.4×.

| Item | Cost |
|---|---|
| 4 training runs × 75k steps | ~31 h of 4-GPU time (124 H200-hours) |
| Wall clock, 2 concurrent on node4900 | **~16 h** (2 waves) |
| ~10 downstream-R² evals @ ~11 min, 1 GPU | ~2 h wall (2-GPU QOS cap serialises them) |
| Recon MSE + counterfactual evals | ~2–4 h wall |
| Disk | 4 runs × (2 retained + 1 frozen) × 2.2 GB ≈ **26 GB** |

**Constraints:**
- `pg_mki_aryeh` has exactly **one** GPU node (node4900, 8×H200); nodes 4701/4702 are CPU-only. So at
  most **2 concurrent 4-GPU jobs**. The partition QOS has *no* limits, and you are currently its only
  user in the queue.
- All 8 H200s are allocated **right now** by your own two 150k jobs (~50 min remaining as of 18:00 EDT).
  Nothing can start before those land.
- `mit_preemptable` (2-day wall, 4-GPU/user cap, includes node4900's H200s) can host **one** extra
  concurrent 4-GPU run. Requeue costs ≤ ~30 min given 5000-step checkpointing.
- POOL: 816.7 / 1024 GB → **207 GB free**. 26 GB is fine. **SCRATCH is hard-full** (1024/1024; `touch`
  fails) — point nothing there. Note the base config's default `shared_checkpoint_dir` resolves to
  `galaxy_images/galaxy_model/checkpoints`, which is a **symlink into scratch**; every arm must set
  absolute paths under `/orcd/pool/007/pablomer/checkpoints_new/scale_ablation/`.

---

## 7. Evaluation protocol

All three metrics reuse existing, already-validated machinery. Every representation-side eval is
encoder-only, so nothing here is objective-specific.

### (1) Downstream R², n=5469 crossmatch, 90/10 MLP probes

Per arm, one sbatch (~11 min, 1 GPU): `prepare_combined.py → predict_combined.py →
plot_aion_vs_variant.py`. Output `outputs/predictions/predict_<arm>.csv`, **78 rows** =
2 latents × (9 physics + 9 HSC-instrument + 21 Legacy-instrument).

- Physics targets (9): `desi_z`, `provabgs_logmstar`, `provabgs_tage_mw`, `provabgs_z_mw`, `ssfr`,
  `legacy_shape_e1/e2`, `hsc_shape_e1_i/e2_i`.
- Instrument: HSC (9) = `a_{g,r,i,z,y}` + `psf_fwhm_{g,r,i,z}`; Legacy (21) = `PSFSIZE/PSFDEPTH/
  GALDEPTH/MW_TRANSMISSION_{G,R,I,Z}` + `EBV` + `NOBS_{G,R,I,Z}`.
- **Off-diagonals come free** — `predict_combined.py:523-525` passes both latents to all three
  families, so every CSV already contains physics-latent-on-instrument and instrument-latent-on-physics.
  Reference (`base`): physics/e1 **0.704**, legacy-instr/e2 **0.649**, off-diagonals **0.524** and
  **0.177**. The *asymmetry gap vs N* is the disentanglement-emergence curve — free, and arguably the
  most interesting output of this whole ablation.
- Report `instrument_hsc` and `instrument_legacy` **separately**; the code never pools them and a
  pooled mean would weight Legacy 21:9.
- Also build the **meanpool twin** per arm (the paper's instrument bar uses pooled 32-d
  `base-meanpool`): `build_meanpool_h5.py --in <arm>.h5 --out <arm>-meanpool.h5 --tokens 4` then one
  `predict_combined.py`. It's a CPU-only post-process on the H5 — no re-encoding. Cheap enough to do
  both flat and pooled.

⚠️ **Compute the means yourself with a 3-line groupby on `predict_<arm>.csv`.** Two traps otherwise:
`build_convergence_summary` hardcodes a 4-variant list and reads `predict_<v>__<arch>.csv` (new arms
land without the `__aion` suffix and would be silently invisible); and `plot_aion_vs_variant.py`'s
group assignment is prefix-based, so filtering `group=='Physics-like Properties'` silently **drops all
4 ellipticity targets**. Use the `task_family` column.

### (2) Reconstruction MSE — C0 only (DECIDED, Pablo 2026-07-24)

**Scope decision: evaluate MSE on the "normal"/intact reconstruction condition (C0) only.** No
donor-shuffled conditions (C1 physics-swap / C2 instrument-swap / C3 both).

This is a real simplification, because **C0 is exactly what `diffusion_ablation/eval_recon_mse.py`
already computes** — it feeds each anchor its own `samegal` and own `sameins` and scores against its
own `x_1`. So `shuffle_ablation/run_shuffle_ablation.py` is **not needed for this ablation at all**:

| | `eval_recon_mse.py` (chosen) | `run_shuffle_ablation.py` (dropped) |
|---|---|---|
| conditions | C0 only | C0/C1/C2/C3 |
| multi-checkpoint | **yes** — `--fm-checkpoint LABEL:PATH` is `action="append"` | no, `--checkpoint` is singular |
| per-arm plumbing | none; one call covers all arms | separate `--out-dir` + job per arm |
| cost | one short job for the whole curve | ~4 h/ckpt at M=8 samples × 4 conditions |
| protocol match | **the script that produced the published FM/DDPM numbers** | imports its helpers, so same protocol |

Also worth noting C1/C2 have no ground truth anyway (there is no image of galaxy A under galaxy B's
observing conditions), so they could only ever have been reported as a relative asymmetry — whereas C0
is a directly interpretable, ground-truthed number. Dropping them costs nothing for the
"how many crossmatches do you need" question.

The disentanglement story is still covered, via the **R² off-diagonals** in §7(1), which come free in
every `predict_<arm>.csv`.

### (2b) The matched C0 protocol

`diffusion_ablation/eval_recon_mse.py` needs **zero changes**: `--fm-checkpoint` is
`action="append"` taking `LABEL:PATH`, repeatable. x₀ noise is redrawn per model from
`manual_seed(1234 + batch_idx)` with `batch_idx` restarting at 0 per model, so noise is *identical*
across checkpoints. Per-survey rows (hsc/legacy) are emitted separately.

Keep `--n 256 --noise-seed 1234`, batch 64, and reuse `results/recon_eval_manifest.json` (256 pinned
anchors, 123 HSC + 133 Legacy) — it hard-errors on any mismatch, which is the safety net we want.
Full-data reference: HSC **0.0499 ± 0.0045**, Legacy **0.1711 ± 0.0068** (but that's from the best-val
58k ckpt; job 18784184 is producing the frozen-75k version — use that as the anchor).

### (3) Free reference lines

Both already computed and cached — zero cost:
- **`untrained-resnet-matched`** — the paper's "Rand. Init. ResNet" bar; two seeded random-init
  `ResNetEncoder`s, `cross_attention_dim=16`, dim-matched to 128-d combined. This is the floor that
  makes "reasonably performant" quantitative — and it also answers JcH2's separate question about
  what random-guessing R² looks like.
- **`aion-base`** — physics-side ceiling (mean R² 0.742 over the 9 physics targets). `has_e2=False`,
  so no instrument column.

---

## 8. Deliverables for the rebuttal

1. **One figure** — mean R² vs N (log x), two panels (physics latent on physics props; instrument
   latent on instrument props), with the RandInit floor as a horizontal band and the full-data point
   at N=98,554. Optionally a third panel: the off-diagonal asymmetry gap.
2. **One table** — rows = N (plus a `distinct images` column, per §1), columns = mean physics R²,
   mean HSC-instr R², mean Legacy-instr R², C0 recon MSE (HSC / Legacy).
3. **One paragraph** for the limitations section, with the quotable number: *"X% of full-data physics
   R² is reached with only N cross-matched galaxies"*, plus the instrument-encoder-needs-no-overlap
   observation and the single-survey-bootstrap follow-on.

No new plotting infrastructure exists for a curve (`plot_r2_4bars.py` is hardcoded to 2 CSVs / 4
series; `--extra-variant` overlays bars per property, not a curve), so `plot_scaling_curve.py` is new
— but it reads `predict_<arm>.csv` and is ~80 lines.

---

## 9. Decisions — LOCKED (Pablo, 2026-07-24)

1. **75,000 steps**, matched verbatim to `neighbors_fm_control.json`. The in-flight 150k continuation
   runs are *not* the protocol for this ablation.
2. **Downstream holdout applied** to every arm. `data_factory.py:248` gate widened to accept `ram48`
   (it was a silent no-op before — the key would have been ignored and we'd have believed otherwise).
   Uses the existing `downstream_evaluation/engaging/outputs/index/holdout_legacy_ids.txt`
   (5,361 ids → **5,466 anchors excluded**, 5.3%).
   → kept **98,275** / train **93,362** / val **4,913** / **364 steps/epoch** / 206 epochs at 75k.
3. **Ladder: 1,000 / 3,162 / 10,000 / 31,622** — 1.07% / 3.39% / 10.71% / 33.87% of the train pool.
   Nested and exactly 50/50 parity-balanced at every rung (verified).
4. **4 runs, no full-data retrain.** Consequence, to be stated in the figure and the text: because
   the holdout shifts `kept_size`, the free `fm-control-75k` checkpoint is **not** a matched
   full-data point — it trained on 98,554 anchors *without* holdout (so it saw the eval galaxies as
   unlabeled images) and had 385 steps/epoch against our 364. Plot it as a **dashed, explicitly
   labelled unmatched reference**, not as the curve's endpoint. If R² at 31,622 already sits near it,
   saturation is demonstrated and a 5th run is unnecessary; if the curve is still climbing, add the
   matched full-data arm (+7.7 h) — we will know within ~16 h and the rebuttal is not due until
   ~2026-07-30.

### Still open (deliberately deferred, none of it blocks training)

- Whether to add the matched full-data arm. Decide after seeing the 4-point curve (see 4 above).
- Averaging convention for the headline number: raw per-target mean (physics over 9, Legacy-instr
  over 21, HSC-instr over 9) vs the paper's band-averaged rows. I will emit **both** from
  `predict_<arm>.csv` and we pick when we see them.
- ~~Whether the counterfactual axis reports C0 MSE only, or also the C1/C2 asymmetry curve.~~
  **RESOLVED (Pablo, 2026-07-24): C0 only.** `shuffle_ablation` is therefore out of scope for this
  ablation; `eval_recon_mse.py` covers it with no code changes. See §7(2).

---

## 9b. Original decision write-up (superseded by §9, kept for the reasoning)

**Q1 — 75k or 150k steps?** `PLAN.md` for the diffusion ablation fixes 75k, and the frozen 75k FM
control is what makes the full-data point free. But jobs `18736113`/`18736114` are right now
extending both arms to **150k** via `--set run.resume_from=… --set trainer.num_steps=150000`, and this
is recorded nowhere in `PROGRESS.md`. If 150k becomes the rebuttal protocol, the free anchor is the
wrong reference and per-run cost goes 7.7 h → 15.5 h (4 runs ≈ 62 h of 4-GPU time, ~31 h wall).
*What are the 150k runs for?*

**Q2 — hold out the downstream eval set?** You asked for this, and it is feasible, but it is not free:

- The mechanism is **silently inert in `ram48` mode**: `data_factory.py:248` gates on
  `config.data.mode == "efficient"`. Set the key today and you'd *believe* you held data out while
  training on it. One-line fix: `in ("efficient", "ram48")`.
- The ids file **does exist on Engaging** (contradicting `diffusion_ablation/PLAN.md:280`):
  `downstream_evaluation/engaging/outputs/index/holdout_legacy_ids.txt`, 5,361 ids, verified
  `set(file) == set(overlap.parquet['legacy_object_id'])` exactly. Matching needs the `b'...'`
  byte-repr strip that `_normalize_legacy_id` already implements. Cost: **5,466 of 103,741 anchors
  (5.3%)**.
- Because the neighbour graph is bipartite, removing anchors removes those galaxies *entirely* — the
  holdout is clean at galaxy level, not merely anchor level.
- **The cost:** `kept_size` changes → the seed-42 val split changes → (a) the free full-data anchor was
  trained *without* holdout and is no longer a matched top-of-curve, so we must **retrain it (+1 run,
  7.7 h)**, and (b) the 256-anchor recon manifest is invalidated, so all arms need a fresh manifest and
  the MSE numbers stop being cross-referenceable to the existing FM/DDPM table.

So: **hold out everywhere + retrain the control (5 runs)**, or **no holdout anywhere (4 runs, free
anchor, matches the paper's own protocol — `base` was never trained with holdout either)**. My lean is
the former since you asked for it and the marginal cost is one run out of a 6-day window; but it is
your call and it changes the run count.

**Q3 — add the 30k rung?** Your 1k/5k/10k tops out at ~10% of the data, leaving a full decade
unmeasured between 10k and 98.5k — which is exactly where saturation probably lives. Adding 30k costs
one run (7.7 h) and makes the curve interpretable. A cleaner alternative is a pure half-decade ladder:
1k / 3.2k / 10k / 32k / 98.5k.

**Minor, will proceed unless told otherwise:** FM-only (no DDPM arm); fixed-75k checkpoints; both flat
and pooled instrument probes; `checkpoint_every_n_train_steps=5000`; nested parity-stratified draws at
seed 42.

---

## 10. Gotchas to bake in (all verified, mostly learned the hard way in `diffusion_ablation`)

1. **Each of the 4 DDP ranks creates its own dated run dir** (`2026-07-23`, `_2`, `_3`, `_4`) and only
   one — nondeterministically — receives checkpoints. The FM arm's landed in `_2` while its log printed
   a different dir. **Never trust the printed "Run dir"; copy to a `frozen/` path per arm immediately.**
   With 4–5 arms this produces 16–20 directories.
2. **Checkpoint filenames are `latest-step=step=<N>.ckpt`** (doubled `step=`, because
   `auto_insert_metric_name` is left True on the periodic callback). Globs must be `latest-step*.ckpt`.
   The `variants.json` resolver takes the **alphabetically first** hit, which is a live hazard when two
   step counts coexist.
3. **`save_top_k=1` on the periodic callback deletes the previous file**, so only one `latest-step`
   survives per Trainer instance. A fixed-75k comparison is unaffected; an R²-vs-steps curve would
   need `save_top_k=-1` (and ~154 GB at the current 1000-step cadence — don't).
4. **Freeze checkpoints before evaluating them.** An eval died today with
   `OSError: [Errno 116] Stale file handle` because `ModelCheckpoint` deleted the file mid-read, leaving
   a CSV with FM rows and no DDPM rows.
5. **Never change `--ntasks-per-node=4`** in the slurm header. `--ntasks=1` silently trains on **one**
   GPU while sharding data 4 ways — no crash, 4× the budget, and a broken step match. Pass criterion:
   the log must contain `Starting with 4 processes`.
6. **`prepare_combined.py:318` sets the CSV `checkpoint` column from the ckpt's *parent directory
   name*.** Co-locating all arms in one folder makes that column useless for grouping (every
   `predict_diffusion-base.csv` row currently says `frozen`). Give each arm its own parent dir named
   after the arm.
7. **Don't clone a `*_dwnval` config as "base + holdout."** They also set `val_ratio=0.01`, enable
   `lens_val` with `exclude_from_train`, and (some) change `max_neighbors`/`random_neighbors`. Start
   from `neighbors_fm_control.json`.
8. **Keep `save_heldout_validation: false`** — `_save_heldout_validation_subset` unpacks 5 values per
   item but `efficient`/`ram48` items are 4-tuples, so it raises `ValueError`.
9. **`auto_adjust_for_h100`** silently replaces `data.batch_size` with `trainer.h100_batch_size` if any
   GPU name contains "h100". H200 names don't match, so it won't fire here — but set it `false` anyway.
10. **RAM48 loads the full 17.4 GB store per rank regardless of subset size** (`np.fromfile` in
    `__init__`, before subsetting). Small rungs save no RAM and no startup time; keep `--mem=256G` at
    every scale. Never "optimise" `np.fromfile` to `np.ascontiguousarray` — it's a no-op on memmaps and
    silently leaves the data on disk.
11. **CRLF → LF** on any new slurm file; `sbatch` rejects DOS line endings.
12. **Ping the Discord webhook** on every submission and completion (already plumbed in
    `train_ddpm.slurm` via `_ping` + an EXIT trap — clone it).
13. `diffusion_ablation/PROGRESS.md` is **stale** (stops 2026-07-23, records nothing about the 150k runs
    or `shuffle_ablation`). Keep this ablation's `PROGRESS.md` current — it's the only reliable status.
