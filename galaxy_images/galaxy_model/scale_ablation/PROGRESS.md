# Scale Ablation — Progress Log

Keep this current. `diffusion_ablation/PROGRESS.md` went stale and cost real time (it recorded
nothing about the 150k continuation runs or `shuffle_ablation`, so the status had to be
re-derived from `squeue` and log files).

---

## 2026-07-24 — design locked, plumbing implemented, smoke submitted

**Reviewer target:** JcH2 Q4 + the Limitations note + minor weakness (line 32) — "how much data does
this need, and how much *overlap* between instruments."

**What we ablate (locked):** the number of **cross-matched anchors**, holding single-survey coverage
at full density. Confirmed to be the dataset's actual structure: the sky-kNN graph is strictly
bipartite anchors→pool (measured over all 103,741 anchors × 20 slots: **zero** anchor→anchor edges),
and the 364,456 pool rows are single-survey-only galaxies. So neighbours legitimately come from the
full pool, and a random anchor subset is the correct design — the contiguous-sky-footprint idea we
first considered is unnecessary.

### Decisions
- 75k steps, matched to `diffusion_ablation/configs/neighbors_fm_control.json` (batch 64 × 4 GPUs =
  effective 256, seed 42, max_neighbors=5, lr 1e-4, bf16-mixed).
- Downstream holdout ON. Ladder 1,000 / 3,162 / 10,000 / 31,622. Four runs, no full-data retrain.
- FM only (the DDPM arm's ε-prediction pathology would swamp the data-scale signal).

### ⚠ The finding that shaped the implementation
`double_train_fm_neighbors.py:962-965` builds `CosineAnnealingLR(T_max=self.trainer.max_epochs)`, and
`train.py` never passes `max_epochs`, so it is **-1**. The LR therefore does not anneal — it
**alternates between 1e-4 and exactly 0.0 at every epoch boundary** (verified by direct simulation in
this env). Two consequences:
- ~50% of optimizer steps run at lr=0, so nominal 75k ≈ 37.5k effective updates. The published base
  model was trained this way, so it is internally consistent.
- **The square-wave period is the epoch length, which scales with training-set size.** Naively
  subsetting would have given the 1k arm ~3-step LR blocks against the full run's 385 — a
  dataset-size-dependent optimization confound that would have been reported as "you need more
  crossmatches." This is the single most important thing the recon caught.

**Fix (implemented): index tiling.** Each rung's index list is cyclically tiled to the full train
length (93,362), so every arm has **364 steps/epoch** and an identical LR waveform. Only the number of
distinct anchors varies. Chosen over `RandomSampler(replacement=True)` because `train.py` builds the
Trainer without `use_distributed_sampler=False`, so Lightning would replace a custom sampler with its
own `DistributedSampler`; tiling needs no sampler plumbing and gives deterministic (balanced) anchor
multiplicity.

### Code changes
| File | Change |
|---|---|
| `config.py` | `DataConfig`: added `train_subset_json`, `train_subset_tile_to` |
| `data_factory.py` | added `_apply_train_subset()`; call site after `random_split`; widened the holdout gate `== "efficient"` → `in ("efficient", "ram48")`; `kept_positions` now always defined so raw→kept translation is uniform |
| `scale_ablation/` | `make_subsets.py`, `make_configs.py`, `configs/scale_{1000,3162,10000,31622}.json`, `train_scale.slurm`, `smoke_train.slurm`, `PLAN.md`, this file |

Subset JSONs store **raw** anchor positions (pre-holdout) so they mean the same thing regardless of
which exclusions are active; `data_factory` translates to kept-space and hard-fails on any position
that is held out, duplicated, or in the validation split.

### Verified numbers
```
anchors total            : 103,741
holdout                  : 5,361 ids -> 5,466 anchors excluded (5.3%)
kept / train / val       : 98,275 / 93,362 / 4,913
steps per epoch          : 364     (206.0 epochs for 75,000 steps)
train parity pool        : 46,689 HSC-role / 46,673 Legacy-role

  scale_1000    n= 1,000    500 HSC /    500 Legacy    1.07% of train  ~93.4 repeats/epoch
  scale_3162    n= 3,162  1,581 HSC /  1,581 Legacy    3.39% of train  ~29.5 repeats/epoch
  scale_10000   n=10,000  5,000 HSC /  5,000 Legacy   10.71% of train  ~ 9.3 repeats/epoch
  scale_31622   n=31,622 15,811 HSC / 15,811 Legacy   33.87% of train  ~ 3.0 repeats/epoch
  nestedness verified: every rung is a strict subset of the next
```

CPU test (`scratchpad/test_subset_plumbing.py`) passed: holdout+split reproduce the manifest; every
rung tiles to 93,362 → 364 steps/epoch; parity exactly 50/50; no rung overlaps the validation split
or the holdout; all five guard rails fire (held-out anchor, validation anchor, duplicates,
`tile_to` < subset, empty subset).

### Status
- **job 18790012** `scale-smoke` (scale_1000, 200 steps, 1 GPU, mit_normal_gpu) — submitted, PENDING
  behind `shuffle-ablation` under the 2-GPU QOS cap. Asserts from the log that the holdout fires in
  ram48 mode, the split is 98,275/93,362/4,913, and the rung tiles to 93,362.
- Training runs **not** submitted. node4900 (8×H200, the only GPU node in `pg_mki_aryeh`) is fully
  allocated by `fm-ctrl-150k` + `ddpm-eps-150k`, ~50 min remaining as of 18:00 EDT.

### Next (as of 2026-07-24)
1. Green smoke → submit rungs 2-at-a-time on `pg_mki_aryeh` (~7.7 h each, ~16 h wall for 4).
   Optionally a 3rd concurrent run via `mit_preemptable` (4-GPU/user cap, requeue costs ≤30 min).
2. Immediately copy each finished ckpt to `frozen/scale_<N>/scale-<N>-75k.ckpt` — **before** any eval.
   Two reasons: `ModelCheckpoint` rotation deleted a file mid-read during the diffusion ablation
   (`OSError: Errno 116 Stale file handle`), and `prepare_combined.py:318` takes the CSV `checkpoint`
   column from the ckpt's **parent directory name**, so each arm needs its own parent dir.
3. Per arm: downstream R² (+ meanpool twin), then **one** `eval_recon_mse.py` call covering all arms
   (`--fm-checkpoint` is repeatable). **C0 / intact reconstruction only** — decided by Pablo
   2026-07-24, so `shuffle_ablation` (C1/C2/C3 donor-shuffled conditions) is out of scope and needs
   no adaptation. A **new** recon manifest is still required: the holdout changed the val split, so
   the diffusion ablation's 256-anchor manifest will hard-fail on mismatch (by design). The new val
   anchors are never trained on in any arm, so they remain a legitimate held-out set.
4. `plot_scaling_curve.py` — not written yet; nothing existing plots a curve.

---

## 2026-07-28 — extra n=100 rung trained; R² + recon-MSE evals submitted

A fifth rung **below** the original ladder, to pin down where the curve bottoms out (the 1k arm
was still well above the untrained floor, so the low end was unresolved).

**Training — job 19010173 `scale-100`, COMPLETED 01:19 EDT, 7 h 34 m, exit 0.** Config
`configs/scale_100.json`, subset `subsets/scale_100.json` (already present in
`subsets/manifest.json` as the first rung: 50 HSC-role / 50 Legacy-role, 700 distinct images,
933.62 repeats/epoch, tiled to 93,362 → the same 364 steps/epoch and identical LR waveform as
every other arm). Final `latest-step=step=75000.ckpt` written at 01:13.

### Which checkpoint — and why it is a symlink
Frozen as `frozen/scale_100/scale-100-75k.ckpt`, pointing at the **step-75,000** checkpoint, not
the best-val/mse one (`best-epoch=184-step=67000.ckpt`). That matches the other four arms: the
tables are labelled "fixed-75k ckpts" and `frozen/scale_<N>/scale-<N>-75k.ckpt` is the final
periodic checkpoint in every case.

It is a **symlink**, not a copy, which departs from step 2 of the 2026-07-24 plan. Both original
reasons for copying were re-checked:
- *Rotation deleting the file mid-read* — no longer possible. Training has COMPLETED, so nothing
  will write to that `checkpoints/` dir again.
- *`prepare_combined.py:318` naming* — still satisfied. It reads
  `args.checkpoint.parent.name` **without** `.resolve()`, so the symlink yields
  `checkpoint_name=scale_100`, exactly as a real copy would. Verified before submitting.

The trigger was cost: POOL is at **92.0 % (942/1024 GB)** and a `cp` of the 2.2 GB file was
running at ~0.7 MB/s against Ceph (374 MB in 10 min, ~1 h projected) while `ddpm-vpred-75k` and
`cf-fm-long` were both hitting the same filesystem. ⚠ The symlink's one weakness is that it dies
if `runs/scale_100/` is ever cleaned up — materialise it into a real copy before any such cleanup.

### Eval jobs
| job | script | partition | writes |
|---|---|---|---|
| 19035900 `engaging-scale-100` | `eval_r2_scale100.slurm` (new) | mit_normal_gpu | `scale-100.h5`, `predict_scale-100.csv`, `scale-100-meanpool.h5`, `predict_scale-100-meanpool.csv` |
| 19035901 `recon-mse-scale100` | `eval_recon_scale_n100.slurm` (new) | pg_mki_aryeh, 1×H200 | `results/recon_mse_scale_n100.csv` |

`eval_r2_scale100.slurm` collapses into one job what `run_variant.slurm` +
`eval_meanpool_scale.slurm` did for the other arms (flat probe **and** the mean-pooled twin). The
twin is mandatory, not cosmetic: `report_scale_results.py` sources the instrument column from
`predict_<v>-meanpool.csv` and silently falls back to the flat 128-d latent with a `†` flag — an
n=100 column on a different latent than the other four would not be comparable.

`eval_recon_scale_n100.slurm` reuses `results/recon_eval_manifest.json` **unchanged**, so it is
pinned to the same 256 held-out anchors and the same per-batch x0 noise seeds as parts 1 and 2;
the script hard-errors on mismatch. Its CSV therefore concatenates with the existing ones
(`report_scale_results.py` globs `recon_mse_scale*.csv` and dedups on
`model_label × anchor_survey`). `--config scale_1000.json` is kept deliberately, matching parts
1/2 — the config only selects the val-split loader and the train-subset key never touches the
val split.

### Code changes
| File | Change |
|---|---|
| `report_scale_results.py` | `ARMS` gains `"scale-100": 100` (first, so it leads the table columns and the log-x curves); `cmd_mse` `order` + `label` gain the `scale-100` / `N=100` row |
| `scale_ablation/` | added `eval_r2_scale100.slurm`, `eval_recon_scale_n100.slurm` |

`configs/`, `subsets/`, `make_subsets.py` and `make_configs.py` already carried the n=100 rung —
no change needed there.

### Results — both jobs COMPLETED, tables + plots regenerated
`19035900` 28 min, `19035901` 53 min, both exit 0. The recon job logged
`[eval] manifest OK`, i.e. it reproduced the same 256 anchors and noise seeds as parts 1/2, so
the three CSVs are legitimately concatenable. `report_scale_results.py` picked up all three
(`recon_mse_scale.csv`, `recon_mse_scale_n100.csv`, `recon_mse_scale_part2.csv` -> 8 models) and
the R2 self-check against the paper's `merged_simplified.csv` still passes at max |d| = 5.55e-17.

|  | n=100 | n=1,000 | n=31,622 | full data | rand-init floor |
|---|---|---|---|---|---|
| mean physics R2 (7) | 0.528 | 0.593 | 0.634 | 0.640 | 0.394 |
| mean Legacy instrument R2 (5) | 0.196 | 0.289 | 0.499 | 0.611 | 0.135 |
| mean HSC instrument R2 (2) | 0.176 | 0.219 | 0.375 | 0.478 | 0.119 |
| recon MSE HSC | 0.2580 +/- 0.0614 | 0.1069 +/- 0.0147 | 0.0534 +/- 0.0055 | 0.0729 +/- 0.0104 | — |
| recon MSE Legacy | 0.2831 +/- 0.0366 | 0.1752 +/- 0.0099 | 0.1876 +/- 0.0072 | 0.1809 +/- 0.0074 | — |

**What the new rung buys us.** It separates the two channels at the bottom of the ladder. At 100
crossmatches the physics latent is already ~63% of the way from the random-init floor to full
data (0.528 vs floor 0.394, full 0.640), while BOTH instrument latents sit barely above their
floors (0.196 vs 0.135; 0.176 vs 0.119). That is a sharper answer to JcH2 Q4 than the four-rung
ladder gave: physics structure is nearly free, instrument disentangling is what actually consumes
crossmatches.

**Two things to keep in mind when writing this up.**
- `t_age` is non-monotone: 0.141 at n=100 vs 0.109 at n=1,000. It is the weakest target in the
  table and n=1,000 was already the outlier there (0.109 against 0.220 at n=3,162), so this looks
  like probe noise on a barely-predictable quantity rather than an n=100 effect. Do not build an
  argument on that row.
- The n=100 recon MSE has a much wider HSC error bar than any other rung (+/- 0.0614 on n=108
  anchors, vs +/- 0.0147 at n=1,000). The HSC-vs-Legacy ordering at n=100 is therefore NOT
  resolved — the two overlap well within 1 SEM. The drop from n=100 to n=1,000 is real and large;
  the HSC/Legacy gap at n=100 specifically is not.
- Unchanged by this rung, but still true and still unexplained: **Legacy recon MSE is U-shaped**
  (0.2831 -> 0.1752 -> 0.1538 -> 0.1611 -> 0.1876), bottoming at n=3,162 and rising back to the
  full-data level by n=31,622. HSC decreases monotonically throughout.

### Next
1. Fold the n=100 column into the rebuttal text (due ~2026-07-30).
2. `snapshot_checkpoints.py` `ARMS` still lists only `[1000, 3162, 10000, 31622]`. Harmless now
   (that archiver only matters *during* training and n=100 is done), but it means there is no
   step-ladder of intermediate n=100 checkpoints for an R²-vs-step curve — add `100` to `ARMS`
   before any re-run if that curve is wanted for this rung.
