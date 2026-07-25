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

### Next
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
