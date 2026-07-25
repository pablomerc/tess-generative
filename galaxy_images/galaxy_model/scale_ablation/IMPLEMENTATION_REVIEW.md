# Scale Ablation — Implementation Review

A walkthrough for verifying what was actually built, written so you can **check the claims rather
than trust them**. Every section states a claim, points at the exact code, and gives a command you
can run yourself.

Companion docs: [`PLAN.md`](PLAN.md) (design + locked decisions), [`PROGRESS.md`](PROGRESS.md) (log).

**Diff footprint:** 3 shared files touched.

```
galaxy_images/galaxy_model/config.py       | +11  DataConfig subset fields, +12 WandbConfig fields
galaxy_images/galaxy_model/data_factory.py | +81  subset mechanism + holdout gate fix
galaxy_images/galaxy_model/train.py        | +6   forward W&B group/tags/job_type/extra_config
galaxy_images/galaxy_model/scale_ablation/ | new (untracked)
```

Every new field defaults to `None`/empty, so **all existing configs behave identically** — the new
code paths are inert unless explicitly switched on. The one behavioural change to existing code is
the holdout gate (§2), whose blast radius is analysed there.

---

## 0. The four claims that carry the science

If any one of these is false, the resulting curve is wrong in a way that would not be obvious from
looking at it. Everything else in this document is supporting detail.

| # | Claim | Why a failure would be invisible |
|---|---|---|
| **C1** | Every arm sees exactly **364 steps/epoch**, so the (epoch-driven) LR waveform is identical across data scales | A confounded curve still looks like a clean saturation curve |
| **C2** | The **downstream eval set is genuinely excluded** from training in `ram48` mode | The mechanism was a silent no-op before; there is no warning |
| **C3** | Every arm validates on the **same 4,913 galaxies**, and no rung overlaps that set | `random_split` silently returns a different val set at a different length |
| **C4** | Rungs are **nested** and **50/50 parity-balanced**; the scale axis is distinct cross-matched anchors | Survey mix shifts change MSE ~3.4× and would masquerade as a data-scale effect |

Jump to: [C1 §3](#3-c1--constant-epoch-length-the-lr-confound-fix) ·
[C2 §2](#2-c2--the-holdout-gate-was-a-silent-no-op) ·
[C3 §4](#4-c3--common-validation-set) ·
[C4 §5](#5-c4--nested-parity-balanced-rungs)

---

## 1. Change-by-change map

### `config.py` — two new fields (lines 33–43)

```python
train_subset_json: Optional[str] = None      # :36
train_subset_tile_to: Optional[int] = None   # :43
```

Both must exist as `DataConfig` dataclass fields: `ExperimentConfig.from_dict` splats each JSON
section into its dataclass, so an unrecognised key raises `TypeError`. Adding them here is what makes
the configs loadable at all. Defaults are `None`, so **every existing config behaves exactly as
before** — the new code path is inert unless `train_subset_json` is set.

### `data_factory.py` — four edits

| Line | Edit | Purpose |
|---|---|---|
| `:3` | `import json` | reading the subset files |
| `:120` | new `_apply_train_subset()` (~60 lines) | the whole mechanism |
| `:314` | holdout gate `== "efficient"` → `in ("efficient", "ram48")` | **C2** |
| `:336` | `kept_positions = list(range(total_size))` in the no-exclusion branch | makes raw→kept translation uniform |
| `:355` | call `_apply_train_subset` after `random_split` | **C1**, **C3** |

Nothing else in the file changed. No existing call signature changed.

### `scale_ablation/` — new files

```
make_subsets.py    builds nested parity-stratified rungs -> subsets/*.json + manifest.json
make_configs.py    derives the 4 training configs from diffusion_ablation's FM control
configs/           scale_{1000,3162,10000,31622}.json
subsets/           scale_{1000,3162,10000,31622}.json + manifest.json
train_scale.slurm  4xH200 trainer, CONFIG_PATH-parameterised (clone of train_ddpm.slurm)
smoke_train.slurm  200-step 1-GPU smoke with log assertions
```

---

## 2. C2 — the holdout gate was a silent no-op

**The bug you were about to hit.** `data_factory.py` gated the downstream holdout on
`config.data.mode == "efficient"`. Every Engaging run uses `mode="ram48"`. So setting
`data.downstream_holdout_ids_txt` would have been **silently ignored** — no error, no warning — and
you would have believed the eval set was held out while training on it.

Fixed at `:314`:

```python
if config.data.downstream_holdout_ids_txt and config.data.mode in ("efficient", "ram48"):
```

Safe because `_resolve_downstream_holdout_positions` only needs `<efficient_data_dir>/catalog.parquet`,
and the ram48 store has it with the required `object_id_legacy` / `source_type` columns. (The two
catalog files — the 205 GB 160×160 store and the 48×48 store — are byte-identical: both 111,844,513
bytes, same schema, same 468,197 rows.)

> **Blast radius:** this changes behaviour for any `ram48` run that sets
> `downstream_holdout_ids_txt`. No existing config does — neither diffusion-ablation config sets it —
> so nothing else in the repo is affected. `lens_val` has the identical gate one block up at `:301`;
> I deliberately left it alone to keep the change minimal (it defaults to disabled and its `lens_h5`
> is not on this cluster).

**Verify:**
```bash
cd /orcd/pool/007/pablomer/tess-generative
sed -n '312,316p' galaxy_images/galaxy_model/data_factory.py
# and confirm nothing else relies on the old behaviour:
grep -rn "downstream_holdout_ids_txt" galaxy_images/galaxy_model --include=*.json
```
The only hits should be the four `*_dwnval*.json` configs (which use `mode="efficient"`, unaffected)
and this ablation's four configs.

**The ids file is real and correct** — it is not something I generated. 5,361 ids at
`downstream_evaluation/engaging/outputs/index/holdout_legacy_ids.txt`, verified to be exactly the id
set of the n=5469 eval crossmatch:
```bash
/orcd/home/002/pablomer/miniforge3/envs/torchenv/bin/python - <<'PY'
import pandas as pd
d="galaxy_images/galaxy_model/downstream_evaluation/engaging/outputs/index/"
ids={l.strip() for l in open(d+"holdout_legacy_ids.txt") if l.strip()}
ov=set(pd.read_parquet(d+"overlap.parquet",columns=["legacy_object_id"])["legacy_object_id"])
print(len(ids),len(ov),"set-equal:",ids==ov)
PY
```
Expect `5361 5361 set-equal: True`. (5,361 < 5,469 because 108 eval rows share a Legacy id — two HSC
objects matching one Legacy object within the 1″ radius. The holdout is a set, so this is harmless.)

⚠️ **Known limitation to state in the paper, not a bug:** the holdout removes **anchors**. It cannot
remove a galaxy from the neighbour pool, because neighbour lists are precomputed row indices baked
into `catalog.parquet`. In practice this is fine — the kNN graph is strictly bipartite anchors→pool
(measured: **zero** anchor→anchor edges across all 103,741 anchors × 20 slots), so a held-out anchor
can never reappear as a neighbour. The one thing I have *not* verified is whether a given physical
galaxy could appear as both an anchor row and a separate pool row. Worth a 5-minute check before
claiming a strict galaxy-level holdout in text — see §7.

---

## 3. C1 — constant epoch length (the LR confound fix)

**The finding.** `double_train_fm_neighbors.py:962-965`:

```python
scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
```

`train.py` passes only `max_steps`, never `max_epochs`, so PyTorch Lightning sets
`trainer.max_epochs == -1`. With `T_max=-1` the cosine does not anneal — it **alternates between
`1e-4` and exactly `0.0` at each epoch boundary**. Verified by direct simulation in this env
(pl 2.6.1 / torch 2.6.0+cu124): per-step LR `[1e-4, 1e-4, 0.0, 0.0, 1e-4, 1e-4, 0.0, 0.0, ...]`.

Two consequences, and the second is what matters for this experiment:
1. ~50% of optimizer steps do nothing. Nominal 75k ≈ 37.5k effective updates. The published base
   model was trained this way, so it is internally consistent — not something to fix here.
2. **The square-wave period is the epoch length, which scales with dataset size.** Naive subsetting
   would give the 1k arm ~3-step LR blocks against the full run's 385. The small arms would
   underperform partly from a pathological LR schedule, and we would have reported that as *"you need
   more crossmatches."*

**The fix** (`_apply_train_subset`, `data_factory.py:120`): tile each rung's index list cyclically up
to the full train length before wrapping it in a `Subset`.

```python
tiled = np.resize(np.asarray(requested_kept, dtype=np.int64), int(tile_to))
return Subset(kept_dataset, tiled.tolist())
```

So `len(train_ds) == 93,362` in **every** arm → `ceil(93362/4)//64 = 364` steps/epoch → identical LR
waveform, identical effective batch, identical total optimizer steps. The only thing that varies is
the count of *distinct* anchors.

**Why tiling and not `RandomSampler(replacement=True)`:** `train.py` constructs the `Trainer` without
`use_distributed_sampler=False`, so Lightning injects its own `DistributedSampler` and would discard a
custom sampler. Tiling needs no sampler plumbing, and gives each anchor a *deterministic* multiplicity
(balanced) rather than a Poisson one.

**Verify the arithmetic independently:**
```bash
/orcd/home/002/pablomer/miniforge3/envs/torchenv/bin/python -c "
import math,json
m=json.load(open('galaxy_images/galaxy_model/scale_ablation/subsets/manifest.json'))
print('train_size',m['train_size'],'tile_to',m['tile_to'])
print('steps/epoch',math.ceil(m['tile_to']/4)//64,'expected',m['steps_per_epoch'])
print('epochs @75k',75000/m['steps_per_epoch'])"
```
Expect `364` and `206.0`.

**Verify the LR claim yourself** (this is the load-bearing surprise, so worth reproducing):
```bash
/orcd/home/002/pablomer/miniforge3/envs/torchenv/bin/python -c "
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
o=torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))],lr=1e-4)
s=CosineAnnealingLR(o,T_max=-1)
print([round(g['lr'],8) for g in o.param_groups]+[ (s.step(), round(o.param_groups[0]['lr'],8))[1] for _ in range(5)])"
```
Expect alternating `1e-04` / `0.0`.

---

## 4. C3 — common validation set

**The trap.** `random_split`'s permutation length is `kept_size`, so *any* change to `kept_size`
completely changes the val set. Measured at seed 42: `len=103741` → val head
`[28083, 35471, 95522, ...]`; `len=51870` → `[42417, 51405, 3403, ...]`. No relation whatsoever.

**The design.** The subset is applied **after** `random_split` (`data_factory.py:355`), and `val_ds`
is never touched. `kept_size` is therefore identical in all four arms (98,275, because all four apply
the same holdout), so all four validate on the same 4,913 galaxies.

```python
train_ds, val_ds = random_split(kept_dataset, [train_size, val_size], generator=generator)
...
if config.data.train_subset_json:
    train_ds = _apply_train_subset(...)     # val_ds untouched
```

`_apply_train_subset` hard-fails if any requested position lands in the val split:

```python
allowed = {int(i) for i in getattr(train_ds, "indices", [])}
stray = [p for p in requested_kept if p not in allowed]
if stray: raise ValueError(...)
```

> ⚠️ **Consequence to be aware of:** because the holdout changed `kept_size` from 103,741 to 98,275,
> the val split differs from the diffusion ablation's. The 256-anchor recon manifest at
> `diffusion_ablation/results/recon_eval_manifest.json` will therefore **hard-fail** on
> `Manifest mismatch` — by design, it compares `dataset_indices`. The scale arms need a **new**
> manifest, and their recon-MSE numbers are not cross-referenceable to the existing FM/DDPM table.
> This is the price of the holdout and it is expected, not a bug.

---

## 5. C4 — nested, parity-balanced rungs

**Why parity.** `neighbors_ram48.py:71` assigns the anchor survey deterministically:
`anchor_is_hsc = (idx % 2 == 0)`. It is **not** a random 50/50 draw (the repo `CLAUDE.md` says it is;
`CLAUDE.md` is wrong). Per-survey recon MSE differs ~3.4× (HSC 0.050 vs Legacy 0.171), so an
unbalanced small draw would shift the survey mix and that would masquerade as a data-scale effect.

`make_subsets.py` shuffles the even-position and odd-position train anchors separately, interleaves
them, and takes prefixes — so every even-length prefix is exactly 50/50 *and* nested by construction.

**Parity survives the indirection.** `Subset.__getitem__(i)` forwards `self.indices[i]`, so the base
dataset receives the original raw position and each anchor keeps its survey role at every scale.
(The alternative — filtering the dataset's internal `anchor_indices` array — would renumber positions
and flip roughly half the anchors' survey role. That is why the implementation goes through `Subset`.)

**Positions in the JSON are RAW** (pre-holdout) positions, so the files mean the same thing regardless
of which exclusions are active; `_apply_train_subset` translates to kept-space via `raw_to_kept` and
raises if a requested position was excluded.

**Verify nestedness, balance, and disjointness from val/holdout in one go** — this is the CPU test,
and it re-derives the holdout and split from scratch rather than trusting the manifest:
```bash
PYTHONPATH=/orcd/pool/007/pablomer/tess-generative \
/orcd/home/002/pablomer/miniforge3/envs/torchenv/bin/python \
  /tmp/claude-239098/-orcd-pool-007-pablomer/1fa7f52f-9fc7-48bb-ac3c-4d4ee5059e82/scratchpad/test_subset_plumbing.py
```
Last run — all green:
```
[ok] holdout+split reproduce manifest: kept=98,275 train=93,362 val=4,913
[ok] n= 1,000  len=93,362  steps/epoch=364     500 HSC /    500 Legacy  no val/holdout leak
[ok] n= 3,162  len=93,362  steps/epoch=364   1,581 HSC /  1,581 Legacy  no val/holdout leak
[ok] n=10,000  len=93,362  steps/epoch=364   5,000 HSC /  5,000 Legacy  no val/holdout leak
[ok] n=31,622  len=93,362  steps/epoch=364  15,811 HSC / 15,811 Legacy  no val/holdout leak
[ok] rejected a held-out anchor / a validation anchor / duplicates / tile_to<subset / empty subset
All subset-plumbing checks passed.
```
Note the file lives in the session scratchpad, not the repo. **If you want it kept, say so and I'll
move it to `scale_ablation/test_subset_plumbing.py`** — it is a genuine regression test for the two
shared-file edits, and I'd argue it belongs in the repo.

---

## 6. Configs — derived, not hand-written

`make_configs.py` deep-copies `diffusion_ablation/configs/neighbors_fm_control.json` and overrides only
six keys, then **asserts** that the step-matching knobs did not drift:

```python
for section, key in [("trainer","num_steps"), ("trainer","devices"), ("trainer","seed"),
                     ("trainer","scale_steps_by_devices"), ("data","batch_size"),
                     ("data","max_neighbors"), ("data","val_ratio"), ("model","lr")]:
    assert cfg[section][key] == base[section][key]
```

Per-arm overrides and why each is needed:

| Key | Value | Why |
|---|---|---|
| `run.output_dir` | `.../scale_ablation/runs/scale_<N>` | the variant is `neighbors_all_attn` for **all** arms, so a shared dir would collide — and each of the 4 DDP ranks creates its own dated subdir, so 4 arms would produce 16 indistinguishable directories |
| `data.train_subset_json` | `subsets/scale_<N>.json` | which rung |
| `data.train_subset_tile_to` | `93362` | **C1** |
| `data.downstream_holdout_ids_txt` | the engaging ids file | **C2** |
| `trainer.checkpoint_every_n_train_steps` | `5000` (was 1000) | cuts write traffic on the shared Ceph FS from ~220 GB to ~44 GB per run; bounds resume loss to ~30 min |
| `wandb.name` | `scale-<N>-75k` | per-arm tag; also names the best-ckpt subdir, so no collision there |

**Verify no drift from the FM control:**
```bash
for n in 1000 3162 10000 31622; do
  echo "== scale_$n"; diff <(/orcd/home/002/pablomer/miniforge3/envs/torchenv/bin/python -m json.tool \
    galaxy_images/galaxy_model/diffusion_ablation/configs/neighbors_fm_control.json) \
    <(/orcd/home/002/pablomer/miniforge3/envs/torchenv/bin/python -m json.tool \
    galaxy_images/galaxy_model/scale_ablation/configs/scale_$n.json)
done
```
Expect exactly the six keys above to differ, nothing else.

---

## 6b. W&B logging

Training metrics already went to W&B before any of my changes (`train.py:115` passes
`config={**asdict(config), ...}`, so the two new `DataConfig` fields are captured automatically).
Three things were missing for this to be a usable sweep, and I added them.

### What was missing

1. **No grouping** — the four arms would scatter into `galaxy-flow-matching-neighbours` alongside
   every other run, with nothing marking them as a family.
2. **No plottable x-axis** — `data.train_subset_json` is a *path*. You cannot chart R² against a
   path. The sweep needs `n_anchors` as a flat numeric config value.
3. **Eval results never reached W&B at all** — downstream R², recon MSE and counterfactual MSE land
   in CSVs on disk only.

### Changes

`config.py` `WandbConfig` gained four fields (`group`, `tags`, `job_type`, `extra_config`), and
`train.py:110-124` forwards them to `WandbLogger` (which passes `group`/`tags`/`job_type` through to
`wandb.init`), merging `extra_config` into the run config. All default to `None`/empty, so **every
existing config logs exactly as before**.

**I kept `project` unchanged** (`galaxy-flow-matching-neighbours`) on purpose, rather than making a
dedicated project: the free `fm-control-ram48-h200-75k` reference run lives there, so keeping the same
project means the scale arms and the unmatched full-data reference are directly comparable in one
workspace. Grouping is done with `group`/`tags` instead, which is the idiomatic W&B way.
Say the word if you'd rather have a separate `galaxy-scale-ablation` project.

Per arm:
```json
"group": "scale-ablation-75k",
"job_type": "train",
"tags": ["scale-ablation", "flow-matching", "75k", "holdout", "n1000"],
"extra_config": {
  "scale/n_anchors": 1000,          "scale/frac_of_train": 0.010711,
  "scale/distinct_neighbour_rows": 4942,  "scale/distinct_images_total": 6942,
  "scale/repeats_per_epoch": 93.36, "scale/train_pool": 93362,
  "scale/tile_to": 93362,           "scale/steps_per_epoch": 364,
  "scale/holdout_applied": true,    "scale/val_size": 4913
}
```

So in W&B you can group by `scale-ablation-75k`, then plot any logged metric against
`scale/n_anchors` directly.

### Distinct-image accounting (also now in the manifest)

`make_subsets.py` now computes this per rung, mirroring the loader exactly (survey side by position
parity, first 5 valid slots). These are the numbers §1 of `PLAN.md` requires next to N in the paper
table — because we ablate *cross-matches*, the model still touches single-survey pool images:

| N anchors | distinct neighbour rows | total distinct images |
|---|---|---|
| 1,000 | 4,942 | **6,942** |
| 3,162 | 15,163 | **21,487** |
| 10,000 | 44,210 | **64,210** |
| 31,622 | 113,366 | **176,610** |

Note the top rung touches **more distinct images (176,610) than the entire anchor catalog (103,741)**,
because pool galaxies are separate objects. That is exactly why both columns have to be published.

### Still to write: `log_results_wandb.py`

Post-training, this will read each arm's `predict_<arm>.csv` + recon/counterfactual CSVs and log a
single summary W&B run (`job_type="eval"`, same group) containing a `wandb.Table` of all arms ×
metrics plus line plots of mean physics R², mean instrument R², and recon MSE against
`scale/n_anchors`. **Not written yet** — it needs the CSVs to exist, so it is post-training work and
does not block submission.

---

## 7. Residual risks — what I have NOT verified

Listed honestly, worst first.

1. **End-to-end GPU run.** As of writing, smoke job **18790012** is still running (I/O-bound loading
   the 17.3 GB RAM48 store; `MaxRSS` had reached 16.1 GiB with only 14 s of CPU, which is the expected
   signature). Until it prints `[smoke-check] PASS`, the claim "this trains" is unproven. The smoke
   asserts C1 and C2 straight out of the log:
   ```
   matched anchors: 5,466 / 103,741
   kept=98,275 | val_ratio=0.05 -> train=93,362, val=4,913
   tiled to 93,362 items
   ```
2. **Anchor-vs-pool galaxy duplication.** Whether one physical galaxy can occupy both an anchor row
   and a pool row. If it can, the holdout is anchor-level rather than strictly galaxy-level. Cheap to
   settle (compare `object_id_hsc` across `source_type` groups) and it only affects the wording of one
   sentence, not the curve.
3. **No full-data arm.** With the holdout on, `fm-control-75k` is **not** a matched endpoint: it
   trained on 98,554 anchors *without* holdout (so it saw the eval galaxies as unlabeled images) and
   had 385 steps/epoch against our 364. It must be plotted as a dashed, explicitly-labelled *unmatched*
   reference. Decision on whether to add the matched arm is deferred until we see the 4-point curve.
4. **The tiling changes the meaning of "epoch."** At N=1,000 an "epoch" is 93.4 passes over the same
   1,000 anchors. Any epoch-indexed logging (including the `best-epoch=NN` in checkpoint filenames)
   is therefore not comparable across arms. Step counts are. Read steps, not epochs.
5. **Duplicate anchors within a batch.** At N=1,000 with ~93 repeats/epoch, a batch of 64 drawn from
   93,362 slots will occasionally contain the same anchor twice. Harmless for the objective (it is
   just a repeated sample) but worth knowing it happens.
6. **Not yet written:** `plot_scaling_curve.py`, the per-arm eval driver, and the new recon manifest.
   All post-training work; none of it blocks submission. Note the recon side got *simpler* after
   Pablo's 2026-07-24 decision to score **C0 (intact reconstruction) only** — that is precisely what
   `diffusion_ablation/eval_recon_mse.py` already computes, and its `--fm-checkpoint LABEL:PATH` flag
   is repeatable, so the whole curve comes from one call with zero code changes. `shuffle_ablation`
   (C1/C2/C3 donor-shuffled conditions) is out of scope.

---

## 8. Review checklist

- [ ] §2 — holdout gate widened, ids file verified set-equal to the n=5469 eval crossmatch, blast radius understood
- [ ] §3 — LR alternation reproduced; `tile_to=93362` → 364 steps/epoch accepted as the fix
- [ ] §4 — subset applied after `random_split`; accept that the recon manifest must be regenerated
- [ ] §5 — CPU test re-run green; decide whether it moves into the repo
- [ ] §6 — config diff shows only the six intended keys
- [ ] §7 — smoke reached `PASS`; residual risks accepted or assigned
- [ ] Approve submitting 4 × 7.7 h on `pg_mki_aryeh` (~124 H200-hours, ~16 h wall)
