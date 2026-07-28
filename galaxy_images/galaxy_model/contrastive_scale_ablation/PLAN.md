# Contrastive × Data-Scale ablation — implementation plan (DRAFT, for Pablo's review)

**What you asked for:** train `contrastive-spatial-conv1x1` at N = 1,000 and ~3k
cross-matched galaxies, in the style of `contrastive_ablation/` crossed with
`scale_ablation/`, and report downstream R² of the latent space as a function of scale.

**Status:** nothing submitted. Every claim below marked *verified* was checked against the
repo/filesystem today (2026-07-27).

**Locked by Pablo 2026-07-27:** all four rungs **1,000 / 3,162 / 10,000 / 31,622** · **one
draw** (nested, seed 42) · **`conv1x1` only** (no `avg`, no `random`) · partition
**`mit_normal_gpu`** · **R² computed automatically** as each arm lands · checkpoints on
**`/orcd/data/mki_aryeh/001`**.

Carried over from my recommendation and *not* explicitly ruled on: **D2, the matched
full-data anchor retrained on Engaging** (§3). It is included as run 0 below because
without it the curve has no matched endpoint. It costs 1.5 h — say the word and it comes
out, and we fall back to arm A with a cluster caveat.

---

## 0. The headline finding this experiment can produce

The FM scale curve is already finished (`scale_ablation/results/r2_table.md`) and it says
something the `scale_ablation/PLAN.md` §1 explicitly predicted *wrong*:

| mean R² | N=1,000 | N=3,162 | N=31,622 | full (93k) |
|---|---|---|---|---|
| physics (7) | 0.593 | 0.612 | 0.634 | 0.640 |
| Legacy instrument (5) | **0.289** | 0.390 | 0.499 | **0.611** |
| HSC instrument (2) | **0.219** | 0.282 | 0.375 | **0.478** |

Physics is nearly flat (93% of full-data R² at N=1,000); the **instrument** latent is the
one that collapses (47% of full-data R²). The plan had predicted the opposite, on the
grounds that the instrument branch feeds on pool images that aren't subsetted.

Now overlay contrastive. Contrastive's whole advantage in the rebuttal is the *instrument*
latent (Galaxy Depth 0.914 vs Ours 0.799; `RESULT_r2_table.csv`). Its instrument branch is
an InfoNCE over same-instrument neighbours drawn from the **non-subsetted pool**, so it has
a mechanistic reason to be more data-efficient exactly where FM is weakest. Two outcomes,
both publishable:

- **Contrastive's instrument curve stays flat** → "InfoNCE on same-instrument neighbours
  needs far fewer crossmatches than the generative route to recover observing conditions."
  A real, quantified limitation of Ours, and the honest version of JcH2 Q1 + Q4 together.
- **It collapses too** → the instrument signal is crossmatch-limited regardless of
  objective, which *defuses* the contrastive result at small survey overlap and directly
  answers "when does this become applicable to a new survey?"

Either way this costs ~8 GPU-hours and answers two reviewer questions with one figure.

---

## 1. The single biggest simplification — the subsets are already correct, *verified*

`scale_ablation/subsets/scale_{1000,3162,10000,31622}.json` can be reused **byte-for-byte**
by the contrastive trainer. This is not an approximation; the two codebases perform
arithmetically identical splits:

| | `make_subsets.py` | `train_contrastive_ablation.py` |
|---|---|---|
| exclusion | `_resolve_downstream_holdout_positions` logic, 5,466 anchors | same resolver, imported (`:229`) |
| kept_size | 98,275 | 98,275 |
| val | `int(kept*0.05)` = 4,913 (`:152`) | `max(1, int(kept*0.05))` = 4,913 (`:277`) |
| split | `random_split(range(98275), [93362,4913], seed 42)` | `random_split(kept_ds, [93362,4913], seed 42)` (`:279`) |

`random_split` only consumes `len(dataset)`, so both draw the *same* `randperm(98275)` under
seed 42. Verified today:

```
contrastive : 93362 4913
make_subsets: 93362 4913
val head (kept-space): [1122, 17722, 45822, 66608, 3010]
```

which maps through `kept_positions` to the manifest's recorded
`val_positions_raw_head = [1127, 17863, 47769, 71789, 3030]`. Same split.

**Consequence:** every contrastive rung trains on the *exact same galaxies* as the
corresponding FM rung. The FM-vs-contrastive comparison at each N is **paired on
galaxies**, which (per `contrastive_ablation/PLAN.md` §6) is worth roughly a 4× reduction
in the error bar on the difference. No new subset-building step is needed.

Reporting obligation carried over from `scale_ablation/PLAN.md` §1 — the table must show
both columns, because "N crossmatches" ≠ "N galaxies":

| N anchors | distinct neighbour rows | total distinct images |
|---|---|---|
| 1,000 | 4,942 | 6,942 |
| 3,162 | 15,163 | 21,487 |
| 10,000 | 44,210 | 64,210 |
| 31,622 | 113,366 | 176,610 |
| 93,362 (full train) | 235,545 | 443,027 |

---

## 2. The confound that must be neutralised — the same LR bug, worse

*Verified:* `contrastive_baseline/dual_encoder_contrastive.py:515` builds

```python
scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
```

and `train_contrastive_ablation.py` passes only `max_steps`, so `max_epochs == -1` and the
LR square-waves between `1e-4` and `0.0` with a period of one epoch — the identical
pathology documented in `scale_ablation/PLAN.md` §3, now confirmed on the contrastive side.

It is **worse here** than in the FM ablation, because contrastive runs `devices=1`:

| | steps/epoch at full data | steps/epoch at N=1,000, untiled |
|---|---|---|
| FM (4 GPU) | 364 | ~3 |
| **contrastive (1 GPU)** | **1,458** | **15** |

An untiled N=1,000 contrastive run would have a 15-on/15-off LR waveform against the
anchor's 1,458-on/1,458-off. The rung would underperform for optimizer reasons and we would
report it as "you need more crossmatches."

**Fix (same as FM, locked there): index tiling.** Cyclically repeat the chosen anchor list
to `tile_to = 93,362`, so every arm has exactly 1,458 steps/epoch, the same LR waveform, the
same number of optimizer steps, and differs only in the number of *distinct* galaxies.

Do **not** fix the scheduler — that would invalidate every existing contrastive arm as a
comparison partner.

**Parity survives tiling.** *Verified:* `neighbors_ram48.py:71` sets
`anchor_is_hsc = (idx % 2 == 0)` on the index the base dataset receives, and
`Subset.__getitem__` forwards raw positions unchanged. Since the JSONs store **raw**
positions and are 50/50 parity-balanced by construction, `Subset(dataset, tiled_raw)`
preserves each anchor's survey role. (This is why the subset must be applied against the
*base* dataset, not by filtering `anchor_indices` — see §7 gotcha 2.)

---

## 3. Protocol decisions — please confirm these four

### D1. Evaluate at a **fixed step (100k)**, not best-val — *this is a change from every existing contrastive arm*

`contrastive_ablation/RUN_INSTRUCTIONS.md` step 4 says to hand back `best-*.ckpt`, and every
published contrastive number came from best-val. But best-val selection is *itself
data-scale dependent* — arm A's best landed at **step 70,000**, not 100,000 (*verified*:
`best-epoch=48-step=70000.ckpt`). At N=1,000 the val curve will turn over much earlier, so a
best-val curve would confound "less data" with "earlier stopping".

`scale_ablation/PLAN.md` §4 already locked fixed-step for exactly this reason. Adopt it here.

**This is free for the anchor:** arm A's directory also contains
`latest-step=100000.ckpt` (*verified*, 269 MB), so it can be re-evaluated at fixed step in
one ~15-min job. Report best-val as a secondary row if you want continuity with the
published table.

### D2. Retrain a **matched full-data anchor on Engaging** (+1 run, ~1.5 h)

Neither existing full-data checkpoint is a matched top-of-curve:

| candidate | holdout? | train_size | cluster | verdict |
|---|---|---|---|---|
| `contrastive-spatial-conv1x1` (published) | no | 98,554 | Engaging L40S | wrong split → subsets don't apply |
| `contrastive-spatial-conv1x1-DWNVAL` (arm A) | yes | 93,362 | **AMD MI210** | right split, wrong hardware |

*Verified* from arm A's `DONE`: `loader=ram48`,
`data_dir=/work1/jeroenaudenaert/pablomer/data/efficient_neighs_48`.

The rungs will train on Engaging, so using arm A as the endpoint puts a cluster change at
the top of the curve. The DWNVAL comparison bounds the combined cluster+holdout effect at
mean +0.006 (stdev 0.030), i.e. within fold noise — but at 1.5 h/run it is not worth
defending. Retrain the full-data point on Engaging, with **no subset flag at all** so it
takes the byte-unchanged code path.

Keep arm A in the figure as a cross-check; if the two full-data points agree, that is a free
cluster-portability result.

### D3. Ladder — all four rungs — **LOCKED**

**1,000 / 3,162 / 10,000 / 31,622**, one nested seed-42 draw, `conv1x1` only. The subsets
already exist and each run is ~1.5 h. Four rungs plus the anchor spans 10³ → 10⁴·⁵ → 93k,
which is enough to show saturation — the thing JcH2 Q4 actually asks for.

### D4. Steps = 100,000, `devices=1`, `batch_size=64` — match the contrastive protocol, not the FM one

The FM ablation used 75k steps × 4 GPUs (effective batch 256). Do **not** copy that: the
free/cheap anchor here is a contrastive run, and every published contrastive number is
100k × 1 GPU × batch 64. Matching the contrastive protocol is what makes the curve
comparable to `RESULT_r2_table.csv`. The FM curve remains comparable *at the level of the
scale axis* (same galaxies at each N), which is what the overlay needs.

---

## 4. Code changes — one file, ~25 lines

Unlike the FM path (which needed `config.py` + `data_factory.py`), the contrastive trainer
owns its own data plumbing, so this is a single additive edit.

### `train_contrastive_ablation.py` — two new flags

```python
p.add_argument("--train-subset-json", default=None,
               help="scale_ablation/subsets/scale_<N>.json: RAW anchor positions to keep "
                    "as the training set. Val is untouched.")
p.add_argument("--train-subset-tile-to", type=int, default=None,
               help="Cyclically repeat the subset to this length so steps/epoch (and thus "
                    "the T_max=-1 LR waveform) is identical at every scale. "
                    "Must equal train_size; defaults to it.")
```

### Insert after `random_split` (currently `:279-282`), before `steps_per_epoch`

```python
n_distinct = None
if args.train_subset_json:
    import json, numpy as np
    spec = json.load(open(args.train_subset_json))

    # Guard 1: the JSON was built under a specific exclusion regime. Mixing them
    # silently shifts the split and would put training galaxies in val.
    if bool(spec["holdout_applied"]) != bool(args.holdout_ids_txt):
        raise SystemExit(f"[FATAL] subset holdout_applied={spec['holdout_applied']} but "
                         f"this run holdout={'yes' if args.holdout_ids_txt else 'no'}")
    # Guard 2: tile length must match this run's train_size, else the LR waveform differs.
    tile_to = args.train_subset_tile_to or int(spec["tile_to"])
    if tile_to != train_size:
        raise SystemExit(f"[FATAL] tile_to={tile_to:,} != train_size={train_size:,}")

    kept_arr  = np.asarray(kept_positions)
    train_raw = kept_arr[np.asarray(train_ds.indices)]      # raw positions in the train split
    chosen    = np.asarray(spec["positions"], dtype=np.int64)

    # Guard 3: no chosen anchor may come from val (the common val set must stay untouched).
    missing = set(chosen.tolist()) - set(train_raw.tolist())
    if missing:
        raise SystemExit(f"[FATAL] {len(missing):,} subset positions are not in the train "
                         f"split (first 5: {sorted(missing)[:5]})")

    tiled = np.resize(chosen, tile_to)                       # cyclic repeat, deterministic
    train_ds = Subset(dataset, tiled.tolist())               # BASE dataset -> parity preserved
    n_distinct = len(chosen)
    n_even = int((chosen % 2 == 0).sum())
    print(f"[subset] {args.train_subset_json}: distinct={n_distinct:,} "
          f"(HSC-role {n_even:,} / Legacy-role {n_distinct - n_even:,}) "
          f"tiled={len(tiled):,} repeats={len(tiled)/n_distinct:.1f}x", flush=True)

train_size = len(train_ds)
```

Also append `n_distinct_anchors` / `tile_to` / `train_subset_json` to the `DONE` marker and
to the wandb config, so a checkpoint is self-documenting.

**Nothing else changes.** `dual_encoder_contrastive_variants.py`, the model, the optimizer,
the loss and the val loader are untouched, so a run with no `--train-subset-json` is
byte-identical to arm A's code path (which is what D2 relies on).

### New files under `contrastive_scale_ablation/`

```
contrastive_scale_ablation/
├── PLAN.md                        # this doc
├── PROGRESS.md                    # running log (the diffusion/scale ablations' discipline)
├── train_contrastive_scale.slurm  # clone of contrastive_baseline/train_contrastive_engaging.slurm,
│                                  #   parameterised by SUBSET_JSON + RUN_TAG, self-chaining
├── eval_contrastive_scale.slurm   # clone of contrastive_ablation/eval_ablation_engaging.slurm
│                                  #   with CKPT pinned to latest-step=100000 and the
│                                  #   DWNVAL-specific compare/plot steps dropped
└── report_contrastive_scale.py    # imports report_scale_results.py helpers (see below)
```

`report_contrastive_scale.py` must **import** `_load_predict`, `_band_average` and
`_column_for` from `scale_ablation/report_scale_results.py` rather than reimplement them.
If the two curves are averaged by different conventions the overlay is meaningless, and
that script's convention is already self-checked against the paper's `merged_simplified.csv`
(max |Δ| = 5.6e-17). Its `ARMS` dict is hardcoded (`:52`), so parameterise it via a
`--arms` flag rather than editing it in place.

---

## 5. Run queue

All on Engaging. Data: `/orcd/pool/007/pablomer/efficient_neighs_48` (*verified* present,
11 GB on disk / 17.4 GB in RAM), `--loader ram48`, `--mem=64G`.

**Partition:** `mit_normal_gpu` (L40S, 6 h wall, self-chaining already implemented) —
**LOCKED**. This also keeps `pg_mki_aryeh`'s 8 H200s free for your `cf-fm-long`
counterfactual job, which is currently queued behind `cf-precompute` there.

| # | arm | `RUN_TAG` | subset | notes |
|---|---|---|---|---|
| 0 | full-data matched anchor | `scale-full` | *(none)* | D2; byte-unchanged code path |
| 1 | **N = 1,000** | `scale-1000` | `scale_1000.json` | you asked for this |
| 2 | **N = 3,162** | `scale-3162` | `scale_3162.json` | you asked for this |
| 3 | N = 10,000 | `scale-10000` | `scale_10000.json` | D3, recommended |
| 4 | N = 31,622 | `scale-31622` | `scale_31622.json` | D3, recommended |

```bash
sbatch --export=ALL,RUN_TAG=scale-1000,\
SUBSET_JSON=galaxy_images/galaxy_model/scale_ablation/subsets/scale_1000.json \
  galaxy_images/galaxy_model/contrastive_scale_ablation/train_contrastive_scale.slurm
```

### Storage — `/orcd/data/mki_aryeh/001`, benchmarked — **LOCKED**

```
CKPT_ROOT=/orcd/data/mki_aryeh/001/pablomer/contrastive_scale_ablation/contrastive-<RUN_TAG>/
```

One parent dir per arm — `prepare_combined.py:318` takes the CSV `checkpoint` column from
the parent directory name, so co-locating arms makes that column useless.

**Is it slower? Marginally on writes, not enough to matter here.** Both volumes are NFSv4
with *identical* mount tuning (`rsize=wsize=1M, hard, tcp, timeo=600`); they differ only in
backing server and how full they are. Measured today with checkpoint-sized (269 MB) I/O,
two passes each:

| | write | read |
|---|---|---|
| POOL `/orcd/pool/007` (hstor012, **91% full**, 94 GB left) | 937 MB/s → 1.2 GB/s | 1.3 GB/s |
| DATA `/orcd/data/mki_aryeh/001` (hstor003, **1% full**, 200 TB) | 262 MB/s → 668 MB/s | 1.0 → 1.2 GB/s |

Reads are equivalent. Writes are ~2× slower cold and ~1.8× slower warm, which in absolute
terms is **1.1 s vs 0.3 s per checkpoint** — irrelevant against a 1.5 h run. Use DATA:
POOL at 91% is the actual risk, and near-full filesystems degrade on allocation anyway.
*(It is group storage, setgid, shared with `twirl`/tehan — your files will be
group-readable. Flagging, not objecting.)*

Leave the **data store on POOL** (`/orcd/pool/007/pablomer/efficient_neighs_48`). It's read
sequentially into RAM once per job (~1 min) and reads are equivalent, so copying 17 GB
across buys nothing.

### ⚠️ Raise `every_n_train_steps` 500 → 5000 — found while benchmarking

`train_contrastive_ablation.py:361` checkpoints every **500** steps. That was written for a
slower setup; at the measured ~37 it/s a 500-step interval is **one 269 MB write every
~13 s**, i.e. ~200 writes and **~54 GB of write traffic per run**, ~270 GB across five runs,
against a shared NFS server. That is both antisocial and a measurable slice of a 1.5 h run.

At 5000 steps it becomes 20 writes / 5.4 GB per run, and resume-on-preemption loses ≤ ~2 min.
**Safe for this experiment:** cadence never touches the weights, and 100,000 is a multiple of
5000, so `latest-step=100000.ckpt` — the file D1 evaluates — is still written exactly.

### Budget — measured, not estimated

*Verified* via `sacct` on the published Engaging arm: 100k steps took **00:32:10 + 00:53:05**
across a cancel/resume pair on 1× L40S ⇒ **~1.5 h/run**.

| item | cost |
|---|---|
| 5 training runs × 100k steps | ~7.5 GPU-hours |
| wall clock at the 2-GPU QOS cap | **~5 h** (3 waves; evals interleave) |
| 5 downstream evals + 5 meanpool twins | ~1.5 h wall, ~15 min each |
| disk on DATA | 5 × ~0.8 GB (best + latest + last @ 269 MB) ≈ **4 GB** of 200 TB |
| write traffic (after the 5000-step fix) | ~27 GB total, down from ~270 GB |

**SCRATCH is hard-full (1024/1024)** — *verified* from `~/orcd/.quota`. Every path must be
absolute; the default `shared_checkpoint_dir` resolves to
`galaxy_images/galaxy_model/checkpoints`, which is a **symlink into scratch** and will fail.

---

## 6. Evaluation — fully automatic, R² lands without you touching anything

**The chain, per arm.** `train_contrastive_scale.slurm` already has the hook: the
`contrastive_baseline` launcher it clones "launches the downstream eval job" once the `DONE`
marker appears. So each arm self-drives:

```
train (100k steps, self-chaining on the 6h wall)
  └─ DONE ──> sbatch eval_contrastive_scale.slurm RUN_TAG=<arm>
                ├─ freeze latest-step=100000.ckpt -> frozen/   (gotcha 4)
                ├─ prepare_combined.py            -> <arm>.h5
                ├─ predict_combined.py            -> predict_contrastive-<arm>__aion.csv     (flat 128-D)
                ├─ build_meanpool_h5.py --tokens 4-> <arm>-meanpool.h5
                ├─ predict_combined.py            -> predict_contrastive-<arm>-meanpool__aion.csv  (pooled 32-D)
                └─ report_contrastive_scale.py --arms <all CSVs present so far>
                     ├─ results/r2_table.md      (regenerated from scratch each time)
                     ├─ results/r2_per_target.csv
                     ├─ results/r2_vs_n.png      (contrastive + FM overlaid)
                     └─ Discord ping with the current table
```

`report_contrastive_scale.py` regenerates the **whole** table from whatever
`predict_contrastive-scale-*__aion.csv` files exist, so it is safe to run after every arm —
the table simply fills in column by column as the five runs land, and the last one to finish
produces the final figure. No manual step, no ordering requirement.

Both latents are needed: the FM scale table reports **physics rows from the flat latent and
instrument rows from the meanpool twin**, so the overlay needs the matching contrastive
twin or the comparison is not like-for-like. The meanpool build is a CPU-only post-process
on the H5 — no re-encoding.

Protocol is byte-identical to every other contrastive arm (`--mlp-arch aion --seed 0`,
n=5,469, 78 rows per CSV).

Off-diagonals come free in every CSV (`predict_combined.py:523-525`), so the
disentanglement-vs-scale panel costs nothing.

Off-diagonals come free in every CSV (`predict_combined.py:523-525`), so the
disentanglement-vs-scale panel costs nothing.

⚠️ Use the `task_family` column for grouping, never the prefix-based group assignment in
`plot_aion_vs_variant.py` — it silently drops all four ellipticity targets
(`scale_ablation/PLAN.md` §7).

### Deliverables

1. **One figure** — mean R² vs N (log x), two panels (physics latent on physics props;
   instrument latent on instrument props), **contrastive and FM overlaid** on the same axes,
   with the `untrained-resnet-matched` floor as a horizontal band.
2. **One table** — rows = N (+ distinct-images column), columns = mean physics R² /
   mean Legacy-instr R² / mean HSC-instr R², contrastive and FM side by side.
3. **The paired ΔR² curve** — (contrastive − FM) at each N, on identical galaxies. This is
   the quantity with the small error bar, and the one that answers whether the contrastive
   instrument advantage is a large-data artefact.
4. **One paragraph** for the rebuttal, joining JcH2 Q1 and Q4.

### Explicitly out of scope

Reconstruction MSE and the counterfactual axis. Contrastive has no decoder — that is the
point of `contrastive_ablation/PLAN.md` exp H, and it is a separate experiment.

---

## 7. Gotchas to bake in

1. **`monitor="val/loss"` is required** — `ModelCheckpoint` already uses it here, but with
   `--no-wandb` the `val/mse`-style metrics logged inside `on_validation_epoch_end`
   early-return with no logger. Keep wandb on, or keep `val/loss`.
2. **Apply the subset to the BASE dataset, never by filtering `anchor_indices`.** Filtering
   renumbers positions and flips roughly half the anchors' survey role
   (`neighbors_ram48.py:71`).
3. **Glob checkpoints as `latest-step*.ckpt`**; `auto_insert_metric_name=False` here so the
   name is clean (`latest-step=100000.ckpt`), unlike the FM runs' doubled `step=step=`.
4. **Freeze the checkpoint before evaluating it.** `ModelCheckpoint` with `save_top_k=1`
   deletes the previous file and has produced `Errno 116 Stale file handle` mid-eval before.
   Copy to a `frozen/` path per arm first.
5. **CRLF → LF** on any new slurm file; `sbatch` rejects DOS line endings.
6. **Ping the Discord webhook** on submission and completion — `train_ablation_amd.slurm`
   has the `_ping` + EXIT-trap pattern to clone.
7. **`--enable-umap`**: default is 1 and arm A used it. Keep it on for all five new runs so
   the anchor stays comparable; turning it off is a legitimate speedup only if applied to
   every arm including the anchor.
8. **Change `CKPT_DIR`'s default in the eval clone.**
   `eval_ablation_engaging.slurm:46` hardcodes
   `CKPT_DIR=/orcd/pool/007/pablomer/checkpoints_new/${VAR}`. With checkpoints now on
   `/orcd/data/mki_aryeh/001` the clone must repoint it, or the eval silently looks in the
   wrong place (or worse, finds a stale same-named arm).
9. **`CKPT` must be pinned to `latest-step=100000.ckpt`, not globbed.** Per D1 we evaluate at
   fixed step; a glob would pick up `best-*.ckpt` and reintroduce data-scale-dependent early
   stopping.
10. **Keep `PROGRESS.md` current.** `diffusion_ablation/PROGRESS.md` went stale and is now
    actively misleading.

---

## 8. Pre-flight checks before any submission

1. `python contrastive_ablation/preflight_holdout.py --loader ram48 --data-dir
   /orcd/pool/007/pablomer/efficient_neighs_48 --holdout-ids-txt …` → must print
   `kept=98,275 train=93,362 val=4,913 steps/epoch=1,458` and `PREFLIGHT OK`.
   **If `steps/epoch` is not 1,458, stop** — the subsets no longer apply.
2. Smoke test, 50 steps, N=1,000:
   ```
   --train-subset-json …/scale_1000.json --max-steps 50 --limit-val-batches 2 --no-wandb
   ```
   Log must show `[subset] distinct=1,000 (HSC-role 500 / Legacy-role 500) tiled=93,362
   repeats=93.4x` and `steps/epoch=1,458` (**not** 15). Loss must fall from ≈8.3 (= 2 ln 64).
3. Assert the tiled subset is disjoint from the val set (Guard 3 does this at runtime; run it
   once for all four JSONs on CPU first).
4. Re-evaluate arm A at `latest-step=100000.ckpt` and confirm it reproduces
   `RESULT_r2_table.csv`'s `A_DWNVAL` column to within probe noise. If fixed-step and
   best-val differ materially, that difference is itself worth a line in the writeup.

---

## 9. Decisions — resolved 2026-07-27

| # | question | decision |
|---|---|---|
| D3 | two rungs or four? | **four** — 1,000 / 3,162 / 10,000 / 31,622 |
| — | draws per rung | **one**, nested, seed 42 |
| — | `encoder_pool` variants | **`conv1x1` only**; no `avg`, no `random-neighbours` |
| — | partition | **`mit_normal_gpu`** (keeps the H200s free for `cf-fm-long`) |
| — | R² reporting | **automatic**, chained off each arm's `DONE` (§6) |
| — | checkpoint storage | **`/orcd/data/mki_aryeh/001`** — benchmarked, ~1.8× slower writes, irrelevant at 269 MB (§5) |
| D1 | best-val or fixed step? | **fixed step 100,000** — best-val is data-scale dependent |
| D4 | training protocol | **100k × 1 GPU × batch 64**, matching the contrastive arms not the FM ones |
| — | checkpoint cadence | **500 → 5000** steps; ~270 GB → ~27 GB of write traffic (§5) |

**Still open — one item, and it does not block submission:**

**D2 — retrain the matched full-data anchor on Engaging (+1.5 h)?** Included as run 0 on my
recommendation. Arm A is the only existing holdout-matched full-data contrastive
checkpoint and it was trained on **AMD MI210** (*verified* from its `DONE` marker), so
using it as the endpoint puts a cluster change at the top of the curve. The bound on that
effect is mean +0.006 / stdev 0.030 — within fold noise, but not worth defending in a
rebuttal for 1.5 h of L40S time. Drop run 0 if you disagree; nothing else changes, and arm
A gets re-evaluated at fixed step either way as a cross-check.
