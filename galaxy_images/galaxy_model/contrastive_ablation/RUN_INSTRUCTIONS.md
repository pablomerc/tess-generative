# Contrastive ablation runs — instructions for the training-cluster agent

**Split of work: you train, Engaging evaluates.** The downstream evaluation needs
image stores that only exist on Engaging, so your job ends when the checkpoint is
trained and copied back. Everything you need is committed on branch
`galaxy-engaging`.

**There are four arms to run (A-D).** They share one launcher; defaults reproduce the
published `contrastive-spatial-conv1x1` arm, so each override isolates one variable.
`PLAN.md` explains why each exists and what we expect. Arm A is described in detail
below; B-D are the same procedure with different env vars (see "Other arms").

## What arm A is

The paper's contrastive ablation (reviewer JcH2 Q1) trained on *all* galaxies,
including the 5,469 used for the downstream R² probes. This run repeats the
Ours-matched contrastive arm with those galaxies **excluded from SSL training**, so
we can show that InfoNCE isn't getting its strong numbers by memorising probe
galaxies. Background: `contrastive_ablation/ANALYSIS.md`.

It must differ from the existing `contrastive-spatial-conv1x1` run in **exactly one
respect** — the holdout. Do not "improve" hyperparameters, the model class, the
optimizer, or the LR schedule, even if something looks wrong (the repo-wide
`T_max=-1` cosine quirk in CLAUDE.md is deliberate here: it applies identically to
both arms and so cannot confound the pair).

## Step 0 — sync the repo

```bash
cd /work1/jeroenaudenaert/pablomer/tess-generative
git checkout galaxy-engaging
git pull                      # brings in galaxy_images/galaxy_model/contrastive_ablation/
```

Per CLAUDE.md: **never force-push a shared branch.** If you need to push work back,
use a snapshot branch (`amd-snapshot`) and let Pablo merge.

## Step 1 — preflight (~1 min CPU, do not skip)

This catches the one silent failure: if `object_id_legacy` is encoded differently
in this cluster's catalog, the holdout list matches 0 anchors and you train a
*non*-holdout run that looks perfectly healthy.

It also benchmarks the data loader, which matters here (see step 1b).

```bash
conda activate torchenv
export PYTHONPATH=/work1/jeroenaudenaert/pablomer/tess-generative:$PYTHONPATH
export TORCH_BLAS_PREFER_HIPBLASLT=0

python galaxy_images/galaxy_model/contrastive_ablation/preflight_holdout.py \
  --loader efficient \
  --data-dir /work1/jeroenaudenaert/pablomer/data/neighbors_efficient \
  --holdout-ids-txt galaxy_images/galaxy_model/contrastive_ablation/holdout_legacy_ids.txt \
  --smoke --benchmark 30 --bench-workers 8
```

**Expected output** (verified on the Engaging copy of the same store):

```
[1/5] store   rows=468,197 (expect 468,197) anchors=103,741 (expect 103,741)
[2/5] holdout ids_in_file=5,361  matched_anchors=5,466  (102.0% of ids)
[3/5] parity  excluded HSC-anchor=2,736 Legacy-anchor=2,730 (HSC fraction 0.501, ...)
[4/5] epochs  kept=98,275 train=93,362 val=4,913 steps/epoch=1,458 ... shift -5.3%
[5/5] exclusion verified: 0 of 5,466 holdout positions in kept subset
PREFLIGHT OK
```

`matched_anchors=5,466` exceeding the 5,361 ids is correct — a few legacy objects
appear in more than one anchor row.

**If numbers differ:** stop and report, don't work around it.
- `matched_anchors` near 0 → id-encoding mismatch. Compare
  `pd.read_parquet(catalog)['object_id_legacy'].head()` against the first lines of
  `holdout_legacy_ids.txt`; the resolver already strips a `b'...'` wrapper, so
  anything else is new.
- row/anchor counts differ → this store is not the one the published runs used; the
  A/B is invalid until that's reconciled.

## Step 1b — read the benchmark before committing a GPU

**This is the one number that decides whether the run is feasible with `efficient`.**
The 160×160 store is random-read bound: on Engaging's Ceph it sustains only
~0.35 batch/s, i.e. **~78h for 100k steps**, which is exactly why the Engaging arm
was trained from the prebaked `ram48` store instead. Whether `/work1` is faster is a
property of your filesystem, so measure it — I can't from here.

The preflight prints:

```
[6/6] bench   30 batches in Xs with 8 workers => Y batch/s (Z ms/batch)
        projected data-only wall clock for 100,000 steps: Hh (excludes GPU compute)
```

- **≥ 1.5 batch/s (< ~19h projected)** → proceed with `efficient` (step 2 as written).
- **< 1.5 batch/s** → the preflight warns. Build the prebaked store once, then train
  from it (needs ~17 GB of `/work1` — check `~/orcd/.quota` or the AMD equivalent first):

  ```bash
  # build_ram48.py now takes a CLI (it used to hardcode Engaging paths).
  # Sequential read of the source store, so it's fast even where random reads are slow.
  srun --partition=mi2101x --cpus-per-task=16 --time=02:00:00 \
    python galaxy_images/galaxy_model/contrastive_baseline/build_ram48.py \
      --src /work1/jeroenaudenaert/pablomer/data/neighbors_efficient \
      --out /work1/jeroenaudenaert/pablomer/data/efficient_neighs_48 \
      --workers 16
  # expect "[build] DONE 468197 rows" then "BUILD_RAM48_OK"

  # re-run preflight against the new store, then train from it
  sbatch --export=ALL,LOADER=ram48,DATA_DIR=/work1/jeroenaudenaert/pablomer/data/efficient_neighs_48 \
    --mem=64G galaxy_images/galaxy_model/contrastive_ablation/train_ablation_amd.slurm
  ```

  `ram48` also makes this arm match the Engaging arm's loader exactly, so it's the
  better outcome anyway — the two loaders otherwise differ only by a float16
  storage round-trip.

## Step 2 — submit training

```bash
sbatch galaxy_images/galaxy_model/contrastive_ablation/train_ablation_amd.slurm
```

Defaults: `RUN_TAG=spatial-conv1x1-DWNVAL`, `ENCODER_POOL=conv1x1`,
`RANDOM_NEIGHBORS=0`, `MAX_STEPS=100000`, `LOADER=efficient`, partition `mi2101x`
(1×MI210, 12h), checkpoints to
`/work1/jeroenaudenaert/pablomer/outputs/contrastive_ablation/contrastive-spatial-conv1x1-DWNVAL/`.

The job **self-chains**: it queues an `afterany` successor before training, so a
wall-limit kill resumes from `last.ckpt`. It re-runs preflight on attempt 1 and
refuses to train if the holdout doesn't resolve. It stops when `DONE` appears.
Reference: the same 100k steps took <6h on one Engaging GPU, so expect roughly
1–2 MI210 links.

Useful overrides:

```bash
# congested queue -> 4-GPU partition, 24h wall (the run still uses devices=1)
sbatch --partition=mi2104x --time=24:00:00 \
  galaxy_images/galaxy_model/contrastive_ablation/train_ablation_amd.slurm

# progress pings (optional; no secret is committed, pass your own webhook)
sbatch --export=ALL,DISCORD_WEBHOOK='https://discord.com/api/webhooks/...' \
  galaxy_images/galaxy_model/contrastive_ablation/train_ablation_amd.slurm
```

`WANDB_MODE` is chosen automatically: online if `WANDB_API_KEY` or a wandb
`~/.netrc` entry exists, else offline. Never let logging fail the run.

### Why `efficient` is the default

It reads the 160×160 mmap store and preprocesses on the fly — the same loader the
existing `*-DWNVAL` flow-matching runs used on this cluster, so it needs no new
build step. Switch to `ram48` only if step 1b's benchmark says to (it also needs a
`--mem=64G` override, since the store is loaded into RAM).

## Step 3 — sanity-check the first log

Within the first few minutes the log should show:

```
[downstream-holdout] holdout list: 5,361 unique legacy IDs | matched anchors: 5,466 / 103,741
[data] total anchors=103,741 | excluded=5,466 (HSC-anchor 2,736 / Legacy-anchor 2,730) | kept=98,275 ...
[data] kept=98,275  train=93,362  val=4,913  steps/epoch=1,458
[cfg] encoder_pool=conv1x1 random_neighbors=False loader=efficient max_steps=100000 batch_size=64
```

Then `train/loss` should fall from ≈8.3 (2 × ln 64 chance) toward ≈0.3–0.8. If it
plateaus near 4.2, only one InfoNCE branch is learning — report it.

## Step 4 — hand the checkpoint back

When `DONE` exists:

```bash
CK=/work1/jeroenaudenaert/pablomer/outputs/contrastive_ablation/contrastive-spatial-conv1x1-DWNVAL
ls -la "$CK"          # expect best-epoch=*-step=*.ckpt, last.ckpt, DONE, attempts
cat "$CK/DONE"        # records global_step + excluded/kept anchor counts
```

Copy **`best-*.ckpt` and `DONE`** (the best-val checkpoint is what every other arm
was evaluated from) to Engaging at:

```
/orcd/pool/007/pablomer/checkpoints_new/contrastive-spatial-conv1x1-DWNVAL/
```

Then report back: final `global_step`, the `[data]` line, final train/val loss,
best-checkpoint filename, and the path you copied to. **Do not run the downstream
eval** — `prepare_combined.py` needs `/home/pablomer/orcd/scratch/hsc_downstream`
and `legacy_downstream_full`, which don't exist on your cluster.

## What happens back on Engaging (for context, not for you)

```bash
sbatch --export=ALL,RUN_TAG=spatial-conv1x1-DWNVAL \
  galaxy_images/galaxy_model/contrastive_ablation/eval_ablation_engaging.slurm
```

Extracts frozen embeddings on the fixed n=5,469 sample → `aion` MLP probes,
seed 0 → `compare_dwnval.py` writes `DWNVAL_RESULT.md` with the per-cell ΔR²
against `contrastive-spatial-conv1x1` → overlays both arms on the paper figure.
The result we're after: mean |ΔR²| at or below the flow-matching arm's already
measured ±0.005, which would show contamination explains none of the contrastive
baseline's R² pattern.

## Files in this directory

| file | role |
|---|---|
| `ANALYSIS.md` | why this run exists; full audit of the contrastive ablation |
| `preflight_holdout.py` | 5 checks + loader smoke test; run before submitting |
| `PLAN.md` | the full experiment queue and what each arm tests |
| `train_contrastive_ablation.py` | trainer; cluster-agnostic, holdout + negative-sampling + head axes |
| `dual_encoder_contrastive_variants.py` | the loss variants (delegates to the baseline when unrestricted) |
| `test_negative_variants.py` | fast CPU tests; asserts default == published baseline |
| `train_ablation_amd.slurm` | AMD launcher (self-chaining, runs preflight, MI210 workaround) |
| `holdout_legacy_ids.txt` | the 5,361 legacy ids to exclude (tracked so both clusters share one list) |
| `eval_ablation_engaging.slurm` | Engaging-only downstream eval (do not run on AMD) |
| `compare_dwnval.py` | ΔR² report vs the non-holdout arm → `DWNVAL_RESULT.md` |

## Other arms (B-D)

Same procedure as arm A: preflight once for the store (it is arm-independent), then
submit. All arms hold out the downstream-eval galaxies (the default), so the set is
internally consistent -- each of B/C/D differs from arm A in exactly one axis, and no
arm can be dismissed on contamination grounds.

```bash
# C -- instrument negatives restricted to the anchor's own survey (highest value)
sbatch --export=ALL,RUN_TAG=instrneg-samesurvey-DWNVAL,INSTRUMENT_NEGATIVES=same_survey \
  galaxy_images/galaxy_model/contrastive_ablation/train_ablation_amd.slurm

# B -- galaxy negatives restricted to the positive's survey
sbatch --export=ALL,RUN_TAG=galneg-samesurvey-DWNVAL,GALAXY_NEGATIVES=same_survey \
  galaxy_images/galaxy_model/contrastive_ablation/train_ablation_amd.slurm

# D -- no SimCLR head: InfoNCE directly on the representation the probes read
sbatch --export=ALL,RUN_TAG=nohead-DWNVAL,PROJECTION_HEAD=0 \
  galaxy_images/galaxy_model/contrastive_ablation/train_ablation_amd.slurm
```

Sanity line to expect in each log (it records the axes actually in force):

```
[cfg] galaxy_negatives=mixed instrument_negatives=same_survey projection_head=True holdout=yes
```

For B and C, if the result moves, also run the negative-COUNT control at double batch
size so "harder negatives" is separated from "fewer negatives":

```bash
sbatch --export=ALL,RUN_TAG=instrneg-samesurvey-DWNVAL-b128,INSTRUMENT_NEGATIVES=same_survey,BATCH_SIZE=128 \
  galaxy_images/galaxy_model/contrastive_ablation/train_ablation_amd.slurm
```

Before running anything, you can verify the loss variants are wired correctly without
a GPU (a few seconds, CPU only) -- it asserts that the default configuration
reproduces the published baseline's loss exactly:

```bash
python galaxy_images/galaxy_model/contrastive_ablation/test_negative_variants.py
# expect: ALL CHECKS PASSED
```
