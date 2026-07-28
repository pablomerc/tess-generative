# Arm I — contrastive base (conv1x1) at 10x batch size

**Being run on Engaging** (`pg_mki_aryeh`, 1x H200) via
`train_ablation_engaging.slurm` — see "Submitting on Engaging" below, which is the live
recipe. Training and evaluation are therefore on the same cluster and no checkpoint copy
is needed. The AMD recipe (`train_ablation_amd.slurm`, steps 0-4 further down) is kept
because it is the cluster arms A-D were trained on, so it stays the reproduction path.

## The batch size in the previous runs was 64

Every published contrastive arm and every DWNVAL arm (A-D) trained at **batch 64**
(`train_contrastive_ablation.py::BATCH_SIZE`, `train_ablation_amd.slurm::BATCH_SIZE`;
confirmed in `contrastive_ablation_checkpoints/contrastive-spatial-conv1x1-DWNVAL/`,
100k steps = 6.4M images). 10x is therefore **640**.

## Why this arm exists

In InfoNCE the batch *is* the negative pool, so batch size is a hyperparameter of the
objective, not just of the optimizer:

| | batch 64 (arms A-D) | batch 640 (arm I) | Ours (`base`) |
|---|---|---|---|
| galaxy negatives / anchor | 63 | **639** | n/a (no contrastive loss) |
| instrument negatives / anchor | ~315 | **~3,195** | n/a |
| global batch | 64 | **640** | 256 (64 x 4 GPUs) |
| images seen at 100k steps | 6.4M | **64M** | 22.3M (87k best-val step) |

Two objections to the current contrastive numbers are answered by one run:

1. **"Your contrastive baseline was starved of negatives."** 63 negatives is small for
   InfoNCE, and every cell where contrastive loses (all spectroscopic physics
   properties, HSC-resolution shapes) is vulnerable to it.
2. **"Your contrastive baseline had a 4x smaller batch than your own model."** Ours
   trained at global batch 256. At 640 the contrastive arm has 2.5x Ours' batch, so the
   comparison can no longer be read as favouring Ours on batch size.

It also directly probes §3 of `PLAN.md` from the other side: if the instrument
advantage is substantially a *survey classifier*, adding 10x more (still ~50/50
cross-survey) negatives should not hurt it — whereas physics R², which needs genuine
instance discrimination, is where extra negatives usually help.

## What is held fixed

Everything except batch size: `lr=1e-4`, temperature 0.1, AdamW wd 1e-4, embedding 64 /
projection 32, `conv1x1` pool, K=5 real sky neighbours, bf16-mixed, 1 GPU, seed 42, 95/5
split, `ram48` loader, and the DWNVAL holdout (`HOLDOUT=1`). The reference arm is
therefore **A** (`contrastive-spatial-conv1x1-DWNVAL`), which differs in exactly one axis.

Deliberately *not* scaled: the LR. Linear/sqrt scaling would make this a two-variable
run. If a reviewer asks for a *tuned* large-batch baseline rather than a controlled one,
that is a second submission — `LR=3.16e-4` (sqrt scaling) — not an edit to this one.

Two things batch size drags along with it, both inherent to the axis rather than fixable:
**BatchNorm** batch statistics are now estimated over 640 images per encoder forward
(3,200 for the instrument encoder's neighbour pass, `B*K`) instead of 64 / 320, and the
**absolute InfoNCE loss** shifts because chance is `2*ln(B)`. So "batch 640 reached a
lower loss" is not a statement about model quality — read `train/acc_*` and downstream R²,
never the raw loss, across these two arms.

Known and accepted, same as every other arm: the `T_max=-1` cosine quirk (CLAUDE.md)
alternates LR between 1e-4 and 0 per epoch. Batch 640 shortens an epoch from 1,458 to
145 steps, so the alternation *period* changes while the dead *fraction* stays ~50%
(arm A: 68.6 epochs over 100k steps; arm I: 690). The trainer prints both so the numbers
are in the log rather than in an argument.

## Three evaluable checkpoints from one run

"Bigger batch" has two defensible conventions and we don't have to choose — permanent
snapshots are written at the steps where the *image* budget matches an existing run
(`--snapshot-steps`, needed because the rolling `latest-step=*` checkpoint is deleted as
soon as the next one lands):

| checkpoint | steps | images | matched to | reads as |
|---|---|---|---|---|
| `snapshot-step=10000.ckpt` | 10k | 6.4M | arm A exactly | 10x negatives at an **equal image budget** (10x fewer updates) |
| `snapshot-step=35000.ckpt` | 35k | 22.4M | Ours (`base`, 22.3M) | the **compute-matched** contrastive arm `ANALYSIS.md` item 10 asks for |
| `best-*.ckpt` | ≤100k | ≤64M | nothing — strictly more | 10x negatives at **equal optimizer steps**; a null result here cannot be blamed on undertraining |

The headline arm is the last one. The other two exist so that "you gave contrastive 10x
the compute" and "you undertrained it" cannot both be raised.

## Submitting on Engaging (the live run)

```bash
cd /orcd/pool/007/pablomer/tess-generative
sbatch --export=ALL,RUN_TAG=spatial-conv1x1-BS640-DWNVAL,BATCH_SIZE=640,\
SNAPSHOT_STEPS=10000:35000,NUM_UMAP_BATCHES=1 \
  galaxy_images/galaxy_model/contrastive_ablation/train_ablation_engaging.slurm
```

Everything else — `ram48` loader, the `HOLDOUT=1` DWNVAL exclusion, `lr=1e-4`,
`MAX_STEPS=100000`, 1 GPU, 16 workers, 96 GB, 48h — is the launcher's default, and the
`#SBATCH` block already carries the right resources so the self-chained successor
inherits them (it re-passes `--gres`, which is the one that would otherwise strand a
resumed link on a CPU node).

Where things land:

| what | path |
|---|---|
| checkpoints | `/orcd/data/mki_aryeh/001/pablomer/contrastive_ablation/contrastive-spatial-conv1x1-BS640-DWNVAL/` (200 TB group allocation — POOL was at 92%) |
| eval-visible alias | `checkpoints_new/contrastive-spatial-conv1x1-BS640-DWNVAL` → symlink to the above, created by the launcher |
| logs | `contrastive_ablation/logs/contr-abl-eng_<jobid>.{out,err}` |

`pg_mki_aryeh` allows 48h and only `node4900` has GPUs (`node4701-4702` are CPU-only in
the same partition, which is why `--gres` doubles as the node selector). Skip to
"Step 3 — sanity-check the first log"; the AMD steps below differ only in paths and in
needing the checkpoint copied back afterwards.

## Step 0 — sync (AMD route)

```bash
cd /work1/jeroenaudenaert/pablomer/tess-generative
git checkout galaxy-engaging && git pull
```

## Step 1 — smoke test first (~10 min, do not skip; AMD route)

The CPU preflight benchmarks the loader but says nothing about whether batch 640 fits in
GPU memory or how fast a real step is. This does both, plus it verifies the snapshots get
written and reload:

```bash
sbatch galaxy_images/galaxy_model/contrastive_ablation/smoke_bigbatch.slurm
```

Expect in the log: `PREFLIGHT OK`, then 30 real steps, then

```
[gpu] peak allocated N.N GiB | peak reserved N.N GiB at batch_size=640
[check] snapshot-step=10.ckpt: global_step=10, 256 MiB  OK
SMOKE OK
```

If MI210 peak memory comes out above ~50 GiB, or the step rate implies more than ~3 days
for 100k steps, stop and report — do not silently reduce the batch. (On the Engaging route
this step is folded into the real run: the launcher's preflight benchmarks the loader at
the actual batch size and the trainer prints peak GPU memory, on a 141 GiB H200 where
memory is not in question.)

## Step 2 — submit training (AMD route)

```bash
sbatch --partition=mi2104x --time=24:00:00 --cpus-per-task=16 --mem=96G \
  --export=ALL,RUN_TAG=spatial-conv1x1-BS640-DWNVAL,BATCH_SIZE=640,LOADER=ram48,\
DATA_DIR=/work1/jeroenaudenaert/pablomer/data/efficient_neighs_48,\
SNAPSHOT_STEPS=10000:35000,NUM_UMAP_BATCHES=1,\
CHAIN_TIME=24:00:00,CHAIN_MEM=96G,CHAIN_CPUS=16,MAX_ATTEMPTS=20 \
  galaxy_images/galaxy_model/contrastive_ablation/train_ablation_amd.slurm
```

Why each override:

- `LOADER=ram48` + that `DATA_DIR` — arm A was trained from the prebaked store (see its
  `DONE` file), so the A/B needs the same loader. It needs `--mem`, hence `--mem=96G`
  (17 GB store + ~165 MB per in-flight batch x workers x prefetch).
- `--cpus-per-task=16` — a batch is 10x more images to fetch and collate; 8 workers is
  what fed batch 64.
- `SNAPSHOT_STEPS=10000:35000` — **colons, not commas**: `sbatch --export` splits on
  commas, so a comma silently truncates the list to the first step.
- `NUM_UMAP_BATCHES=1` — the UMAP panel is `batch_size * 2 * N` points, so leaving it at
  8 would UMAP 10,240 points on every validation (~100x per run) for a plot. 1 at batch
  640 = 1,280 points, i.e. the same panel arm A produced.
- `CHAIN_*` — sbatch flags are **not** inherited by the self-chained successor, only the
  environment is. Without these, link 2 silently drops to 12h / no `--mem` / 8 cpus and
  either OOMs on the ram48 store or crawls.
- `MAX_ATTEMPTS=20` — more links than arm A needed (it finished inside one).

The run self-chains on `afterany` and resumes from `last.ckpt`, stopping when it writes
`DONE`. `mi2104x` is 4 GPUs but the trainer uses `devices=1` — take `mi2101x` instead if
the queue prefers it, just pass `CHAIN_TIME=12:00:00` to match.

## Step 3 — sanity-check the first log

```
[cfg] encoder_pool=conv1x1 random_neighbors=False loader=ram48 max_steps=100000 batch_size=640
[cfg] lr=0.0001  negatives/anchor: galaxy=639 instrument~3,195  |  InfoNCE chance loss = 2*ln(640) = 12.93
[data] total anchors=103,741 | excluded=5,466 (HSC-anchor 2,736 / Legacy-anchor 2,730) | kept=98,275 ...
[data] kept=98,275  train=93,362  val=4,913  steps/epoch=145
[data] images/epoch=93,362  epochs at max_steps=689.7  images seen=64.0M (published arms: 6.4M, Ours: 22.3M)
[snapshot] permanent checkpoints requested at steps 10,000, 35,000 (= 6.4M, 22.4M images)
```

`train/loss` starts near **12.9** (not 8.3 — chance is `2*ln(batch)`, so a larger batch
starts higher and the two runs' losses are **not** comparable in absolute terms; compare
`train/acc_*` and downstream R² instead).

## Step 4 — hand back (AMD route only — the Engaging run needs no copy)

Copy **`best-*.ckpt`, both `snapshot-step=*.ckpt`, and `DONE`** (≈1.1 GB total) to
Engaging:

```
/orcd/pool/007/pablomer/checkpoints_new/contrastive-spatial-conv1x1-BS640-DWNVAL/
```

Engaging POOL quota was at 92% on 2026-07-28 — check `~/orcd/.quota` before the rsync and
say so in the hand-back message if it is tight.

Report: final `global_step`, the two `[data]` lines, `[gpu] peak allocated`, final
train/val loss and `train/acc_galaxy` / `train/acc_instrument`, and the wall clock per
100k steps (so the number in this file can be replaced with a real one).

## What runs on Engaging afterwards (context, not for the AMD agent)

Three evals, one per checkpoint, all against arm A:

```bash
CA=galaxy_images/galaxy_model/contrastive_ablation
CK=/orcd/pool/007/pablomer/checkpoints_new/contrastive-spatial-conv1x1-BS640-DWNVAL

# headline: equal steps, 10x negatives
sbatch --export=ALL,RUN_TAG=spatial-conv1x1-BS640-DWNVAL,\
REF_VAR=contrastive-spatial-conv1x1-DWNVAL,CKPT_DIR=${CK} $CA/eval_ablation_engaging.slurm

# equal image budget (6.4M, same as arm A)
sbatch --export=ALL,RUN_TAG=spatial-conv1x1-BS640-eq6M-DWNVAL,\
REF_VAR=contrastive-spatial-conv1x1-DWNVAL,CKPT_DIR=${CK},SNAP_STEP=10000 \
  $CA/eval_ablation_engaging.slurm

# compute-matched with Ours (22.4M)
sbatch --export=ALL,RUN_TAG=spatial-conv1x1-BS640-eq22M-DWNVAL,\
REF_VAR=contrastive-spatial-conv1x1-DWNVAL,CKPT_DIR=${CK},SNAP_STEP=35000 \
  $CA/eval_ablation_engaging.slurm
```

`SNAP_STEP` exists because a snapshot filename contains an `=` and `sbatch --export`
parses `name=value` pairs — passing the path directly as `CKPT` is easy to mangle.

Each writes `RESULT_contrastive-<tag>.md` with the per-cell ΔR² against arm A. Read the
deltas against the 5-fold noise floor already measured in `RESULT_folds.md` (typical
paired std 0.020, so |Δ| below ~0.04 is not resolvable in a single split) — and note
`compare_dwnval.py` labels its columns "holdout"/"reference"; here they mean
"batch 640"/"batch 64".

## Measured numbers

### Batch 640 on one H200 (`pg_mki_aryeh`, job 19105116, 2026-07-28) — the live run

| | measured |
|---|---|
| step rate | **10.42 steps/s** (steps 500 → 1,000 in 48s, from consecutive `latest-step` mtimes) |
| throughput | 6,667 anchors/s ≈ 46,700 images/s through the two encoders |
| loader | 33.6 batch/s at batch 640, 16 workers, in-RAM store — **~3x more than training consumes**, so the GPU is the limiter. Read it as a floor: with 16 persistent workers and prefetch, a 20-batch timing is partly served from the pre-filled queue |
| 100k steps | **~2.6h** of pure stepping; ~3-3.5h with the 100 validations (8 val batches + a UMAP panel each) |
| snapshots | step 10,000 ≈ 16 min in, step 35,000 ≈ 1h in |

That is ~13x the per-anchor throughput of the MI210 reference below, and it makes the
whole arm a single-evening run rather than a multi-link one. Practical consequence: the
follow-up arms this directory lists as "optional because expensive" (BS=128 negative-count
controls on B/C, a sqrt-scaled `LR=3.16e-4` tuned-large-batch arm) cost ~3h each here.

Peak GPU memory prints at the end of `fit` (`[gpu] peak allocated ...`); on a 143 GiB H200
it was never in question, but the number is what tells you whether batch 1,280 is possible.

### Prior expectations, kept for the AMD route

`smoke_bigbatch.slurm` exists to produce the equivalent number on an MI210, where memory
(64 GiB) and step rate are genuinely open questions. What was known before the H200 run:

- Arm A, the reference: **100k steps at batch 64 in ~3h10m on one MI210** with
  `LOADER=ram48` (from the checkpoint mtimes in
  `contrastive_ablation_checkpoints/contrastive-spatial-conv1x1-DWNVAL/`: step 70,000 at
  04:45, 89,500 at 05:24, 100,000 at 05:48 → **7-8 steps/s**, i.e. ~500 items/s).
- That rate is ~1 TFLOP/s of resnet18 work, roughly 2% of MI210 peak, so batch 64 is
  launch-overhead / small-batch bound rather than compute bound. Batch 640 should
  therefore cost **well under** 10x per step — the honest bracket for 100k steps is
  **~15-30h** (2-3 links), with the loader (needs ~640 items/s sustained, vs the ~500
  already demonstrated) the most likely limiter. Verify with the smoke instead of
  trusting this paragraph.
- The step-10,000 snapshot — the equal-image-budget checkpoint — lands after roughly the
  same wall clock as arm A's *entire* run (~3h on MI210; it was 16 min on the H200), so
  there is an evaluable checkpoint early even if the full 100k never finishes.
- GPU memory is not expected to be the constraint: ~5,120 images of 4x48x48 per step
  through resnet18 at bf16 is a few GiB of activations, against 64 GiB on an MI210. The
  smoke prints `[gpu] peak allocated` so this stops being an estimate.
- Verified on Engaging (CPU, batch 8, `--snapshot-steps 2:4`): the callback fires at the
  requested global steps and writes `snapshot-step=2.ckpt` / `snapshot-step=4.ckpt`
  (269 MB each). Reload was not re-checked there — login-node I/O was saturated — but
  `smoke_bigbatch.slurm` step [3/3] asserts `global_step` out of each snapshot, so the AMD
  smoke covers it.
- The Engaging **GPU** smoke could not be run: the `mit_normal_gpu` QOS allows 2 GPUs per
  user and the scale-ablation / recon-mse runs were holding the slot.
