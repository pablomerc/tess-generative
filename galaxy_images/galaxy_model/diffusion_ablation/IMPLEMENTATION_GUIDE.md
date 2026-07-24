# Implementation Guide — Diffusion-vs-FM Ablation (for the executing agent)

You are implementing the diffusion ablation described in [`PLAN.md`](PLAN.md) (read it first; the
scientific rationale and locked decisions live there — this file is the *how*). Everything runs on
the **Engaging** cluster from the repo at `/orcd/pool/007/pablomer/tess-generative` (alias
`/home/pablomer/orcd/pool/tess-generative`), branch `galaxy-engaging`, conda env `torchenv`
(`/home/pablomer/miniforge3/envs/torchenv`; diffusers 0.38.0, torch 2.6.0+cu124 — verified).

**Goal:** train two models that differ ONLY in the generative objective —
(1) `ddpm` arm: DDPM ε-prediction, cosine schedule; (2) `fm-control` arm: the existing
flow-matching objective — under identical Engaging conditions (RAM48 data, 4×H200, batch 64/GPU,
75k steps, seed 42), then produce: the n=5469 downstream-R² comparison, a matched 256-galaxy
reconstruction-MSE table (DDIM η=0 headline + η=1 robustness row), and side-by-side UMAPs.

---

## ⚠ POST-AUDIT CORRECTIONS (2026-07-23, verified against the repo — these OVERRIDE the text below)

A 4-agent audit fact-checked this guide against the actual code. Corrections, in execution order:

1. **[CRITICAL, fixed in `train_ddpm.slurm`] 4-GPU launch:** with PL 2.6.1, `srun` + `--ntasks=1`
   makes Lightning detect `SLURMEnvironment` and NOT self-spawn ranks → a silent 1-GPU run that
   shards data 4 ways (trains on ¼ of the data, no crash). Correct header: `--ntasks-per-node=4`
   + `--cpus-per-task=8` (4×32 CPUs would exceed node4900's 120 and never schedule; plain
   `--ntasks=4` without ntasks-per-node makes PL raise). **Pass criterion for the real jobs: the
   log must say "Starting with 4 processes".**
2. **[CRITICAL for smoke, fixed in `smoke_train.slurm`]** with `wandb.enabled=false`, trainer gets
   `logger=None`, `on_validation_epoch_end` early-returns (`double_train_fm_neighbors.py:769`) →
   `val/mse`, sample grid, and UMAP are all skipped, and `ModelCheckpoint(monitor='val/mse')`
   raises `MisconfigurationException`. Logger-less smokes must `--set trainer.monitor_metric=val/loss`
   and cannot use val/mse/grid/UMAP as pass criteria. Real runs (wandb on) are unaffected.
3. **Checkpoint filenames:** the periodic callback keeps `auto_insert_metric_name=True`, so files
   are **`latest-step=step=<N>.ckpt`** (not `latest-step=<N>.ckpt`). Only one exists at a time
   (`save_top_k=1`), so tolerant `latest-step*.ckpt` globs are safe everywhere (smoke checks,
   `run.resume_from`, downstream `variants.json` — note its resolver takes the alphabetically
   FIRST glob hit, so never point it at a dir of many ckpts).
4. **kwargs filtering (Phase 1.1/gotcha 5):** a `**kwargs` constructor DISABLES
   `filter_supported_model_kwargs` (variants.py:130-135) — all `ModelConfig` keys then reach the
   PARENT `__init__`, which rejects `experiment_config`, `instrument_pooling`,
   `disable_global_physics` → guaranteed `TypeError`. The subclass must filter parent kwargs
   itself via `inspect.signature` (already done in `double_train_ddpm_neighbors.py:30-34`).
5. **Batch shapes (Phase 5.1):** `collate_neighbors` pads to the per-BATCH max neighbor count →
   assert `sameins.shape[1] <= 5` and `masks.dtype == torch.bool`, not `== 5`.
6. **CPU smoke of the dataset:** `load_into_ram=False` is NOT plumbed through `data_factory`;
   instantiate `NeighborsRAM48Dataset(data_dir=..., max_neighbors=5, load_into_ram=False)`
   directly for login-node checks (never pull 16 GB onto a login node).
7. **UMAP (Phase 7.3):** the RAM48 store is NOT compatible with the UMAP scripts' dataset — use
   the full efficient store `/orcd/pool/007/pablomer/efficient_neighs` as `--data-dir`; all args
   (`--checkpoint --data-dir --num-anchors --tag`) are required, and a reused `--tag` raises
   `FileExistsError`. Per Pablo: implement as a NEW script; leave `regenerate_umap_base.py`
   untouched.
8. **Mean-pool twins on the plot (Phase 7.1):** `--extra-variant '<v>:Label'` reads the FLAT
   `combined_e2` from `predict_<v>__aion.csv`; pooled instrument bars require their own CSVs named
   `predict_<v>-meanpool__aion.csv` and their own `--extra-variant '<v>-meanpool:...'` flags.
9. **save_hyperparameters (gotcha 11):** in PL 2.6.1 the parent's own call already captures
   subclass init args via frame inspection; keep the subclass call as belt-and-suspenders (the
   "load breaks without it" claim was wrong).
10. **Environment facts:** pool quota has ~230 GB headroom (fine); **scratch is 100% full — nothing
    may write to scratch**; `prepare_combined.py`'s default `--hsc-dir/--legacy-dir` exist on
    Engaging (no need to pass them); two concurrent 4-GPU `pg_mki_aryeh` jobs are QOS-permitted.

---

## 0. Ground rules (non-negotiable)

1. **Frozen files — do not modify:** `double_train_fm_neighbors.py`, existing files in `configs/`,
   anything under `downstream_evaluation/engaging/outputs/` except *adding new* files,
   `downstream_evaluation/engaging/build_overlap_dataset.py` outputs (`outputs/index/*.parquet` —
   NEVER rerun the build step). New outputs always get new names/suffixes; never overwrite paper
   artifacts.
2. **No git commits or pushes** unless Pablo explicitly asks. Work in the working tree.
3. **Shared HPC etiquette** (see `/etc/claude-code/CLAUDE.md` policy): scoped commands only, never
   scan `/orcd/pool` or `/orcd/scratch` broadly; check quota via `cat ~/orcd/.quota` (cached) before
   writing multi-GB outputs; read `/etc/motd.d/*` at session start.
4. **Discord ping on every job submission and completion** — reuse the `_ping` curl helper pattern
   from `contrastive_baseline/train_contrastive_engaging.slurm` (webhook URL is hardcoded there).
   One short line at kickoff, one at finish/failure. No pings for reads/smoke commands.
5. **Smoke-test before submitting anything real** (Phase 5). Never submit a 75k-step job from
   untested code.
6. **Keep a progress log**: append dated entries to `diffusion_ablation/PROGRESS.md` as you complete
   each phase (created by you; one bullet per milestone, include job IDs and checkpoint paths).
7. Read these files IN FULL before writing any code:
   `double_train_fm_neighbors.py`, `variants.py`, `config.py`, `train.py`, `data_factory.py`,
   `contrastive_baseline/neighbors_ram48.py`, `contrastive_baseline/train_contrastive_engaging.slurm`,
   `configs/neighbors_efficient.json`, `downstream_evaluation/engaging/README.md` (note: partially
   stale, see Phase 7), `downstream_evaluation/engaging/variants.json`.

All paths below are relative to `galaxy_images/galaxy_model/` unless absolute.

---

## Phase 1 — The model: `diffusion_ablation/double_train_ddpm_neighbors.py`

Create `ConditionalDDPMModule(ConditionalFlowMatchingModule)` plus an empty
`diffusion_ablation/__init__.py`. The parent is in `double_train_fm_neighbors.py`; import it with
the same package-path style used by `variants.py`.

### 1.1 Constructor

```python
def __init__(self, prediction_type: str = "epsilon", num_train_timesteps: int = 1000,
             beta_schedule: str = "squaredcos_cap_v2", **kwargs):
    super().__init__(**kwargs)
    self.save_hyperparameters()   # REQUIRED: subclass args must land in hparams for load_from_checkpoint
    self.train_scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps, beta_schedule=beta_schedule,
        prediction_type=prediction_type, clip_sample=False)
    self.inference_scheduler = DDIMScheduler(
        num_train_timesteps=num_train_timesteps, beta_schedule=beta_schedule,
        prediction_type=prediction_type, clip_sample=False)
```

- `clip_sample=False` on BOTH schedulers is **critical**: the data is z-scored with heavy tails,
  not in [-1,1]; the diffusers default clamps predicted x₀ to [-1,1] inside `step()` and silently
  corrupts reconstructions.
- The parent constructor builds the UNet (`self.velocity_model`), both encoders, and all
  validation plumbing — do not duplicate any of that.
- Verify after implementing: `ConditionalDDPMModule.load_from_checkpoint(<smoke ckpt>)` restores
  `prediction_type` etc. without passing them manually.

### 1.2 `compute_loss(self, batch)` (override)

Mirror the parent's structure exactly (read `double_train_fm_neighbors.py:395-485` and reproduce
its contract), replacing ONLY the corruption/target block:

```python
x_1, cond_image_samegal, cond_image_sameins, masks, metadata = <same 4/5-tuple unpack as parent>
noise = torch.randn_like(x_1)
t = torch.randint(0, self.train_scheduler.config.num_train_timesteps,
                  (x_1.shape[0],), device=x_1.device)
x_t = self.train_scheduler.add_noise(x_1, noise, t)
if self.hparams.prediction_type == "epsilon":
    target = noise
else:  # "v_prediction"
    target = self.train_scheduler.get_velocity(x_1, noise, t)
pred = self(x_t, t.float() / 1000.0, cond_image_samegal, cond_image_sameins, masks)
```

Then keep, identically to the parent: per-example MSE reduction, the per-survey split by
`metadata[i]['anchor_survey']`, the stashed logging attributes the parent sets
(`self._loss_hsc`, `self._loss_legacy`, `self._loss_generative_total` — check exact names in the
parent), the `lambda_generative` gate, and the same return value semantics. Drop the Sinkhorn /
`lambda_geometric` branch (it is 0.0 in all runs here); keep `mask_center` handling only if trivial
to copy (it's off in our configs).

**The `t/1000` trick:** the inherited `forward` multiplies its `t` argument by 1000 before the UNet
call (`double_train_fm_neighbors.py:339`). Passing `t_int.float()/1000.0` therefore delivers the
integer timestep to the UNet unchanged, with zero edits to `forward`. Do not "fix" the parent.

### 1.3 `sample(...)` (override)

Keep the **exact parent signature** — `compute_mse`, the validation sample grid, and downstream
generative scripts all call it positionally/by keyword as the parent defines it
(`double_train_fm_neighbors.py:642-697`): `(cond_image_samegal, cond_image_sameins, masks=None,
num_steps=None, x_noise=None)` — plus add a trailing `eta: float = 0.0` keyword.

```python
sched = self.inference_scheduler
sched.set_timesteps(num_steps or self.num_integration_steps, device=device)   # 250 in config
x = x_noise if x_noise is not None else torch.randn(B, self.in_channels, self.image_size, self.image_size, device=device)
x = x * sched.init_noise_sigma          # 1.0 for VP schedules; harmless, keeps API correctness
for t in sched.timesteps:               # descending ints, e.g. 996, 992, ..., 0
    pred = self(x, t.expand(B).float() / 1000.0, cond_image_samegal, cond_image_sameins, masks)
    step_kwargs = {"eta": eta}
    if eta > 0:                          # reproducible stochastic sampling for the η=1 pass
        step_kwargs["generator"] = <torch.Generator seeded by caller-provided seed, see eval script>
    x = sched.step(pred, t, x, **step_kwargs).prev_sample
return x
```

Wrap in `@torch.no_grad()`, default masks to all-ones like the parent does. `num_integration_steps`
is inherited config plumbing — for the DDPM arm it means "DDIM inference steps"; do not rename it.

### 1.4 Things NOT to change

`forward`, `training_step`, `validation_step`, `on_validation_epoch_end`, `compute_mse`,
`configure_optimizers`, the UMAP code — all inherited. `val/loss` will now mean ε-MSE (fine,
within-run only); cross-arm model selection uses `val/mse` (see Phase 2 config) and the fixed
75k-step checkpoint.

---

## Phase 2 — Config fields, variant registration, config files

### 2.1 `config.py`

- Add to `ModelConfig`: `prediction_type: str = "epsilon"`, `num_train_timesteps: int = 1000`,
  `beta_schedule: str = "squaredcos_cap_v2"`. (Unknown JSON keys raise `TypeError` in
  `ExperimentConfig.from_dict`, so they MUST be dataclass fields; `filter_supported_model_kwargs`
  silently drops them for model classes that don't accept them, so FM configs are unaffected.)
- Widen the `data.mode` validation whitelist (currently `precomputed|neighbors|efficient` around
  `config.py:146-153`) to accept `"ram48"`. Note the `efficient` branch requires
  `efficient_data_dir`; apply the same requirement to `ram48`.

### 2.2 `variants.py`

Register one new variant (FM control reuses the existing `neighbors_all_attn`):

```python
"neighbors_all_attn_ddpm": VariantSpec(
    name="neighbors_all_attn_ddpm",
    description="DDPM-objective ablation of neighbors_all_attn (identical arch/data, ε-prediction)",
    model_cls=ConditionalDDPMModule,
    model_overrides={"all_attention": True},
),
```

Import at top of `variants.py` following the existing import style. Remember
`model_overrides` win over JSON values and silently drop if not in the constructor signature.

### 2.3 `data_factory.py` — RAM48 branch

In `build_neighbors_dataloaders`, add an `elif config.data.mode == "ram48":` branch next to the
`efficient` branch (~`data_factory.py:197-217`), constructing
`contrastive_baseline.neighbors_ram48.NeighborsRAM48Dataset` with the store dir from
`config.data.efficient_data_dir`, `max_neighbors=config.data.max_neighbors`, and
`random_neighbors=config.data.random_neighbors` — **read the actual constructor signature in
`neighbors_ram48.py` and match it exactly** (arg names in this guide are indicative). Collate with
the same `collate_neighbors` the efficient branch uses. The downstream-holdout and lens-val gates
are conditioned on `mode == "efficient"` (`data_factory.py:224,237,294`) — leave them alone; our
runs use neither (no holdout: locked decision, PLAN §5).

Store: `/orcd/pool/007/pablomer/efficient_neighs_48` (`hsc48.bin`, `legacy48.bin`,
`catalog.parquet`, `meta.json` — verify present with a plain `ls` before wiring).

### 2.4 Config JSONs (new, in `diffusion_ablation/configs/`)

`neighbors_ddpm_eps.json` — start from a copy of `configs/neighbors_efficient.json` and change:

```
run.variant                 = "neighbors_all_attn_ddpm"
run.output_dir              = "/orcd/pool/007/pablomer/checkpoints_new/diffusion_ablation/runs"
run.shared_checkpoint_dir   = "/orcd/pool/007/pablomer/checkpoints_new/diffusion_ablation/best"
data.mode                   = "ram48"
data.efficient_data_dir     = "/orcd/pool/007/pablomer/efficient_neighs_48"
data.num_workers            = 4          # per rank; keep OMP/MKL threads=1 in slurm
model.prediction_type       = "epsilon"
model.num_train_timesteps   = 1000
model.beta_schedule         = "squaredcos_cap_v2"
trainer.num_steps           = 75000
trainer.scale_steps_by_devices = false   # IMPORTANT: we want max_steps=75000 GLOBAL optimizer
                                         # steps; the original config's 1500000/devices semantics
                                         # (train.py:173-176) must NOT apply. Verify in the smoke
                                         # log that Trainer reports max_steps=75000.
trainer.devices             = 4
trainer.monitor_metric      = "val/mse"  # objective-neutral best-checkpoint selection
trainer.auto_adjust_for_h100 = false     # determinism; batch stays 64/device
wandb.name                  = "ddpm-eps-ram48-h200-75k"
```

All else identical to `neighbors_efficient.json` (batch 64, lr 1e-4, bf16-mixed,
val_check_interval 1000, checkpoint_every_n_train_steps 1000, seed 42, max_neighbors 5,
lambda_geometric 0.0, num_integration_steps 250, cross_attention_dim 16, image_size 48).

`neighbors_fm_control.json` — identical except:
`run.variant = "neighbors_all_attn"`, no `model.prediction_type/num_train_timesteps/beta_schedule`
keys, `wandb.name = "fm-control-ram48-h200-75k"`.

Note: `shared_checkpoint_dir` subfolders are keyed by `wandb.name` (train.py:131-134), so distinct
names keep the two arms' best-checkpoints separate. Periodic `latest-step={step}.ckpt` files land
under `run.output_dir/<variant>/<date>/checkpoints/`; there is **no `last.ckpt`**
(`save_last=False` in train.py) — resume, if ever needed, is
`--set run.resume_from=<newest latest-step ckpt>`.

---

## Phase 3 — SLURM: `diffusion_ablation/train_ddpm.slurm`

One script, parameterized by `--export=ALL,CONFIG_PATH=...` so both arms share it. Model it on
`train_neighbors_efficient.slurm` (invocation style) + `contrastive_baseline/
train_contrastive_engaging.slurm` (Engaging env, `_ping` helper, netrc-based W&B detection).

Header (primary partition):

```
#SBATCH --partition=pg_mki_aryeh
#SBATCH --nodes=1
#SBATCH --ntasks=1                 # Lightning launches the 4 DDP ranks itself
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G                 # 4 ranks × 16 GB RAM48 copy + workers + headroom (node has 2 TB)
#SBATCH --time=24:00:00            # expect ~8h; 3× safety, well under the 48h partition cap
#SBATCH --output=.../diffusion_ablation/logs/%x_%j.out
```

Body: `source ~/.bashrc; conda activate torchenv`; `cd $REPO; export PYTHONPATH=$REPO`;
`export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`;
`export TORCH_BLAS_PREFER_HIPBLASLT=0` (harmless no-op on NVIDIA, keeps script portable);
W&B online iff `~/.netrc` has `api.wandb.ai` (copy the contrastive script's detection block);
Discord `_ping` at start and at exit (success/failure paths); then:

```
srun python -u -m galaxy_images.galaxy_model.train --config "$CONFIG_PATH" $EXTRA_SET_ARGS
```

Submission (Phase 6): two `sbatch` calls, one per config. **Submit as two 4-GPU jobs, not one
8-GPU job** (gentler preemption of scavenger jobs on node4900). Before submitting, check
`sinfo -p pg_mki_aryeh` and `scontrol show node node4900 | grep AllocTRES`. Fallback if jobs sit
PENDING > 24h: the `mit_normal_gpu` self-chaining pattern per PLAN §4 (build only if actually
needed).

---

## Phase 4 — `diffusion_ablation/eval_recon_mse.py`

Standalone matched reconstruction-MSE eval. Spec:

- **CLI:** `--fm-checkpoint PATH` (repeatable label:path pairs are fine too), `--ddpm-checkpoint
  PATH`, `--config PATH` (the ddpm JSON, used only to rebuild the dataloaders), `--n 256`,
  `--num-steps 250`, `--eta 0.0` (repeatable → one result row per η, applied to DDPM only),
  `--noise-seed 1234`, `--out CSV`, `--device cuda`.
- **Protocol:** build loaders via `data_factory.build_neighbors_dataloaders` with the seed-42
  config → the val split is deterministic and identical for every checkpoint (same dataset, same
  seeded split). Iterate the val loader in order (no shuffling) and take the first `--n` anchors.
  For each batch: draw `x_noise` from a `torch.Generator(device).manual_seed(noise_seed + batch_idx)`
  — the SAME noise tensors are then fed to every model via `sample(..., x_noise=...)`.
  FM checkpoints load with `ConditionalFlowMatchingModule.load_from_checkpoint`, DDPM with
  `ConditionalDDPMModule...`; run `model.sample(samegal, sameins, masks=masks, num_steps=250,
  x_noise=noise)` (+`eta=` for DDPM rows), compute per-image MSE vs the target over the full 48×48,
  and aggregate mean±sem split by `metadata[i]['anchor_survey']` ('hsc' vs 'legacy').
- **Output:** long-form CSV (model_label, objective, eta, anchor_survey, n, mse_mean, mse_sem) +
  a printed markdown table. For the η=1 row, seed the per-step generator from `--noise-seed` so the
  number is reproducible.
- **Persist the eval set as an explicit artifact:** on first run, write
  `diffusion_ablation/results/recon_eval_manifest.json` recording the dataset indices and catalog
  row ids (`metadata['idx']`) of the selected anchors, plus noise seed, num_steps, val_ratio and
  split seed. On subsequent runs, load the manifest if present and **hard-error if the rebuilt
  split disagrees** — the val set must be a stored artifact, not an implicit consequence of
  seed-42 splitting. (Do NOT use the repo's `data.save_heldout_validation` exporter for this — it
  has a latent bug: `_save_heldout_validation_subset` unpacks 5 values per item at
  `data_factory.py:165` but efficient/ram48 items are 4-tuples, so it raises `ValueError`.)
- Also evaluate the **paper base checkpoint** (`checkpoints/base/snapshot.ckpt`, class
  `ConditionalFlowMatchingModule`) as a third row — context only, flagged in the output as
  "different training setup (paper run)".
- Run as a 1-GPU sbatch on `mit_normal_gpu` (≤1h) or on a pg node; ~256 galaxies × 250 steps ×
  4 model/η rows is minutes of GPU time.

---

## Phase 5 — Smoke tests (all must pass before Phase 6)

1. **Dataset-only, CPU, login-safe (small!):** instantiate the ram48 dataset through
   `data_factory` with the new config but `load_into_ram=False`-equivalent if the constructor
   allows (avoid pulling 16 GB onto a login node — if the class always loads to RAM, do this step
   inside the GPU smoke job instead). Check one collated batch: shapes
   `(B,4,48,48)/(B,4,48,48)/(B,5,4,48,48)`, bool mask `(B,5)`, metadata has `anchor_survey`.
2. **200-step GPU smoke of the DDPM arm** (sbatch, 1 GPU, `mit_normal_gpu`, 30 min):
   `--set trainer.devices=1 --set trainer.num_steps=200 --set trainer.val_check_interval=100
   --set data.batch_size=16 --set wandb.enabled=false --set model.num_integration_steps=10
   --set model.num_mse_images=8 --set model.num_sample_images=2`.
   Pass criteria: train loss finite and decreasing; validation completes (val/mse logged, sample
   grid + UMAP produced without error); `latest-step=200.ckpt` written;
   `ConditionalDDPMModule.load_from_checkpoint` round-trips; a manual `sample()` call returns
   `(N,4,48,48)` with values outside [-1,1] present (proves `clip_sample=False` took effect);
   Trainer log shows `max_steps=200` (validates the `scale_steps_by_devices=false` semantics).
3. **Same smoke for the FM control config** (exercises the ram48 plumbing against the unmodified
   parent class — a pure regression test of Phase 2.3).
4. **4-GPU sanity (optional but recommended):** 200 steps with `devices=4` on pg_mki_aryeh to
   confirm DDP + per-rank RAM48 loading + memory headroom before the real submission.

---

## Phase 6 — Real training

1. Check `~/orcd/.quota` (each arm writes ~2–3 GB of checkpoints + W&B files).
2. `sbatch --export=ALL,CONFIG_PATH=.../neighbors_ddpm_eps.json diffusion_ablation/train_ddpm.slurm`
   and same for `neighbors_fm_control.json`. Discord ping each. The two jobs are independent and
   are **expected to run concurrently** on node4900 (4+4 of its 8 H200s) once preemption frees the
   GPUs; if only 4 free up, slurm serializes them automatically — no manual sequencing. Do NOT
   merge them into one 8-GPU job: separate failure domains (one crash shouldn't kill both arms),
   two clean single-run DDP groups, and gentler preemption of scavenger jobs.
3. Monitor: `squeue -u pablomer`; W&B project `galaxy-flow-matching-neighbours`, runs
   `ddpm-eps-ram48-h200-75k` / `fm-control-ram48-h200-75k`. Sanity milestones: both arms' `val/mse`
   should be broadly similar in trajectory after ~10k steps (they share data + architecture); if
   the DDPM arm's val/mse is pathological (>5× FM) by 10k steps, stop and debug (usual suspects:
   clip_sample, timestep scaling, prediction_type mismatch between schedulers).
4. Completion: confirm `latest-step=75000.ckpt` exists for both arms. These fixed-step checkpoints
   are THE comparison artifacts (best-val/mse checkpoints are secondary).

## Phase 7 — Evaluation

1. **Downstream R²** (`downstream_evaluation/engaging/`; README is stale about dataset composition
   — the shipped index IS the single n=5469 crossmatch; never rerun `build_overlap_dataset.py`):
   - Append to `variants.json`: entries `diffusion-base` and `fm-control`, each with `ckpt_glob`
     pointing at the 75k checkpoint, `module": "diffusion_ablation/double_train_ddpm_neighbors.py"`
     / `"double_train_fm_neighbors.py"` respectively, and the matching `model_class`. Match the
     existing entries' schema exactly.
   - `ONLY=diffusion-base bash run_per_variant.sh`, then `ONLY=fm-control ...` (idempotent,
     ~10 min each on 1 GPU). `prepare_combined.py` duck-types on `encoder_1/encoder_2`, so the base
     branch handles both classes with no code edits.
   - Paper-style CSVs: `predict_combined.py --embeddings outputs/embeddings/<v>.h5 --mlp-arch aion
     --out outputs/predictions/predict_<v>__aion.csv` for both.
   - Instrument-side twin: `build_meanpool_h5.py` on each embeddings H5 → probe those too (mirrors
     how the paper's "Ours (Instr)" bar = `base-meanpool` was produced; check that script's CLI).
   - Comparison plot, zero code edits:
     `plot_unified_three_probes.py --mlp-arch aion --extra-variant 'fm-control:FM (matched)'
     --extra-variant 'diffusion-base:DDPM (ε)'`.
2. **Recon MSE:** run `eval_recon_mse.py` (Phase 4) with η=0 and η=1 rows. Output CSV + table into
   `diffusion_ablation/results/`.
3. **UMAPs:** `visualization_scripts/regenerate_umap_base.py --checkpoint <ckpt> ...` for both
   arms. It hardcodes the `ConditionalFlowMatchingModule` import — add a small `--module/--model-class`
   (or importlib) parameter, defaulting to current behavior, rather than editing the class name in
   place. Render via its companion `plot_umap_from_file.py`.
4. **Deliverable for Pablo:** a short `diffusion_ablation/RESULTS.md` with: the R² comparison table
   (FM-paper / FM-control / DDPM columns for the headline properties + instrument properties), the
   recon-MSE table (incl. η rows), UMAP figure paths, W&B run links, checkpoint paths, and any
   deviations from this guide.

---

## Gotcha index (things that WILL bite if ignored)

| # | Gotcha |
|---|---|
| 1 | `clip_sample=False` on both schedulers — data is not in [-1,1]. |
| 2 | Pass `t_int/1000.0` to the inherited `forward` (it multiplies by 1000). Never edit the parent. |
| 3 | `scale_steps_by_devices=false` + `num_steps=75000`, else 4-GPU DDP divides your steps by 4. |
| 4 | `val/loss` is not comparable across objectives — cross-arm selection = fixed 75k ckpt, `monitor_metric=val/mse`. |
| 5 | New model kwargs must be `ModelConfig` dataclass fields (unknown JSON keys → `TypeError`); they're silently filtered per-class, so typos in `model_overrides` also vanish silently — double-check spelling against the constructor. |
| 6 | Each DDP rank loads its own 16 GB RAM48 copy → `--mem=256G`. `np.fromfile` is what forces RAM (do not "optimize" to `np.ascontiguousarray` — no-op on memmaps). |
| 7 | No `last.ckpt` exists; resume via `--set run.resume_from=<latest-step ckpt>`. |
| 8 | Submit two 4-GPU jobs, not one 8-GPU job (preemption etiquette on node4900). |
| 9 | Never rerun `build_overlap_dataset.py`; never overwrite files in `outputs/{index,embeddings,predictions}` — only add. |
| 10 | No downstream holdout — deliberate, matches the paper (PLAN §5). Don't "improve" this. |
| 11 | `save_hyperparameters()` in the subclass `__init__`, or `load_from_checkpoint` breaks. |
| 12 | Keep `strategy: ddp_find_unused_parameters_true` (conditional encoder paths). |
