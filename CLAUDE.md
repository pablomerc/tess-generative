# tess-generative — Quick Orientation for Claude

> Catch-up notes for new sessions. The user (Pablo, p.mercader01@gmail.com) is an ML researcher working on self-supervised / generative representation learning for galaxy survey images (HSC + Legacy Survey). This file lives at the repo root.

---

## What this codebase does

Trains dual-encoder generative models that learn **physics** and **instrument** latent spaces from paired galaxy images across two surveys (HSC and Legacy Survey). The main approach is **flow matching** (conditional velocity prediction) with two ResNet18 encoders feeding a `diffusers` UNet via cross-attention. A separate **contrastive baseline** (InfoNCE on dual encoders) lives in `galaxy_images/galaxy_model/contrastive_baseline/`.

**Active clusters: AMD (this one) and Engaging.** Development happens on both; they share the GitHub remote `git@github.com:pablomerc/tess-generative.git` and the working branch is `galaxy-engaging`.

- **AMD** (this cluster, `/work1/jeroenaudenaert/pablomer/`): AMD GPUs only — MI210 (`mi2101x` 1×/12h, `mi2104x` 4×/24h), MI250 (`mi2508x` 8×/12h), MI300X/MI325X/MI350X 8-GPU nodes (`mi3008x`/`mi3258x`/`mi3508x`, 12h batch). No NVIDIA hardware here; H100/H200 are on Engaging. **Needs the hipBLASLt workaround** (see below).
- **Engaging** (`/orcd/pool/007/pablomer/tess-generative`, alias `/home/pablomer/orcd/pool/tess-generative`): NVIDIA GPUs (CUDA). **Does not need the hipBLASLt workaround.**

The old MIT/CSAIL cluster (`/data/vision/billf/scratch/pablomer/`) is **no longer in active use** — treat its hardcoded paths in older downstream scripts as legacy to parameterize before running anywhere. **Only the neighbors dataset is available on the AMD cluster.**

---

## Working across clusters (AMD ⇄ Engaging)

Both clusters develop on `galaxy-engaging` and sync through the shared GitHub remote.

- **Never force-push a shared branch** (`galaxy-engaging`, `main`). To move work between clusters, push to a per-cluster snapshot branch (`amd-snapshot`, `engaging-snapshot`) and merge from there, so nothing gets clobbered.
- **Cluster-specific job scripts diverge on purpose.** SLURM/`.sh` submission scripts and training entry points carry per-cluster settings (partitions, module loads, the hipBLASLt workaround on AMD). When merging, **keep both variants** rather than overwriting one cluster's scripts with the other's.
- For code that must run on **both** clusters, use the guarded hipBLASLt form (`try/except` around `preferred_blas_library`, as in `downstream_evaluation/engaging/`) — it's a no-op on NVIDIA and a fix on AMD.

---

## Submitting jobs on each cluster

Same repo, same conda env name (`torchenv`) on both — the differences are the SLURM partitions, absolute paths, and the hipBLASLt workaround. `sbatch` the cluster-appropriate script (each cluster's scripts hardcode its own paths).

### AMD (this cluster)
- Repo `/work1/jeroenaudenaert/pablomer/tess-generative` · `conda activate torchenv` · logs → `/work1/jeroenaudenaert/pablomer/logs/`
- GPU partitions: **`mi2104x`** (4×MI210) and **`mi2101x`** (1×MI210). GPUs come with the partition — no `--gres` line. Some scripts `--exclude` known-bad nodes.
- **Every GPU job needs the hipBLASLt workaround**: `export TORCH_BLAS_PREFER_HIPBLASLT=0` in the script *and* `torch.backends.cuda.preferred_blas_library("hipblas")` in the Python entry point.
```bash
sbatch galaxy_images/galaxy_model/train_neighbors_efficient.slurm            # baseline training
sbatch galaxy_images/galaxy_model/slurm_jobs/base_ins4x4.slurm               # variant sweep: sets $CONFIG_PATH, sources _train_template.slurm
```

### Engaging
- Repo `/orcd/pool/007/pablomer/tess-generative` (alias `/home/pablomer/orcd/pool/tess-generative`) · `conda activate torchenv`
- GPU partitions: **`mit_normal_gpu`** (H100/H200/L40S) and **`mit_preemptable`** (A100/A40, preemptible). Request GPUs explicitly: `#SBATCH --gres=gpu:1` + a `--mem` line. Logs default to `%x_%j.out` in the submit dir.
- **No hipBLASLt workaround** (NVIDIA); the env var, if present, is a harmless no-op. Engaging-specific pipelines live in `downstream_evaluation/engaging/`, `cross_predict_experiment/`, `anomaly_detection/engaging/`, and are parameterized via `--export`:
```bash
sbatch galaxy_images/galaxy_model/downstream_evaluation/engaging/run_predict.slurm
sbatch --export=ALL,XPRED_DIRECTION=hsc_to_legacy \
  galaxy_images/galaxy_model/cross_predict_experiment/run_engaging.slurm     # also legacy_to_hsc
```
- Engaging data: `/orcd/pool/007/pablomer/neighbours_v2.h5`.

> **Secrets note:** a few scripts embed credentials in plaintext — a WandB API key (AMD SLURM scripts) and a Discord webhook (Engaging scripts). Since the repo is on GitHub, prefer moving these into `~/.bashrc`/env vars and rotating them.

---

## Top-level layout

```
tess-generative/
├── galaxy_images/
│   └── galaxy_model/
│       ├── train.py                  # unified Lightning entry point
│       ├── variants.py               # variant → model class registry
│       ├── data_factory.py           # builds NeighborsEfficientDataset loaders
│       ├── neighbors_efficient.py    # mmap+Parquet dataset
│       ├── neighbors.py              # collate_neighbors (pads neighbors, returns masks)
│       ├── config.py                 # ExperimentConfig dataclass + JSON loader
│       │
│       ├── configs/                  # per-variant JSON configs
│       │   ├── neighbors_default.json
│       │   ├── neighbors_efficient.json          # baseline (neighbors_all_attn)
│       │   ├── neighbors_efficient_hier.json     # hierarchical (neighbors_hier_global_ins)
│       │   └── ...
│       │
│       ├── double_train_fm_neighbors.py          # ConditionalFlowMatchingModule (baseline)
│       ├── train_neighbors_efficient.slurm       # SLURM for baseline
│       ├── neighbors_efficient_train.py          # python entry for baseline (delegates to train.py)
│       │
│       ├── hierarchical_attention/               # hierarchical multi-level cross-attn variant
│       │   ├── double_train_fm_neighbors_hier_global_ins.py  # HierarchicalGlobalInstrumentFlowMatchingModule
│       │   ├── train_experiments.py              # ConfigurableEncoder, ConditionedUNet, EXPERIMENTS dict
│       │   ├── train_neighbors_efficient_hier.slurm
│       │   ├── neighbors_efficient_train_hier.py
│       │   └── INSTRUMENT_LEAK_OPTIONS.md        # diagnosis: physics latent leaks survey identity
│       │
│       ├── single_encoder_ablation/              # single-encoder control (no physics invariance)
│       ├── contrastive_baseline/                 # dual-encoder InfoNCE (no flow matching)
│       │   ├── dual_encoder_contrastive.py
│       │   └── downstream_evaluation/
│       │
│       ├── downstream_evaluation/                # MLP probes on top of frozen embeddings
│       │   ├── prepare_data.py                   # generate embeddings, save HDF5
│       │   ├── predict.py                        # train MLP, report R²
│       │   ├── predict_neighbors.py              # variant-specific
│       │   └── prepare_neighbors.py
│       │
│       ├── umap_exploration/                     # offline UMAP scripts (validation also produces UMAPs)
│       ├── outputs/                              # checkpoints + figures (per variant subdirs)
│       └── ...
│
├── lightning_logs/
└── requirements.txt
```

---

## How to run training

**Always go through `train.py`.** Both the baseline and hierarchical SLURM scripts call into the same Python entry — they only differ by which config they pass.

### Baseline (recommended starting point)
```bash
sbatch /work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model/train_neighbors_efficient.slurm
```
- variant: `neighbors_all_attn`
- model: `ConditionalFlowMatchingModule`
- both encoders output spatial tokens, concatenated into UNet cross-attention

### Hierarchical
```bash
sbatch /work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model/hierarchical_attention/train_neighbors_efficient_hier.slurm
```
- variant: `neighbors_hier_global_ins`
- model: `HierarchicalGlobalInstrumentFlowMatchingModule`
- encoder_1: `ConfigurableEncoder` (multi-level spatial tokens + global vec)
- encoder_2: ResNet18 with `mean_pool=True` → single global vector → time embedding (NOT cross-attention)
- **Known issue:** the physics latent UMAP shows HSC/Legacy separation due to the missing spatial instrument pathway. See `hierarchical_attention/INSTRUMENT_LEAK_OPTIONS.md`.

### Direct invocation (for debugging)
```bash
python -m galaxy_images.galaxy_model.train \
  --config galaxy_images/galaxy_model/configs/neighbors_efficient.json \
  --set trainer.devices=1 \
  --set trainer.num_steps=5000 \
  --set wandb.enabled=false
```

### How the variant system works
1. SLURM script → `python -m galaxy_images.galaxy_model.train --config <config>.json`
2. `train.py` loads config, reads `run.variant`, looks it up in `variants.VARIANTS`
3. Variant entry maps to a model class (`model_cls`) and optional `model_overrides`
4. Model kwargs are filtered by the constructor signature (`filter_supported_model_kwargs`) — config keys not accepted by the chosen model class are silently dropped
5. Same `data_factory.build_neighbors_dataloaders` is used for all variants

To add a new variant: register it in `variants.py` and create a matching config in `configs/`.

---

## HipBLASLt bug (MI210 GPUs — AMD cluster only)

**Applies only on the AMD cluster.** On Engaging (NVIDIA/CUDA) the workaround is unnecessary — the guarded form is a harmless no-op, and the unguarded `preferred_blas_library("hipblas")` call may error, so use the guarded form in any code shared across clusters.

**Symptom:** training crashes with `HIPBLAS_STATUS_INVALID_VALUE` on certain matrix shapes when running on AMD MI210 GPUs.

**Workaround (already applied in the SLURM scripts and python entry points):**
```bash
# In SLURM
export TORCH_BLAS_PREFER_HIPBLASLT=0
```
```python
# At top of *_train.py entry points
import torch
torch.backends.cuda.preferred_blas_library("hipblas")
```
Both are required (env var for new processes, Python call for the main process). If you see HIPBLAS errors, confirm both are present. (The `auto_adjust_for_h100` config flag is only relevant on NVIDIA clusters, i.e. Engaging — there are no H100s on the AMD cluster.)

---

## Data

All data on this cluster lives under `/work1/jeroenaudenaert/pablomer/data/`:
- `neighbors_efficient/` — primary mmap+Parquet store used by `NeighborsEfficientDataset` (preferred path, `mode="efficient"`)
- `neighbours_v2.h5` — older HDF5 neighbors dataset
- `train_neighbors.vds` / `val_neighbors.vds` — precomputed virtual datasets (`mode="precomputed"`)
- `heldout_validation/` — saved heldout val subsets (controlled by `data.save_heldout_validation` in config)
- `preprocessed_hsc_legacy_48x48.h5` — paired survey crops at 48×48
- `hsc_ivar_*.h5`, `lens_*` — auxiliary HSC inverse-variance and lens datasets

**The dataloader** randomly chooses an HSC or Legacy image as the anchor (50/50), the same-galaxy match in the other survey, and `max_neighbors=5` same-instrument neighbors. `collate_neighbors` pads neighbors to `max_neighbors` and returns a `(B, k)` boolean mask — the model uses the mask to zero out padded entries.

---

## Outputs

- Trained checkpoints: `galaxy_images/galaxy_model/outputs/<variant_name>/<YYYY-MM-DD>/checkpoints/`
  - `latest-step=step={step}.ckpt` — note the **doubled `step=`**: the periodic callback leaves
    `auto_insert_metric_name=True`, so PL rewrites `{step}` → `step={step}`. Globs must be
    `latest-step*.ckpt`. Saved every `checkpoint_every_n_train_steps`, and `save_top_k=1` means each
    new one **DELETES the previous** — only one ever exists per Trainer instance.
  - `best-epoch={epoch}-step={step}.ckpt` — selected on `trainer.monitor_metric`, which is
    **`val/mse`** in the ablation configs (the 250-step *sampled reconstruction* MSE), **not**
    `val/loss` (the velocity-regression objective). Those two can diverge: see the note in
    `scale_ablation/PROGRESS.md`.
  - Each of the 4 DDP ranks creates its own dated run dir (`<date>`, `<date>_2`, …) and only **one**,
    nondeterministically, receives checkpoints. Never trust the printed "Run dir"; glob all of them,
    and copy anything you intend to evaluate to a `frozen/` path first (a live `ModelCheckpoint` has
    deleted a file mid-read before, giving `OSError: [Errno 116] Stale file handle`).
- Hierarchical run dir: `galaxy_images/galaxy_model/hierarchical_attention/outputs/...`
- Contrastive baseline runs land under `/work1/jeroenaudenaert/pablomer/outputs/contrastive_baseline/`
- WandB project: `galaxy-flow-matching-neighbours` (contains all flow-matching runs and their UMAP
  grids). Ablations use `wandb.group` / `tags` / `job_type` to stay a comparable family, and put the
  sweep's independent variable in `wandb.extra_config` as a flat numeric key (e.g.
  `scale/n_anchors`) so W&B can chart metrics against it — a config *path* string is not plottable.

---

## Storage layout — four filesystems, know which one to use

`~/orcd/.quota` (refreshed ~every 30 min) reports only the three **personal** allocations. There is a
fourth, much larger **group** allocation that it does not mention.

| Mount | Convenience path | Size | Use for |
|---|---|---|---|
| `/home/pablomer` | — | 200 GB (9% used) | code, small outputs. Backed up — don't put bulk binaries here |
| `/orcd/pool/007/pablomer` | `~/orcd/pool` | 1 TB (**~81% used**) | repo, active run dirs, `checkpoints_new/` |
| `/orcd/scratch/orcd/009/pablomer` | `~/orcd/scratch` | 1 TB (**100% — HARD FULL**) | the downstream image stores; **nothing may be written here** |
| `/orcd/data/mki_aryeh/001/pablomer` | `~/aryeh_data` | **200 TB (~1% used)** | bulk checkpoint archives, anything large |

**`~/orcd/scratch` is completely full and unusable for writes** — `touch` fails with
`Disk quota exceeded`. It holds ~855 GB of *irreplaceable* downstream cutouts that the eval pipeline
depends on by default (`prepare_combined.py` → `--hsc-dir ~/orcd/scratch/hsc_downstream`,
`--legacy-dir ~/orcd/scratch/legacy_downstream_full/full_1M`), plus ~8 GB of paper checkpoints. Note
that the repo's `galaxy_images/galaxy_model/checkpoints` symlinks **into scratch**, which is also
`RunConfig.shared_checkpoint_dir`'s default — so any config that does not set an absolute
`shared_checkpoint_dir` will try to write 2 GB into a full filesystem and fail. Always set it
explicitly. (There is ~237 GB of reclaimable space in
`legacy_downstream_full/test_parallel/` — a superseded trial build, referenced nowhere in the repo,
superseded by `full_1M/` which was rebuilt six weeks later and carries the `selection.parquet`
curation artifacts that `test_parallel` lacks.)

**`/orcd/data/mki_aryeh/001` is the place for bulk data.** It is the `mki_aryeh` group allocation
(same group as the `pg_mki_aryeh` GPU partition), 200 TB at ~1% used, group
`orcd_rg_hstor003_pg_mki_aryeh` with the setgid bit set so new files inherit group access. **Verified
mounted and writable on both `node4900` (GPU/training) and `node4701` (CPU)**, and it does **not**
count against the personal POOL quota. Convention there is one dir per user at the top level.
Measured throughput POOL → here: ~646 MB/s.

Pattern used by `scale_ablation`: keep the canonical path inside `checkpoints_new/` and make it a
**symlink** into the group allocation, so code paths never change and POOL stays lean —

```
/orcd/pool/007/pablomer/checkpoints_new/scale_ablation/snapshots
    -> /orcd/data/mki_aryeh/001/pablomer/scale_ablation/snapshots
```

plus a second symlink at `galaxy_images/galaxy_model/scale_ablation/snapshots` so the archive shows up
in the VSCode tree where you actually work (VSCode only displays paths under the workspace root, so a
symlink in `$HOME` is invisible when the repo is the workspace). Both are `.gitignore`d, and
`.vscode/settings.json` excludes them from the file watcher and from search — without that, VSCode
churns the extension host against a network mount and every search crawls 2.2 GB binaries.

> **Cross-filesystem `mv` is a copy.** Moving a few GB from POOL to `/orcd/data` can exceed a
> command timeout and leave a **truncated** file at the destination while the source is still intact.
> Use copy → verify sizes → delete source, not `mv`.

---

## UMAPs

UMAP visualizations are produced **as part of training** (no separate command needed):
- During each validation epoch, `validation_step` collects `num_umap_batches` (default 8) of HSC and Legacy images
- `on_validation_epoch_end` calls `plot_latent_space(hsc_batch, legacy_batch)`
- Plots: per-token + combined for the baseline; per-level + combined-spatial + global + instrument-global for the hierarchical
- Saved to `<run_dir>/figures/umap_latent_space_step{step}.png` and logged to WandB as `latent_space/umap_grid`

For **offline / one-off UMAPs** on a checkpoint, scripts live in `galaxy_images/galaxy_model/umap_exploration/`:
```bash
sbatch /work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model/umap_exploration/run_umap.slurm
```

---

## Downstream evaluation

The pattern is **two stages**: (1) generate frozen embeddings from a checkpoint, (2) train a small MLP per property and report R² on a held-out 10% split.

### For the flow-matching variants
```bash
# 1. Generate embeddings
python galaxy_images/galaxy_model/downstream_evaluation/prepare_neighbors.py \
  --checkpoint <path-to-ckpt> \
  --output <out>.h5

# 2. Train MLP probes per target property
python galaxy_images/galaxy_model/downstream_evaluation/predict_neighbors.py \
  --embeddings <out>.h5
```

### For the contrastive baseline
Use `galaxy_images/galaxy_model/contrastive_baseline/downstream_evaluation/` (parallel structure). Note: some scripts in `downstream_evaluation/` have hardcoded MIT paths (`/data/vision/billf/scratch/pablomer/...`) — they need a `--neighbors-hdf5` arg or manual edit when run on this cluster.

### What gets predicted
Galaxy properties from PROVABGS / MMU catalogs (mass, age, metallicity, SFR, ...) plus instrument-side properties (PSF FWHM, sky brightness, etc.). Instrument R² on the **physics** encoder is expected to be ~0 if disentanglement worked.

---

## Key scripts at a glance

| Want to...                                          | Run                                                                                              |
|-----------------------------------------------------|--------------------------------------------------------------------------------------------------|
| Train baseline                                      | `sbatch galaxy_images/galaxy_model/train_neighbors_efficient.slurm`                              |
| Train hierarchical                                  | `sbatch galaxy_images/galaxy_model/hierarchical_attention/train_neighbors_efficient_hier.slurm`  |
| Train single-encoder ablation                       | `sbatch galaxy_images/galaxy_model/single_encoder_ablation/train.slurm`                          |
| Train contrastive baseline                          | `sbatch galaxy_images/galaxy_model/contrastive_baseline/train_contrastive_amd.slurm`             |
| Quick smoke test (no GPU/SLURM, dataset only)       | `bash galaxy_images/galaxy_model/smoke_dataset_only.sh`                                          |
| Smoke test of efficient dataloader + 1 train step   | `bash galaxy_images/galaxy_model/smoke_neighbors_efficient.sh`                                   |
| Offline UMAP from checkpoint                        | `sbatch galaxy_images/galaxy_model/umap_exploration/run_umap.slurm`                              |
| Generate downstream embeddings                      | `python galaxy_images/galaxy_model/downstream_evaluation/prepare_neighbors.py ...`               |
| Train downstream MLP probes                         | `python galaxy_images/galaxy_model/downstream_evaluation/predict_neighbors.py ...`               |

---

## Reviews / rebuttal materials — hard rule

`reviews/` holds **official** peer-review files and Pablo's **official** responses
(`review_*.md`, `response_*.md`). **Never edit, overwrite, or reformat anything
directly in `reviews/`.** Treat those files as read-only source material.

Write all drafts, suggested wording, and analysis to **`reviews/claude-suggestions/`**
instead (e.g. `reviews/claude-suggestions/response_1_SYJm_q2.md`). Pablo merges
from there into the official response himself.

---

## Conventions

- **Image size**: 48×48 throughout. The center 48×48 is the science region (some datasets crop from 64×64 stamps).
- **Channels**: 4 (HSC has g/r/i/z analog; Legacy is matched to 4 channels). Configured by `cond_channels` / `in_channels`.
- **Anchor survey**: assigned **deterministically by dataset-index parity**, NOT randomly — `anchor_is_hsc = (idx % 2 == 0)` (`neighbors_ram48.py:71`, `neighbors_efficient.py:123`). Even positions are always HSC-target/Legacy-samegal, odd the reverse. `metadata[i]['anchor_survey']` is `'hsc'` or `'legacy'`. (This file previously claimed a random 50/50 draw; that was wrong. It matters: any subsampling of anchors must be parity-stratified or it will skew the survey mix, and per-survey recon MSE differs ~3.4×.)
- **Mask convention**: `masks: (B, k) bool`, `True` = real neighbor, `False` = padding.
- **Per-survey logging**: `train/loss_generative_hsc`, `train/loss_generative_legacy`, plus `val/mse_hsc`, `val/mse_legacy`.
- **Multi-GPU**: `ddp_find_unused_parameters_true` (some encoder paths are conditional, so `find_unused_parameters` must be on). `scale_steps_by_devices: true` divides `num_steps` by the device count.
- **Mixed precision**: `bf16-mixed` (both AMD and H100). `auto_adjust_for_h100` swaps batch_size to `h100_batch_size` if an H100 is detected.

---

## LR schedule — a latent quirk in every published run (read before training)

**All step-based training in this repo runs at learning rate 0 for roughly half of its
optimizer steps.** This is not a hypothesis; it is verified in this exact environment
(pl 2.6.1 / torch 2.6.0+cu124).

The cause is one line, historically present in every FM module *and* in the contrastive
baseline:

```python
scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
```

`train.py` (and `contrastive_baseline/train_contrastive_engaging.py`) pass only `max_steps`
and never `max_epochs`, so Lightning sets `trainer.max_epochs == -1`. With `T_max=-1` the
cosine does not anneal — **the LR alternates between `lr` and exactly `0.0` at every epoch
boundary**, and the default scheduler interval is `"epoch"`.

What that means concretely:

| | During an `lr=0` block |
|---|---|
| forward + backward | computed in full — the wasted cost |
| parameter update | `p -= lr·(…)` = **zero**; params frozen |
| AdamW decoupled weight decay | `p *= (1 − lr·wd)` → **also paused** |
| Adam moments, BatchNorm running stats | **still update** |

So a nominal 75k-step run delivers **≈37.5k effective updates**. The residual value of a dead
block is small (BN statistics keep refining, and the last ~10 steps feed `exp_avg` at frozen
parameters, giving one variance-reduced step per block). It is **not** a warm-restart /
SGDR-style benefit — parameters do not move at all during a pause, so there is no
exploration cycle.

**The alternation *period* is the epoch length, so it is dataset-size dependent.** At full
data an epoch is 385 steps (`ceil(98554/4)//64`); on 1% of the data it is ~3 steps. Any
ablation that varies dataset size at matched step count therefore varies the optimization
schedule too, unless you hold epoch length fixed. `scale_ablation/` handles this by tiling
its index list to the full train length — see `scale_ablation/PLAN.md` §3.

**Affected files:** `double_train_fm_neighbors.py` (the paper baseline), `double_train_fm.py`,
`double_train_fm_no-attn.py`, `train_fm.py`, `train_fm_no-attn.py`,
`contrastive_baseline/dual_encoder_contrastive.py`. Because the FM model and the contrastive
baseline are affected *identically*, the generative-vs-contrastive comparison is **not**
confounded by this. The AION probe heads (`aion_benchmark/predict_aion_*.py`) use the same
line but are trained *with* `max_epochs=50`, so there it is correct.

### The `model.lr_schedule` switch

`double_train_fm_neighbors.py::configure_optimizers` now takes a `lr_schedule` option
(`ModelConfig.lr_schedule`, i.e. `--set model.lr_schedule=…`). **The default is `"paper"`,
which reproduces published runs bit-for-bit — do not change the default.**

| Value | Behaviour | Use for |
|---|---|---|
| `"paper"` | **default.** The original line, alternating `lr` / `0.0` per epoch | reproducing or extending any published result; anything that must stay comparable to `checkpoints_new/base/` or `diffusion_ablation/frozen/*` |
| `"constant"` | flat `lr` every step, no scheduler | ~2× the effective updates for the same wall clock; the cheapest real fix |
| `"cosine"` | `CosineAnnealingLR(T_max=max_steps)`, `interval="step"` — what the original line evidently intended | **preferred for new runs**; reaches exactly 0 at `max_steps` |
| `"linear"` | `LinearLR` 1.0 → 0.0 over `max_steps`, `interval="step"` | alternative decay |

`"cosine"` and `"linear"` require `trainer.max_steps > 0` and raise a clear error otherwise.
Only `double_train_fm_neighbors.py` has the switch (and `ConditionalDDPMModule`, which
inherits it); the legacy modules above still hard-code the `"paper"` behaviour.

> ⚠️ **Do not "fix" this for the camera-ready without thinking it through.** Retraining the
> base model changes *every* number in the paper — all tables, the Fig. 2 UMAPs, the anomaly
> top-12, the cross-predict baseline. The published "4×H100, ~8h" figure is honest about wall
> clock; it just describes a model that received half the updates a reader would assume,
> which if anything is a favourable framing.

**Open probe:** `scale_ablation/configs/probe_base_constlr_37500.json` trains the base recipe
on **full data** with `lr_schedule="constant"` for **37,500** steps — i.e. the same ~37.5k
*effective* updates as the published 75k run, in half the wall clock. It is matched to
`diffusion_ablation/frozen/fm-control-75k-fixed.ckpt` in every other respect (same data, no
holdout, so `kept_size=103,741` and the existing 256-anchor recon manifest stays valid). If it
matches or beats that checkpoint, a scheduled run is strictly better and worth adopting for
future work. W&B group `lr-schedule-probe`, run name `base-constlr-37k5`.

---

## Common gotchas / debugging

1. **HipBLAS crash** — see workaround section above. Always confirm both env var and Python call are in place.
2. **Hardcoded MIT paths in downstream scripts** — `prepare_*.py` and some `predict_*.py` files have absolute MIT cluster paths. Search for `/data/vision/billf/` before running on this cluster.
3. **`val/loss` looks frozen at 0** — check `lambda_generative > 0` in the config; the generative branch is short-circuited otherwise.
4. **`find_unused_parameters` warnings** — expected for variants where some encoder paths are skipped at runtime. Don't silence them by changing the strategy.
5. **`encoder_2` outputs differ between baseline and hierarchical**:
   - baseline: `mean_pool=False` → spatial tokens, joined with encoder_1 in cross-attention
   - hierarchical: `mean_pool=True` → single vector, injected into time embedding only
   - This is the root cause of the hierarchical "instrument leak into physics latent" — see `hierarchical_attention/INSTRUMENT_LEAK_OPTIONS.md`.
6. **`save_heldout_validation: true` once, then turn off** — first run with this flag exports a fixed validation subset to `data.heldout_validation_dir`; subsequent runs should set it to false.

---

## Pointers to past investigations / write-ups

- `galaxy_images/galaxy_model/hierarchical_attention/INSTRUMENT_LEAK_OPTIONS.md` — diagnosis and four candidate fixes for the hierarchical physics-latent instrument leak
- `galaxy_images/galaxy_model/contrastive_baseline/README.md` — contrastive A/B variants (A/B/C)
- `galaxy_images/galaxy_model/README.md` — minimal training notes (older, partially superseded by this file)

---

## Working preferences (Pablo)

- Trained on PyTorch Lightning + SLURM. Prefers running multiple training jobs in parallel.
- Comfortable with contrastive learning, InfoNCE, ResNet encoders, MLP probes.
- Default downstream-eval style: train small MLP per property, report R² on a 10% held-out split.
- When asked to plan a fix, present the architectural reasoning first, then the smallest experiment that confirms-or-fixes the hypothesis.
