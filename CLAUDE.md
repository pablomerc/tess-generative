# tess-generative — Quick Orientation for Claude

> Catch-up notes for new sessions. The user (Pablo, p.mercader01@gmail.com) is an ML researcher working on self-supervised / generative representation learning for galaxy survey images (HSC + Legacy Survey). This file lives at the repo root.

---

## What this codebase does

Trains dual-encoder generative models that learn **physics** and **instrument** latent spaces from paired galaxy images across two surveys (HSC and Legacy Survey). The main approach is **flow matching** (conditional velocity prediction) with two ResNet18 encoders feeding a `diffusers` UNet via cross-attention. A separate **contrastive baseline** (InfoNCE on dual encoders) lives in `galaxy_images/galaxy_model/contrastive_baseline/`.

**Active clusters: AMD (this one) and Engaging.** Development happens on both; they share the GitHub remote `git@github.com:pablomerc/tess-generative.git` and the working branch is `galaxy-engaging`.

- **AMD** (this cluster, `/work1/jeroenaudenaert/pablomer/`): AMD MI210 GPUs plus an H100 partition. **Needs the hipBLASLt workaround** (see below).
- **Engaging** (`/home/<user>/... — confirm exact path on Engaging`): NVIDIA GPUs (CUDA). **Does not need the hipBLASLt workaround.**

The old MIT/CSAIL cluster (`/data/vision/billf/scratch/pablomer/`) is **no longer in active use** — treat its hardcoded paths in older downstream scripts as legacy to parameterize before running anywhere. **Only the neighbors dataset is available on the AMD cluster.**

---

## Working across clusters (AMD ⇄ Engaging)

Both clusters develop on `galaxy-engaging` and sync through the shared GitHub remote.

- **Never force-push a shared branch** (`galaxy-engaging`, `main`). To move work between clusters, push to a per-cluster snapshot branch (`amd-snapshot`, `engaging-snapshot`) and merge from there, so nothing gets clobbered.
- **Cluster-specific job scripts diverge on purpose.** SLURM/`.sh` submission scripts and training entry points carry per-cluster settings (partitions, module loads, the hipBLASLt workaround on AMD). When merging, **keep both variants** rather than overwriting one cluster's scripts with the other's.
- For code that must run on **both** clusters, use the guarded hipBLASLt form (`try/except` around `preferred_blas_library`, as in `downstream_evaluation/engaging/`) — it's a no-op on NVIDIA and a fix on AMD.

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
Both are required (env var for new processes, Python call for the main process). If you see HIPBLAS errors, confirm both are present. The H100 partition is unaffected (config has `auto_adjust_for_h100: true`).

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
  - `latest-step={step}.ckpt` (saved every `checkpoint_every_n_train_steps`)
  - `best-epoch={epoch}-step={step}.ckpt` (best `val/loss`)
- Hierarchical run dir: `galaxy_images/galaxy_model/hierarchical_attention/outputs/...`
- Contrastive baseline runs land under `/work1/jeroenaudenaert/pablomer/outputs/contrastive_baseline/`
- WandB project: `galaxy-flow-matching-neighbours` (contains all flow-matching runs and their UMAP grids)

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

## Conventions

- **Image size**: 48×48 throughout. The center 48×48 is the science region (some datasets crop from 64×64 stamps).
- **Channels**: 4 (HSC has g/r/i/z analog; Legacy is matched to 4 channels). Configured by `cond_channels` / `in_channels`.
- **Anchor survey**: dataset randomizes per-sample (50/50 HSC vs Legacy). `metadata[i]['anchor_survey']` is `'hsc'` or `'legacy'`.
- **Mask convention**: `masks: (B, k) bool`, `True` = real neighbor, `False` = padding.
- **Per-survey logging**: `train/loss_generative_hsc`, `train/loss_generative_legacy`, plus `val/mse_hsc`, `val/mse_legacy`.
- **Multi-GPU**: `ddp_find_unused_parameters_true` (some encoder paths are conditional, so `find_unused_parameters` must be on). `scale_steps_by_devices: true` divides `num_steps` by the device count.
- **Mixed precision**: `bf16-mixed` (both AMD and H100). `auto_adjust_for_h100` swaps batch_size to `h100_batch_size` if an H100 is detected.

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
