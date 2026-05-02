# Engaging downstream evaluation

Unified HSC + Legacy downstream eval. Uses the precomputed binaries at
`/work1/jeroenaudenaert/pablomer/data/hsc_downstream/` and
`/work1/jeroenaudenaert/pablomer/data/legacy_downstream_quick/` directly — no
runtime mixing of MIT-cluster sources (unlike `../final/`).

This branch (`galaxy-engaging`, forked from `galaxy-amd`) is intended to be
pulled and run on the **engaging** cluster.

## What it produces

For each checkpoint in the registry (`base`, `hier`, `hier-small`,
`single-baseline`), trains MLP probes on three task families × three latent
variants (HSC-only / Legacy-only / Combined) and produces a comparison plot
in the spirit of `../final/makeplot_v2.py`.

## Pipeline (3 stages)

```
build_overlap_dataset.py       # crossmatch + sample 3 subsets, write index parquets
        ↓  outputs/index/{overlap,hsc_only,legacy_only}.parquet
prepare_combined.py            # embed each subset with one checkpoint
        ↓  outputs/embeddings/{ckpt_name}.h5     (one per checkpoint)
predict_combined.py            # train MLPs across (task × latent × ckpt)
        ↓  outputs/predictions/predict_{ckpt_name}.csv
makeplot.py                    # bar chart per ckpt + cross-ckpt summary
        ↓  outputs/plots/compare_{ckpt_name}.png  +  compare_all_checkpoints.png
```

## Subsets and sampling

Three independent samples, capped at **16 000** rows each (random,
`seed=42`):

- `overlap` — galaxies present in both surveys (matched within 1″ on the sky).
  Used for HSC-only, Legacy-only, **and** combined-latent comparisons on
  physics tasks.
- `hsc_only` — HSC galaxies *not* in the overlap. Used for HSC-instrument
  tasks; gives statistical power on the HSC side.
- `legacy_only` — Legacy galaxies not in the overlap. Same role for Legacy.

## Combined latents

For an overlap galaxy with HSC image `I_h` and Legacy image `I_l`:

```
combined_e1 = concat([encoder_1(I_h), encoder_1(I_l)])    # physics
combined_e2 = concat([encoder_2(I_h), encoder_2(I_l)])    # instrument
```

Doubles the latent dim; the downstream MLP is free to weight each survey.

## Checkpoint sweep

The pipeline runs over all four registered variants. Per-variant load info is
in `../../checkpoints/README.md`. Default checkpoint for a single-variant run
is `base`.

| Variant            | Module                                                                                          | Class                                              |
|--------------------|-------------------------------------------------------------------------------------------------|----------------------------------------------------|
| `base`             | `double_train_fm_neighbors.py`                                                                  | `ConditionalFlowMatchingModule`                    |
| `hier`             | `hierarchical_attention/double_train_fm_neighbors_hier_global_ins.py`                           | `HierarchicalGlobalInstrumentFlowMatchingModule`   |
| `hier-small`       | `hierarchical_attention/double_train_fm_neighbors_hier_global_ins.py`                           | `HierarchicalGlobalInstrumentFlowMatchingModule`   |
| `single-baseline`  | `single_encoder_ablation/model.py`                                                              | `SingleEncoderFlowMatchingModule`                  |

Note: `single-baseline` has only one encoder. For that variant the pipeline
will produce only `*_e1` embeddings (no `_e2`), and physics/instrument tasks
share the same latent.

## ⚠ Known blocker — HSC binary doesn't currently overlap Legacy

As of this branch, `hsc_flux.bin` contains only the first **57 564** of the
474 954 catalog rows (HEALPix tiles **1104, 1105** — the XMM-LSS region,
RA ~33-37°, Dec ~-6 to -4°). Legacy lives at RA 55-155°, Dec -54 to +5°.
**Zero galaxies overlap.**

The full HSC catalog *would* yield **19 622** matches with Legacy at 1″
separation, but **every one of them is in HEALPix tile 1703**, which is not
in the current binary. Until that tile is added to (or `hsc_flux.bin` is
rebuilt from) HSC PDR3, the `overlap` subset will be empty and only the
`hsc_only` and `legacy_only` evaluations will produce results.

`build_overlap_dataset.py` detects this case and prints a clear diagnostic.

## Data invariants

- HSC images: `(N, 5, 160, 160) float16` in `hsc_flux.bin`. Bands g/r/i/z/y.
  We center-crop to 48×48, arcsinh-normalize, drop y → `(4, 48, 48)`.
- Legacy images: `(N, 4, 160, 160) float16` in `legacy_flux.bin`. Bands
  g/r/i/z. Center-crop to 48×48, arcsinh-normalize → `(4, 48, 48)`.
- HSC binary contains a **subset** of the catalog (~57 565 of 474 954 rows
  as of writing). `build_overlap_dataset.py` derives N from file size and
  clips to `image_idx < N_actual`.
- Legacy binary matches its catalog exactly (330 581 rows).

## Quick start (on engaging, after pulling this branch)

```bash
# 1. Build the index parquets (one-shot, ~minutes — pure CPU)
python build_overlap_dataset.py \
    --hsc-dir    /path/to/hsc_downstream \
    --legacy-dir /path/to/legacy_downstream_quick \
    --out-dir    outputs/index \
    --max-per-subset 16000

# 2. Embed for one checkpoint (GPU, ~30 min full / ~2 min smoke)
python prepare_combined.py \
    --checkpoint   ../../checkpoints/base/snapshot.ckpt \
    --module       double_train_fm_neighbors.py \
    --model-class  ConditionalFlowMatchingModule \
    --index-dir    outputs/index \
    --out          outputs/embeddings/base.h5

# 3. Train MLP probes
python predict_combined.py \
    --embeddings outputs/embeddings/base.h5 \
    --out        outputs/predictions/predict_base.csv

# 4. Plot
python makeplot.py --predictions-dir outputs/predictions --out-dir outputs/plots
```

Or end-to-end on a tiny subset:

```bash
bash smoke_test.sh
```

## Files

| File                       | Status      | Notes                                                  |
|----------------------------|-------------|--------------------------------------------------------|
| `README.md`                | written     | This file                                              |
| `build_overlap_dataset.py` | **working** | Crossmatch + sampling, no GPU needed                   |
| `prepare_combined.py`      | stub        | Fill in embedding loop on engaging                     |
| `predict_combined.py`      | stub        | Fill in MLP loop on engaging                           |
| `makeplot.py`              | stub        | Fill in plotting on engaging                           |
| `run_build.slurm`          | stub        | Sbatch wrapper for build_overlap_dataset.py            |
| `run_prepare.slurm`        | stub        | Sbatch wrapper that loops over the 4 checkpoints       |
| `run_predict.slurm`        | stub        | Sbatch wrapper for predict_combined.py                 |
| `smoke_test.sh`            | stub        | End-to-end on 512 rows per subset, `base` only         |

Outputs land under `outputs/` (gitignored).

## Reuses

- `../new_data_downstream/prepare_hsc_downstream.py` — `load_trained_model`,
  `load_untrained_model`, `HSCBinaryDataset` file-seek pattern.
- `../new_data_downstream/predict_hsc_downstream.py` — MLP training loop.
- `../final/makeplot_v2.py` — bar-chart styling and band-averaging regex.
- `galaxy_images/image_preprocessing.py` — `preprocess_image_v2` for HSC and
  Legacy.
- `../../checkpoints/` — registry; pull `snapshot.ckpt` from each variant dir.
