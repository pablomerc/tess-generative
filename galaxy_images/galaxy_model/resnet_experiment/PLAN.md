# ResNet Ellipticity Experiment — Plan

## Goal
Train a ResNet-18 to predict galaxy shape parameters (SHAPE_E1, SHAPE_E2) from
4-channel HSC images, then evaluate how well a flow-matching generative model
preserves those shape parameters by running the same ResNet on generated images.

---

## Data pipeline

### Source
Parquet files in `/data/vision/billf/scratch/pablomer/legacysurvey_hsc/data`
- 396 files (`train-*-of-00396.parquet`)
- ~101,771 rows total
- Each row has `hsc_image` (nested dict), `SHAPE_E1`, `SHAPE_E2`,
  `legacysurvey_object_id`

### `hsc_image` dict structure (inside each parquet row)
```
hsc_image:
  band      → list of 5 band names: HSC-G, HSC-R, HSC-I, HSC-Z, HSC-Y
  flux      → list of 5 bands, each is an array of 160 1D arrays (160 pixels)
              i.e., flux[b] is NOT a 2D array — must use np.stack(flux[b])
  ivar, mask, psf_fwhm, scale
```
Use bands 0–3 (G, R, I, Z). Skip band 4 (Y). Stack → (4, 160, 160),
then center-crop to 48×48 (offsets: `[56:104, 56:104]`).

### `prepare_data.py` (DONE)
Reads all 396 parquets → `resnet_data.h5`:
- `hsc_images`   (N, 4, 48, 48)  float32  raw flux
- `shape_e1`     (N,)             float32
- `shape_e2`     (N,)             float32
- `ls_object_id` (N,)             bytes    legacysurvey_object_id

Normalization parameters (from data.py): mean=0.022, std=0.05 — applied on-the-fly.

---

## Stage 1 — Train ResNet

### `train.py` (TO CREATE)

#### Dataset: `HSCEllipticityDataset`
- Reads `resnet_data.h5` lazily (keep file open)
- `__getitem__`: normalize image with `(img - 0.022) / 0.05`
- Returns `(image, e1, e2)`

#### Model: `ResNet18Ellipticity`
Standard torchvision ResNet-18 with two modifications:
1. `model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)`
   (3→4 input channels)
2. `model.fc = nn.Linear(512, 2)` (output 2 values: e1, e2)
Note: 48×48 input → after 7×7 conv (stride 2) → 24×24 → pool → 12×12 → ...
→ GlobalAvgPool → 512. Works fine despite smaller than ImageNet's 224×224.

#### Training
- 80/20 train/test split, seed=42, test indices saved to `split_indices.json`
- Loss: MSE on (SHAPE_E1, SHAPE_E2)
- Optimizer: Adam, lr=1e-4
- Epochs: 50 (or until converged)
- Save best checkpoint by validation loss → `resnet_best.pth`
- Log train/val loss each epoch

#### Evaluation (test set)
- Scatter plots: predicted vs. true SHAPE_E1 and SHAPE_E2
- R² score for each component
- Histogram of residuals
- Save plots to `outputs/`

---

## Stage 2 — Evaluate on Generated Images

### Concept
We train on real HSC images. Now we want to check: if the generative model produces
a plausible galaxy image given neighbors as conditioning, does the ResNet (trained on
real data) predict the correct ellipticity for that generated image?

A systematic bias (generated images → wrong ellipticity) would indicate the generative
model is NOT preserving shape information faithfully.

### `evaluate_generated.py` (TO CREATE)

#### Step 1: Load a shard from neighbors dataset
Use a validation shard (NOT from training data for the flow-matching model).
E.g., `/data/vision/billf/scratch/pablomer/data/neighbor_batches/neighbors_shard_NNNN.h5`

Shard has:
- `targets`        (N, 4, 48, 48) — real galaxy images (HSC or Legacy)
- `samegals`       (N, 4, 48, 48) — same galaxy, opposite survey
- `sameins`        (N, k, 4, 48, 48) — same-instrument neighbors
- `neighbor_masks` (N, k)
- `meta_survey`    (N,) — b'hsc' or b'legacy'
- `meta_idx`       (N,) — index into `neighbours_v2.h5`

**Filter**: keep only rows where `meta_survey == b'hsc'` (we trained on HSC).

#### Step 2: Generate 1 sample per galaxy
Load flow-matching checkpoint (e.g., `g2g9kvr4`, the nogeom neighbors model).
For each selected galaxy: call `model.sample(samegal, sameins, masks, num_steps=250)`
→ 1 generated image of shape (4, 48, 48).

#### Step 3: Crossmatch to get ground truth ellipticities
Critical: NEVER match by position. Always join by object ID.

Chain:
```
shard['meta_idx'][i]
  → neighbours_v2.h5['object_id_legacy'][meta_idx]   (bytes ID)
  → lookup dict built from resnet_data.h5:
      {ls_object_id[j] → (shape_e1[j], shape_e2[j]) for j in range(N)}
  → ground truth (SHAPE_E1, SHAPE_E2) for galaxy i
```

Note: `neighbours_v2.h5` has `legacy_SHAPE_E1` but NOT `legacy_SHAPE_E2`.
So SHAPE_E2 must come from the parquet-derived lookup (resnet_data.h5).

#### Step 4: Run ResNet
Run the trained ResNet on:
- Real targets  → predicted (e1_real, e2_real)
- Generated images → predicted (e1_gen, e2_gen)

#### Step 5: Plots
- Scatter: true vs. real-image predictions (sanity check)
- Scatter: true vs. generated-image predictions
- Comparison: |e1_real - e1_true| vs. |e1_gen - e1_true| (bias in generated)
- Histograms of residuals for real vs. generated

---

## File layout

```
resnet_experiment/
  prepare_data.py       DONE   — parquets → resnet_data.h5
  train.py              TODO   — ResNet-18 training
  evaluate_generated.py TODO   — Stage 2 evaluation
  PLAN.md                       this file
  resnet_data.h5        (generated by prepare_data.py)
  resnet_best.pth       (generated by train.py)
  split_indices.json    (generated by train.py)
  outputs/
    train_val_loss.png
    test_scatter_e1.png
    test_scatter_e2.png
    generated_scatter_e1.png
    generated_scatter_e2.png
```

---

## Key constraints (do NOT violate)

1. **No positional alignment** between parquet rows and preprocessed H5 rows
   (46 rows were dropped during preprocessing). Always join by `legacysurvey_object_id`.

2. **Normalization**: raw flux images, normalize on-the-fly with mean=0.022, std=0.05.

3. **Band order**: HSC bands in parquet are [G, R, I, Z, Y] at indices 0–4.
   Use only 0–3 (skip Y). This matches how `data.py` and `neighbours_train.py` use HSC images.

4. **`meta_idx` is in the shard, not in `full_samples.h5`**. For Stage 2, load the
   shard directly (don't reuse `xi_squared_experiment/outputs/full_samples.h5`).

5. **`neighbours_v2.h5` lacks SHAPE_E2**. Must use the parquet-based lookup dict
   (or `resnet_data.h5['ls_object_id']`) to retrieve SHAPE_E2.

---

## Model checkpoint for Stage 2

Flow-matching model: `g2g9kvr4` (zdim16_nogeom_neighbors)
Checkpoint: `/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy-flow-matching-neighbours/g2g9kvr4/checkpoints/latest-step=step=75000.ckpt`

---

## Python environment
Always use: `/data/vision/billf/scratch/pablomer/miniconda3/envs/py310-torch/bin/python`
