# Contrastive Dual-Encoder Baseline

This folder contains a baseline model to compare against:

- `galaxy_images/galaxy_model/double_train_fm_neighbors.py`

The baseline trains a **dual encoder** with a **contrastive objective only** (no reconstruction or generative loss).

## Files

- `dual_encoder_contrastive.py`
  - `ResNetEncoder`: ResNet18-based image encoder (4-channel input support).
  - `ProjectionHead`: MLP projection head for contrastive learning.
  - `DualEncoderContrastiveModule` (PyTorch Lightning):
    - `encoder_galaxy` + `head_galaxy`
    - `encoder_instrument` + `head_instrument`
    - training/validation with two contrastive losses.
- `train_dual_encoder_contrastive.py`
  - Training entrypoint using the same neighbors precomputed data format used in neighbors FM training.
  - UMAP logging is optional and currently enabled.

## Batch format

The model expects batches from `NeighborsPrecomputedDataset` + `simple_collate`:

- `targets`: `(B, C, H, W)`
- `samegals`: `(B, C, H, W)` (same galaxy, other survey)
- `sameins`: `(B, K, C, H, W)` (neighbors from same survey/instrument)
- `masks`: `(B, K)` bool mask for valid neighbor slots
- `metadata`: list of dicts

## Losses

Total loss:

`L = lambda_galaxy * L_galaxy + lambda_instrument * L_instrument`

- `L_galaxy`:
  - CLIP/InfoNCE-style symmetric in-batch contrastive loss.
  - Positive pair: `(target_i, samegal_i)`.
  - Negatives: other items in the batch.
- `L_instrument`:
  - Multi-positive InfoNCE.
  - For anchor `target_i`, all valid `sameins_i,*` are positives.
  - Valid neighbors from other anchors are negatives.
  - `masks` are used to ignore padded neighbor slots.

## Run

From `tess-generative` root (or with equivalent `PYTHONPATH`):

```bash
python galaxy_images/galaxy_model/contrastive_baseline/train_dual_encoder_contrastive.py
```

## Notes

- The code is intentionally aligned with the existing neighbors training setup:
  - same precomputed dataset source
  - same train/val split pattern
  - same Lightning + WandB style
- Optional UMAP logging:
  - Controlled by `enable_umap_logging` in `DualEncoderContrastiveModule`.
  - Uses validation targets split by `anchor_survey` and logs one 1x2 panel:
    `Galaxy Encoder UMAP` and `Instrument Encoder UMAP`.
- There is no decoder and no reconstruction term in this baseline.
