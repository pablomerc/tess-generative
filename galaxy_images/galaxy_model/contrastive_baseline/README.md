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

## Encoder pooling variants

`ResNetEncoder` supports two ways of collapsing the ResNet18 `(B, 512, H', W')`
feature map into the per-image embedding, selected by `encoder_pool`:

- `"avg"` (default): global average pool over `H'×W'` then `Linear(512 → embedding_dim)`.
  Spatially invariant, so position/texture cues (PSF/blur) are discarded.
  Output width = `embedding_dim`.
- `"conv1x1"`: `Conv2d(512 → token_dim, 1)` keeping every spatial location as a token,
  then flatten. Mirrors the encoder in `double_train_fm_neighbors.py`
  (`nn.Conv2d(512, cross_attention_dim, 1)` with spatial tokens preserved), so spatial
  cues survive into the embedding. By default `token_dim = embedding_dim // (H'×W')`,
  so the flattened output width **equals `embedding_dim`** — i.e. dim-matched to `avg`
  (e.g. 48×48 → 2×2 map → `token_dim = 64/4 = 16` → 16×4 = 64-d). Override with
  `encoder_token_dim` for a wider latent. `embedding_dim` must be divisible by `H'×W'`.

The projection heads and all downstream code size themselves from
`encoder.out_dim`, so both variants are drop-in. Existing `avg` checkpoints keep
loading unchanged (missing hparams default to `avg`).

## Run

From `tess-generative` root (or with equivalent `PYTHONPATH`):

```bash
# Variant 1: average pooling (original)
python galaxy_images/galaxy_model/contrastive_baseline/train_dual_encoder_contrastive.py

# Variant 2: no pooling / 1x1-conv spatial tokens (main-model-consistent)
ENCODER_POOL=conv1x1 python galaxy_images/galaxy_model/contrastive_baseline/train_dual_encoder_contrastive.py
```

`ENCODER_POOL` feeds the run name and checkpoint directory
(`dual-encoder-contrastive-resnet18-{avg,conv1x1}`), so the two runs never
collide.

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
