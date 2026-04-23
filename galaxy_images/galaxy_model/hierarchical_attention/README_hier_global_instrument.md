# Hierarchical Galaxy + Global Instrument Neighbors

## Purpose

This variant splits conditioning into two paths:

- the same-galaxy image provides hierarchical spatial context plus a galaxy-global summary
- the same-instrument neighbor set provides one pooled global instrument summary

The main difference from `double_train_fm_neighbors_hier.py` is that same-instrument neighbors no longer contribute token sequences to cross-attention. Cross-attention is galaxy-only, while the pooled instrument signal is injected separately into the UNet embedding path.

## Architecture

- `ConfigurableEncoder` encodes the same-galaxy image into:
  - `spatial_levels`: level-routed spatial tokens for cross-attention
  - `global_vec`: a galaxy-global summary
- `ResNetEncoder(mean_pool=True)` encodes each same-instrument neighbor into one token of shape `(B, 1, instrument_zdim)`
- masked mean pooling aggregates valid neighbor embeddings into `instrument_global`
- `DualGlobalConditionedUNet` keeps the existing galaxy-global projection path and adds a second learned MLP for `instrument_global`
- UNet cross-attention receives only galaxy spatial tokens, so standard hierarchical RoPE is reused unchanged

## Tensor Shapes

Assuming:

- target batch: `(B, C, H, W)`
- same-galaxy batch: `(B, C, H, W)`
- same-instrument neighbors: `(B, K, C, H, W)`

Then the main tensors are:

- galaxy spatial level `i`: `(B, N_i, token_dim)`
- galaxy global vector: `(B, global_dim)`
- per-neighbor instrument token: `(B, K, instrument_zdim)` after squeezing the singleton token axis
- pooled instrument global: `(B, instrument_zdim)`
- UNet cross-attention inputs: one galaxy spatial level at a time
- UNet global embedding inputs:
  - galaxy global via `class_labels`
  - instrument global via `instrument_global_labels`

## Data Flow

1. Encode the same-galaxy image with `ConfigurableEncoder`.
2. Encode every same-instrument neighbor with `ResNetEncoder(mean_pool=True)`.
3. Aggregate neighbor embeddings with masked mean pooling.
4. Feed galaxy spatial levels into the routed UNet cross-attention blocks.
5. Add the galaxy-global and instrument-global projections into the UNet embedding stream.
6. Predict flow-matching velocity and compute the usual generative loss.
7. If enabled, compute geometric loss from flattened galaxy spatial embeddings only.

## Public Encoding API

`HierarchicalGlobalInstrumentFlowMatchingModule.encode_image(image)` is the image-only checkpoint analysis interface. It does not perform pair reconstruction or neighbor pooling.

It returns:

- `physics.spatial_levels`: list of dicts with `tokens`, `height`, `width`, and `rope`
- `physics.level_flats`: per-level flattened embeddings
- `physics.spatial_concat`: all spatial tokens concatenated along the sequence axis
- `physics.spatial_flat`: flattened concatenation of all spatial tokens
- `physics.global_vec`: galaxy-global summary
- `instrument.tokens`: mean-pooled instrument tokens of shape `(B, 1, instrument_zdim)`
- `instrument.flat`: flattened instrument embeddings of shape `(B, instrument_zdim)`

## Difference From The Current Hierarchical Neighbors Model

- old model: appends neighbor tokens to every galaxy spatial level
- new model: compresses the neighbor set into one pooled instrument-global vector
- old model: needs a neighbor-aware RoPE processor
- new model: reuses standard hierarchical RoPE because keys are spatial-only again
- old model: mixes same-instrument conditioning into cross-attention
- new model: keeps instrument conditioning explicit and separate in the global path

## Assumptions And Limitations

- instrument information is assumed to be mostly global rather than spatially aligned
- masked mean gives every valid neighbor equal weight
- all-false neighbor masks fall back to a zero vector, which weakens conditioning but keeps the path well-defined

## Downstream Hooks

- swap `instrument_pooling` for alternate aggregators
- add survey-aware or metadata-aware instrument pooling
- attach classifier or regressor heads to `physics.global_vec` or `instrument.flat`
- run ablations for no instrument path, merged globals, or tokenized instrument conditioning
