 ready# Hierarchical Global-Instrument Variant

## Summary
- Add a new hierarchical neighbors variant where the galaxy path stays hierarchical, the instrument path becomes global via `ResNetEncoder(mean_pool=True)`, and the UNet receives separate galaxy-global and instrument-global conditioning.
- Set `lambda_geometric=0.0` by default and keep the geometric-loss metric logged as zero for W&B/dashboard parity.
- Integrate the variant into the same unified training/config pipeline used by `train_neighbours.slurm`, while also adding the requested standalone wrapper script.
- Fix latent-space visualization so each validation pair contributes both the HSC image and the Legacy image to the UMAP, not only the alternating anchor side.

## Public API / Interfaces
- Add a public `encode_image(image: torch.Tensor) -> dict` method on the new model for checkpoint-side analysis, downstream tasks, UMAP generation, and offline metrics.
- `encode_image` is image-only:
  - input shape `(B, C, H, W)`
  - no neighbor pooling
  - no pair logic
- Return `physics` with raw + convenience outputs:
  - `spatial_levels`: list of dicts with `tokens`, `height`, `width`, `rope`
  - `level_flats`: list of `(B, N_i * D)`
  - `spatial_concat`: `(B, sum_i N_i, D)`
  - `spatial_flat`: `(B, sum_i N_i * D)`
  - `global_vec`: `(B, global_dim)`
- Return `instrument` with raw + convenience outputs:
  - `tokens`: `(B, 1, instrument_zdim)`
  - `flat`: `(B, instrument_zdim)`
- Keep neighbor-pooling as a separate internal helper used only by the generative forward path.

## Implementation Changes
- Add `hierarchical_attention/double_train_fm_neighbors_hier_global_ins.py` with:
  - `HierarchicalGlobalInstrumentFlowMatchingModule`
  - local `DualGlobalConditionedUNet`
- Keep `encoder_1 = ConfigurableEncoder(...)` for physics latents and galaxy-only cross-attention.
- Use `encoder_2 = ResNetEncoder(mean_pool=True)` for per-image instrument latents.
- Add an internal masked-mean helper for training-time neighbor pooling:
  - `(B, k, instrument_zdim) -> (B, instrument_zdim)`
  - clamp denominator to avoid divide-by-zero
  - all-false masks return zeros
- Remove neighbor-token concatenation and the neighbor-aware RoPE processor; reuse the standard hierarchical RoPE path.
- Register a unified variant such as `neighbors_hier_global_ins` in `variants.py`.
- Extend shared config support with hierarchical-only fields:
  - `model.experiment_config`
  - `model.instrument_zdim`
  - `model.instrument_pooling`
- Update model construction in the unified trainer to pass only constructor-supported kwargs so existing non-hierarchical variants remain unaffected.
- Add a dedicated config JSON for the new variant and a thin wrapper script `hierarchical_attention/neighbours_train_hier_global_ins.py`.
- Add `hierarchical_attention/README_hier_global_instrument.md` documenting architecture, shapes, conditioning split, and the encoding API.

## Tracking and Visualization
- Keep W&B metric/image keys aligned with the standard neighbors runs:
  - `train/loss`, `val/loss`
  - `train/val loss_generative_*`
  - `train/val loss_geom_total`
  - `val/mse`, `val/mse_32`, `val/mse_hsc`, `val/mse_legacy`
  - `val/sample_grid`, `val/sample_grid_row_scaled`
  - `latent_space/umap_grid`
- Fix the current UMAP bug by reconstructing both sides of each validation pair:
  - if `anchor_survey == "hsc"`: `hsc = anchor`, `legacy = samegal`
  - if `anchor_survey == "legacy"`: `hsc = samegal`, `legacy = anchor`
- Build UMAPs from `2N` encoded images:
  - physics: one panel per spatial level from `level_flats`, one combined spatial panel from `spatial_flat`, and one `global_vec` panel
  - instrument: one paired-image panel from `instrument.flat`
- Use `encode_image(...)` inside the new UMAP path so visualization and downstream extraction share the same latent definitions.
- Where practical, refactor the existing neighbors UMAP code to reuse the same pair-reconstruction helper so the bug is fixed consistently, not only in the new variant.

## Test Plan
- Forward/sampling smoke tests with mixed masks and with all-neighbor masks false.
- Shape tests for:
  - `encode_image.physics.spatial_levels`
  - `encode_image.physics.level_flats`
  - `encode_image.physics.spatial_flat`
  - `encode_image.physics.global_vec`
  - `encode_image.instrument.tokens`
  - `encode_image.instrument.flat`
  - pooled training-time instrument latent
  - model output `(B, C, H, W)`
- Pooling correctness tests:
  - padded neighbors do not affect the pooled instrument latent
  - changing valid neighbors changes the pooled latent
  - all-false masks produce a zero vector
- UMAP data-path test:
  - validation reconstruction includes both HSC and Legacy images
  - point count is `2N` per encoder input set
- Unified integration test:
  - new config builds through `train.py`
  - existing variants still build after shared config changes
- Logging regression test:
  - W&B keys match the standard neighbors setup
  - `loss_geom_total` is present and zero by default

## Assumptions and Defaults
- Default experiment config is `bn_36x16`.
- Default instrument visualization is paired-image only; pooled neighbor-set instrument conditionings are not shown by default.
- `encode_image` is the public checkpoint-analysis API; pair reconstruction and neighbor pooling stay outside that method.
- Convenience embeddings are returned alongside raw hierarchical outputs so downstream scripts do not have to duplicate flatten/combine logic.
