# Hierarchical-Galaxy + Global-Instrument Neighbor Model

  ## Summary

  Create a new hierarchical-neighbors variant that keeps the same-galaxy encoder on the existing hierarchical-attention path, but changes the same-instrument
  encoder from cross-attention neighbor tokens to a single global conditioning signal derived from masked mean-pooled neighbor embeddings.

  The implementation should be delivered as:

  - a new model module under galaxy_images/galaxy_model/hierarchical_attention/
  - a companion standalone training script
  - a colocated Markdown design document describing architecture, tensor shapes, data flow, and downstream-task hooks

  The main design choice is to keep the instrument-global signal explicit and separate from the galaxy-global signal. Concretely: the galaxy global vector
  continues to feed the UNet’s existing class_embedding path, while the instrument global vector gets its own projection path into the UNet timestep embedding,
  instead of being merged into the galaxy vector first.

  ## Files To Add

  ### New model file

  Add:

  - galaxy_images/galaxy_model/hierarchical_attention/double_train_fm_neighbors_hier_global_ins.py

  This file should be based on double_train_fm_neighbors_hier.py, but with these structural changes:

  - retain hierarchical galaxy conditioning from ConfigurableEncoder
  - replace neighbor-token concatenation with masked global aggregation
  - remove the neighbor-aware RoPE processor, because cross-attention keys will now contain only galaxy spatial tokens
  - introduce a small wrapper around ConditionedUNet for the separate instrument-global path

  ### New training entrypoint

  Add:

  - galaxy_images/galaxy_model/hierarchical_attention/neighbours_train_hier_global_ins.py

  This should mirror neighbours_train_hier.py, but import and instantiate the new module and use a distinct default W&B run name.

  ### New documentation

  Add:

  - galaxy_images/galaxy_model/hierarchical_attention/README_hier_global_instrument.md

  This should describe the model in enough detail that a later agent can implement evaluation, ablations, and downstream uses without re-reading the code.

  ## Model Architecture

  ### 1. Galaxy encoder stays hierarchical

  Keep the existing galaxy path unchanged in spirit:

  - encoder_1 = ConfigurableEncoder(...)
  - output:
      - spatial_levels: list[(tokens, h, w)]
      - global_vec: (B, global_dim)
      - rope_flags

  Use the hierarchical spatial levels exactly as in the current hierarchical model:

  - level-routed cross-attention into the UNet
  - standard RoPE on Q and K for compatible spatial levels
  - global_vec continues to act as the galaxy-level global summary

  ### 2. Instrument encoder becomes global

  Use the existing ResNetEncoder from double_train_fm_neighbors.py with:

  - mean_pool=True

  Per neighbor image, this returns:

  - (B*k, 1, instrument_zdim)

  Then in the new module:

  1. flatten neighbors to (B*k, C, H, W)
  2. encode with encoder_2
  3. squeeze the singleton token dimension to (B*k, instrument_zdim)
  4. unflatten to (B, k, instrument_zdim)
  5. apply mask-aware aggregation across neighbors

  Default aggregation rule:

  - masked mean over valid neighbors
  - denominator should be clamp(valid_count, min=1) to avoid divide-by-zero
  - if a sample has no valid neighbors, the aggregated vector should become all zeros

  This yields:

  - instrument_global: (B, instrument_zdim)

  If instrument_zdim differs from the projection input needed by the instrument path, add a learned linear projection before injection.

  ### 3. Keep instrument global separate from galaxy global

  Do not concatenate the instrument global vector into global_vec and do not collapse both through one projection.

  Instead, add a new local wrapper class in the new file, for example:

  - DualGlobalConditionedUNet

  It should:

  - wrap the existing UNet2DConditionModel
  - preserve current class_labels behavior for galaxy global conditioning
  - add a second learned projection for instrument global conditioning
  - sum both projected globals into the timestep embedding before down/mid/up blocks

  Forward contract of the wrapper should be conceptually:

  - sample
  - timestep
  - spatial_levels
  - class_labels for galaxy global
  - instrument_global_labels for instrument global

  Embedding composition inside the wrapper:

  1. compute time embedding as current ConditionedUNet does
  2. if class_labels is present, add self.unet.class_embedding(class_labels)
  3. if instrument_global_labels is present, add self.instrument_global_proj(instrument_global_labels)
  4. continue through the UNet exactly as current ConditionedUNet.forward

  Recommended instrument projection design:

  - a 2-layer MLP with GELU mapping instrument_global_dim -> time_embed_dim
  - keep this local to the new file so shared hierarchical code is not modified unless later needed elsewhere

  Reason:

  - this keeps the architecture explicit
  - avoids changing shared train_experiments.py interfaces
  - avoids forcing a new combined projection_class_embeddings_input_dim

  ### 4. Cross-attention becomes galaxy-only

  The current hierarchical-neighbors model appends neighbor tokens to every galaxy spatial level. The new variant should not.

  So:

  - remove _encode_neighbors() returning flattened token sequences
  - remove _augment_spatial_levels()
  - remove RoPEWithNeighborTokensProcessor
  - use only galaxy spatial tokens as encoder_hidden_states for cross-attention blocks

  Because K is spatial-only again, RoPE handling should revert to the standard hierarchical-attention behavior already used in train_experiments.py.

  Implementation default:

  - either reuse the existing standard RoPE setup logic from train_experiments.py
  - or copy the minimal standard variant locally if import friction makes reuse awkward

  Preferred option:

  - reuse existing standard hierarchical setup rather than keeping the neighbor-aware processor

  ## Training Module Behavior

  ### Forward path

  The new module’s forward(...) should do:

  1. convert t to diffusion-style timesteps with timesteps = t * 1000
  2. encode cond_image_samegal through encoder_1
  3. encode cond_image_sameins through encoder_2(mean_pool=True)
  4. aggregate masked neighbor embeddings to one instrument_global
  5. call the wrapped conditioned UNet with:
      - sample=x_t
      - timestep=timesteps
      - spatial_levels=spatial_levels
      - class_labels=global_vec
      - instrument_global_labels=instrument_global

  ### Geometric loss

  Keep the geometric loss aligned with galaxy structure, same as the current hierarchical model:

  - use flattened hierarchical galaxy embeddings from encoder_1
  - do not include instrument global embeddings in the geometric loss

  ### Sampling / MSE / validation

  Keep behavior aligned with double_train_fm_neighbors_hier.py:

  - same batch compatibility with or without masks
  - same Euler integration sampler
  - same logging structure where practical
  - same center-mask logic for loss/MSE if enabled

  ## Public API / Interface Changes

  ### New module class

  Expose a new Lightning module class in the new file, for example:

  - HierarchicalGlobalInstrumentFlowMatchingModule

  Constructor should stay close to the existing hierarchical class, with these intended defaults:

  - experiment_config
  - in_channels
  - cond_channels
  - image_size
  - model_channels
  - channel_mult
  - layers_per_block
  - attention_head_dim
  - instrument_zdim=None
  - pretrained_encoder=False
  - lr=1e-4
  - num_sample_images=8
  - num_mse_images=64
  - num_integration_steps=250
  - lambda_generative=1.0
  - lambda_geometric=0.0
  - num_umap_batches=8
  - mask_center=False
  - figures_dir=None

  Add one new optional argument:

  - instrument_pooling: str = "masked_mean"

  Even if only "masked_mean" is implemented initially, exposing it makes the design and doc clearer for future ablations.

  ### New UNet wrapper interface

  The local wrapper should extend the existing conceptual API with:

  - instrument_global_labels: Optional[torch.Tensor] = None

  No existing shared module needs to change unless reuse turns out to require a tiny helper extraction. The default plan is to avoid public changes to
  train_experiments.py.

  ### New training script CLI

  The new training script should mirror the existing CLI and keep defaults identical where possible.

  Differences:

  - import the new module
  - default wandb_name should distinguish the variant, for example hier-global-ins-{experiment}
  - header printouts should state that instrument conditioning is global rather than token-based

  ## Documentation Content

  The new Markdown file should include these sections:

  ### 1. Purpose

  State the exact conditioning split:

  - same-galaxy image provides hierarchical spatial context plus galaxy global summary
  - same-instrument neighbor set provides one pooled global instrument context

  ### 2. Architecture

  Document each component:

  - ConfigurableEncoder
  - ResNetEncoder(mean_pool=True)
  - masked mean pooling across neighbors
  - dual-global conditioned UNet
  - galaxy-only cross-attention
  - separate galaxy-global and instrument-global embedding paths

  ### 3. Tensor shapes

  Include a shape table for:

  - input batch
  - galaxy spatial levels
  - galaxy global vector
  - per-neighbor pooled embeddings
  - masked aggregated instrument vector
  - UNet conditioning inputs

  ### 4. Data flow

  Step-by-step forward pass from batch to loss.

  ### 5. Why this differs from the current hierarchical neighbors model

  Explicitly contrast against double_train_fm_neighbors_hier.py:

  - old: neighbor tokens appended to every cross-attention level
  - new: neighbor set summarized globally and injected separately into the UNet embedding path

  ### 6. Assumptions and limitations

  Include:

  - instrument information is assumed to be mostly global rather than spatially aligned
  - masked mean treats all valid neighbors equally
  - zero-vector fallback for missing neighbors may weaken conditioning quality

  ### 7. Downstream hooks

  Document likely extension points for later tasks:

  - alternate neighbor aggregators
  - survey-aware instrument pooling
  - classifier/regressor heads on galaxy global or instrument global
  - ablations: no instrument path, merged globals, tokenized instrument path

  ## Test Cases And Validation

  ### Static correctness checks

  Add or run targeted checks for:

  1. forward pass with valid masks and mixed valid/padded neighbors
  2. forward pass with all-neighbor masks false for some batch elements
  3. shape consistency for:
      - spatial_levels
      - global_vec
      - instrument_global
      - model output (B, C, H, W)

  ### Behavioral checks

  Validate:

  1. padded neighbors do not affect instrument_global
  2. changing valid neighbors changes the output
  3. changing only galaxy input affects cross-attention conditioning
  4. disabling instrument input by zero masks produces deterministic zero-global fallback behavior

  ### Script-level smoke test

  The new training script should support a short smoke run equivalent in structure to the existing hierarchical trainer:
  - build model

  ### Optional regression comparison

  If convenient during implementation, compare parameter counts and printouts against the current hierarchical neighbors model to confirm:

  - galaxy encoder unchanged
  - instrument encoder simpler in token flow
  - cross-attention conditioning length reduced to galaxy-only tokens

  ## Explicit Assumptions And Defaults

  - File names will be:
      - double_train_fm_neighbors_hier_global_ins.py
      - neighbours_train_hier_global_ins.py
      - README_hier_global_instrument.md
  - Instrument conditioning will use ResNetEncoder(mean_pool=True) rather than a new encoder class.
  - Neighbor aggregation will be masked mean, not attention pooling.
  - The instrument global path will be implemented as a separate MLP added to the timestep embedding, not merged into class_labels.
  - The shared train_experiments.py file should remain unchanged unless a small non-behavioral helper extraction is required for RoPE setup reuse.
  - The new training script will mirror neighbours_train_hier.py rather than replacing it.
  - Documentation will live in galaxy_images/galaxy_model/hierarchical_attention/ next to the new code.