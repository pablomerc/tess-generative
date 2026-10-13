# Instrument properties from concat(physics, instrument)

Folds aggregated: **[0, 1, 2, 3, 4]** (5 of 5).

Frozen checkpoints, no training: **Ours** = the paper base model
(`checkpoints_new/base/snapshot.ckpt`, `neighbors_all_attn/2026-04-05`, `best-epoch=228-step=87000`); **Contrastive** = `checkpoints_new/contrastive-spatial-conv1x1/best-epoch=53-step=83000.ckpt`.
Probe: 2-layer MLP (`Linear(d,256)->GELU->Linear`), `--mlp-arch aion --seed 0 --n-folds 5`, `max_epochs=50`, early stopping on `val/loss`; R2 on the held-out fold.

The 256-D feature is **64 x 4** — 64 per image per latent x 2 surveys x 2 latents:

```
[ hsc_e1(64) | legacy_e1(64) | hsc_e2(64) | legacy_e2(64) ]  = 256
  physics      physics         instrument   instrument
```

and each 64 = 4 spatial tokens x 16 channels (the 2x2 grid of a 48x48 input,
flattened). The published "Ours" column is the right-hand half only: 128-D.

## Main table

The four right-hand columns are new. `balanced` rescales each 64-D block to unit RMS before concatenating; it is the fair measurement, because the physics blocks carry ~1.9x the RMS of the instrument blocks and the raw concat therefore down-weights the informative half. Caveat: the 256-D concat has **2x the probe input width** of the 128-D columns — see the dimension-matched control below.

| property | Ours (instr, flat 128-D) | Contrastive (instr, flat 128-D) | Ours phys+instr (256-D) | Contrastive phys+instr (256-D) | Ours phys+instr (256-D, balanced) | Contrastive phys+instr (256-D, balanced) |
|---|---|---|---|---|---|---|
| Galaxy Depth | +0.751 ± 0.011 | +0.852 ± 0.016 | +0.727 ± 0.014 | +0.825 ± 0.019 | +0.745 ± 0.011 | +0.835 ± 0.019 |
| PSF Depth | +0.742 ± 0.013 | +0.846 ± 0.020 | +0.717 ± 0.017 | +0.820 ± 0.022 | +0.738 ± 0.012 | +0.828 ± 0.022 |
| # Observations | +0.535 ± 0.014 | +0.699 ± 0.009 | +0.500 ± 0.013 | +0.653 ± 0.011 | +0.525 ± 0.014 | +0.664 ± 0.011 |
| E(B-V) | +0.591 ± 0.025 | +0.770 ± 0.017 | +0.548 ± 0.027 | +0.713 ± 0.014 | +0.570 ± 0.024 | +0.735 ± 0.014 |
| PSF FWHM (HSC) | +0.500 ± 0.017 | +0.684 ± 0.018 | +0.471 ± 0.021 | +0.621 ± 0.019 | +0.501 ± 0.018 | +0.638 ± 0.019 |

## Supplementary — controls

`matched` = the survey providing the target only (Legacy latents for Legacy targets, HSC for HSC), so no cross-survey information. `pooled 32-D` and `physics only` are published cells, recomputed here on the same folds.

| property | Ours phys+instr matched (128-D) | Contrastive phys+instr matched (128-D) | Ours instr matched (64-D) | Contrastive instr matched (64-D) | Ours instr pooled (32-D) | Contrastive instr pooled (32-D) | Ours physics only (128-D) | Contrastive physics only (128-D) |
|---|---|---|---|---|---|---|---|---|
| Galaxy Depth | +0.714 ± 0.017 | +0.802 ± 0.026 | +0.733 ± 0.013 | +0.841 ± 0.016 | +0.688 ± 0.020 | +0.750 ± 0.027 | +0.170 ± 0.010 | -0.008 ± 0.006 |
| PSF Depth | +0.698 ± 0.019 | +0.789 ± 0.031 | +0.719 ± 0.015 | +0.831 ± 0.020 | +0.671 ± 0.023 | +0.741 ± 0.032 | +0.166 ± 0.009 | -0.006 ± 0.007 |
| # Observations | +0.474 ± 0.013 | +0.593 ± 0.013 | +0.499 ± 0.015 | +0.630 ± 0.014 | +0.482 ± 0.012 | +0.567 ± 0.011 | +0.059 ± 0.013 | -0.025 ± 0.007 |
| E(B-V) | +0.450 ± 0.019 | +0.598 ± 0.024 | +0.472 ± 0.026 | +0.631 ± 0.025 | +0.541 ± 0.023 | +0.681 ± 0.020 | +0.176 ± 0.020 | +0.066 ± 0.020 |
| PSF FWHM (HSC) | +0.432 ± 0.020 | +0.566 ± 0.021 | +0.460 ± 0.020 | +0.628 ± 0.023 | +0.420 ± 0.017 | +0.526 ± 0.019 | +0.070 ± 0.011 | +0.024 ± 0.008 |

## Validation — `combined_e1` identity gate

Every derived H5 carries `combined_e1` through verbatim, so its probe scores must reproduce the published pipeline's. This validates the derived-feature path end-to-end.

**Status: PASS**

| derived variant | reference | comparisons | max abs delta | verdict |
|---|---|---|---|---|
| `base-e1e2-256` | `base` | 195 | 0.00e+00 | PASS |
| `base-e1e2-256-bal` | `base` | 195 | 0.00e+00 | PASS |
| `base-e1e2m-hsc` | `base` | 195 | 0.00e+00 | PASS |
| `base-e1e2m-leg` | `base` | 195 | 0.00e+00 | PASS |
| `base-e2m-hsc` | `base` | 195 | 0.00e+00 | PASS |
| `base-e2m-leg` | `base` | 195 | 0.00e+00 | PASS |
| `contr-e1e2-256` | `contrastive-spatial-conv1x1` | 195 | 0.00e+00 | PASS |
| `contr-e1e2-256-bal` | `contrastive-spatial-conv1x1` | 195 | 0.00e+00 | PASS |
| `contr-e1e2m-hsc` | `contrastive-spatial-conv1x1` | 195 | 0.00e+00 | PASS |
| `contr-e1e2m-leg` | `contrastive-spatial-conv1x1` | 195 | 0.00e+00 | PASS |
| `contr-e2m-hsc` | `contrastive-spatial-conv1x1` | 195 | 0.00e+00 | PASS |
| `contr-e2m-leg` | `contrastive-spatial-conv1x1` | 195 | 0.00e+00 | PASS |

Thresholds: PASS <= 1e-06, WARN <= 0.001 (BLAS reduction-order noise), FAIL above.

## Provenance

- published fold CSVs (read-only): `/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model/downstream_evaluation/engaging/outputs/predictions/folds`
- new fold CSVs: `/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model/downstream_evaluation/rebuttal_experiments/outputs/predictions/folds`
- derived features: `build_derived_h5.py` (post-hoc H5 transform of the cached embeddings; no re-encoding, no GPU)
- probe driver: `../engaging/predict_combined.py`, **unmodified**
- aggregation math replicates `contrastive_ablation/aggregate_folds.py:88-114`
