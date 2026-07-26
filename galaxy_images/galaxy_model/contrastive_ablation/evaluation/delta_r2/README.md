# Delta-R2 survey-view evaluation

This standalone evaluation trains the same AION-style probe
(`Linear -> GELU -> Linear`, hidden width 256) on three frozen physics
representations from a `prepare_combined.py` H5:

- `hsc_e1`
- `legacy_e1`
- `combined_e1`

It uses five deterministic, disjoint outer folds based on the absolute row
position in `overlap.parquet`. Fold assignment happens before finite-label
filtering, so every model and representation uses identical object membership.
Targets are standardized using finite outer-training-fold labels only. A deterministic
10% inner split of each training fold controls early stopping; outer-fold
objects are used only for scoring.

## Probe one model

```bash
python run_delta_r2.py \
  --embeddings /path/to/outputs/embeddings/base-5NB-DWNVAL.h5 \
  --overlap-parquet /path/to/outputs/index/overlap.parquet \
  --model-name base-5NB-DWNVAL \
  --out-dir outputs/base-5NB-DWNVAL
```

Run every model with the same `--seed` (default 0), overlap parquet, target
list, and training options. Non-finite embeddings are rejected instead of
silently inducing representation-specific masks.

Outputs:

- `fold_scores.csv`: one row per target, representation, and fold, containing
  R2, RMSE divided by the test-target standard deviation, and MAE divided by
  the test-target standard deviation.
- `oof_predictions.csv`: long-form object predictions keyed by positional
  `row_index` plus ID columns copied from the overlap index.
- `provenance.json`: command/configuration, input/output SHA-256 hashes,
  split/mask hashes, software versions, and sanity-check results.

The default target list matches the existing physics probe. Override it with,
for example, `--targets legacy_shape_e1,hsc_shape_e1_i`.

## Compare model pairs

`--pair` is repeatable and has no hardcoded model names. Each occurrence is:

```text
--pair NAME OURS_SCORES OURS_PREDS CONTRASTIVE_SCORES CONTRASTIVE_PREDS
```

For a primary holdout-matched pair and a secondary published pair:

```bash
python analyze_delta_r2.py \
  --pair primary \
    outputs/base-5NB-DWNVAL/fold_scores.csv \
    outputs/base-5NB-DWNVAL/oof_predictions.csv \
    outputs/contrastive-spatial-conv1x1-DWNVAL/fold_scores.csv \
    outputs/contrastive-spatial-conv1x1-DWNVAL/oof_predictions.csv \
  --pair secondary \
    outputs/base/fold_scores.csv \
    outputs/base/oof_predictions.csv \
    outputs/contrastive-spatial-conv1x1/fold_scores.csv \
    outputs/contrastive-spatial-conv1x1/oof_predictions.csv \
  --out-dir outputs/analysis
```

The analysis reports:

- `delta_survey(model) = R2(hsc_e1) - R2(legacy_e1)`
- `interaction = delta_survey(ours) - delta_survey(contrastive)`
- `combined_gain(model) = R2(combined_e1) - max(R2(hsc_e1), R2(legacy_e1))`
- paired fold means, standard deviations, and standard errors
- paired object-level bootstrap confidence intervals, with each resampled
  object shared across models and all three representations

Input split and finite-mask hashes must match across each pair.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests create a tiny synthetic H5 and overlap parquet, run all 15 probe
fits, validate OOF coverage and provenance, and run a paired self-control
bootstrap whose interaction must be zero.

For Slurm, `run_delta_r2.slurm` accepts `EMBEDDINGS`, `OVERLAP`, `MODEL_NAME`,
and `OUT_DIR` environment variables.
