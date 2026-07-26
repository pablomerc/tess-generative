# Contrastive resolution-collapse evaluation

This directory contains two linked, non-destructive evaluations of the frozen
physics embeddings:

- `delta_r2/`: compare physics probes trained on HSC, Legacy, and concatenated
  HSC+Legacy embeddings.
- `adversarial_testset/`: rank paired images by independently measured survey
  quality differences, inspect the ranking visually, freeze a manifest, and
  evaluate out-of-fold probe predictions by quality stratum.

The existing downstream pipeline remains the source of embeddings:

```text
downstream_evaluation/engaging/outputs/embeddings/<variant>.h5
downstream_evaluation/engaging/outputs/index/overlap.parquet
```

## Headline comparisons

The primary, holdout-matched comparison is:

```text
base-5NB-DWNVAL
contrastive-spatial-conv1x1-DWNVAL
```

The published `base` versus `contrastive-spatial-conv1x1` pair is a secondary
continuity check. Other contrastive arms use
`contrastive-spatial-conv1x1-DWNVAL` as their reference.

The key quantity is

```text
Delta_survey(model) = R2(HSC embedding) - R2(Legacy embedding)
interaction = Delta_survey(Ours) - Delta_survey(Contrastive)
```

A positive interaction supports the claim that HSC contains physics information
retained by Ours but suppressed by cross-survey contrastive alignment.

## Required ordering

1. Run `audit_inputs.py`.
2. Build the quality table and galleries.
3. Inspect the top/middle/bottom galleries and freeze a metric version.
4. Run the survey-view probes and save out-of-fold predictions.
5. Freeze the adversarial manifest.
6. Run the stratified evaluation.

Selection never uses model errors. The adversarial manifest is based only on
catalog/image quality metadata, and all reported predictions are out-of-fold.

Run each subdirectory's `README.md` commands for details. Generated data belongs
under `outputs/`, which is ignored locally.
