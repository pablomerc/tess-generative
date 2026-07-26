# Quality gallery and adversarial test set

This directory implements a quality-only ranking of the fixed HSC × Legacy
`overlap.parquet`, visual inspection, immutable manifest creation, and
stratified evaluation of out-of-fold (OOF) predictions from sibling
`../delta_r2/`.

No model error is used to select an adversarial object. Freeze the manifest
before inspecting prediction losses.

## Prerequisites

Python packages: `numpy`, `pandas`, `pyarrow`, `matplotlib`, and `Pillow`.
Tests additionally use only the standard-library `unittest`.

The scripts fail clearly when these inputs are absent:

- the fixed overlap parquet;
- HSC and Legacy catalog parquets;
- HSC `(5,160,160)` and Legacy `(4,160,160)` float16 image binaries for galleries;
- OOF prediction CSVs for evaluation.

The scripts do not require checkpoints and do not claim that galleries or
evaluation outputs exist until the corresponding command succeeds.

## 1. Audit catalog schemas

```bash
HERE=galaxy_images/galaxy_model/contrastive_ablation/evaluation/adversarial_testset
ENG=galaxy_images/galaxy_model/downstream_evaluation/engaging

python "$HERE/quality_table.py" audit \
  --overlap "$ENG/outputs/index/overlap.parquet" \
  --hsc-catalog /home/pablomer/orcd/scratch/hsc_downstream/catalog.parquet \
  --legacy-catalog /home/pablomer/orcd/scratch/legacy_downstream_full/full_1M/catalog.parquet \
  --report "$HERE/outputs/schema_audit.json"
```

Discovery supports scalar per-band and list-valued griz(/y) HSC fields. It
requires adaptive object `xx/yy/xy`, adaptive PSF `xx/yy/xy`, flux/error, and
PSF FWHM for each matched griz band. Legacy requires `PSFSIZE_*`, `FLUX_*`, and
`FLUX_IVAR_*`. Exact aliases are preferred; an ambiguous heuristic match is an
error, never a silent choice. The JSON report records the chosen source field
for every metric.

## 2. Build quality table and candidate list

```bash
python "$HERE/quality_table.py" build \
  --overlap "$ENG/outputs/index/overlap.parquet" \
  --hsc-catalog /home/pablomer/orcd/scratch/hsc_downstream/catalog.parquet \
  --legacy-catalog /home/pablomer/orcd/scratch/legacy_downstream_full/full_1M/catalog.parquet \
  --out "$HERE/outputs/quality_table.parquet" \
  --candidates "$HERE/outputs/quality_candidates_top200.csv" \
  --top-n 200 \
  --moments-unit pixel2 \
  --hsc-pixel-scale 0.168
```

If the HSC moment columns are already in arcsec², use
`--moments-unit arcsec2`. The metric is:

- intrinsic HSC size:
  `FWHM = 2.35482 det(M_object - M_psf)^(1/4)`, only for positive-definite
  object, PSF, and corrected moment matrices;
- dimensionless HSC size proxy:
  `det(M_object-M_psf)^(1/4) / det(M_psf)^(1/4)`;
- survey resolution: that same intrinsic angular size divided by each survey's
  catalog PSF FWHM;
- HSC SNR: `flux/error`; Legacy SNR: `flux*sqrt(ivar)`;
- gaps: median over valid griz bands of HSC minus Legacy resolution and
  `log1p(SNR)`;
- joint score: mean of the two within-table percentile ranks.

At least two valid bands are required for each axis. Per-band validity,
coverage flags, source fields, the formula, and `metric_version` are retained.

## 3. Render contact sheets

```bash
python "$HERE/make_gallery.py" \
  --quality-table "$HERE/outputs/quality_table.parquet" \
  --hsc-bin /home/pablomer/orcd/scratch/hsc_downstream/hsc_flux.bin \
  --legacy-bin /home/pablomer/orcd/scratch/legacy_downstream_full/full_1M/legacy_flux.bin \
  --out-dir "$HERE/outputs/gallery" \
  --n-per-stratum 12 --seed 0
```

This randomly samples the top 20%, middle 40–60%, and bottom 20% strata and
writes one paired contact sheet per stratum plus a summary plot. Raw binary
records are bounds-checked. Images use the model's center-crop and Legacy
field-of-view zoom idea, but visualization scaling is intentionally separate:
each survey is background-subtracted and robust-asinh scaled independently.

To post generated PNGs, pass `--webhook "$DISCORD_WEBHOOK"` or set that
environment variable. No webhook or secret is stored in this repository.

## 4. Freeze the quality-only manifest

After visual inspection and accepting the recorded metric version:

```bash
python "$HERE/freeze_manifest.py" \
  --quality-table "$HERE/outputs/quality_table.parquet" \
  --out "$HERE/outputs/adversarial_manifest.csv" \
  --top-n 200 --n-quantile-bins 5
```

The CLI refuses to overwrite an existing manifest. Its JSON sidecar records
source and selection hashes, metric version, and confirms that selection did
not use predictions.

## 5. Evaluate sibling `delta_r2` OOF predictions

Each OOF CSV must contain one object-ID column (`hsc_object_id` or
`legacy_object_id`), truth (`y_true`), prediction (`y_pred`), and optionally a
`target` column. One finite row per object and target is required. Supply every
model/view file with a label:

```bash
python "$HERE/evaluate_adversarial.py" \
  --manifest "$HERE/outputs/adversarial_manifest.csv" \
  --pred "ours:hsc=../delta_r2/outputs/base-5NB-DWNVAL/oof_predictions.csv" \
  --pred "contrastive:hsc=../delta_r2/outputs/contrastive-spatial-conv1x1-DWNVAL/oof_predictions.csv" \
  --pred "ours:legacy=../delta_r2/outputs/base-5NB-DWNVAL/oof_predictions.csv" \
  --pred "contrastive:legacy=../delta_r2/outputs/contrastive-spatial-conv1x1-DWNVAL/oof_predictions.csv" \
  --out-dir "$HERE/outputs/adversarial_eval" \
  --bootstrap-replicates 2000 --seed 0
```

When a prediction CSV contains all three `latent_variant` values, the view in
the spec automatically selects `hsc_e1`, `legacy_e1`, or `combined_e1`. The
standard delta-R2 OOF file may carry both object IDs; HSC ID is used as the
manifest join key and the paired manifest still retains both.

Outputs:

- `stratified_metrics.csv`: R², nRMSE, and nMAE for all objects, top-N,
  quantile bins, and matched random controls along resolution, SNR, and joint
  axes;
- `paired_object_losses.csv`: paired squared and absolute losses and model-A
  minus model-B deltas;
- `paired_bootstrap.csv`: object-paired 95% bootstrap intervals for metric
  differences;
- `survey_interaction_bootstrap.csv`: the model-by-survey difference of
  differences, including the top-N, quality bins, and matched controls;
- `evaluation_metadata.json`: conventions, inputs, metric version, and control
  matching feasibility.

nRMSE and nMAE divide by the within-stratum standard deviation of truth.
Controls are sampled without replacement from outside top-N, matched on the
other two axes' quality quantile bins where feasible; fallback counts are
reported rather than hidden.

## Tests

```bash
python -m unittest discover -s "$HERE/tests" -v
```

The optional `run_quality_gallery.slurm` runs audit, table construction, and
gallery generation. It intentionally does not freeze a manifest or evaluate
predictions because visual acceptance and OOF inputs are explicit gates.
