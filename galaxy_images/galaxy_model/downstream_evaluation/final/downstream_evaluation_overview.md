# Downstream Evaluation Pipeline — Overview

## What This Directory Does

Unified downstream evaluation pipeline for a **dual-encoder flow-matching model** trained on paired HSC + Legacy Survey galaxy images. The goal is to measure how well the learned latent space captures physics (galaxy properties) vs. instrumental effects (observing conditions).

### Pipeline

```
prepare_all.py  →  predict_all.py  →  makeplot_v2.py
   (H5 files)      (CSV w/ R²/MAE)    (bar chart)
```

| Script | What it does |
|---|---|
| `prepare_all.py` | Generates 3 H5 files (one per dataset) with 18 embedding variants + labels |
| `predict_all.py` | Trains MLP regressors on embeddings; outputs R² and MAE per target |
| `makeplot_v2.py` | Bar chart of R² grouped into Physics / Legacy Prop. / HSC Prop. |

### Three Datasets Used

| Dataset | H5 file | Labels from |
|---|---|---|
| MMU | `downstream_mmu_{suffix}.h5` | `preprocessed_hsc_legacy_metadata_8192.h5` |
| Legacy ProvaBGS | `downstream_legacy_provabgs_{suffix}.h5` | `provabgs_legacysurvey_{train,eval}_v2.fits` |
| Neighbors | `downstream_neighbors_{suffix}.h5` | `neighbours_v2.h5` |

### Embeddings Stored (18 per galaxy)

6 encoder combinations × 3 variants (real / untrained / random):

| Key | Description |
|---|---|
| `hsc_encoder1` | HSC image → encoder 1 (physics latent) |
| `hsc_encoder2` | HSC image → encoder 2 (instrument latent) |
| `legacy_encoder1` | Legacy image → encoder 1 |
| `legacy_encoder2` | Legacy image → encoder 2 |
| `hsc_legacy_encoder1` | Concat(HSC enc1, Legacy enc1) |
| `hsc_legacy_encoder2` | Concat(HSC enc2, Legacy enc2) |
| `*_untrained` | Same architecture, random weights |
| `*_random` | N(0,1) random vectors, same shape |

### Prediction Tasks

| Task | Dataset | Encoder used | Targets |
|---|---|---|---|
| `physics_mmu` | MMU | encoder1 | SHAPE_E1, SHAPE_E2, SHAPE_R |
| `instrument_mmu` | MMU | encoder2 | EBV, a_g/i/r/y/z |
| `physics_provabgs` | ProvaBGS | encoder1 | desi_Z, LOG_MSTAR, TAGE_MW, LOG_Z_MW, sSFR |
| `instrument_neighbors_legacy` | Neighbors | encoder2 | legacy_PSFSIZE/PSFDEPTH/GALDEPTH/NOBS (×4 bands) |
| `instrument_neighbors_hsc` | Neighbors | encoder2 | hsc_*_variance_value, hsc_*_psf_fwhm (derived) |

---

## The "Perfect" Unified File - Minimal Version Actually Used Here

This is the smallest merged, per-object schema that covers the code in:

- `downstream_evaluation/`
- `downstream_evaluation/4x4/`
- `downstream_evaluation/final/`

Scope:

- includes fields used directly as prediction targets,
- includes raw columns used to derive `hsc_*_psf_fwhm`,
- includes the main identifier block that makes a merged file usable,
- excludes pass-through columns that are copied into intermediate H5 files but are never explicitly selected by the downstream scripts.

### Prep / Join Keys Used by the Scripts

These are not really "science labels", but they are genuinely used during dataset construction and filtering.

| Field | Source | Purpose |
|---|---|---|
| `indices` | Preprocessed H5 / MMU metadata H5 | Align MMU metadata rows to image rows |
| `abs_index` | Overlap CSVs | Match ProvaBGS rows back to the preprocessed H5 |
| `TARGETID` | Overlap CSVs | Bridge ID used during ProvaBGS matching |
| `legacy_object_id` | Legacy ProvaBGS FITS | Merge key in `prepare_legacy_provabgs.py` / `prepare_all.py` |
| `hsc_object_id` | HSC ProvaBGS FITS | Merge key in `prepare_hsc_provabgs.py` / `predict_all_ours_instrument4x4.py` |
| `source_type` | Neighbors H5 | Used to filter the MMU subset in `baseline_all.py` |

---

### Identifiers

These are the practical per-object identifiers worth keeping in a merged file. The current scripts do not predict them, but they make the unified table much more usable.

| Field | Source |
|---|---|
| `object_id_hsc` | Neighbors H5 |
| `object_id_legacy` | Neighbors H5 |
| `ra` | Neighbors H5 |
| `dec` | Neighbors H5 |

---

### Physics Labels - ProvaBGS (DESI spectroscopy)

| Field | Description |
|---|---|
| `desi_Z` | Spectroscopic redshift |
| `LOG_MSTAR` | Log stellar mass |
| `TAGE_MW` | Mass-weighted stellar age |
| `LOG_Z_MW` | Mass-weighted stellar metallicity |
| `sSFR` | Specific star formation rate |
| `hsc_g_extendedness_value` | HSC g-band star/galaxy separation |
| `DEC` | Declination from the ProvaBGS FITS tables; predicted in the legacy ProvaBGS scripts |

---

### Shape Labels - MMU Catalog

| Field | Description |
|---|---|
| `SHAPE_E1` | Ellipticity component 1 |
| `SHAPE_E2` | Ellipticity component 2 |
| `SHAPE_R` | Half-light radius |

---

### MMU Photometry / Observing Terms

These are explicitly used by the older MMU downstream scripts in `downstream_evaluation/`.

| Field | Description |
|---|---|
| `EBV` | Galactic dust extinction |
| `FLUX_G/I/R/W1/W2/W3/W4/Z` | Fluxes used by `predict_mmu.py` / `predict.py` |
| `a_g/i/r/y/z` | HSC-band extinction terms |
| `g/i/r/y/z_cmodel_mag` | CModel magnitudes |
| `g/i/y_cmodel_magerr` | CModel magnitude errors that are explicitly targeted |
| `i_extendedness_value` | HSC i-band star/galaxy separation |
| `y_extendedness_value` | HSC y-band star/galaxy separation |
| `g/i/r/z_sdssshape_psf_shape11/12/22` | HSC PSF second-moment terms explicitly predicted by the MMU scripts |

---

### Legacy Survey Instrument Labels

These are the Legacy-side targets used by `predict_neighbors.py`, `predict_all.py`, `predict_baseline.py`, and the `4x4/compare_all.py` summaries.

| Field | Description |
|---|---|
| `legacy_PSFSIZE_G/R/I/Z` | Legacy PSF size per band |
| `legacy_PSFDEPTH_G/R/I/Z` | Legacy PSF depth per band |
| `legacy_GALDEPTH_G/R/I/Z` | Legacy galaxy depth per band |
| `legacy_NOBS_G/R/I/Z` | Number of Legacy observations per band |

---

### HSC Instrument Labels From Neighbors

| Field | Description |
|---|---|
| `hsc_i/r/z_variance_value` | HSC variance terms explicitly predicted downstream |
| `hsc_g/i/r/z_sdssshape_psf_shape11` | Raw PSF moment needed to derive HSC seeing |
| `hsc_g/i/r/z_sdssshape_psf_shape22` | Raw PSF moment needed to derive HSC seeing |
| `hsc_g/i/r/z_psf_fwhm` | Derived target used downstream; computed from `shape11` and `shape22` |

Derived formula:

`hsc_*_psf_fwhm = 2.355 * sqrt((shape11 + shape22) / 2) * 0.168`

---

### Plotting Note

Some fields are used in prediction scripts but then dropped from the compact comparison plots:

- `DEC`
- `hsc_g_extendedness_value`
- `i_extendedness_value`
- `hsc_z_psf_fwhm` is predicted, but excluded from the averaged `hsc_psf_fwhm` display in the `4x4/` and `final/` comparison plots

---

## Running the Pipeline

```bash
conda activate torchenv

# 1. Prepare embeddings (run from galaxy_model/)
python downstream_evaluation/final/prepare_all.py \
    --checkpoint PATH_TO_CKPT \
    --module double_train_fm_neighbors.py \
    --suffix zdim16_nogeom_neighbors

# 2. Predict all targets
python downstream_evaluation/final/predict_all.py \
    --suffix zdim16_nogeom_neighbors \
    --output-dir downstream_evaluation/final

# 3. Plot results
python downstream_evaluation/final/makeplot_v2.py \
    --suffix zdim16_nogeom_neighbors \
    --output-dir downstream_evaluation/final
```

### Output files

| File | Description |
|---|---|
| `downstream_mmu_{suffix}.h5` | MMU embeddings + labels |
| `downstream_legacy_provabgs_{suffix}.h5` | ProvaBGS embeddings + labels |
| `downstream_neighbors_{suffix}.h5` | Neighbors embeddings + labels |
| `predict_all_{suffix}.csv` | R² and MAE per task/target |
| `predict_all_{suffix}_plot_v2.png` | Bar chart |

---

## Data Paths

| Dataset | Path |
|---|---|
| MMU H5 | `/data/vision/billf/scratch/pablomer/legacysurvey_hsc/preprocessed_hsc_legacy_48x48_all.h5` |
| MMU metadata | `/data/vision/billf/scratch/pablomer/legacysurvey_hsc/preprocessed_hsc_legacy_metadata_8192.h5` |
| ProvaBGS train FITS | `/data/vision/billf/scratch/pablomer/data/provabgs_legacysurvey_train_v2.fits` |
| ProvaBGS eval FITS | `/data/vision/billf/scratch/pablomer/data/provabgs_legacysurvey_eval_v2.fits` |
| Neighbors H5 | `/data/vision/billf/scratch/pablomer/data/neighbours_v2.h5` |
| Neighbors H5 (local) | `/work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5` |
