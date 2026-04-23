# Ideal Unified File Schema

This is a collaborator-facing proposal for what a single merged per-object file should contain.

Goal:

- one row per object,
- stable IDs and coordinates,
- the main science labels from ProvaBGS,
- the main HSC and Legacy observing / photometry fields,
- consistent canonical column names even when the original files used different names.

Conventions:

- `Unified field` is the canonical column name to keep in the merged file.
- `Original key(s)` lists the matching raw key names from the source files.
- `Original source(s)` lists which source file(s) those keys came from.
- if multiple raw keys are listed, they are intended to map into the same unified field.

## Core Fields

### Identifiers and Coordinates

| Unified field | Original key(s) | Original source(s) | Notes |
| --- | --- | --- | --- |
| `object_id_hsc` | `hsc_object_id`, `object_id_hsc` | MMU, Neighbors, ProvaBGS | Primary HSC object identifier |
| `object_id_legacy` | `legacysurvey_object_id`, `object_id_legacy`, `TARGETID`, `legacy_object_id` | MMU, Neighbors, ProvaBGS | Primary Legacy / DESI-side object identifier |
| `ra` | `ra`, `RA` | Neighbors, ProvaBGS | Right ascension |
| `dec` | `dec`, `DEC` | Neighbors, ProvaBGS | Declination |
| `healpix` | `healpix`, `HPX_64` | Neighbors, ProvaBGS | Sky-pixel index; useful for joins / regional splits |
| `sample_origin` | `catalog`, `source_type` | MMU, Neighbors | Provenance / subset label |

---

### Spectroscopy and Physics Labels

| Unified field | Original key(s) | Original source(s) | Notes |
| --- | --- | --- | --- |
| `desi_targetid` | `TARGETID` | ProvaBGS | Keep explicitly even if also used as Legacy-side ID |
| `desi_z` | `desi_Z` | ProvaBGS | Spectroscopic redshift |
| `z_hp` | `Z_HP` | ProvaBGS | Additional redshift estimate |
| `z_err` | `ZERR` | ProvaBGS | Redshift uncertainty |
| `log_mstar` | `LOG_MSTAR`, `PROVABGS_LOGMSTAR` | ProvaBGS | Log stellar mass |
| `tage_mw` | `TAGE_MW` | ProvaBGS | Mass-weighted stellar age |
| `log_z_mw` | `LOG_Z_MW` | ProvaBGS | Mass-weighted stellar metallicity |
| `ssfr` | `sSFR` | ProvaBGS | Specific star formation rate |
| `tsnr2_bgs` | `TSNR2_BGS` | ProvaBGS | Spectroscopic quality / signal metric |
| `is_bgs_bright` | `IS_BGS_BRIGHT` | ProvaBGS | Sample membership flag |
| `is_bgs_faint` | `IS_BGS_FAINT` | ProvaBGS | Sample membership flag |
| `provabgs_w_zfail` | `PROVABGS_W_ZFAIL` | ProvaBGS | Weight / quality flag |
| `provabgs_w_fibassign` | `PROVABGS_W_FIBASSIGN` | ProvaBGS | Weight / quality flag |
| `provabgs_z_max` | `PROVABGS_Z_MAX` | ProvaBGS | Redshift bound / diagnostic |

---

### Shape and Morphology

| Unified field | Original key(s) | Original source(s) | Notes |
| --- | --- | --- | --- |
| `shape_e1` | `SHAPE_E1` | MMU | Main ellipticity component |
| `shape_e2` | `SHAPE_E2` | MMU | Main ellipticity component |
| `shape_r` | `SHAPE_R` | MMU | Size / half-light radius |
| `legacy_shape_e1` | `legacy_SHAPE_E1` | Neighbors | Legacy-side shape summary |
| `legacy_shape_r` | `legacy_SHAPE_R` | Neighbors | Legacy-side size summary |
| `legacy_type` | `legacy_TYPE` | Neighbors | Legacy morphology / type label |
| `legacy_sersic` | `legacy_SERSIC` | Neighbors | Sersic parameter |

---

### HSC Photometry and HSC-side Metadata

| Unified field | Original key(s) | Original source(s) | Notes |
| --- | --- | --- | --- |
| `hsc_flux_g` | `FLUX_G` | MMU | Broad-band flux from MMU metadata |
| `hsc_flux_r` | `FLUX_R` | MMU | Broad-band flux from MMU metadata |
| `hsc_flux_i` | `FLUX_I` | MMU | Broad-band flux from MMU metadata |
| `hsc_flux_z` | `FLUX_Z` | MMU | Broad-band flux from MMU metadata |
| `wise_flux_w1` | `FLUX_W1` | MMU | WISE flux |
| `wise_flux_w2` | `FLUX_W2` | MMU | WISE flux |
| `wise_flux_w3` | `FLUX_W3` | MMU | WISE flux |
| `wise_flux_w4` | `FLUX_W4` | MMU | WISE flux |
| `hsc_a_g` | `a_g`, `hsc_a_g` | MMU, ProvaBGS | Extinction term |
| `hsc_a_r` | `a_r`, `hsc_a_r` | MMU, ProvaBGS | Extinction term |
| `hsc_a_i` | `a_i`, `hsc_a_i` | MMU, ProvaBGS | Extinction term |
| `hsc_a_z` | `a_z`, `hsc_a_z` | MMU, ProvaBGS | Extinction term |
| `hsc_a_y` | `a_y`, `hsc_a_y` | MMU, ProvaBGS | Extinction term |
| `hsc_g_cmodel_mag` | `g_cmodel_mag`, `hsc_g_cmodel_mag` | MMU, ProvaBGS | Canonical g-band magnitude |
| `hsc_r_cmodel_mag` | `r_cmodel_mag`, `hsc_r_cmodel_mag` | MMU, ProvaBGS | Canonical r-band magnitude |
| `hsc_i_cmodel_mag` | `i_cmodel_mag`, `hsc_i_cmodel_mag` | MMU, ProvaBGS | Canonical i-band magnitude |
| `hsc_z_cmodel_mag` | `z_cmodel_mag`, `hsc_z_cmodel_mag` | MMU, ProvaBGS | Canonical z-band magnitude |
| `hsc_y_cmodel_mag` | `y_cmodel_mag`, `hsc_y_cmodel_mag` | MMU, ProvaBGS | Canonical y-band magnitude |
| `hsc_g_cmodel_magerr` | `g_cmodel_magerr`, `hsc_g_cmodel_magerr` | MMU, ProvaBGS | Magnitude uncertainty |
| `hsc_r_cmodel_magerr` | `hsc_r_cmodel_magerr` | ProvaBGS | Magnitude uncertainty |
| `hsc_i_cmodel_magerr` | `i_cmodel_magerr`, `hsc_i_cmodel_magerr` | MMU, ProvaBGS | Magnitude uncertainty |
| `hsc_z_cmodel_magerr` | `hsc_z_cmodel_magerr` | ProvaBGS | Magnitude uncertainty |
| `hsc_y_cmodel_magerr` | `y_cmodel_magerr`, `hsc_y_cmodel_magerr` | MMU, ProvaBGS | Magnitude uncertainty |
| `hsc_g_extendedness_value` | `hsc_g_extendedness_value` | ProvaBGS | Star / galaxy separation |
| `hsc_i_extendedness_value` | `i_extendedness_value`, `hsc_i_extendedness_value` | MMU, ProvaBGS | Star / galaxy separation |
| `hsc_y_extendedness_value` | `y_extendedness_value`, `hsc_y_extendedness_value` | MMU, ProvaBGS | Star / galaxy separation |
| `hsc_z_extendedness_value` | `z_extendedness_value`, `hsc_z_extendedness_value` | MMU, ProvaBGS | Star / galaxy separation |

---

### HSC PSF and Observing Conditions

| Unified field | Original key(s) | Original source(s) | Notes |
| --- | --- | --- | --- |
| `hsc_g_cmodel_flux` | `hsc_g_cmodel_flux` | Neighbors, ProvaBGS | Flux-space version of HSC photometry |
| `hsc_r_cmodel_flux` | `hsc_r_cmodel_flux` | Neighbors, ProvaBGS | Flux-space version of HSC photometry |
| `hsc_i_cmodel_flux` | `hsc_i_cmodel_flux` | Neighbors, ProvaBGS | Flux-space version of HSC photometry |
| `hsc_z_cmodel_flux` | `hsc_z_cmodel_flux` | Neighbors, ProvaBGS | Flux-space version of HSC photometry |
| `hsc_g_cmodel_fluxerr` | `hsc_g_cmodel_fluxerr` | Neighbors, ProvaBGS | Flux uncertainty |
| `hsc_r_cmodel_fluxerr` | `hsc_r_cmodel_fluxerr` | Neighbors, ProvaBGS | Flux uncertainty |
| `hsc_i_cmodel_fluxerr` | `hsc_i_cmodel_fluxerr` | Neighbors, ProvaBGS | Flux uncertainty |
| `hsc_z_cmodel_fluxerr` | `hsc_z_cmodel_fluxerr` | Neighbors, ProvaBGS | Flux uncertainty |
| `hsc_g_psf_shape11` | `g_sdssshape_psf_shape11`, `hsc_g_sdssshape_psf_shape11` | MMU, Neighbors, ProvaBGS | Raw PSF second moment |
| `hsc_g_psf_shape12` | `g_sdssshape_psf_shape12`, `hsc_g_sdssshape_psf_shape12` | MMU, Neighbors, ProvaBGS | Raw PSF second moment |
| `hsc_g_psf_shape22` | `g_sdssshape_psf_shape22`, `hsc_g_sdssshape_psf_shape22` | MMU, Neighbors, ProvaBGS | Raw PSF second moment |
| `hsc_r_psf_shape11` | `r_sdssshape_psf_shape11`, `hsc_r_sdssshape_psf_shape11` | MMU, Neighbors, ProvaBGS | Raw PSF second moment |
| `hsc_r_psf_shape12` | `r_sdssshape_psf_shape12`, `hsc_r_sdssshape_psf_shape12` | MMU, Neighbors, ProvaBGS | Raw PSF second moment |
| `hsc_r_psf_shape22` | `r_sdssshape_psf_shape22`, `hsc_r_sdssshape_psf_shape22` | MMU, Neighbors, ProvaBGS | Raw PSF second moment |
| `hsc_i_psf_shape11` | `i_sdssshape_psf_shape11`, `hsc_i_sdssshape_psf_shape11` | MMU, Neighbors, ProvaBGS | Raw PSF second moment |
| `hsc_i_psf_shape12` | `i_sdssshape_psf_shape12`, `hsc_i_sdssshape_psf_shape12` | MMU, Neighbors, ProvaBGS | Raw PSF second moment |
| `hsc_i_psf_shape22` | `i_sdssshape_psf_shape22`, `hsc_i_sdssshape_psf_shape22` | MMU, Neighbors, ProvaBGS | Raw PSF second moment |
| `hsc_z_psf_shape11` | `z_sdssshape_psf_shape11`, `hsc_z_sdssshape_psf_shape11` | MMU, Neighbors, ProvaBGS | Raw PSF second moment |
| `hsc_z_psf_shape12` | `z_sdssshape_psf_shape12`, `hsc_z_sdssshape_psf_shape12` | MMU, Neighbors, ProvaBGS | Raw PSF second moment |
| `hsc_z_psf_shape22` | `z_sdssshape_psf_shape22`, `hsc_z_sdssshape_psf_shape22` | MMU, Neighbors, ProvaBGS | Raw PSF second moment |
| `hsc_y_psf_shape11` | `y_sdssshape_psf_shape11`, `hsc_y_sdssshape_psf_shape11` | MMU, ProvaBGS | Raw PSF second moment |
| `hsc_y_psf_shape12` | `y_sdssshape_psf_shape12`, `hsc_y_sdssshape_psf_shape12` | MMU, ProvaBGS | Raw PSF second moment |
| `hsc_y_psf_shape22` | `y_sdssshape_psf_shape22`, `hsc_y_sdssshape_psf_shape22` | MMU, ProvaBGS | Raw PSF second moment |
| `hsc_i_variance_value` | `hsc_i_variance_value` | Neighbors | Sky / variance term |
| `hsc_r_variance_value` | `hsc_r_variance_value` | Neighbors | Sky / variance term |
| `hsc_z_variance_value` | `hsc_z_variance_value` | Neighbors | Sky / variance term |
| `hsc_g_psf_fwhm` | derived from `hsc_g_*psf_shape11/22` | MMU, Neighbors, ProvaBGS | Derived PSF FWHM |
| `hsc_r_psf_fwhm` | derived from `hsc_r_*psf_shape11/22` | MMU, Neighbors, ProvaBGS | Derived PSF FWHM |
| `hsc_i_psf_fwhm` | derived from `hsc_i_*psf_shape11/22` | MMU, Neighbors, ProvaBGS | Derived PSF FWHM |
| `hsc_z_psf_fwhm` | derived from `hsc_z_*psf_shape11/22` | MMU, Neighbors, ProvaBGS | Derived PSF FWHM |
| `hsc_y_psf_fwhm` | derived from `hsc_y_*psf_shape11/22` | MMU, ProvaBGS | Derived PSF FWHM when y-band moments exist |

Suggested derived formula:

`psf_fwhm = 2.355 * sqrt((shape11 + shape22) / 2) * 0.168`

---

### Legacy / DESI-side Photometry and Imaging Conditions

| Unified field | Original key(s) | Original source(s) | Notes |
| --- | --- | --- | --- |
| `legacy_ebv` | `legacy_EBV`, `desi_EBV` | Neighbors, ProvaBGS | Galactic extinction on the Legacy / DESI side |
| `legacy_flux_g` | `legacy_FLUX_G`, `desi_FLUX_G` | Neighbors, ProvaBGS | Flux |
| `legacy_flux_r` | `legacy_FLUX_R`, `desi_FLUX_R` | Neighbors, ProvaBGS | Flux |
| `legacy_flux_i` | `legacy_FLUX_I` | Neighbors | Flux; only present in Neighbors |
| `legacy_flux_z` | `legacy_FLUX_Z`, `desi_FLUX_Z` | Neighbors, ProvaBGS | Flux |
| `legacy_flux_ivar_g` | `legacy_FLUX_IVAR_G`, `desi_FLUX_IVAR_G` | Neighbors, ProvaBGS | Flux inverse variance |
| `legacy_flux_ivar_r` | `legacy_FLUX_IVAR_R`, `desi_FLUX_IVAR_R` | Neighbors, ProvaBGS | Flux inverse variance |
| `legacy_flux_ivar_i` | `legacy_FLUX_IVAR_I` | Neighbors | Flux inverse variance; only present in Neighbors |
| `legacy_flux_ivar_z` | `legacy_FLUX_IVAR_Z`, `desi_FLUX_IVAR_Z` | Neighbors, ProvaBGS | Flux inverse variance |
| `legacy_fiberflux_g` | `desi_FIBERFLUX_G` | ProvaBGS | Fiber flux |
| `legacy_fiberflux_r` | `desi_FIBERFLUX_R` | ProvaBGS | Fiber flux |
| `legacy_fiberflux_z` | `desi_FIBERFLUX_Z` | ProvaBGS | Fiber flux |
| `legacy_fibertotflux_g` | `desi_FIBERTOTFLUX_G` | ProvaBGS | Total fiber-related flux |
| `legacy_fibertotflux_r` | `desi_FIBERTOTFLUX_R` | ProvaBGS | Total fiber-related flux |
| `legacy_fibertotflux_z` | `desi_FIBERTOTFLUX_Z` | ProvaBGS | Total fiber-related flux |
| `legacy_psfsize_g` | `legacy_PSFSIZE_G` | Neighbors | Legacy PSF size |
| `legacy_psfsize_r` | `legacy_PSFSIZE_R` | Neighbors | Legacy PSF size |
| `legacy_psfsize_i` | `legacy_PSFSIZE_I` | Neighbors | Legacy PSF size |
| `legacy_psfsize_z` | `legacy_PSFSIZE_Z` | Neighbors | Legacy PSF size |
| `legacy_psfdepth_g` | `legacy_PSFDEPTH_G` | Neighbors | Legacy PSF depth |
| `legacy_psfdepth_r` | `legacy_PSFDEPTH_R` | Neighbors | Legacy PSF depth |
| `legacy_psfdepth_i` | `legacy_PSFDEPTH_I` | Neighbors | Legacy PSF depth |
| `legacy_psfdepth_z` | `legacy_PSFDEPTH_Z` | Neighbors | Legacy PSF depth |
| `legacy_galdepth_g` | `legacy_GALDEPTH_G` | Neighbors | Galaxy depth |
| `legacy_galdepth_r` | `legacy_GALDEPTH_R` | Neighbors | Galaxy depth |
| `legacy_galdepth_i` | `legacy_GALDEPTH_I` | Neighbors | Galaxy depth |
| `legacy_galdepth_z` | `legacy_GALDEPTH_Z` | Neighbors | Galaxy depth |
| `legacy_nobs_g` | `legacy_NOBS_G` | Neighbors | Number of observations |
| `legacy_nobs_r` | `legacy_NOBS_R` | Neighbors | Number of observations |
| `legacy_nobs_i` | `legacy_NOBS_I` | Neighbors | Number of observations |
| `legacy_nobs_z` | `legacy_NOBS_Z` | Neighbors | Number of observations |
| `legacy_image_ivar` | `legacy_image_ivar` | Neighbors | Image-level inverse variance summary |
| `legacy_image_psf_fwhm` | `legacy_image_psf_fwhm` | Neighbors | Image-level PSF summary |

---

## Optional but Useful Columns

These are not part of the minimal core schema, but they are worth keeping if the collaborator can provide them without much effort.

| Unified field | Original key(s) | Original source(s) | Notes |
| --- | --- | --- | --- |
| `hsc_image` | `hsc_image`, `images_hsc` | MMU, Neighbors | Raw or preprocessed HSC cutout |
| `legacy_image` | `legacysurvey_image`, `images_legacy` | MMU, Neighbors | Raw or preprocessed Legacy cutout |
| `rgb_preview` | `rgb` | MMU | Convenience visualization channel |
| `neighbor_dist_hsc` | `neighbor_dist_hsc` | Neighbors | Only useful if neighbor relationships matter |
| `neighbor_dist_legacy` | `neighbor_dist_legacy` | Neighbors | Only useful if neighbor relationships matter |
| `neighbor_idx_hsc` | `neighbor_idx_hsc` | Neighbors | Only useful if neighbor relationships matter |
| `neighbor_idx_legacy` | `neighbor_idx_legacy` | Neighbors | Only useful if neighbor relationships matter |
| `object_mask` | `object_mask` | MMU | Image / segmentation quality mask |
| `blbdmodel` | `blbdmodel` | MMU | Keep only if this morphology field is meaningful downstream |
| `schlegel_color` | `SCHLEGEL_COLOR` | ProvaBGS | Additional catalog diagnostic |

---

## Fields I Would Not Put in the Main Unified Table

These are better kept in a sidecar file or left out unless there is a concrete use case.

- tokenized / transformed copies such as `tok_*`
- large chain outputs such as `PROVABGS_MCMC`
- fit parameter vectors such as `PROVABGS_THETA_BF`
- duplicated versions of the same concept when a cleaner canonical column already exists

---

## Practical Request to Collaborator

If the collaborator is preparing a single merged file, the cleanest version would be:

1. One row per object.
2. Canonical columns named like the `Unified field` column above.
3. Raw source-specific names preserved in a separate mapping file or metadata block.
4. HSC and Legacy IDs both included even if one is missing for some objects.
5. Raw PSF moment fields included, so PSF FWHM can be recomputed later.
6. Image arrays either stored in the same file or referenced by stable external paths.
