# Contrastive DWNVAL — SSL-contamination result

- holdout run   : `predict_contrastive-galneg-samesurvey-DWNVAL__aion.csv`
- reference run : `predict_contrastive-spatial-conv1x1__aion.csv`
- shared cells  : 78

## Aggregate ΔR² (holdout − reference)

| statistic | value |
|---|---|
| mean | +0.0000 |
| median | +0.0030 |
| stdev | 0.0290 |
| max abs | -0.0963 |
| cells with abs(Δ) > 0.05 | 7 / 78 |

Flow-matching reference for the same comparison (already measured): `base-5NB` mean -0.0047, `base-15NB` mean +0.0052.

## By encoder x task family

| latent | task family | n | mean Δ | median Δ | max abs Δ |
|---|---|---|---|---|---|
| combined_e1 | instrument_hsc | 9 | +0.0032 | +0.0038 | -0.0562 |
| combined_e1 | instrument_legacy | 21 | +0.0139 | +0.0179 | +0.0834 |
| combined_e1 | physics_provabgs | 9 | +0.0011 | +0.0034 | -0.0151 |
| combined_e2 | instrument_hsc | 9 | +0.0161 | +0.0152 | +0.0578 |
| combined_e2 | instrument_legacy | 21 | -0.0029 | -0.0078 | -0.0206 |
| combined_e2 | physics_provabgs | 9 | -0.0460 | -0.0436 | -0.0963 |

## Headline cells (paper Table 5 / Fig. 4)

| cell | reference | holdout | Δ |
|---|---|---|---|
| Redshift z (physics latent) | +0.740 | +0.737 | -0.003 |
| log M* (physics latent) | +0.675 | +0.685 | +0.010 |
| Ellipticity e1 (physics latent) | +0.918 | +0.921 | +0.003 |
| HSC ellipticity e1 (physics latent) | +0.882 | +0.888 | +0.005 |
| Galaxy Depth R (instrument latent) | +0.914 | +0.917 | +0.003 |
| PSF Depth R (instrument latent) | +0.909 | +0.911 | +0.003 |
| # Observations R (instrument latent) | +0.659 | +0.671 | +0.012 |
| E(B-V) (instrument latent) | +0.740 | +0.750 | +0.010 |
| Galaxy Depth R (physics latent — expect ~0) | +0.010 | +0.052 | +0.041 |
| Redshift z (instrument latent — the leak) | +0.568 | +0.528 | -0.040 |

## Top 8 movers

| task family | target | latent | reference | holdout | Δ |
|---|---|---|---|---|---|
| physics_provabgs | ssfr | combined_e2 | +0.397 | +0.301 | -0.096 |
| instrument_legacy | PSFSIZE_G | combined_e1 | +0.014 | +0.097 | +0.083 |
| physics_provabgs | hsc_shape_e2_i | combined_e2 | +0.532 | +0.453 | -0.079 |
| physics_provabgs | provabgs_logmstar | combined_e2 | +0.563 | +0.501 | -0.062 |
| instrument_hsc | psf_fwhm_g | combined_e2 | +0.462 | +0.520 | +0.058 |
| instrument_hsc | psf_fwhm_i | combined_e1 | +0.068 | +0.012 | -0.056 |
| physics_provabgs | legacy_shape_e2 | combined_e2 | +0.545 | +0.493 | -0.052 |
| instrument_legacy | NOBS_G | combined_e1 | -0.020 | +0.024 | +0.045 |

## Verdict

Mean |ΔR²| = 0.0215, LARGER than the flow-matching arm's bound (0.0052). Contamination affects the contrastive arm more than the generative one — report the holdout numbers as the headline and say so.
