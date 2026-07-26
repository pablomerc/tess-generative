# Contrastive DWNVAL — SSL-contamination result

- holdout run   : `predict_contrastive-instrneg-samesurvey-DWNVAL__aion.csv`
- reference run : `predict_contrastive-spatial-conv1x1__aion.csv`
- shared cells  : 78

## Aggregate ΔR² (holdout − reference)

| statistic | value |
|---|---|
| mean | -0.0020 |
| median | +0.0015 |
| stdev | 0.0236 |
| max abs | -0.0633 |
| cells with abs(Δ) > 0.05 | 2 / 78 |

Flow-matching reference for the same comparison (already measured): `base-5NB` mean -0.0047, `base-15NB` mean +0.0052.

## By encoder x task family

| latent | task family | n | mean Δ | median Δ | max abs Δ |
|---|---|---|---|---|---|
| combined_e1 | instrument_hsc | 9 | -0.0036 | -0.0079 | +0.0434 |
| combined_e1 | instrument_legacy | 21 | +0.0107 | +0.0078 | +0.0373 |
| combined_e1 | physics_provabgs | 9 | +0.0009 | +0.0088 | -0.0360 |
| combined_e2 | instrument_hsc | 9 | +0.0145 | +0.0218 | -0.0590 |
| combined_e2 | instrument_legacy | 21 | -0.0170 | -0.0179 | -0.0633 |
| combined_e2 | physics_provabgs | 9 | -0.0149 | -0.0194 | -0.0428 |

## Headline cells (paper Table 5 / Fig. 4)

| cell | reference | holdout | Δ |
|---|---|---|---|
| Redshift z (physics latent) | +0.740 | +0.749 | +0.009 |
| log M* (physics latent) | +0.675 | +0.703 | +0.028 |
| Ellipticity e1 (physics latent) | +0.918 | +0.913 | -0.004 |
| HSC ellipticity e1 (physics latent) | +0.882 | +0.891 | +0.009 |
| Galaxy Depth R (instrument latent) | +0.914 | +0.908 | -0.005 |
| PSF Depth R (instrument latent) | +0.909 | +0.902 | -0.006 |
| # Observations R (instrument latent) | +0.659 | +0.660 | +0.002 |
| E(B-V) (instrument latent) | +0.740 | +0.743 | +0.003 |
| Galaxy Depth R (physics latent — expect ~0) | +0.010 | +0.018 | +0.008 |
| Redshift z (instrument latent — the leak) | +0.568 | +0.549 | -0.019 |

## Top 8 movers

| task family | target | latent | reference | holdout | Δ |
|---|---|---|---|---|---|
| instrument_legacy | NOBS_Z | combined_e2 | +0.770 | +0.707 | -0.063 |
| instrument_hsc | psf_fwhm_r | combined_e2 | +0.628 | +0.569 | -0.059 |
| instrument_hsc | psf_fwhm_g | combined_e2 | +0.462 | +0.511 | +0.049 |
| instrument_hsc | a_z | combined_e1 | +0.006 | +0.049 | +0.043 |
| physics_provabgs | provabgs_logmstar | combined_e2 | +0.563 | +0.520 | -0.043 |
| instrument_legacy | PSFSIZE_I | combined_e2 | +0.851 | +0.809 | -0.042 |
| physics_provabgs | hsc_shape_e2_i | combined_e2 | +0.532 | +0.492 | -0.040 |
| physics_provabgs | hsc_shape_e1_i | combined_e2 | +0.371 | +0.332 | -0.039 |

## Verdict

Mean |ΔR²| = 0.0189, LARGER than the flow-matching arm's bound (0.0052). Contamination affects the contrastive arm more than the generative one — report the holdout numbers as the headline and say so.
