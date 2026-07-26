# Contrastive DWNVAL — SSL-contamination result

- holdout run   : `predict_contrastive-spatial-conv1x1-DWNVAL__aion.csv`
- reference run : `predict_contrastive-spatial-conv1x1__aion.csv`
- shared cells  : 78

## Aggregate ΔR² (holdout − reference)

| statistic | value |
|---|---|
| mean | +0.0059 |
| median | -0.0004 |
| stdev | 0.0304 |
| max abs | +0.1718 |
| cells with abs(Δ) > 0.05 | 3 / 78 |

Flow-matching reference for the same comparison (already measured): `base-5NB` mean -0.0047, `base-15NB` mean +0.0052.

## By encoder x task family

| latent | task family | n | mean Δ | median Δ | max abs Δ |
|---|---|---|---|---|---|
| combined_e1 | instrument_hsc | 9 | +0.0046 | -0.0002 | +0.0457 |
| combined_e1 | instrument_legacy | 21 | +0.0293 | +0.0288 | +0.1718 |
| combined_e1 | physics_provabgs | 9 | +0.0028 | +0.0026 | +0.0159 |
| combined_e2 | instrument_hsc | 9 | -0.0038 | -0.0007 | -0.0376 |
| combined_e2 | instrument_legacy | 21 | -0.0127 | -0.0126 | -0.0326 |
| combined_e2 | physics_provabgs | 9 | +0.0090 | +0.0004 | +0.0722 |

## Headline cells (paper Table 5 / Fig. 4)

| cell | reference | holdout | Δ |
|---|---|---|---|
| Redshift z (physics latent) | +0.740 | +0.743 | +0.003 |
| log M* (physics latent) | +0.675 | +0.683 | +0.008 |
| Ellipticity e1 (physics latent) | +0.918 | +0.913 | -0.005 |
| HSC ellipticity e1 (physics latent) | +0.882 | +0.891 | +0.009 |
| Galaxy Depth R (instrument latent) | +0.914 | +0.914 | +0.001 |
| PSF Depth R (instrument latent) | +0.909 | +0.909 | +0.001 |
| # Observations R (instrument latent) | +0.659 | +0.657 | -0.002 |
| E(B-V) (instrument latent) | +0.740 | +0.725 | -0.016 |
| Galaxy Depth R (physics latent — expect ~0) | +0.010 | +0.043 | +0.033 |
| Redshift z (instrument latent — the leak) | +0.568 | +0.569 | +0.000 |

## Top 8 movers

| task family | target | latent | reference | holdout | Δ |
|---|---|---|---|---|---|
| instrument_legacy | PSFSIZE_G | combined_e1 | +0.014 | +0.186 | +0.172 |
| physics_provabgs | legacy_shape_e1 | combined_e2 | +0.380 | +0.452 | +0.072 |
| physics_provabgs | hsc_shape_e1_i | combined_e2 | +0.371 | +0.442 | +0.071 |
| instrument_legacy | PSFSIZE_R | combined_e1 | +0.021 | +0.069 | +0.048 |
| instrument_legacy | NOBS_I | combined_e1 | -0.002 | +0.044 | +0.047 |
| instrument_hsc | a_z | combined_e1 | +0.006 | +0.051 | +0.046 |
| instrument_legacy | NOBS_G | combined_e1 | -0.020 | +0.025 | +0.045 |
| physics_provabgs | ssfr | combined_e2 | +0.397 | +0.355 | -0.042 |

## Verdict

Mean |ΔR²| = 0.0202, LARGER than the flow-matching arm's bound (0.0052). Contamination affects the contrastive arm more than the generative one — report the holdout numbers as the headline and say so.
