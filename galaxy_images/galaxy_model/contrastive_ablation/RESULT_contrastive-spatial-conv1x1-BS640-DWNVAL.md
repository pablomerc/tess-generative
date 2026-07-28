# Contrastive DWNVAL — SSL-contamination result

- holdout run   : `predict_contrastive-spatial-conv1x1-BS640-DWNVAL__aion.csv`
- reference run : `predict_contrastive-spatial-conv1x1-DWNVAL__aion.csv`
- shared cells  : 78

## Aggregate ΔR² (holdout − reference)

| statistic | value |
|---|---|
| mean | -0.0568 |
| median | -0.0432 |
| stdev | 0.0565 |
| max abs | -0.2252 |
| cells with abs(Δ) > 0.05 | 36 / 78 |

Flow-matching reference for the same comparison (already measured): `base-5NB` mean -0.0047, `base-15NB` mean +0.0052.

## By encoder x task family

| latent | task family | n | mean Δ | median Δ | max abs Δ |
|---|---|---|---|---|---|
| combined_e1 | instrument_hsc | 9 | -0.0252 | -0.0288 | -0.0419 |
| combined_e1 | instrument_legacy | 21 | -0.0304 | -0.0185 | -0.1848 |
| combined_e1 | physics_provabgs | 9 | -0.0251 | -0.0183 | -0.0882 |
| combined_e2 | instrument_hsc | 9 | -0.0449 | -0.0411 | -0.1518 |
| combined_e2 | instrument_legacy | 21 | -0.0772 | -0.0709 | -0.1483 |
| combined_e2 | physics_provabgs | 9 | -0.1455 | -0.1471 | -0.2252 |

## Headline cells (paper Table 5 / Fig. 4)

| cell | reference | holdout | Δ |
|---|---|---|---|
| Redshift z (physics latent) | +0.743 | +0.716 | -0.027 |
| log M* (physics latent) | +0.683 | +0.673 | -0.010 |
| Ellipticity e1 (physics latent) | +0.913 | +0.894 | -0.018 |
| HSC ellipticity e1 (physics latent) | +0.891 | +0.875 | -0.016 |
| Galaxy Depth R (instrument latent) | +0.914 | +0.818 | -0.097 |
| PSF Depth R (instrument latent) | +0.909 | +0.817 | -0.093 |
| # Observations R (instrument latent) | +0.657 | +0.508 | -0.148 |
| E(B-V) (instrument latent) | +0.725 | +0.664 | -0.061 |
| Galaxy Depth R (physics latent — expect ~0) | +0.043 | +0.008 | -0.035 |
| Redshift z (instrument latent — the leak) | +0.569 | +0.458 | -0.111 |

## Top 8 movers

| task family | target | latent | reference | holdout | Δ |
|---|---|---|---|---|---|
| physics_provabgs | legacy_shape_e1 | combined_e2 | +0.452 | +0.227 | -0.225 |
| physics_provabgs | legacy_shape_e2 | combined_e2 | +0.575 | +0.352 | -0.223 |
| physics_provabgs | hsc_shape_e2_i | combined_e2 | +0.545 | +0.337 | -0.208 |
| physics_provabgs | hsc_shape_e1_i | combined_e2 | +0.442 | +0.238 | -0.204 |
| instrument_legacy | PSFSIZE_G | combined_e1 | +0.186 | +0.001 | -0.185 |
| instrument_hsc | psf_fwhm_r | combined_e2 | +0.641 | +0.489 | -0.152 |
| instrument_legacy | NOBS_R | combined_e2 | +0.657 | +0.508 | -0.148 |
| physics_provabgs | provabgs_logmstar | combined_e2 | +0.540 | +0.393 | -0.147 |

## Verdict

Mean |ΔR²| = 0.0603, LARGER than the flow-matching arm's bound (0.0052). Contamination affects the contrastive arm more than the generative one — report the holdout numbers as the headline and say so.
