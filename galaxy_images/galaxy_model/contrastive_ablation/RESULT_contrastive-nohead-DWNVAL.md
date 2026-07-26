# Contrastive DWNVAL — SSL-contamination result

- holdout run   : `predict_contrastive-nohead-DWNVAL__aion.csv`
- reference run : `predict_contrastive-spatial-conv1x1__aion.csv`
- shared cells  : 78

## Aggregate ΔR² (holdout − reference)

| statistic | value |
|---|---|
| mean | -0.0206 |
| median | -0.0184 |
| stdev | 0.0346 |
| max abs | -0.1203 |
| cells with abs(Δ) > 0.05 | 11 / 78 |

Flow-matching reference for the same comparison (already measured): `base-5NB` mean -0.0047, `base-15NB` mean +0.0052.

## By encoder x task family

| latent | task family | n | mean Δ | median Δ | max abs Δ |
|---|---|---|---|---|---|
| combined_e1 | instrument_hsc | 9 | -0.0175 | -0.0109 | -0.0638 |
| combined_e1 | instrument_legacy | 21 | -0.0080 | -0.0090 | +0.0423 |
| combined_e1 | physics_provabgs | 9 | -0.0471 | -0.0481 | -0.0914 |
| combined_e2 | instrument_hsc | 9 | +0.0078 | +0.0035 | +0.0674 |
| combined_e2 | instrument_legacy | 21 | -0.0223 | -0.0214 | -0.0468 |
| combined_e2 | physics_provabgs | 9 | -0.0516 | -0.0363 | -0.1203 |

## Headline cells (paper Table 5 / Fig. 4)

| cell | reference | holdout | Δ |
|---|---|---|---|
| Redshift z (physics latent) | +0.740 | +0.715 | -0.025 |
| log M* (physics latent) | +0.675 | +0.671 | -0.004 |
| Ellipticity e1 (physics latent) | +0.918 | +0.849 | -0.069 |
| HSC ellipticity e1 (physics latent) | +0.882 | +0.836 | -0.047 |
| Galaxy Depth R (instrument latent) | +0.914 | +0.894 | -0.020 |
| PSF Depth R (instrument latent) | +0.909 | +0.891 | -0.018 |
| # Observations R (instrument latent) | +0.659 | +0.644 | -0.015 |
| E(B-V) (instrument latent) | +0.740 | +0.718 | -0.022 |
| Galaxy Depth R (physics latent — expect ~0) | +0.010 | -0.015 | -0.025 |
| Redshift z (instrument latent — the leak) | +0.568 | +0.569 | +0.001 |

## Top 8 movers

| task family | target | latent | reference | holdout | Δ |
|---|---|---|---|---|---|
| physics_provabgs | hsc_shape_e2_i | combined_e2 | +0.532 | +0.412 | -0.120 |
| physics_provabgs | legacy_shape_e2 | combined_e2 | +0.545 | +0.426 | -0.119 |
| physics_provabgs | hsc_shape_e1_i | combined_e2 | +0.371 | +0.259 | -0.112 |
| physics_provabgs | legacy_shape_e1 | combined_e2 | +0.380 | +0.275 | -0.105 |
| physics_provabgs | legacy_shape_e2 | combined_e1 | +0.895 | +0.803 | -0.091 |
| physics_provabgs | hsc_shape_e2_i | combined_e1 | +0.893 | +0.824 | -0.070 |
| physics_provabgs | provabgs_z_mw | combined_e1 | +0.272 | +0.203 | -0.069 |
| physics_provabgs | legacy_shape_e1 | combined_e1 | +0.918 | +0.849 | -0.069 |

## Verdict

Mean |ΔR²| = 0.0291, LARGER than the flow-matching arm's bound (0.0052). Contamination affects the contrastive arm more than the generative one — report the holdout numbers as the headline and say so.
