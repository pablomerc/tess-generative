# 5-fold R² with error bars

Disjoint 5-way partition (seed 0), `--mlp-arch aion`; every galaxy tested exactly once. Values are mean ± std across folds.

Variants: `base`, `base-meanpool`, `contrastive-spatial-conv1x1`, `contrastive-spatial-conv1x1-meanpool`

| cell | base | base-meanpool | contrastive-spatial-conv1x1 | contrastive-spatial-conv1x1-meanpool |
|---|---|---|---|---|
| Redshift z | +0.784±0.005 | +0.717±0.010 | +0.752±0.006 | +0.693±0.016 |
| log M* | +0.745±0.008 | +0.624±0.023 | +0.697±0.004 | +0.597±0.011 |
| sSFR | +0.511±0.014 | +0.472±0.014 | +0.484±0.029 | +0.415±0.023 |
| Ellipticity (legacy) | +0.933±0.005 | +0.729±0.014 | +0.920±0.006 | +0.860±0.010 |
| Ellipticity (HSC) | +0.938±0.006 | +0.695±0.016 | +0.899±0.007 | +0.840±0.011 |
| Galaxy Depth | +0.751±0.011 | +0.688±0.020 | +0.852±0.016 | +0.750±0.027 |
| PSF Depth | +0.742±0.013 | +0.671±0.023 | +0.846±0.020 | +0.741±0.032 |
| # Observations | +0.535±0.014 | +0.482±0.012 | +0.699±0.009 | +0.567±0.011 |
| E(B-V) | +0.591±0.025 | +0.541±0.023 | +0.770±0.017 | +0.681±0.020 |
| PSF FWHM (HSC) | +0.500±0.017 | +0.420±0.017 | +0.684±0.018 | +0.526±0.019 |
| LEAK: z from instrument | +0.669±0.010 | +0.513±0.012 | +0.536±0.026 | +0.397±0.015 |
| LEAK: depth from physics | +0.170±0.010 | +0.098±0.011 | -0.008±0.006 | +0.001±0.007 |

**Typical fold-to-fold spread:** median std 0.020, 90th pct 0.032. Differences smaller than ~2x this are not resolvable with n=5,469.
