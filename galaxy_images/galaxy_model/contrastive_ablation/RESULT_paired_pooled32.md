# Paired 5-fold comparison: `base-meanpool` vs `contrastive-spatial-conv1x1-meanpool`

Positive = **base-meanpool** higher. SE is over folds, paired (same test galaxies both sides). |t| >= 4 required to call a difference resolved.

| cell | A − B | SE | t | verdict |
|---|---|---|---|---|
| Redshift z (phys) | +0.024 | 0.010 | +2.5 | marginal |
| log M* (phys) | +0.027 | 0.011 | +2.5 | marginal |
| sSFR (phys) | +0.057 | 0.011 | +5.4 | RESOLVED |
| Ellipticity LEGACY (phys) | -0.130 | 0.003 | -50.9 | RESOLVED |
| Ellipticity HSC (phys) | -0.145 | 0.006 | -25.6 | RESOLVED |
| Galaxy Depth (instr) | -0.062 | 0.007 | -9.5 | RESOLVED |
| PSF Depth (instr) | -0.069 | 0.007 | -9.6 | RESOLVED |
| # Observations (instr) | -0.086 | 0.004 | -22.4 | RESOLVED |
| E(B-V) (instr) | -0.140 | 0.013 | -10.7 | RESOLVED |
| PSF FWHM HSC (instr) | -0.106 | 0.013 | -7.9 | RESOLVED |
| LEAK z from instr | +0.115 | 0.009 | +12.8 | RESOLVED |
| LEAK depth from phys | +0.096 | 0.007 | +13.2 | RESOLVED |

## Resolution-retention test (difference of differences)

(A−B on **HSC** shapes) − (A−B on **Legacy** shapes) = **-0.014 ± 0.005** (t = -3.1, marginal)

Positive means B loses more on the higher-resolution survey's shapes than on the lower-resolution one's — the signature of an objective that keeps only what the two views share. Being a difference of differences, it cannot be explained by either model simply being better overall.
