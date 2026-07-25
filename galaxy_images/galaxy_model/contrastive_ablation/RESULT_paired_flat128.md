# Paired 5-fold comparison: `base` vs `contrastive-spatial-conv1x1`

Positive = **base** higher. SE is over folds, paired (same test galaxies both sides). |t| >= 4 required to call a difference resolved.

| cell | A − B | SE | t | verdict |
|---|---|---|---|---|
| Redshift z (phys) | +0.032 | 0.002 | +13.6 | RESOLVED |
| log M* (phys) | +0.048 | 0.004 | +12.8 | RESOLVED |
| sSFR (phys) | +0.027 | 0.010 | +2.8 | marginal |
| Ellipticity LEGACY (phys) | +0.013 | 0.001 | +10.8 | RESOLVED |
| Ellipticity HSC (phys) | +0.040 | 0.002 | +20.1 | RESOLVED |
| Galaxy Depth (instr) | -0.100 | 0.002 | -49.7 | RESOLVED |
| PSF Depth (instr) | -0.104 | 0.003 | -41.0 | RESOLVED |
| # Observations (instr) | -0.164 | 0.007 | -23.6 | RESOLVED |
| E(B-V) (instr) | -0.179 | 0.009 | -20.8 | RESOLVED |
| PSF FWHM HSC (instr) | -0.184 | 0.006 | -28.8 | RESOLVED |
| LEAK z from instr | +0.133 | 0.014 | +9.8 | RESOLVED |
| LEAK depth from phys | +0.178 | 0.003 | +65.3 | RESOLVED |

## Resolution-retention test (difference of differences)

(A−B on **HSC** shapes) − (A−B on **Legacy** shapes) = **+0.026 ± 0.003** (t = +10.6, RESOLVED)

Positive means B loses more on the higher-resolution survey's shapes than on the lower-resolution one's — the signature of an objective that keeps only what the two views share. Being a difference of differences, it cannot be explained by either model simply being better overall.
