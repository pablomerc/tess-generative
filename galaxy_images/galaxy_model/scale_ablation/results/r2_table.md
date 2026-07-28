## Downstream R² vs # crossmatches (n=5469 crossmatch, 90/10 MLP probes, fixed-75k ckpts)

physics rows → physics latent (combined_e1, flat); instrument rows → instrument latent (combined_e2, **meanpool** twin where available — flagged † when flat fallback was used)

| property | scale-100 | scale-1000 | scale-3162 | scale-10000 | scale-31622 | fm-control | base | untrained |
|---|---|---|---|---|---|---|---|---|
| **Physics-related properties** |  |  |  |  |  |  |  |  |
| Ellipticity e1 | 0.827 | 0.918 | 0.919 | 0.927 | 0.927 | 0.924 | 0.918 | 0.613 |
| Ellipticity e2 | 0.851 | 0.924 | 0.916 | 0.918 | 0.921 | 0.917 | 0.934 | 0.616 |
| Redshift z | 0.659 | 0.753 | 0.763 | 0.773 | 0.791 | 0.804 | 0.801 | 0.514 |
| log M* | 0.611 | 0.709 | 0.708 | 0.740 | 0.729 | 0.763 | 0.755 | 0.485 |
| sSFR | 0.427 | 0.496 | 0.516 | 0.531 | 0.529 | 0.542 | 0.558 | 0.312 |
| t_age | 0.141 | 0.109 | 0.220 | 0.230 | 0.276 | 0.283 | 0.257 | 0.091 |
| log Z_Met | 0.179 | 0.241 | 0.244 | 0.259 | 0.265 | 0.244 | 0.250 | 0.123 |
| **mean physics (7)** | **0.528** | **0.593** | **0.612** | **0.625** | **0.634** | **0.640** | **0.639** | **0.394** |
| **Legacy instrument properties** |  |  |  |  |  |  |  |  |
| Galaxy Depth (L) | 0.256 | 0.367 | 0.474 | 0.568 | 0.594 | 0.720 | 0.682 | 0.195 |
| PSF Depth (L) | 0.241 | 0.351 | 0.458 | 0.553 | 0.581 | 0.702 | 0.667 | 0.201 |
| PSF Size (L) | 0.118 | 0.214 | 0.309 | 0.422 | 0.473 | 0.582 | 0.556 | 0.077 |
| Extinction E(B-V) (L) | 0.276 | 0.335 | 0.371 | 0.422 | 0.453 | 0.544 | 0.551 | 0.169 |
| # Observations (L) | 0.087 | 0.175 | 0.340 | 0.391 | 0.393 | 0.509 | 0.501 | 0.034 |
| **mean Legacy instrument (5)** | **0.196** | **0.289** | **0.390** | **0.471** | **0.499** | **0.611** | **0.591** | **0.135** |
| **HSC instrument properties** |  |  |  |  |  |  |  |  |
| Extinction a (H) | 0.289 | 0.338 | 0.369 | 0.417 | 0.452 | 0.548 | 0.546 | 0.169 |
| PSF FWHM (H) | 0.063 | 0.100 | 0.196 | 0.289 | 0.298 | 0.408 | 0.438 | 0.069 |
| **mean HSC instrument (2)** | **0.176** | **0.219** | **0.282** | **0.353** | **0.375** | **0.478** | **0.492** | **0.119** |
| _pooled instrument (7), Legacy-weighted 5:2 — reference only_ | _0.190_ | _0.269_ | _0.360_ | _0.437_ | _0.463_ | _0.573_ | _0.563_ | _0.131_ |

_Averaging follows the paper table exactly: 7 physics rows (HSC ellipticities excluded as in the paper), band-averaged instrument rows (channels averaged within each quantity, MW_TRANSMISSION excluded). **Legacy and HSC instrument means are reported separately** — the surveys differ in depth/resolution and the paper groups them separately; the pooled row is shown only for reference because it silently weights Legacy 5:2._

_self-check vs paper merged_simplified.csv: 14 rows, max |Δ| = 5.55e-17 → OK_
