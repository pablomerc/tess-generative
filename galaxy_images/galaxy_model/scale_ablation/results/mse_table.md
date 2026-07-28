## Reconstruction MSE vs # crossmatches (C0 / intact, 256 held-out anchors, 250 Euler steps)

| model | N crossmatches | distinct images | MSE HSC | MSE Legacy |
|---|---|---|---|---|
| N=100 | 100 | 700 | 0.2580 ± 0.0614 | 0.2831 ± 0.0366 |
| N=1,000 | 1,000 | 6,942 | 0.1069 ± 0.0147 | 0.1752 ± 0.0099 |
| N=3,162 | 3,162 | 21,487 | 0.0746 ± 0.0084 | 0.1538 ± 0.0063 |
| N=10,000 | 10,000 | 64,210 | 0.0591 ± 0.0058 | 0.1611 ± 0.0062 |
| N=31,622 | 31,622 | 176,610 | 0.0534 ± 0.0055 | 0.1876 ± 0.0072 |
| full data, 75k (fm-control) * | 93,362 (full)* | — | 0.0729 ± 0.0104 | 0.1809 ± 0.0074 |
| full data, 87k (paper ckpt) * | 93,362 (full)* | — | 0.0742 ± 0.0078 | 0.1997 ± 0.0072 |
| full data, constant-LR 35k * | 93,362 (full)* | — | 0.0679 ± 0.0086 | 0.1960 ± 0.0077 |

\* trained **without** the downstream holdout → saw ~95% of these eval galaxies during training (unlabeled); their MSE is optimistic. The scale arms never saw any eval galaxy — enforced at load time. Conservative direction for the scaling claim.
