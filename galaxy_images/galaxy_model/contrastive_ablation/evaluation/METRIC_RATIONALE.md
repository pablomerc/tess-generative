# Quality-gap metric rationale

## Resolution

Legacy `PSFSIZE_{G,R,I,Z}` is the weighted PSF full width at half maximum
(FWHM) in arcseconds. The conventional dimensionless resolution proxy is

```text
resolution = apparent galaxy size / PSF FWHM.
```

Consequently, `R_eff / (PSFSIZE / 2)` is not invalid dimensionally, but it is
exactly twice `R_eff / PSFSIZE`; thresholds from one definition cannot be
transferred to the other without rescaling. We use the FWHM definition and
calibrate bins empirically with paired images.

The HSC catalog does not necessarily provide a validated Sérsic half-light
radius for every object. The initial metric therefore derives a size proxy from
the adaptive second moments:

```text
r_det = (Ixx * Iyy - Ixy^2)^(1/4)
```

and, where both measurements are valid, subtracts the local PSF covariance
before taking the determinant. This is a Gaussian adaptive-moment scale, not a
true half-light radius, and outputs must retain that name. Ratios of HSC object
and HSC PSF moments do not require a pixel-scale conversion. Comparison against
Legacy FWHM does require conversion of the HSC size proxy to arcseconds.

The stress-test score is two-sided:

```text
Legacy is poorly resolved AND HSC is well resolved.
```

A large PSF ratio alone is insufficient if both images are poor.

## Signal-to-noise

Per-band catalog SNR is computed independently:

```text
HSC:    cmodel_flux / cmodel_fluxerr
Legacy: FLUX * sqrt(FLUX_IVAR)
```

Only finite, positive-error measurements enter the robust cross-band summary.
Resolution and SNR remain separate reported axes. A joint score may combine
their percentile ranks, but must expose both components and its exact version.

## Leakage controls

- Candidate selection uses no embedding, prediction, target residual, or model
  identity.
- The ranking is frozen before the headline adversarial evaluation.
- Probe predictions are out-of-fold.
- The top-N result is accompanied by quantile curves and matched random
  controls.
- Restricted-subset R2 is reported with normalized MAE/RMSE because low target
  variance can make R2 unstable.
