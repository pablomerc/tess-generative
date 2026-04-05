# Posterior Calibration (Paper Description)

## Suggested paragraph

To assess the calibration of our conditional generative model, we evaluate
whether the learned posterior accurately captures the uncertainty of the true
galaxy image. For each galaxy in a held-out set, we draw $M = 20$ independent
posterior samples using 250-step Euler integration and compute the per-pixel
sample mean $\hat\mu$ and standard deviation $\hat\sigma$ across the $M$ draws.
We then form pixel-wise z-scores $z = (x^{*} - \hat\mu) / \hat\sigma$, where
$x^{*}$ is the true anchor image. If the posterior is well-calibrated, the
pooled z-scores across all galaxies, channels, and spatial positions should
follow a standard normal distribution $\mathcal{N}(0, 1)$.
We verify this by inspecting histograms of the z-score distribution overlaid
with the $\mathcal{N}(0, 1)$ density and by comparing empirical quantiles to
theoretical quantiles in a Q-Q plot. We report the mean and standard deviation
of the pooled z-scores as summary statistics. We perform this analysis
separately for galaxies whose anchor observations come from the HSC and DECaLS
Legacy surveys.
