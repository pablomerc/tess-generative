# Latent Traversal

Latent traversal experiments on the dual-encoder galaxy model.

## Current: Property Distribution Plots

### Context

Follows the same patterns established in:

- `umap_exploration/umap_neighbors_exploration.py` for metadata extraction, HSC PSF FWHM derivation, and channel averaging via `AVERAGE_PATTERNS`
- `downstream_evaluation/final/baseline_all.py` for direct h5py data loading from `neighbours_v2.h5`

### Data Source

- HDF5: `/work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5`
- Filter to `source_type == 0` (MMU rows), take first 8000 examples
- No model/encoder needed -- only raw metadata from the HDF5

### What `plot_distributions.py` Does

1. **Loads metadata** directly from `neighbours_v2.h5` using h5py (no torch/model needed), reading all columns except image arrays and neighbor indices (same `NEIGHBORS_SIMPLE_EXCLUDE_KEYS` pattern)
2. **Derives HSC PSF FWHM** from `hsc_{band}_sdssshape_psf_shape11/22` using:
   ```
   fwhm = 2.355 * sqrt((shape11 + shape22) / 2) * 0.168
   ```
3. **Averages multi-band properties** across channels using the same `AVERAGE_PATTERNS` regex logic from `umap_neighbors_exploration.py`:
   - `legacy_PSFSIZE_{G,I,R,Z}` -> `legacy_PSFSIZE`
   - `hsc_{g,i,r,z}_psf_fwhm` -> `hsc_psf_fwhm`
   - Plus: `legacy_PSFDEPTH`, `legacy_GALDEPTH`, `legacy_NOBS`, `hsc_variance_value`
4. **Plots histograms** of all averaged properties, saving two figures

### Outputs

- `property_distributions.png` -- Grid of histograms, one per averaged property, with median lines
- `psf_distributions.png` -- Dedicated HSC vs Legacy PSF size comparison (per-band overlaid + channel-averaged side-by-side)

### How to Run

```bash
sbatch plot_distributions.slurm
```

Or directly:

```bash
python plot_distributions.py --n-examples 8000
```
