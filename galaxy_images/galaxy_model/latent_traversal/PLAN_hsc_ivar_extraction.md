# Plan: Extract HSC ivar + PSF stats from parquets

**Status: DEFERRED — revisit after normalization exploration is done.**

## Goal

One-time batch job that reads all 396 parquet files, extracts scalar HSC stats
per example, and saves an HDF5 aligned row-for-row with neighbours_v2.h5, so
downstream traversal scripts can load it with zero join overhead.

## Why parquets and not HDF5?

The neighbours_v2.h5 already has `legacy_image_ivar` (Legacy) and
`legacy_PSFSIZE_*` (Legacy). The **HSC** ivar lives only in the raw parquets at
`/work1/jeroenaudenaert/pablomer/data/raw_mmu/data/` as
`hsc_image['ivar']` (per-band pixel cutout). HSC PSF is `hsc_image['psf_fwhm']`
(5-band float array, already clean scalars).

## Data sources

- **396 parquet files:** `train-XXXXX-of-00396.parquet`
- Per row:
  - `hsc_object_id` (str) → join key to `object_id_hsc` (bytes) in HDF5
  - `hsc_image['psf_fwhm']` → (5,) float32, bands g/r/i/z/y
  - `hsc_image['ivar']` → per-band pixel arrays (shape TBD, use `.mean()` robustly)
  - `g/r/i/z/y_cmodel_mag` → HSC catalog magnitudes

## Output HDF5: `/work1/jeroenaudenaert/pablomer/data/hsc_ivar_psf_stats.h5`

Arrays of length 468,197 (matching neighbours_v2.h5), NaN for unmatched rows:

| Key | Shape | Description |
|-----|-------|-------------|
| `hsc_mean_ivar` | (468197,) | mean ivar across all bands × pixels |
| `hsc_ivar_per_band` | (468197, 5) | per-band mean ivar, bands g/r/i/z/y |
| `hsc_psf_fwhm` | (468197, 5) | PSF FWHM per band (arcsec) |
| `hsc_psf_fwhm_avg` | (468197,) | mean PSF FWHM across bands |
| `hsc_cmodel_mag` | (468197, 5) | HSC cmodel magnitudes g/r/i/z/y |

## Script: `extract_hsc_stats.py`

1. Load `object_id_hsc` from HDF5 → build `{id_str → hdf5_row}` dict
2. `multiprocessing.Pool` over 396 parquet files; per file:
   - `pd.read_parquet(f, columns=['hsc_object_id','hsc_image','g_cmodel_mag',...])`
   - Per row: extract psf_fwhm, `[np.asarray(ivar[b]).mean() for b in range(5)]`, cmodel mags
3. Write pre-allocated HDF5 arrays, fill by hdf5_row index, NaN fill for misses
4. Log n_matched / n_total

## Slurm: `extract_hsc_stats.slurm`

- Partition: `mi2104x`, time: 2h, cpus-per-task: 16

## Verification

```python
import h5py, numpy as np
f = h5py.File('/work1/jeroenaudenaert/pablomer/data/hsc_ivar_psf_stats.h5', 'r')
for k in f: print(k, np.array(f[k]).shape, 'nans:', np.isnan(f[k]).sum())
```


---
User notes:

this plan looks good.

I think the output wont be ~400k examples but more like ~103k.

lets make two jobs one that does this "preprocessing" for the first 1 parquet file and saves it, and then generates the 2D density plot for that result (for testing quickly).

And a longe one that does the full dataset and also generates the 2D plot but will take longer to run.

After this is done, we will begin to think about how to combine PSF and IVAR in a smart way and get a latent traversal.