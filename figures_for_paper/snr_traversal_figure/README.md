# SNR-traversal figure (paper)

Per-target figure showing how the model reconstructs an HSC galaxy when conditioned on instrument latents drawn from progressively higher-SNR HSC neighbors. Layout:

```
[HSC target]  [Legacy pair]  ||  [recon @ SNR p10]  ...  [recon @ SNR p90]
                              <--------- SNR low → SNR high --------->
```

No conditioning thumbnails are shown — only the inputs (target HSC + same-galaxy Legacy) and the resulting reconstructions per SNR bucket. Each bucket's average SNR (from the K nearest-by-SNR HSC neighbors used as instrument conditioning) is annotated under the panel.

## Files

```
snr_traversal_figure/
├── README.md                            # this file
├── paper_snr_traversal.py               # standalone renderer (numpy + matplotlib + h5py)
├── _cache/
│   └── snr_traversal_arrays.h5          # bundled intermediate; one group per (target, mode)
└── paper_snr_traversal_<idx>_k10_indep.{png,pdf}
```

## Cache schema

`_cache/snr_traversal_arrays.h5` is the sidecar produced by the upstream traversal script. Each `target_{stats_idx:05d}/{mode}` group contains:

| Dataset / attr | Shape | Meaning |
|---|---|---|
| `target_hsc`    | `(4, 48, 48)` | The HSC stamp for the chosen target (preprocessed, 4-band) |
| `target_legacy` | `(4, 48, 48)` | The same-galaxy Legacy stamp |
| `generated`     | `(5, 4, 48, 48)` | One reconstruction per SNR bucket, ordered noisy→clean (matches upstream `SNR_LEVELS`) |
| `buckets/percentiles` | `(5,)` | The neg-SNR percentiles used to pick each bucket (95, 75, 50, 25, 5) |
| `buckets/snr_avgs`    | `(5,)` | Average SNR of the K HSC neighbors that conditioned each reconstruction |
| `buckets/ivar_avgs`, `buckets/psf_avgs` | `(5,)` | Auxiliary noise stats per bucket |
| `buckets/labels`      | `(5,)` S64 | Human-readable bucket labels |
| `buckets/cond_images_unique` | `(5, n_select, 4, 48, 48)` | The conditioning HSC neighbors (kept for reproducibility, not used by this paper plot) |
| attr `target_snr`, `target_psf`, `target_ivar`, `target_stats_idx`, `target_hdf5_row` | scalars | Target-level metadata |
| attr `mode`, `n_select`, `n_pass`, `repeat_one` | scalars | Upstream sampling configuration |

The bundled cache holds k10 results for the paper-figure target set: `53601, 55427, 83907, 15780, 33241`. `53601` is the original favorite; the other four were picked from the random p60-SNR gallery sent to Discord by `pick_random_targets.py`.

## Recreate the figures

### From the bundled cache (no GPU needed)

```bash
cd figures_for_paper/snr_traversal_figure
python paper_snr_traversal.py                            # all 5 targets, PNG+PDF, indep vis
python paper_snr_traversal.py --target-idxs 53601        # one target only
python paper_snr_traversal.py --vis rowscale             # row-scale to target HSC instead
python paper_snr_traversal.py --formats png              # skip PDF
```

Only `numpy`, `matplotlib`, and `h5py` are needed.

### From scratch (regenerating the cache)

The cache was built by running:

```
galaxy_images/galaxy_model/latent_traversal/snr_traversal_full.py
```

via the SLURM script:

```
galaxy_images/galaxy_model/latent_traversal/snr_traversal_paper.slurm
```

That job loads the trained model checkpoint, picks K=10 HSC neighbors per SNR percentile bucket per target, generates one reconstruction per bucket, and appends every (target, mode) record into a master `snr_traversal_arrays.h5` under the SLURM `--output-dir`. After it finishes, copy that file into `./_cache/`.

Default upstream paths:
- checkpoint: `outputs/neighbors_all_attn/checkpoints/best-epoch=228-step=87000.ckpt`
- HSC/Legacy stamps: `/work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5`
- per-target SNR / ivar / PSF metrics: `/work1/jeroenaudenaert/pablomer/data/hsc_noise_metrics.h5`, `/work1/jeroenaudenaert/pablomer/data/hsc_ivar_psf_stats.h5`

## Styling notes

- Panel highlights and inner/bottom labels follow `visualization_scripts/for_paper/replot_reconstruction.py` (green = Target, gray = Input, blue = Output).
- Default visualization is per-image 2-98 percentile (`indep`) — the same style as the original traversal `*_indep.png`. Pass `--vis rowscale` to use the target-HSC row-scale used for the rowscale variant of the upstream plots.
- Bucket labels show the **positive**-SNR percentile (e.g. `SNR p95` = clean), converted from the upstream neg-SNR percentile, so high label = high SNR — matching the arrow direction.
