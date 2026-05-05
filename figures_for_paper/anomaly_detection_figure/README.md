# Anomaly detection figure (paper)

3-column comparison of the top anomalies from three models:

| Column | Encoder | Score source (HDF5) | Score key |
|---|---|---|---|
| Ours (Physics) | dual-encoder flow matching, physics latent, mean-pooled | `anomaly_scores_ours_100k.h5` | `ours/hsc_mean/flow` |
| AION-1 | AION embeddings, PCA-64, mean-pooled | `anomaly_scores_aion_100k.h5` | `aion/hsc_mean_pca64/flow` |
| Ours (Instrument) | dual-encoder flow matching, instrument latent, flat | `anomaly_scores_ins_100k.h5` | `ours/hsc_flat/flow` |

Anomaly scores are negative log-likelihoods from a normalizing flow fit on top of each model's frozen embedding space (100k galaxies). Top-N is taken in descending NLL order; the displayed thumbnails are the corresponding HSC images from `neighbours_v2.h5`.

## Files in this folder

```
anomaly_detection_figure/
├── README.md                  # this file
├── paper_anomaly_figure.py    # standalone, cache-only renderer (no h5py needed)
├── _cache/                    # bundled top-24 per source (top_raw, top_pcts, hsc_imgs)
│   ├── anomaly_scores_ours_100k__ours_hsc_mean_flow__top24.npz
│   ├── anomaly_scores_aion_100k__aion_hsc_mean_pca64_flow__top24.npz
│   └── anomaly_scores_ins_100k__ours_hsc_flat_flow__top24.npz
├── paper_anomaly_8.{png,pdf}    # 2 rows x 4 cols per group
├── paper_anomaly_9.{png,pdf}    # 3 rows x 3 cols per group  (the "3x3" version)
└── paper_anomaly_12.{png,pdf}   # 3 rows x 4 cols per group
```

Each `_cache/*.npz` holds, for one source:
- `top_raw`  — `int` array of length 24, raw row indices into `neighbours_v2.h5`
- `top_pcts` — `float` array of length 24, NLL percentile among all finite scores in that score file
- `hsc_imgs` — `(24, 5, 160, 160) float32` HSC stamps, ordered by descending NLL

The cache only stores 24 examples (well over what the figure needs), so it's tiny (~12 MB / file) but still lets you experiment with any layout up to top-24 without touching the full datasets.

## Recreating the figure

### From the bundle (no external data required)
```bash
cd figures_for_paper/anomaly_detection_figure
python paper_anomaly_figure.py            # produces 8, 9 (3x3), and 12
python paper_anomaly_figure.py --top-n 9  # just the 3x3
python paper_anomaly_figure.py --top-n 16 --n-cols 4   # any layout up to top-24
```

Only `numpy` and `matplotlib` are needed. PNG and PDF are written side by side.

### From scratch (regenerating the cache)
The cache was built by the upstream script:

```
galaxy_images/galaxy_model/anomaly_detection/paper_anomaly_figure.py
```

That script reads three big files on the cluster:
- `outputs/anomaly_scores_{ours,aion,ins}_100k.h5` — per-model NLL scores keyed by `raw_index`
- `/work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5` — the HSC image array

For each (source, top_n) it computes the top-N indices + percentiles, pulls the matching HSC stamps, and writes a `.npz` to `outputs/figures_compare/_cache/`. To regenerate the bundled cache here for a larger top_n, run that upstream script (it caches automatically) and copy the `top{N}.npz` files into `./_cache/`.

The two scripts are kept in sync; the only differences are:
- Upstream uses `h5py` to read scores + neighbours and writes to its own cache dir.
- The bundled copy reads only from `_cache/` and refuses `top_n > 24` with a clear error.

## Layout / styling notes

- Background tints per column are lightened versions of the palette used in `downstream_eval/final/makeplot_v2.py` (Physics blue / AION green / Instrument red).
- Each thumbnail is per-channel percentile-stretched (1st / 99th) on the first 3 channels for display only — scores were computed on the raw embeddings, not these RGBs.
- Tile labels `#1, #2, ...` are the rank by NLL within that model. They are NOT comparable across columns.

## Provenance

- Models / scoring: `galaxy_images/galaxy_model/anomaly_detection/{encode_latents_*,fit_and_score}.py`
- Score files used: the `*_100k.h5` variants (100k galaxies each); `*_10k.h5` versions also exist for sanity checks.
- Image source: `/work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5`, dataset `images_hsc`.
