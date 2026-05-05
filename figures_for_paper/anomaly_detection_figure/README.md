# Anomaly detection figure (paper)

3-column comparison of the top-12 deduplicated normalizing-flow anomalies from three encoders, evaluated on the **367k** HSC pool (`source_type ∈ {0, 1}` of `neighbours_v2.h5`, i.e. all rows that carry a valid HSC stamp = 366,706 galaxies).

| Column | Encoder | Score source (HDF5) | Score key |
|---|---|---|---|
| Ours (Physics) | dual-encoder flow matching, **physics** latent (encoder_1), mean-pooled | `anomaly_scores_ours_367k.h5` | `ours/hsc_mean/flow` |
| AION-1 | AION embeddings, PCA→64, mean-pooled | `anomaly_scores_aion_367k.h5` | `aion/hsc_mean_pca64/flow` |
| Ours (Instrument) | dual-encoder flow matching, **instrument** latent (encoder_2), flat | `anomaly_scores_ins_367k.h5` | `ours/hsc_flat/flow` |

Anomaly scores are negative log-likelihoods from a normalizing flow (zuko NSF, 50 epochs) fit on top of each model's frozen embedding space. The displayed thumbnails are the corresponding HSC stamps from `neighbours_v2.h5`.

## Sky-position deduplication

The `neighbours_v2.h5` catalog overlaps spatially (each anchor brings up to 5 neighbors, many of which live within a few arcseconds of each other), so naive top-N lists contain visually identical stamps. To avoid that, candidates are walked top-down by score and a candidate is kept only if its angular separation from every previously-kept entry exceeds `DEDUP_ARCSEC = 10″` — wider than the ~8″ HSC stamp size. The kept set is the top 12 score-ranked HSC stamps with sky positions at least 10″ apart.

Implementation: small-angle approximation
```
sep_arcsec ≈ sqrt((Δra · cos(dec))² + Δdec²) · 3600
```
Brute-force O(n_keep × n_candidates), so a candidate pool of `12 × 500 = 6000` is more than enough to fill 12 deduplicated slots.

## Files

```
anomaly_detection_figure/
├── README.md                                 # this file
├── build_cache.py                            # rebuild _cache from raw HDF5s
├── make_figure.py                            # render figure from _cache
├── _cache/                                   # bundled top-12 per source
│   ├── anomaly_scores_ours_367k__ours_hsc_mean_flow__top12.npz
│   ├── anomaly_scores_aion_367k__aion_hsc_mean_pca64_flow__top12.npz
│   └── anomaly_scores_ins_367k__ours_hsc_flat_flow__top12.npz
├── anomaly_detection_figure.{png,pdf}         # plain matplotlib
└── anomaly_detection_figure_science.{png,pdf} # scienceplots styling
```

Each `_cache/*.npz` holds, for one source:
- `top_raw`  — `int64[12]` raw row indices into `neighbours_v2.h5`
- `top_pcts` — `float[12]` per-model NLL percentile (rank/N_finite × 100)
- `hsc_imgs` — `float[12, 5, 160, 160]` HSC g/r/i/z/y stamps
- `label`, `score_key`, `scores_file`, `dedup_arcsec` — provenance

## How to reproduce

```bash
# 1. (re)build the cache from the raw HDF5s
python build_cache.py

# 2. render PNGs + PDFs
python make_figure.py
```

`build_cache.py` reads:
- `/work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5` — for `images_hsc`, `ra`, `dec`
- `…/anomaly_detection/outputs/anomaly_scores_{ours,aion,ins}_367k.h5` — produced by jobs `job_{ours,instrument,aion}_367k.sh` (encode → fit_and_score → visualize) on the same `mi2104x` partition.

Edit those constants at the top of `build_cache.py` if your data is in a different location. `make_figure.py` only needs `_cache/`, so it can be re-run on a different machine without the upstream HDF5s.

## Styling notes

- 4 rows × 3 columns per source group; 3 source groups side-by-side.
- Gray dashed vertical separator (`#888888`, `linestyle="--"`, `linewidth=1.2`, `dashes=(6, 4)`) between groups.
- Per-stamp rank label `#i` rendered inside the image at the bottom in a translucent white box (`boxstyle="square,pad=0.2"`, `facecolor="white"`, `alpha=0.7`, `linewidth=0`) — copied from the inner-label style in `figures_for_paper/datadriven_noise_model_figure/make_figure_scienceplots.py`.
- Tight top margin, tight inner spacing (`hspace=wspace=0.04`).
- Two output styles: plain matplotlib and `scienceplots(["science", "no-latex"])`.
