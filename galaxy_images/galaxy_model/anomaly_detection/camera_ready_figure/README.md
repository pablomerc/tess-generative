# Camera-ready anomaly figure: "normal-looking galaxies" row

Goal: add a 5th row of *typical* galaxies to the paper's Fig. 3
(`tess-gen-disentanglement_neurips2026_paper/figures/new_figures/anomaly_detection_figure_science.pdf`).

## Camera-ready candidates (5-row Fig. 3)

| File | Row 5 |
|---|---|
| `anomaly_detection_figure_science_row5_band.{pdf,png}` | Per column: 3 galaxies drawn at random from that model's 40-60th score percentile, labelled with the percentile |
| `anomaly_detection_figure_science_row5_random.{pdf,png}` | One full-width row of 9 galaxies drawn at random from the 367k pool; inset = percentile under Ours-P / AION-1 / Ours-I |
| `row5_{band,random}.csv` | The row-5 galaxies: `row_474k`, `object_id`, ra/dec, percentile under all three models |

Rows 1-4 are the **published stamps, pixel-identical**, at the published page coordinates.
`extract_published_stamps.py` pulls them out of the paper PDF into `published_top12_stamps.npz`.
Draws use `SEED = 0`, first draw, not re-rolled.

```bash
~/venvs/pptx/bin/python extract_published_stamps.py                      # once (needs PyMuPDF)
~/miniforge3/envs/torchenv/bin/python make_camera_ready.py --variant band
~/miniforge3/envs/torchenv/bin/python make_camera_ready.py --variant random
```

Gotcha: scienceplots sets `savefig.bbox: tight`, which would shift the published geometry and re-rasterize the
stamps. The script overrides it to `standard` and embeds stamps with `interpolation="none"`.

Caption text:
- band: "Bottom row: for reference, three galaxies drawn uniformly at random from the 40-60th percentile of each model's anomaly score."
- random: "Bottom row: nine galaxies drawn uniformly at random from the pool; insets give each galaxy's anomaly-score percentile under each model."

## Exploration figure

| File | What |
|---|---|
| `make_top_median_bottom.py` | Exploration figure: top-12 / 9 at the median score / 9 at the lowest score, per space |
| `top_median_bottom_367k.{png,pdf}` | Output on the paper's 367k pool |
| `top_median_bottom_367k.csv` | Every displayed stamp: `row_474k`, `object_id`, ra/dec, flow NLL, percentile |

```bash
~/miniforge3/envs/torchenv/bin/python make_top_median_bottom.py            # 367k pool, ~10 s on a login node
~/miniforge3/envs/torchenv/bin/python make_top_median_bottom.py --pool 474k
```

No retraining or re-encoding. It reuses the flow scores of the 474k HSC-downstream rerun
(`../outputs/anomaly_scores_{ours,aion,ins}_hscdown_474k.h5`, made by `../engaging/`),
restricted to the 367k paper pool via `/orcd/pool/007/pablomer/outputs/contrastive_anomalies/pool367k_rows.npy`.

## How this differs from the published figure

The published figure's scores (`anomaly_scores_*_367k.h5`) and `_cache/` were made on AMD and are not on Engaging.
The rerun differs in two places:

- **Physics**: `ours/hsc_flat` (64-d) here vs `ours/hsc_mean` (16-d) in the paper. Top-12 is mostly the same galaxies, in a different order.
- **AION**: `aion/hsc_mean_full` (768-d, no PCA) here vs PCA->64 in the paper. The top-12 differs noticeably (fewer bleed trails).
- **Instrument**: same key. Top-12 is mostly the same objects.

## Decision for Pablo

Which set do we use for the 5th row: **median (~50th pct.)** or **lowest score**? Both look like ordinary
small, faint HSC galaxies. Recommendation: **median**. In a 64-768-d flow, the highest-density points sit at the mode,
and the mode is not a typical sample (typical-set argument). The median is the honest "what a normal galaxy looks like" row.
Hand-pick 3 per column from the CSV. Skip `Ours (Physics)` median #2, which is mostly empty with a bright galaxy at the edge.
