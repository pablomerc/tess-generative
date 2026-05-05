# Paper figures

Each subdirectory below renders **one figure** for the paper. The structure is
uniform: a `make_figure.py` reads from a local `_cache/` (npz / images), and
writes both `.png` and `.pdf` in place. None require upstream HDF5s or model
checkpoints to *re-render* — only to *regenerate* the cache, which lives in
that subdir's own `build_cache.py` (where present).

## Index

| Subdirectory | What it shows | Key script | Cache | Status |
|---|---|---|---|---|
| `latent_umap_figure/` | Standalone 2-panel UMAP — Physics + Instrument latent of our base ckpt on 4096 paired galaxies, SciencePlots, no pair lines | (renders via `galaxy_images/galaxy_model/visualization_scripts/plot_umap_from_file.py` from saved npz) | (npz lives under the model's `visualization_scripts/neighbors_visualization/latent_space/`) | self-contained PDF only |
| `UMAP_anomalies_combined/` | **Combined**: 1×3 UMAP (Physics \| Instrument \| AION) on top + 1×3 anomaly-stamp grid (Physics \| Instrument \| AION) on bottom | `make_figure.py` | local `_cache/` (UMAP npz + 3 anomaly npz) | self-contained ✅ |
| `anomaly_detection_figure/` | 3-column comparison of top-12 deduped normalizing-flow HSC anomalies (Ours-Physics, AION-1, Ours-Instrument) on the 367k pool | `make_figure.py` | local `_cache/` | self-contained ✅ |
| `nearest_neighbors_figure/` | HSC↔Legacy nearest-neighbour qualitative comparison | `make_figure.py` | local `_cache/` | self-contained ✅ |
| `lens_neighbors_figure/` | For two galaxy-galaxy lens queries, the top physics-latent neighbours retrieved from a 103k gallery | `make_figure.py` | local `_cache/` | self-contained ✅ |
| `snr_traversal_figure/` | Reconstruction of an HSC galaxy as the conditioning instrument-latent SNR is traversed | `make_figure.py` | local `_cache/` | self-contained ✅ |
| `datadriven_noise_model_figure/` | Data-driven instrument-noise-model verification figure | `make_figure_scienceplots.py` | local `_cache/` | self-contained ✅ |
| `artifact_correction_random_row/` | Random-row artifact-correction visualization | `make_figure.py` | local tensor file | self-contained ✅ |

## How to reproduce a figure

```bash
cd <subdir>
python make_figure.py    # writes <subdir>/<figure>.{png,pdf}
```

Each subdir's `README.md` documents the column order, data sources, and any
styling knobs. PDFs are vector for paper inclusion; PNGs are raster previews
at dpi=150–200.

## Reproducing the cache (advanced)

`make_figure.py` is intentionally lightweight — it never touches the parent
project's training checkpoints or 100s of GB of raw data. To **rebuild** a
cache from scratch (e.g. to use a different ckpt or a larger pool), use the
sibling `build_cache.py` in the relevant subdir, or follow the upstream
pipeline notes in that subdir's `README.md`.

For `UMAP_anomalies_combined/` specifically, the upstream cache producers are:
- UMAP coords: `galaxy_images/galaxy_model/visualization_scripts/regenerate_umap_base.py`
  (Physics + Instrument) and
  `galaxy_images/galaxy_model/aion_benchmark/aion_umap/aion_umap_neighbors_efficient.py`
  (AION).
- Anomaly stamps: `figures_for_paper/anomaly_detection_figure/build_cache.py`.

## Conventions

- **SciencePlots**: most figures provide a paper-styling variant via
  `plt.style.context(["science", "no-latex"])`. Install `pip install
  scienceplots` to use it; otherwise the script falls back to plain
  matplotlib.
- **Color palette**: HSC = `#e8c4a0`, Legacy = `#8eb8e8`; pair markers cycle
  over `#70a845`, `#b460bd`, `#4aac8d`, `#c85979`, `#b49041` × shapes
  `["x", "s", "o", "^"]`.
- **No upstream paths**: all `make_figure.py` files read only from their own
  subdir; no absolute paths to scratch / HOME / cluster-specific locations.
