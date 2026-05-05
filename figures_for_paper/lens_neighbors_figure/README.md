# Lens nearest-neighbour figure (paper)

Two known galaxy-galaxy lens systems are used as queries; for each, we display the top "Physics" nearest neighbours retrieved from the dual-encoder flow-matching model's physics latent space (cosine/Euclidean kNN over a pre-encoded gallery of ~103k galaxies covering both HSC and Legacy Survey).

| Lens | HSC Object ID | Default neighbour ranks shown (rank label = `rank − 1`) |
|---|---|---|
| 33 | 41619422804208581 | `[2, 3, 4, 7, 8, 9]`  → NN #1, #2, #3, #6, #7, #8 |
| 48 | 42692125246129290 | `[2, 4, 5, 6, 7, 9]`  → NN #1, #3, #4, #5, #6, #8 |

Rank 1 is intentionally skipped: in the gallery it is the query itself (the lens stamp is in the kNN index).

## Files in this folder

```
lens_neighbors_figure/
├── README.md
├── lens_final_figure.py             # standalone, cache-only renderer
├── _cache/
│   ├── lens_33.npz                  # query + top-12 NNs (image, survey, obj id)
│   └── lens_48.npz
├── lens_neighbors_final_figure.png  # default 2x7 figure (2 lenses × (1 query + 6 NNs))
└── lens_neighbors_final_figure.pdf
```

Each `_cache/lens_{user_idx}.npz` contains:
- `query_img`: `(3, 64, 64) float32` — center-cropped HSC stamp of the query lens (channels g/r/i for display)
- `nn_imgs`:   `(12, 3, 64, 64) float32` — top-12 neighbours, ordered by physics-latent distance
- `nn_survey`: list of 12 strings, each `"hsc"` or `"legacy"`
- `nn_raw_h5_row`: original row indices into `neighbours_v2.h5` (provenance only)
- `nn_obj_id`:  list of 12 catalog object IDs
- `obj_id`, `h5_row`, `user_idx`: query-side metadata

Total bundle is ~1.3 MB.

## Recreating the figure

### Default (matches the paper)
```bash
cd figures_for_paper/lens_neighbors_figure
python lens_final_figure.py
```

### More neighbours (uses cached top-12)
Pass per-lens rank lists explicitly:
```bash
python lens_final_figure.py --nn-ranks 33:2,3,4,5,6,7,8,9 48:2,3,4,5,6,7,8,9
```

Or use the convenience flag to take the first N neighbours (skipping rank 1):
```bash
python lens_final_figure.py --n-nn 8                  # ranks 2..9 for both default lenses
python lens_final_figure.py --n-nn 11                 # uses all bundled NNs (ranks 2..12)
python lens_final_figure.py --n-nn 6 --extra-lens 33  # just lens 33, 6 NNs
```

The bundle holds 12 NNs per lens, so you can show up to 11 (skipping rank 1) without regenerating anything. Going beyond that, or adding new query lenses, requires running the upstream pipeline (next section).

Only `numpy` and `matplotlib` are needed.

## Provenance / regenerating from scratch

The bundle was extracted from these upstream artifacts:
- Script:     `galaxy_images/galaxy_model/neighbor_search_lenses/lens_final_figure.py`
- NN cache:   `neighbor_search_lenses/outputs/lens_neighbors_final_figure_cache.h5`
- Embeddings: `galaxy_images/galaxy_model/neighbor_search/neighbor_latents_103k.h5`
- Checkpoint: `outputs/neighbors_all_attn/2026-04-05/checkpoints/best-epoch=228-step=87000.ckpt`
- Lens stamps: `galaxy_images/galaxy_model/lense_reconstruction/lens_reconstruction_dataset.h5`
- Image source: `/work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5` (`images_hsc`, `images_legacy`, `object_id_*`)

The upstream script:
1. Loads pre-computed gallery embeddings (HSC physics + Legacy physics, ~103k items).
2. Loads the model checkpoint, re-encodes the lens HSC stamps through `encoder_1` (the physics encoder), flattens.
3. Runs sklearn `NearestNeighbors` (Euclidean) over the combined HSC+Legacy gallery; keeps `K_SEARCH = 12` neighbours per lens after dropping the trivial self-match.
4. Caches `(combined_pos, survey, raw_h5_row)` per lens to its own HDF5 cache.
5. Pulls 64×64 raw center-crops from `neighbours_v2.h5` and renders the figure.

To regenerate this folder's bundle for **more lenses or a deeper top-K**:
1. In the upstream `lens_final_figure.py`, edit `LENSES` (add user_idx) and bump `K_SEARCH`.
2. Delete `outputs/lens_neighbors_final_figure_cache.h5` and re-run on a GPU node.
3. Re-run the bundling snippet that extracted these `.npz` files (the lookup is straightforward — for each cache entry, slice `images_hsc[h5_row][:3]` for the query, and `images_{survey}[raw_row][:3]` for each NN, then center-crop to 64).

## Visual style

- Per-row alternating gray shading; dashed vertical separator between query and NNs.
- Legacy-survey neighbours have a goldenrod label color (`#B8860B`); HSC-survey neighbours stay black.
- Per-channel percentile-stretched (0.5 / 99.5) on channels g/r/i; same as the upstream figure.
- The "NN #k" label uses `rank − 1` so the closest non-self neighbour is labelled #1.
