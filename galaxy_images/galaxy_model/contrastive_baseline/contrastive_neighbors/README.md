# Contrastive Neighbor Search

Equivalent neighbor-search workflow for the contrastive baseline.

## Files

- `make_latents_all.py`
  - Extracts latent embeddings for neighbors dataset from a contrastive checkpoint.
  - Saves an H5 with:
    - `idx`, `index_mmu`
    - `physics_embedding`, `instrument_embedding`
    - `legacy_physics_embedding`, `legacy_instrument_embedding`
- `search_neighbors.py`
  - Loads the latent H5, runs kNN in combined HSC+Legacy latent spaces, and plots query neighbors.
  - Supports single index mode and batch mode (`--batch-start/--batch-end`) with CSV summaries.
- `run_make_latents.sh`
  - Easy launcher for latent extraction.

## Quick run

1. Edit checkpoint/suffix in `run_make_latents.sh`.
2. Run:

```bash
cd /data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy_images/galaxy_model
./contrastive_baseline/contrastive_neighbors/run_make_latents.sh
```

3. Visualize one query:

```bash
python contrastive_baseline/contrastive_neighbors/search_neighbors.py \
  --latents contrastive_baseline/contrastive_neighbors/contrastive_neighbor_latents_<suffix>.h5 \
  --index 10
```

4. Batch mode (writes `query_results/neighbors_summary.csv` and `query_*.png`):

```bash
python contrastive_baseline/contrastive_neighbors/search_neighbors.py \
  --latents contrastive_baseline/contrastive_neighbors/contrastive_neighbor_latents_<suffix>.h5 \
  --batch-start 0 --batch-end 199 --out contrastive_baseline/contrastive_neighbors/query_results
```
