# Hierarchical-latent comparison experiments

Compares 4 ways of defining the **physics** latent on top of the trained
`HierarchicalGlobalInstrumentFlowMatchingModule` checkpoint. The instrument
latent is fixed: `model.encode_image(image)["instrument"]["flat"]` (16-d).

| Variant key | Physics latent | Dim (bn_36x16) |
|---|---|---|
| `spatial_flat` | `physics.spatial_flat` | 576 |
| `global_vec` | `physics.global_vec` | 64 |
| `spatial_pooled` | `mean(physics.spatial_concat, dim=1)` | 16 |
| `global_concat` | `concat(global_vec, spatial_flat)` | 640 |

Three downstream pipelines are run for each variant; each posts plots and short
status messages to the configured Discord webhook.

```
hierarchical_latent_experiments/
├── common/                    # shared utilities
├── umap_pairs/                # 2-panel UMAP of HSC/Legacy pairs
├── anomalies/                 # k-NN, IsolationForest, Mahalanobis, NSF on the latents
└── downstream/                # neighbors-task probe MLP (R²/MAE)
```

## Submitting

```
cd /work1/jeroenaudenaert/pablomer/tess-generative

sbatch galaxy_images/galaxy_model/hierarchical_latent_experiments/umap_pairs/job_umap_pairs.slurm
sbatch galaxy_images/galaxy_model/hierarchical_latent_experiments/anomalies/job_anomalies.slurm
sbatch galaxy_images/galaxy_model/hierarchical_latent_experiments/downstream/job_downstream.slurm
```

Each is a 4-task array (one per variant). Tunable env vars:

- `CKPT` — checkpoint path (default: step 201k of the trained run)
- `N` — number of HSC/Legacy pairs for UMAP (default 4096)
- `N_MAX` — anomaly latents cap (default 8000)
- `NSF_EPOCHS` — NSF training epochs for anomaly scoring (default 50)

To run only one variant, use `--array=1` (for `global_vec`), etc.

## Cluster limitation

The contrastive `prepare_all_contrastive.py` script supports four downstream
datasets (`mmu`, `legacy_provabgs`, `hsc_provabgs`, `neighbors`). On this
cluster only `neighbors` data is available; the other three depend on
`/data/vision/billf/scratch/pablomer/...` paths which are MIT-only. The
`downstream` job runs the `neighbors` task only and posts a notice when run.

## Smoke test (no SLURM)

```
cd /work1/jeroenaudenaert/pablomer/tess-generative
PYTHONPATH=. python -m galaxy_images.galaxy_model.hierarchical_latent_experiments.umap_pairs.run_umap_pairs \
    --variant global_vec --n 256 --no-discord \
    --out-dir galaxy_images/galaxy_model/hierarchical_latent_experiments/umap_pairs/outputs/_smoke
```
