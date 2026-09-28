# single_latent — early single-survey exploration on LSST DP2

Everything in here trains ONE latent space on LSST stamps alone (no HSC/Legacy
pairing, no physics/instrument split). It predates the HSC × LSST dual-encoder
work and is kept as the architecture-controlled reference for it.

| module | model | latent |
|---|---|---|
| `ae_baseline/` | ResNet encoder + residual-upsample decoder, MSE | 128-d |
| `fm_ae/` | joint flow-matching autoencoder (token encoder → cross-attn UNet) | 4 × 32 = 128-d |
| `jepa/` | I-JEPA (ViT patch-4, EMA target encoder) | 384-d mean-pooled |

All three share `ae_baseline/{data,anomaly,notify}.py` (grizy dataset + id-hash
split, NSF/IForest/recon anomaly stage, Discord updates). `run_matrix.sbatch`
launches any of them on `mi2101x`:

    sbatch galaxy_images/galaxy_model/lsst/single_latent/run_matrix.sbatch \
        galaxy_images.galaxy_model.lsst.single_latent.fm_ae.train --field cosmos

Results and the 7-run AE / FM-AE / JEPA × field matrix: `ae_baseline/README.md`.
Outputs live outside the repo under `/work1/jeroenaudenaert/pablomer/outputs/lsst_*`.
The dual-encoder line of work is `../hsc_lsst_dataset.py` + `../DECISIONS.md`.
