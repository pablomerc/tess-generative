# LSST AE baseline — handoff summary

> Orientation for a fresh session. Everything here lives on the AMD cluster
> (`/work1/jeroenaudenaert/pablomer/`), repo `tess-generative`, **branch `lsst`**.
> As of 2026-09-02 this module is **uncommitted** (untracked on the branch —
> commit in topical chunks, no co-author trailer, per Pablo's convention).

## What this is

A single-survey autoencoder baseline on the new **LSST DP2** image data, built
as the first model on the third survey view (HSC + Legacy + now LSST) for the
galaxy physics/instrument disentanglement work. Purposes: (1) sanity-check the
LSST data + preprocessing end to end with a real training run, (2) provide an
architecture-controlled comparison anchor for the contrastive/flow-matching
models, (3) drive the normalizing-flow anomaly pipeline on LSST galaxies.

Background on the data itself (fields, depth regimes, filters, future Rubin
releases): read `../data_exploration/DATASET_NOTES.md` — it's the primer, and
ends with the agreed research strategy this baseline slots into.

## Data path

- Store: `/work1/jeroenaudenaert/pablomer/data/lsst_efficient/` —
  `lsst_image.bin` float16 `(65163, 6, 160, 160)`, bands **g,r,i,z,u,y**
  (griz prefix), flux in **nanomaggies**, NaN planes for missing bands,
  row-aligned `lsst_meta.parquet` (per-band `psf_fwhm`, `n_input_images`,
  `has_*` flags), provenance in `meta.json`.
- `data.py: LsstGrizyDataset` — grizy subset (**64,918** objects; DEEP2-3 has
  no u), deterministic 90/10 split by `sha1(object_id_hsc)` →
  58,496 train / 6,422 val; per-run `val_object_ids.txt` freezes the split
  for later model comparisons. Rows read with seek+read (login-node safe).
- Preprocessing: `..preprocessing.preprocess_lsst` (crop 48 → clamp →
  arcsinh → FoV zoom ×0.84 to 8.0" HSC parity → scalar grizy norm
  (0.0322, 0.0696)). Constants are pinned in source and regression-tested
  against a committed fixture (`python -m galaxy_images.galaxy_model.lsst.preprocessing`).

## Model (`model.py`)

- Encoder: the contrastive baseline's `ResNetEncoder` imported read-only
  (`in_channels=5`, `embedding_dim=128`, avg pool) — 11.2M params,
  architecture-matched to the dual-encoder work on purpose.
- Decoder: `ResUpDecoder` — Linear(128→512·3·3) then 4× [nearest-upsample ×2 +
  2 residual GroupNorm/SiLU conv blocks], 512→256→128→64→32 → 5 channels,
  4.7M params. **No skip connections** (bottleneck must carry everything);
  upsample+conv avoids transposed-conv checkerboard.
- MSE loss, AdamW 3e-4, bf16-mixed. Per-field val metrics
  (`val/mse_cosmos`, `val/mse_deep23`), recon grids and a 3-panel latent UMAP
  (colored by field / n_input_i / psf_fwhm_i) logged every validation.

## First run — results (job 396758, 2026-09-01)

Run dir: `/work1/jeroenaudenaert/pablomer/outputs/lsst_ae_baseline/lsst-ae-grizy-z128-2026-09-01_1628/`
WandB: `tess-ml/lsst-disentanglement`, run `24pmefkr`.

- 40k steps, batch 256, **~45 min** on one MI210 (`mi2101x`) — far faster than
  estimated; **best ckpt at step 14k** (val rose afterward → use ~15–20k steps
  in future runs).
- `val/mse` 0.2254 — **COSMOS 0.1139 vs DEEP2-3 0.3410**: the 3× depth gap
  (median 89 vs 7 i-band exposures) measured in model space. This is the
  reference number for any depth-robustness claim.
- Anomaly stage (auto-runs after training; `anomaly.py`): encodes all 64,918 →
  `latents.h5`, `scores.parquet`, three rankings + top-16 gallery PNG each:
  - **NSF** (zuko, `run_flow_only.py` recipe, fit on train latents),
  - **recon MSE** (global + per-field — global tail is 89% COSMOS:
    bright/structured galaxies, NOT noisy DEEP2-3),
  - **IsolationForest** (train latents).
  Key structure: NSF∩IForest top-100 overlap = 84 (latent methods agree);
  recon overlaps **zero** with either — pixel-weird and latent-weird are
  different populations.
  - Candidate worth eyeballing: COSMOS `44218655407344105` — reconstructs
    with a strong green (= r-band) excess; possible emission-line object.
  - Cosmetic: mild dark-edge artifacts in reconstructions (decoder boundary).

## How to run

```bash
cd /work1/jeroenaudenaert/pablomer/tess-generative
sbatch galaxy_images/galaxy_model/lsst/single_latent/ae_baseline/run_ae.sbatch            # full
# args pass through: sbatch ... run_ae.sbatch --steps 15000 --latent 64
# local smoke (no wandb/discord): PYTHONPATH=. python -m ...ae_baseline.train --smoke
```

- Partition `mi2101x` (user is NOT authorized for `mi3501x`; never pass
  `--mem` — the submission filter rejects it). hipBLASLt workaround is baked
  into sbatch + train.py.
- WandB auth via `~/.netrc` (already logged in, entity `tess-ml`).
- Discord updates (start / every 5th val / final anomaly report with
  galleries) via webhook in `~/.config/discord_webhook` (chmod 600 — never
  commit it) or `$DISCORD_WEBHOOK_URL`.
- Outputs go to a fresh dated dir under `outputs/lsst_ae_baseline/`
  (no-overwrite policy).

## Files

| file | role |
|---|---|
| `data.py` | grizy Dataset + id-hash split |
| `model.py` | LsstAE Lightning module (encoder/decoder, val figures) |
| `train.py` | entry point; runs anomaly stage after fit |
| `anomaly.py` | encode-all + NSF / recon / IsolationForest + galleries + Discord |
| `notify.py` | webhook helper (never commits the URL) |
| `run_ae.sbatch` | SLURM wrapper (mi2101x) |

## Follow-up analysis (2026-09-02, in the run dir)

- `anomalies_*_fullstamp160.png` — top-16 per method on **raw 160×160**
  stamps (per-stamp asinh). Finding: NSF/IForest top-3 are near-empty
  stamps with `n_input_i=1` (coverage failures + hot pixels), and the
  recon-MSE tops are almost all shallow-coverage stamps (`n_input_i` 2–11,
  even in COSMOS) — consider an `n_input` floor cut before science claims.
- `umap_by_field_ninp_psf.png` + `umap_all_latents.npz` — joint UMAP on all
  64,918 latents, per-field panels colored by `n_input_i` / `psf_fwhm_i`.
  The detached arm = low-`n_input` COSMOS cluster (same objects as above).
- WandB curves confirm overfitting past ~14k steps: val/mse 0.1932 →
  0.2254 at 40k while train keeps falling; 40k steps ≈ **175 epochs** with
  no augmentation, and DEEP2-3 degrades ~2× more than COSMOS (noise
  memorization). Add flips/rot90 augmentation next run.

## Experiment matrix (2026-09-02, in progress)

Decided with Pablo: per-field runs use the **run-1 data recipe unchanged**
(no augmentation, no coverage cut — comparability over cleanliness).
Sibling modules under `lsst/single_latent/`: `fm_ae/` (joint flow-matching AE, main
neighbors-FM architecture: token ResNet encoder → all-CrossAttn UNet,
token_dim 32 × 4 tokens = 128-d flattened latent) and `jepa/` (I-JEPA,
small ViT patch-4, EMA target encoder, mean-pooled 384-d latent).
`anomaly.py` is generalized: NSF+IForest on latents for every model
(fit on own-field train latents; `train_field` arg), recon-MSE only for
models whose forward reconstructs, `gallery_recon_fn` for FM ODE galleries.
Early stopping everywhere. Launcher: `lsst/single_latent/run_matrix.sbatch <module> [args]`.
Runs: AE×{cosmos,deep23}, FM-AE×{cosmos,deep23}, JEPA×{all,cosmos,deep23};
mixed-AE anchors = run 1 + the 2026-09-02_1552 early-stop rerun
(job 401401: stopped ~20k, best again step 14k, 27 min).

## Open threads (good next steps for a fresh session)

1. **Per-band normalization rerun** (`norm_key="per_band"`, constants already
   in `..preprocessing.LSST_NORM_PER_BAND`) — does the DEEP2-3 gap shrink and
   does field-separability in the latent UMAP weaken?
2. **Depth conditioning**: feed `n_input_*`/`psf_fwhm_*` (in `lsst_meta`) to
   the model — the "depth regimes, not separate instruments" strategy.
3. **Downstream probes on `latents.h5`** using the repo's MLP-R² protocol —
   AE-vs-contrastive embedding comparison (dim-matched at 128 to the
   embedding-FM work; contrastive default is 256, mind the mismatch).
4. **Shorter schedule** (~15–20k steps) given the 14k best.
5. **Commit the module** (plus `../data_exploration/`) on branch `lsst`.
