# Embedding-conditioned Flow-Matching Decoder

Trains a flow-matching decoder that **reconstructs a galaxy image from a single frozen
contrastive embedding**. It is a probe: how much image information do the contrastive
`encoder_galaxy` / `encoder_instrument` representations actually retain?

Pipeline: **(1)** encode anchor images once into an `[embedding, image]` dataset → **(2)** train
an FM decoder conditioned on `concat(galaxy, instrument)` (128-d), logging reconstructions to a
separate W&B project.

## Files
- `precompute_embeddings.py` — Stage 1. Runs anchors through the frozen contrastive checkpoint,
  writes `emb_galaxy (N,64)`, `emb_instrument (N,64)`, `targets (N,4,48,48)`, `survey`, `row_idx`.
  Reuses `NeighborsEfficientDataset._preprocess`, so images match training-time preprocessing.
- `embedding_dataset.py` — `EmbeddingImageDataset` (loads the H5 to RAM) + collate + split.
- `embedding_fm_module.py` — `EmbeddingConditionedFlowMatching` LightningModule.
- `train_embedding_fm.py` — Stage 2 entrypoint (argparse), resumable, separate W&B project.
- `precompute_embeddings.slurm`, `train_embedding_fm_6h.slurm`, `train_embedding_fm_long.slurm`.

## Conditioning design (why AdaGN, not tokens)
The conditioning is **one global 128-d vector per image**, not a token sequence. So we inject it
via the **class-embedding projection pathway as true AdaGN**:
`class_embed_type="projection"` + `class_labels=emb128` + **`resnet_time_scale_shift="scale_shift"`**
(multiplicative FiLM; the plain default is additive and weaker), with cross-attention removed
(`DownBlock2D`/`UpBlock2D`, `mid_block_type="UNetMidBlock2D"`, `encoder_hidden_states=None`).
This is the DiT/ADM/StyleGAN precedent for a global code and reuses the FM math/sampler/metrics
unchanged. Faking the vector as ~N cross-attn tokens (IP-Adapter style) is worse for fidelity.

`--injection` swaps the pathway with no other code change:
`adagn` (default) · `hybrid` (AdaGN + 2×256 cross-attn tokens; max-bandwidth reference) ·
`xattn` (tokens only) · `concat` (broadcast to input channels).
`--cond-mode` zeroes a half for content ablations: `concat128` · `galaxy64` · `instrument64`.

## Run
```bash
# Stage 1 (one-time)
sbatch precompute_embeddings.slurm    # or: python -m ...precompute_embeddings --limit 100000

# Stage 2 — long partition (pg_mki_aryeh / ou_mki_gpu), no resume chaining:
sbatch train_embedding_fm_long.slurm

# Stage 2 — 6 h partition (mit_normal_gpu), auto-resubmits until max-steps:
export WANDB_API_KEY=...   # provide via env; not committed
sbatch train_embedding_fm_6h.slurm
```
Edit the `REPO`, `TORCHENV_PY`, and data/output paths at the top of each `.slurm` for your
cluster. This repo's env is **CUDA** (torch 2.6+cu124); no ROCm/hipBLAS workaround is needed here.

## Resume
Periodic `ModelCheckpoint(save_last=True)` → `<run-dir>/checkpoints/last.ckpt` (model + optimizer
+ scheduler + step). Resume with `--resume-from <that path>`; the W&B run continues
(`resume="allow"`, stable id). The 6 h wrapper queues a `--dependency=afterany` successor before
training, so a timeout kill is followed by an automatic resume; a `DONE` marker + resubmit
counter stop the chain when `max-steps` is reached.

## W&B metrics (mirrors `double_train_fm_neighbors.py`)
`train/loss`, `val/loss` (+ per-survey `*_hsc`/`*_legacy`); reconstruction MSE
`val/mse`, `val/mse_32`, `val/mse_hsc`, `val/mse_legacy`; image panels `val/recon_grid` and
`val/recon_grid_row_scaled` (columns **[Target | Sample1..N | Mean]**); `val/embedding_umap`
(conditioning embeddings colored by survey). Reconstruction is stochastic conditional generation
— the samples are draws consistent with the embedding, and MSE(target, sample) measures fidelity.

## Data
Default source `/orcd/pool/007/pablomer/efficient_neighs` (468,197 rows / 103,741 anchors; same
corpus + preprocessing the contrastive model trained on). Anchors alternate HSC/Legacy by index,
so the first 100k are a balanced ~50/50 mix. For a large Stage-1 run, stage `efficient_neighs`
onto local NVMe scratch first (mmap random access is slow over Ceph).
