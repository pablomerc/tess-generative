# Counterfactual Frozen-Embedding FM Decoder

Flow-matching decoder that reconstructs a galaxy image from **frozen contrastive
embeddings of other images** — never from the target itself:

| Slot | Image | Frozen tower | Role |
|------|-------|--------------|------|
| Physics | `samegal` (cross-survey pair) | `encoder_galaxy` | 64-d |
| Instrument | ≤5 same-survey neighbors | `encoder_instrument` | K×64-d + mask |
| Target | anchor | **not encoded** | FM `x_1` only |

This mirrors the paper dual-encoder CF objective, but freezes a pretrained
contrastive dual-encoder and trains only the FM UNet.

Contrast with [`../embedding_fm_decoder/`](../embedding_fm_decoder/): that probe
embeds the **same** anchor and reconstructs it (autoencoding-style, not CF).

## Conditioning styles

**B — `tokens` + `xattn` (default, train first)**  
Sequence `[gal(samegal), inst(n1), …, inst(nK)]` → `(B, 1+K, 64)` cross-attn.
Pad neighbor slots are zeroed.

**A — `meanpool` + `adagn`**  
`concat(gal, masked_mean(inst))` → `(B, 128)` AdaGN. Same injection path as the
non-CF embedding probe; only the embedding *sources* are CF.

One precomputed H5 serves both.

## Pipeline

```bash
# 1) Precompute CF embeddings (all ~104k anchors)
sbatch precompute_cf_embeddings.slurm

# 2) Train B (tokens / xattn) — 6 h auto-resubmit
export WANDB_API_KEY=...
sbatch train_cf_fm_tokens.slurm

# 3) Matched recon MSE vs paper base + fm-control
sbatch eval_cf_recon_mse.slurm

# 4) Train A (meanpool / AdaGN), then re-run eval
sbatch train_cf_fm_meanpool.slurm
sbatch eval_cf_recon_mse.slurm
```

Long partition (no chaining): `sbatch train_cf_fm_long.slurm`  
(or `COND_STYLE=meanpool INJECTION=adagn sbatch train_cf_fm_long.slurm`).

### Manual train

```bash
python -m galaxy_images.galaxy_model.contrastive_baseline.counterfactual_fm_decoder.train_cf_fm \
    --data-h5 /orcd/pool/007/pablomer/outputs/counterfactual_fm/cf_embeddings_all.h5 \
    --cond-style tokens --injection xattn \
    --max-steps 100000 --val-ratio 0.05 --seed 42 --devices 1
```

## Defaults (matched to embedding_fm + paper split)

| Setting | Value |
|---------|-------|
| Frozen encoder | `/orcd/pool/007/pablomer/outputs/contrastive/best-epoch=599-step=240000.ckpt` |
| Data | `/orcd/pool/007/pablomer/efficient_neighs`, K=5 |
| H5 out | `/orcd/pool/007/pablomer/outputs/counterfactual_fm/cf_embeddings_all.h5` |
| val_ratio / seed | **0.05 / 42** (same as `data_factory` / paper / fm-control) |
| batch / lr / steps | 128 / 1e-4 / 100k |
| ODE steps | 250 |
| W&B project | `galaxy-counterfactual-fm-decoder` |

## Comparison protocol

[`eval_cf_recon_mse.py`](eval_cf_recon_mse.py) follows the diffusion-ablation
matched harness:

1. Rebuild val via `neighbors_fm_control.json` (ram48, 5%, seed 42).
2. First `--n` (default 256) anchors in loader order; shared `x_noise` per batch.
3. Evaluate:
   - **fm-control** — `checkpoints_new/.../fm-control-ram48-h200-75k/`
   - **fm-paper** — `galaxy_model/checkpoints/base/snapshot.ckpt`
   - **cf-fm-*** — CF Lightning ckpts; cond looked up in the CF H5 by
     `(catalog_idx, survey)` so stores can differ.
4. Full-frame 48×48 MSE, mean±SEM by survey → CSV + markdown.

Manifest: `results/recon_eval_manifest.json` (refuses silent set drift).

## Files

| File | Role |
|------|------|
| `precompute_cf_embeddings.py` | Stage 1 H5 |
| `cf_embedding_dataset.py` | tokens / meanpool loaders + split |
| `cf_fm_module.py` | Lightning FM module |
| `train_cf_fm.py` | Train CLI |
| `eval_cf_recon_mse.py` | Three-way (+ CF) recon MSE |
| `*.slurm` | Engaging launchers |
