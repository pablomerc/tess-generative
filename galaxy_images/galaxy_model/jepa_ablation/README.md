# jepa_ablation

> **See [`REPORT.md`](REPORT.md) for the single comprehensive write-up** (design, results, bugs, caveats, next steps). This file is one slice of it.


JEPA-style (predict-in-embedding-space, no negatives) objectives benchmarked
against the published contrastive dual-encoder arms. Design + locked decisions
live in **PLAN.md** — read that first. This file is the operational reference.

## The four arms

| arm (`--arm`) | objective | data | encoders | eval dim |
|---|---|---|---|---|
| `hsc-resnet` | masked-image JEPA: enc(masked) → predict EMA enc(clean) | 366,706 HSC stamps | 1× ResNet18 | 64 |
| `hsc-vit` | true I-JEPA, latent multi-block masking | same 367k HSC pool | ViT p4/d192/depth6 | 192 |
| `cf` | counterfactual: partner→anchor (physics), neighbor-mean→anchor (instrument) | 103,741 anchor triplets | 2× ResNet18 | 64 + 64 |
| `cross` | cross-survey, both directions (HSC↔Legacy of the same galaxy) | 103,741 anchor pairs | 1× ResNet18 | 64 |

All use BYOL cosine loss `2 − 2·cos(pred, stopgrad(target))` except `hsc-vit`,
which uses I-JEPA's token MSE. Batch 64, 100k steps, AdamW 1e-4/wd 1e-4,
**step-based cosine LR with 2k warmup** (deliberately NOT the repo's
`T_max=max_epochs` quirk — see CLAUDE.md "LR schedule" and PLAN.md decision 4),
EMA momentum cosine 0.996→1.0.

## Running

```bash
sbatch smoke_test.slurm            # 30-step micro-run of all 4 arms + every eval surface
bash   submit_all_jepa.sh          # the 4 real runs; each auto-chains its own eval
sbatch --export=ALL,ARM=cf --job-name=jepa-eval-cf eval_jepa.slurm   # re-run one eval
```

Training self-chains (`afterany` successor queued before training starts; a
`DONE` marker stops the chain), resumes from `last.ckpt`, and caps at
`MAX_ATTEMPTS=10`. On `DONE` it submits `eval_jepa.slurm`, which runs the n=5469
R² probe, the overlay plots, the paper-style UMAP, and then submits the anomaly
pipeline. Discord pings at every transition.

Checkpoints: `/orcd/pool/007/pablomer/checkpoints_new/jepa-<arm>/` — a **symlink**
into `/orcd/data/mki_aryeh/001/pablomer/jepa_ablation/<arm>/` (the 200 TB group
allocation; POOL sits at ~93%). wandb project `galaxy-jepa-engaging`.

## The integration trick (why the eval stack needed almost no changes)

Every module exposes its **EMA target** encoders as `encoder_galaxy` /
`encoder_instrument` properties; single-encoder arms alias one module under both
names. `prepare_combined.py:_is_contrastive`, `encode_latents.py`, and the UMAP
tooling all duck-type on exactly those attributes, so the whole downstream stack
works unchanged. The only edit outside this directory is the JEPA branch in
`contrastive_anomalies/encode_latents.py::load_contrastive`'s class sniff.

Consequence for single-encoder arms: `hsc_e1 == hsc_e2` in the embeddings H5, so
`predict_combined.py` probes the *same* space for both the physics and the
instrument target families. That is intended — for those arms the question is
what one space captures, not how two spaces divide.

## Watching for collapse

BYOL-family objectives can collapse to a constant embedding, and the loss looks
*good* when they do. Every arm logs, every 10 steps:

- `train/std_online`, `train/std_target` — mean per-dim std of the L2-normalized
  projections. Healthy 32-d ≈ **0.18** (`1/√32`); **trending toward 0 = collapse**.
- `train/align` (or `align_galaxy` / `align_h2l`) — mean cos(pred, target).
  Rising to ~1.0 *while* std falls is the collapse signature.
- `train/ema_tau` — sanity-check the momentum ramp.

If an arm collapses: first try predictor-only lr ×10, then `ema_base_momentum`
0.99. Both are hparams, no code change.

`check_collapse.py` measures this directly from a checkpoint, independent of the
training loss (`norm_std` relative to the isotropic `1/√D`, plus PCA effective
rank — a space can show non-zero std while living on one direction). `eval_jepa.slurm`
runs it before the probe and posts the verdict, because an R² number from a
collapsed encoder is not interpretable.

### Why eval uses the FINAL checkpoint, not `best-*.ckpt`

`monitor="val/loss"` still writes `best-*.ckpt`, but **the eval deliberately
ignores it** (this is where `eval_jepa.slurm` diverges from
`eval_contrastive_engaging.slurm`). A JEPA/BYOL prediction loss is not a
model-selection criterion: the target side is *learned*, so the loss can fall by
making the target trivially predictable (collapse), and it legitimately *rises* as
the representation gets richer and harder to predict. Observed here: the `hsc-vit`
arm's best val/loss was at **step 2,000 of 100,000** — selecting it would have
evaluated an essentially untrained model. Final EMA weights are the standard
choice for this family; the `best-*` files are diagnostics only.

## Files

| file | role |
|---|---|
| `datasets.py` | `HSC48PoolDataset` (367k), `AnchorPairs48Dataset` (anchors), `load_umap_pairs` |
| `jepa_base.py` | EMA machinery, cosine loss, schedules, diagnostics, wandb UMAP hook |
| `jepa_modules.py` | the three ResNet arms |
| `jepa_vit.py` | `PatchViT`, `ViTPredictor`, `IJEPAViTModule` |
| `train_jepa.py` | single entry for all arms |
| `train_jepa.slurm` / `eval_jepa.slurm` / `submit_all_jepa.sh` / `smoke_test.slurm` | orchestration |
| `regenerate_umap_jepa.py` | checkpoint → npz → `visualization_scripts/plot_umap_from_file` |

Reads the prebuilt in-RAM store `/orcd/pool/007/pablomer/efficient_neighs_48`
(float16, preprocessing baked in). **Loading must use `np.fromfile`** —
`np.ascontiguousarray`/`np.array` on an already-contiguous memmap is a no-op that
silently leaves the data on Ceph.

## Caveats to carry into the writeup

- The fixed LR schedule is a real deviation from the published contrastive arms.
  If a JEPA arm *beats* them, rerun `contrastive-spatial-conv1x1` with the fixed
  schedule as a matched control before claiming an objective-level win.
- `hsc-vit`'s architecture and 192-d width differ from every other arm — treat it
  as exploratory, not headline.
- `hsc-*` arms are trained on HSC only, so their Legacy-side R² measures transfer.
  Their UMAP Legacy points are pushed through an HSC-trained encoder (never
  trained on) — visualization only.
- No DWNVAL holdout, matching the published protocol (PLAN.md decision 3).
