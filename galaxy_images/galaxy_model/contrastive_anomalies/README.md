# Anomaly detection in the contrastive physics / instrument latent spaces

Does the contrastive dual-encoder's **physics** tower surface rare *morphology*
while its **instrument** tower surfaces rare *observing conditions*? This runs the
standard `anomaly_detection/` experiment separately on each tower, over the
103,741-anchor corpus, and scores both against 74 catalogued strong lenses.

Engaging-only (NVIDIA). Nothing here writes outside
`/orcd/pool/007/pablomer/outputs/contrastive_anomalies/`.

```bash
mkdir -p logs && sbatch run_pipeline.slurm        # ~1 h on one H200
```

---

## What gets encoded

| | |
|---|---|
| Corpus | the **103,741 `source_type==0` anchors** of `/orcd/pool/007/pablomer/efficient_neighs_48` (the ram48 store) |
| Checkpoint | `checkpoints_new/contrastive-spatial-conv1x1/best-epoch=53-step=83000.ckpt` |
| Pooling | `conv1x1` — 48×48 → 2×2 ResNet18 map, `token_dim = 64/4 = 16`, so both towers emit **64-d** |
| Lens truth | 74 lenses from `/orcd/pool/007/pablomer/lens_indices.txt`, joined on `hsc_object_id`; all 74 are anchors |

Five representations, all **self**-encodings of the anchor's own image:

| key | definition |
|---|---|
| `physics_hsc` | `encoder_galaxy(hsc48[row])` |
| `physics_legacy` | `encoder_galaxy(legacy48[row])` |
| `physics_mean` | mean of the two |
| `instrument_hsc` | `encoder_instrument(hsc48[row])` |
| `instrument_legacy` | `encoder_instrument(legacy48[row])` |

Raw encoder outputs, never the projection heads — same convention as
`downstream_evaluation/engaging/prepare_combined.py`, so these are the vectors the
R² probes describe.

> **Why not reuse `outputs/counterfactual_fm/cf_embeddings_conv1x1.h5`?** It holds
> conv1x1 embeddings of this exact corpus from the same checkpoint, but they are
> *counterfactual*: `emb_galaxy` is the **cross-survey partner** image and
> `emb_instrument` is **five sky-neighbours**, never the anchor. Scoring those
> would answer "is this galaxy's neighbourhood unusual", not "is this galaxy
> unusual". Hence the fresh pass — which is cheap (~1 min on an H200).

Each representation is one point per galaxy, so no space mixes HSC and Legacy
images; the instrument spaces are not secretly bimodal on survey identity.

## Protocol

Detectors are fit on an 80% train split (seed 42) and score all anchors:
`knn` (k=5, mean squared L2), `iforest` (200 trees), `mahal`, `flow`
(zuko NSF, 6 transforms, `[64,64]`, 50 epochs). Same four as
`anomaly_detection/fit_and_score.py`, with two deliberate deviations — both
recorded in the output attrs, both revertible by flag:

1. **kNN excludes self-matches** (`--knn-allow-self` restores the old behaviour).
   The AMD script added the train split to a FAISS index and queried *all* points
   against it, so every training point matched itself at distance 0 and scored as
   less anomalous — biasing the top-anomaly list ~5× towards the held-out 20%.
2. **Per-dimension standardisation** on train-split statistics
   (`--no-standardize` restores the old behaviour). Raw conv1x1 physics latents
   have per-dimension σ spanning ~0.7–3.4, which silently weights the wide
   dimensions in kNN and hurts NSF conditioning. Mahalanobis and Isolation Forest
   are ≈invariant either way.

FAISS is not in `torchenv` on Engaging, so kNN is a chunked GPU `torch.cdist` —
same quantity (squared L2), faster here.

## Outputs

Everything lands in `/orcd/pool/007/pablomer/outputs/contrastive_anomalies/`:

| file | contents |
|---|---|
| `latents_<tag>.h5` | the five representations + `row_idx`, `ra`, `dec`, `is_lens`, object ids |
| `scores_<tag>.h5` | `<representation>/<method>` score arrays + `in_train` |
| `lens_metrics_<tag>.csv` | ROC-AUC, AP, recall@{100,500,1k,5k}, lift, median lens percentile |
| `space_agreement_<tag>.csv` | Spearman ρ and top-1000 Jaccard for every pair of score arrays |
| `summary_<tag>.md` | the two tables above, rendered; posted to Discord by the job |
| `figures_<tag>/joint_top12_<method>_<tag>.png` | physics vs instrument, side by side |
| `figures_<tag>/top12_<rep>_<method>_<tag>.png` | one grid per representation |
| `figures_<tag>/roc_lens_<tag>.png` | lens-retrieval ROC, one panel per method |

Stamps default to the ram48 48×48 images the encoder scored (RGB = g/r/i). Known
lenses get a green border; captions carry the object's pre-dedup rank in the corpus
(percentiles are useless at top-12-of-103,741 — they all round to 0.00%). Grids are
RA/Dec-deduped at 10″ so one bright neighbourhood cannot fill the panel.

**Paper-figure view.** `--stamp-store full160` re-renders the same grids from the raw
160×160 `efficient_neighs` cutouts, writing `*_160.png` alongside. Worth doing: satellite
trails, chip-edge discontinuities and scattered-light gradients are the instrument tower's
top anomalies and are largely invisible in the 48×48 crop.

```bash
python plot_anomalies.py --scores <scores_*.h5> --stamp-store full160 \
    --out-dir <out>/figures_<tag>        # add --stamp-crop 96 to zoom in
```

A *cold* random row on that 120 GB store costs **~4.4 s** — the same Ceph latency cliff that
killed the 160×160 precompute pass. So the renderer resolves every grid's top-N first, dedupes
(240 slots → 134 distinct stamps) and prefetches with 16 threads (`--stamp-workers`). The reads
are latency-bound, not bandwidth-bound, so that turns ~25 min of serial waiting into ~2 min.

## Variants

```bash
# the random-neighbour arm of the 2x2 sweep
CKPT=/orcd/pool/007/pablomer/checkpoints_new/contrastive-random-conv1x1/best-epoch=59-step=91000.ckpt \
TAG=random-conv1x1 sbatch run_pipeline.slurm

# the avg-pool arm (dimension-matched, so it just works)
CKPT=/orcd/pool/007/pablomer/checkpoints_new/contrastive-spatial-avg/best-epoch=*.ckpt \
TAG=spatial-avg sbatch run_pipeline.slurm

# reproduce the legacy AMD scoring protocol
FIT_EXTRA="--knn-allow-self --no-standardize" TAG=spatial-conv1x1-legacyproto sbatch run_pipeline.slurm
```

Stage 1 is skipped if `latents_<tag>.h5` already exists, so re-running to change
only the scoring protocol costs nothing. Use a distinct `TAG` per encoder — all
arms emit 64-d, so a mismatched pair fails silently rather than loudly.

## Reading the results

- **Lens metrics are indicative, not a benchmark.** The encoder was trained on this
  corpus with no lens holdout, and 74 positives is small: one lens crossing the
  top-100 boundary moves recall@100 by 1.4 points. AUC and AP are the stable
  numbers.
- Anomaly detectors surface *artifacts* as readily as astrophysics. The instrument
  column is expected to be full of noise, striping and saturation — that is the
  tower working, not failing.
- The number that answers the actual question is the **top-1000 Jaccard between
  `physics_hsc/<m>` and `instrument_hsc/<m>`** in `space_agreement_<tag>.csv`. If
  the towers were redundant it would be near 1.

## Files

| file | role |
|---|---|
| `anchor_data.py` | anchor catalog + lens truth + the sequential ram48 reader |
| `encode_latents.py` | stage 1 — the five representations |
| `fit_and_score.py` | stage 2 — four detectors × five spaces |
| `evaluate.py` | stage 3 — lens metrics + space-agreement tables + `summary_<tag>.md` |
| `plot_anomalies.py` | stage 4 — stamp grids, joint panel, ROC |
| `run_pipeline.slurm` | all four stages in one job, with Discord pings |
