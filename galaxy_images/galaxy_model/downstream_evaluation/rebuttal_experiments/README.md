# Rebuttal experiment — instrument properties from concat(physics, instrument)

Answers Reviewer JcH2 (Review 2, W1/Q1). The drafted response
(`reviews/review_2_response.md:56-63`) shows the contrastive baseline **beating** our model at
predicting instrument properties from the instrument latent:

| property | Ours (instr 128-D) | Contrastive | Ours − Contr (paired) |
|---|---|---|---|
| Galaxy Depth | 0.751 ± 0.011 | 0.852 ± 0.016 | −0.100 ± 0.002 |
| PSF Depth | 0.742 ± 0.013 | 0.846 ± 0.020 | −0.104 ± 0.003 |
| # Observations | 0.535 ± 0.014 | 0.699 ± 0.009 | −0.164 ± 0.007 |
| E(B−V) | 0.591 ± 0.025 | 0.770 ± 0.017 | −0.179 ± 0.009 |
| PSF FWHM (HSC) | 0.500 ± 0.017 | 0.684 ± 0.018 | −0.184 ± 0.006 |

**The question:** is that gap lost information, or just *factorization* — our model splitting
instrument information across two latents while the contrastive model concentrates it in one?
Probing **concat(physics, instrument)** measures the total instrument information the paper's
base model carries. Useful anchor, already on disk at the same folds: the physics latent *alone*
gives base 0.170/0.166/0.059/0.176/0.070 vs contrastive ≈0 — so our physics latent does retain
instrument signal the contrastive one does not.

## Two corrections to the original framing

1. **The published "Ours" column is already flattened tokens**, not pooled. `combined_e2` =
   `hsc_e2` (4 tokens × 16 = 64-D, flattened) ∥ `legacy_e2` (64-D) = 128-D
   (`../engaging/prepare_combined.py:187-188,347`). The *pooled* 32-D variant exists separately
   (`base-meanpool`: 0.688/0.671/0.482/0.541/0.420) and is what the paper **figure**
   `phys_flat_inst_pooled_all.png` uses — the likely source of the pooled impression. Both are
   carried as supplementary columns here.
2. The instrument grid is **2×2 (4 tokens)**, not 4×4 — `(B, 4, 16)`, where the 4 is `H*W`. No
   trained variant has a 4×4=16-token grid (`base3x3` → 9, `base6x6` → 36).

## Where the 256 comes from — 64 × 4

```
combined_e1e2 = [ hsc_e1(64) | legacy_e1(64) | hsc_e2(64) | legacy_e2(64) ]  = 256
                  physics      physics         instrument   instrument
```

64 per image per latent × 2 surveys × 2 latents. Each 64 = 4 spatial tokens × 16 channels (the
2×2 grid of a 48×48 input, flattened). The published "Ours" column is the right-hand half only.

## Design: post-hoc feature transforms, unmodified probe driver

The cached embeddings (`../engaging/outputs/embeddings/{base,contrastive-spatial-conv1x1}.h5`)
already hold all four per-survey latents at `(5469, 64)` plus 187 label columns, so **no
re-encoding, no GPU, no image reads**. `build_derived_h5.py` writes new H5s whose `combined_e2`
holds the derived feature vector, and `../engaging/predict_combined.py` runs on them
**unmodified**. Nothing is written into the published `../engaging/outputs/predictions/folds/`.

Frozen checkpoints, no training:
- **Ours** — `/orcd/pool/007/pablomer/checkpoints_new/base/snapshot.ckpt` (the paper base:
  `neighbors_all_attn/2026-04-05`, `best-epoch=228-step=87000`)
- **Contrastive** — `checkpoints_new/contrastive-spatial-conv1x1/best-epoch=53-step=83000.ckpt`

### The 10 derived variants (5 recipes × 2 checkpoints)

| variant | `combined_e2` := | dim | rows used |
|---|---|---|---|
| `{base,contr}-e1e2-256` | `[hsc_e1 ∥ legacy_e1 ∥ hsc_e2 ∥ legacy_e2]` | 256 | both instrument families |
| `{base,contr}-e1e2m-hsc` | `[hsc_e1 ∥ hsc_e2]` | 128 | `instrument_hsc` only |
| `{base,contr}-e1e2m-leg` | `[legacy_e1 ∥ legacy_e2]` | 128 | `instrument_legacy` only |
| `{base,contr}-e2m-hsc` | `hsc_e2` | 64 | `instrument_hsc` only |
| `{base,contr}-e2m-leg` | `legacy_e2` | 64 | `instrument_legacy` only |

The `-hsc`/`-leg` pairs exist because `combined_e2` is a single dataset that
`predict_combined.py` feeds to *both* instrument families; a survey-matched probe therefore needs
one derived file per survey, and the mismatched family's rows are dropped at aggregation.
`e2m_*` is the genuinely-new survey-matched flat-token probe (the published column mixes both
surveys' instrument latents, so it could infer Legacy depth partly from the HSC latent).

### Comparability contract

Identical to the published run: `--mlp-arch aion --seed 0 --n-folds 5 --fold F --device cpu`,
`max_epochs=50`. `seed=0` fixes fold membership (`predict_combined.py:302-312`), so all 5469 rows
land in the same folds as the published columns and cells are directly comparable.
`--cpus-per-task=8` / `OMP_NUM_THREADS=8` matches `contrastive_ablation/folds_engaging.slurm` so
BLAS reduction order matches too, keeping the identity gate tight.

### Measured block scales — why raw concatenation is fine

`base.h5` RMS: `hsc_e1` 1.841, `legacy_e1` 1.702, `hsc_e2` 0.922, `legacy_e2` 0.967.
`contrastive`: 2.057 / 1.970 / 1.562 / 1.074. All finite. The physics block is only ~1.9× the
instrument block, so raw concatenation will not make the MLP's first layer ignore the instrument
half — and the published pipeline never standardizes `X` either (`predict_combined.py:331` feeds
raw features; only `Y` is z-scored at `:317-329`).

**Decision rule:** if the raw 256-D concat scores *below* the 128-D instrument-only baseline,
re-run with `--balance-blocks` (rescales each 64-D block to unit RMS — one scalar per block, so
label-independent and effectively leak-free) before interpreting, because that would be the
scale-imbalance signature rather than a real result.

## Results (5 folds, 2026-07-28)

Full tables in `RESULT_instrument_concat.md`. The published columns were **recomputed** by this
pipeline and reproduce `RESULT_folds.md` exactly (0.751 ± 0.011, 0.852 ± 0.016, …), and the
`combined_e1` identity gate passed at max |Δ| = **0.00e+00**.

| property | Ours instr 128-D | Ours phys⊕instr 256-D | Ours phys⊕instr 256-D **balanced** | Contrastive instr 128-D |
|---|---|---|---|---|
| Galaxy Depth | 0.751 ± 0.011 | 0.727 ± 0.014 | 0.745 ± 0.011 | 0.852 ± 0.016 |
| PSF Depth | 0.742 ± 0.013 | 0.717 ± 0.017 | 0.738 ± 0.012 | 0.846 ± 0.020 |
| # Observations | 0.535 ± 0.014 | 0.500 ± 0.013 | 0.525 ± 0.014 | 0.699 ± 0.009 |
| E(B−V) | 0.591 ± 0.025 | 0.548 ± 0.027 | 0.570 ± 0.024 | 0.770 ± 0.017 |
| PSF FWHM (HSC) | 0.500 ± 0.017 | 0.471 ± 0.021 | 0.501 ± 0.018 | 0.684 ± 0.018 |

**1. The decision rule fired, and the block-scale artifact was real.** The raw concat came in
0.024–0.043 *below* instrument-only. Rescaling each 64-D block to unit RMS recovers nearly all of
it (residuals −0.006, −0.004, −0.010, −0.021, +0.001). So the raw number understates the
information content, and the balanced column is the one to quote.

**2. Adding the physics latent contributes nothing.** Balanced phys⊕instr is within error bars of
instrument-only on all five properties. Note this is *not* because the physics latent is empty of
instrument signal — alone it reaches 0.059–0.176 — but because that signal is entirely redundant
with the instrument latent. The factorization is clean in the sense that matters: the instrument
latent already carries everything the model knows about the instrument.

**3. The gap to contrastive is genuine, not a factorization artifact.** This is the answer to the
question the experiment was built to settle. Even using *both* latents, our model reaches
0.745 vs the contrastive baseline's 0.852 from its instrument latent alone — the deficit persists
at ≈ −0.107 on Galaxy Depth and −0.183 on PSF FWHM. Our model encodes less instrument
information, full stop; it is not merely spreading it across two spaces.

**4. The same dilution appears in the contrastive model**, whose balanced concat (0.835) also
fails to beat its instrument latent alone (0.852). That cross-check is reassuring: the effect is
a general probe-dilution property, not something specific to our architecture.

**5. Survey-matched probes (new).** Restricting to the survey that provides the target costs
little: 64-D matched instrument gives 0.733 vs 0.751 for the 128-D both-surveys latent on Galaxy
Depth. So the published column is not materially inflated by cross-survey leakage.

Not run, and now the one thing that would sharpen claim 2:
`contrastive_ablation/paired_fold_test.py` would turn "within error bars" into a t-statistic.
Fold membership is identical, so it is valid — it just needs the published and new fold CSVs
reachable from one directory.

## Running it

```bash
# 1. derived feature sets (seconds, CPU, ~10 MB each)
for src in base:base contrastive-spatial-conv1x1:contr; do
  for r in e1e2_256 e1e2m_hsc e1e2m_leg e2m_hsc e2m_leg; do
    python build_derived_h5.py \
      --in  ../engaging/outputs/embeddings/${src%%:*}.h5 \
      --out outputs/embeddings/${src##*:}-${r//_/-}.h5 --recipe $r
  done
done

# 2. probes — fold 0 first for a quick preliminary table, then folds 1-4
sbatch --array=0-9     --export=ALL,FOLDS="0",DISCORD_WEBHOOK="$DISCORD_WEBHOOK" run_folds.slurm
sbatch --array=0-39%10 --export=ALL,FOLDS="1 2 3 4",DISCORD_WEBHOOK="$DISCORD_WEBHOOK" run_folds.slurm

# 3. table
python aggregate_rebuttal_table.py --folds 0            # preliminary, no error bars
python aggregate_rebuttal_table.py --folds 0,1,2,3,4    # final
```

`run_folds.slurm` is resumable (a task whose CSV exists exits immediately) and runs from
node-local `/tmp`: `predict_combined.py:343` builds `ModelCheckpoint` with no `dirpath`, so probe
checkpoints land in the CWD — that is how the published run littered 2101 files / 2.7 GB into the
repo root. POOL is at ~93%, so the scratch dir is deleted on exit.

## Validation

- **`combined_e1` identity gate.** Every derived H5 copies `combined_e1` through verbatim, so its
  probe scores must reproduce the published `base` / `contrastive-spatial-conv1x1` `combined_e1`
  scores. `aggregate_rebuttal_table.py` checks this and hard-fails above 1e-3. This validates the
  whole derived-feature path end-to-end.
- **Sanity floor.** The 256-D concat input is a superset of `combined_e2`, so it should not score
  materially *below* it — if it does, see the decision rule above.
- **Preliminary table honesty.** With `--folds 0` every column, published ones included, is
  recomputed on fold 0 alone, so the comparison stays apples-to-apples rather than mixing a
  1-fold number against a 5-fold mean.

## Caveats to carry into any response text

- The 256-D concat has **2× the probe input width** of the published columns. The 128-D
  survey-matched concat is the dimension-matched control; both are reported.
- The ± on the Ours/Contrastive columns is a *population* std over band-level fold-stds combined
  in quadrature; the paired ± in `RESULT_paired_flat128.md` is a *standard error of the mean* over
  5 folds. Different statistics — do not compare their magnitudes.
- **Pre-existing discrepancy, not introduced here.** `reviews/review_2_response.md:19` claims both
  models exclude the 5469 downstream galaxies from SSL training, but the published table came from
  `base` and `contrastive-spatial-conv1x1`, not the `-DWNVAL` checkpoints. The `-DWNVAL`
  embeddings exist (`contrastive-spatial-conv1x1-DWNVAL.h5`, `fm-control.h5`) but were never run
  through the 5-fold script. Worth settling before camera-ready.
- `reviews/` is read-only. Proposed wording goes to `reviews/claude-suggestions/`.

## Not run (available cheaply if wanted)

- `contrastive_ablation/paired_fold_test.py` on the new columns → ±SE and t-statistics.
- A shuffled-physics 256-D capacity control (same width, physics information destroyed).
- The same 5 recipes on the `-DWNVAL` checkpoint pair, which would settle the caveat above.

## Files

| file | role |
|---|---|
| `build_derived_h5.py` | post-hoc H5 feature builder (5 recipes, `--balance-blocks`) |
| `run_folds.slurm` | job array over (variant × fold), resumable, Discord failure ping |
| `aggregate_rebuttal_table.py` | merges published + new fold CSVs, identity gate, emits the table |
| `outputs/embeddings/` | 10 derived H5s (~86 MB total) |
| `outputs/predictions/folds/` | per-(variant, fold) probe CSVs |
| `RESULT_instrument_concat.md` / `.csv` | the table |
