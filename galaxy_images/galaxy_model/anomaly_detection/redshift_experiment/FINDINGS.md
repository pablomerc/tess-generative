# Redshift-aware anomaly detection — findings & changelog

Living notes for this experiment. Update each iteration. See plan file
`~/.claude/plans/would-it-be-possible-frolicking-bee.md` for the design plan.

---

## What the experiment is

Run the existing NSF anomaly detector on a **redshift-labeled HSC sample** and
study how anomalies relate to physical properties. Two flows run in parallel
on the same physics latent:

- **Job A** — unconditional NSF (the standard anomaly detector, redshift-agnostic).
- **Job B** — conditional NSF modeling p(latent | z), surfacing galaxies that
  are unusual *for their redshift*.

Working set: the **9,210** HSC galaxies in `data/hsc_downstream/catalog.parquet`
with a finite, positive `desi_z` (range 0.0038 – 0.586, median 0.243).

"Ours-Physics" latent = `encoder_1(img).mean(dim=1)` from
`outputs/neighbors_all_attn/2026-04-05/checkpoints/best-epoch=228-step=87000.ckpt`
(default checkpoint registered as `checkpoints/base/snapshot.ckpt`). Latent
dim **D = 16**.

---

## Preprocessing parity — verified

`hsc_downstream/hsc_flux.bin` and `neighbors_efficient/hsc_image.bin` hold
**byte-identical raw pixels**. 366,706 HSC galaxies common to both binaries;
per-galaxy i-band pixel correlation = 1.000, per-band p99.9 ratio = 1.0 across
g/r/i/z/y for randomly sampled common galaxies. So encoding with the training
chain `preprocess_image_v2(crop_size=48, survey="hsc") → (img - NORM_DICT["hsc"]) / std → [:4]`
gives the encoder the same input distribution it saw during training.

---

## Findings — by iteration

### v1 (initial run, 2026-06-18)

Three SLURM jobs (encode 64s, Job A 37s, Job B 31s on MI210). Posted 5 figures
to Discord:

- `A1_top_anomalies.png` — top-32 unconditional NSF anomalies
- `A2_redshift_distribution.png` — z histogram of all + top-32 rug
- `A3_top_anomalies_zfiltered.png` — top-32 with z ≥ p25 (lowest 25% dropped)
- `B1_top_anomalies.png` — top-32 conditional NSF anomalies
- `B2_redshift_distribution.png` — z histogram + top-32 conditional rug

User-visible issues that drove v2:

1. **Anomaly tiles looked washed out** vs the paper figures. Root cause:
   v1 passed the encoder-input tensor `(img - 0.022) / 0.05` (4 channels, 48×48)
   into the percentile-stretch RGB function. The published anomaly figures
   instead read `neighbours_v2.h5["images_hsc"]` (5 channels, 160×160,
   arcsinh-compressed but **not** divided by NORM std). Verified by reading
   `images_hsc` directly: per-band median ~0.005–0.025, p99 0.15–1.2 — matches
   `preprocess_image_v2` output without normalization. The encoder-tensor's
   wide negative-to-positive range pushes the 1–99 percentile stretch to anchor
   on bright cores, washing out the body of the galaxy.

2. **Job B ≈ Job A** — concerning. Cheap diagnostic on the v1 scores:

   ```
   Spearman(uncond, cond)    = 0.9428
   Pearson (uncond, cond)    = 0.9660
   top-32 overlap            = 29 / 32
   Spearman(uncond, z)       = -0.1486
   Spearman(cond,   z)       = -0.1318
   Ridge   R²(z | latent)    = 0.5934
   RandomForest R²(z | latent) = 0.6231
   ```

   **The physics latent strongly encodes redshift** (R² ≈ 0.6), so conditioning
   on z *should* substantially change the ranking. The cond flow producing a
   near-identical ranking is the diagnostic signature of "the cond NSF is
   effectively ignoring its context."

   Likely cause: a 1-D context entering NSF transforms with
   `hidden_features=[64,64]` — the small hyper-net concatenates the 1-D context
   alongside a 16-D latent at each conditioner, so context can be
   under-weighted by ~1/16 in the first layer and never recovered.

### v2 (this iteration)

Four changes, applied in `common.py` + 2 scoring scripts + new diagnostic:

#### Fix 1 — separate display path from encoder path

New `common.preprocess_hsc_for_display(raw)` → `preprocess_image_v2(crop_size=160, survey="hsc")`
**without** the `(img - mean) / std` step. Matches
`neighbours_v2.h5["images_hsc"]` storage convention exactly (verified per-band
medians on a random sample: 0.0007–0.009, p99 0.06–0.17 — same order of magnitude
as the audit's reading of `images_hsc`).

New `HSCBinDisplayDataset` + `load_hsc_images_for_display()` give a clean (n, 5, 160, 160)
loader for the grids. `to_rgb` (unchanged) takes `[:3]` = (g, r, i) and maps to
matplotlib (R, G, B) — backwards from astronomy convention but **matches the
paper figure verbatim**, deliberately.

Encoder path (4-channel, 48×48, normalized) is untouched and continues to feed
the encoder; only the display path changed.

#### Fix 2 — middle-rank sanity grid

Both `fit_score_plot_uncond.py` and `fit_score_plot_cond.py` now also pick
ranks `[N//2 - 16, N//2 + 16)` of the descending score sort and render a tile
grid via the same display path. Posted to Discord as `A4_mid_rank_anomalies.png`
and `B3_mid_rank_anomalies.png`. If middle-rank tiles look "typical" relative
to top tiles, the score ordering is meaningful.

#### Fix 3 — 5-property distribution figure

Replaces v1's single-panel `*_redshift_distribution.png` with a 1×5 panel
figure (`A2_property_distributions.png`, `B2_property_distributions.png`):
one panel per property, each showing the full-sample histogram + top-32
anomaly values as a red rug + faint red twin-axis histogram.

Properties (all 9,210/9,210 finite on the working set):

| Column | Units | Range | x-axis scale |
|---|---|---|---|
| `desi_z` | dimensionless | 0.0038 – 0.586 | log |
| `provabgs_logmstar` | log10(M⊙) | 7.42 – 12.69 | linear |
| `provabgs_tage_mw` | Gyr | 0.06 – 12.3 | log |
| `provabgs_z_mw` (mass-wt metallicity, **NOT redshift**) | Z/Z⊙ | 4.6e-5 – 0.045 | log |
| `provabgs_avg_sfr` | M⊙/yr | 1.0e-6 – 445 | log |

The encode step now pulls these 4 PROVABGS columns from `catalog.parquet`
(aligned to the working-set indices) and writes them under `props/<name>` in
`outputs/latents_redshift.h5`, so scoring jobs don't need to re-read the
catalog.

#### Fix 4a — diagnostic (cheap, no GPU)

Standalone `diagnostics_cond_vs_uncond.py`:

- Spearman/Pearson + top-K overlap between uncond and cond scores.
- Ridge + RandomForest R²(z | latent) — "can conditioning *possibly* help?"
- **Context-sensitivity sweep:** retrain a fresh cond-flow replica, hold 8
  fixed latents (4 random + 4 top-anomaly), sweep z across [z_min, z_max],
  measure stdev of `log p(x* | z)` across z. Flat curves (σ ≈ 0) ≡ flow
  ignored context.
- Posts `D1_cond_vs_uncond.png`, `D2_z_sweep.png`, `D3_metrics.txt` to Discord
  with an interpretation line ("flow appears to be **USING / IGNORING** context").

The v1 cheap diagnostic (run prior to v2 code changes) returned:
**latent has z signal → cond flow ranking matches uncond → bug**, justifying
Fix 4b without further investigation.

#### Fix 4b — wider conditional NSF

`common.score_nsf` gained a `profile` parameter:

| Profile | transforms | hidden_features |
|---|---|---|
| `default` (v1) | 6 | [64, 64] |
| `wide` (v2 cond) | 8 | [128, 128] |

Job B now defaults to `profile=wide`, `z_context=z_z2` (2-D context
`[z_std, z_std**2]` instead of bare standardized z), `nsf_epochs=100`,
`cosine_lr=True`. Rationale: more hyper-net capacity for the conditioner to
modulate spline coefficients with z; an explicit z² term gives a direct
nonlinear coupling without forcing the hyper-net to discover it.

Unconditional Job A is unchanged (`profile=default`, 50 epochs) for
comparability with the v1 ranking.

---

## What v2 does NOT change

- Encoder preprocessing (still `crop_size=48, survey="hsc"` + normalize +
  drop y, matching training).
- The Ours-Physics latent (`encoder_1.mean(dim=1)`, D=16).
- The checkpoint (`checkpoints/base/snapshot.ckpt`).
- The working-set selection (`desi_z` finite & > 0 → 9,210 rows).
- The 80/20 train-fit / score-everyone split (seeded with `np.random.default_rng(42)`).
- SLURM partition / env / cluster (mi2101x, `torchenv`, `TORCH_BLAS_PREFER_HIPBLASLT=0`).

So Job A's v2 ranking should be **identical** to v1's (same code path);
v2's improvements there are display-only. Job B's v2 ranking will differ
from v1's by design (wider flow, richer context, more epochs).

---

## Files

- `common.py` — preprocessing (encoder + display), NSF (default + wide),
  z-context builder, RGB stretch, grid + multi-property plots, Discord wrapper.
- `encode_redshift_subset.py` — encode hsc_downstream desi_z>0 subset → 16-D
  latent + property columns → `outputs/latents_redshift.h5`.
- `fit_score_plot_uncond.py` — Job A: NSF (default profile) → A1/A2/A3/A4 → Discord.
- `fit_score_plot_cond.py` — Job B: NSF (wide profile, [z, z²] ctx) → B1/B2/B3 → Discord.
- `diagnostics_cond_vs_uncond.py` — Fix 4a (correlation + R² + z-sweep) → D1/D2/D3 → Discord.
- `job_encode.sh`, `job_uncond.sh`, `job_cond.sh`, `job_diag.sh` — SLURM wrappers.
- `submit_all.sh` — submit-and-chain driver (encode → A & B & diag in parallel).
- `outputs/` — all H5 / .npy / .png artifacts (git-ignored).

---

## v1 vs v2 figure naming

| v1 | v2 | Notes |
|---|---|---|
| A1_top_anomalies.png | A1_top_anomalies.png | display path fixed |
| A2_redshift_distribution.png | **A2_property_distributions.png** | 5 panels |
| A3_top_anomalies_zfiltered.png | A3_top_anomalies_zfiltered.png | display path fixed |
| — | **A4_mid_rank_anomalies.png** | new |
| B1_top_anomalies.png | B1_top_anomalies.png | display path fixed + wide cond flow |
| B2_redshift_distribution.png | **B2_property_distributions.png** | 5 panels |
| — | **B3_mid_rank_anomalies.png** | new |
| — | **D1_cond_vs_uncond.png** | diagnostic |
| — | **D2_z_sweep.png** | diagnostic |
| — | **D3_metrics.txt** | diagnostic |

---

## v2 run logbook

**2026-06-30 16:46 — v2 chain submitted (339133–339136)**

- Encode (339133): 52s, latents H5 N=9210/D=16 + props/{logmstar, tage_mw,
  z_mw, avg_sfr} all 9210/9210 finite. ✓
- Job A (339134): 43s. A1/A2/A3/A4 written.
- Job B (339135): 45s. **Caught a script-vs-job-script discrepancy after the
  fact:** `job_cond.sh` v1 explicitly passed `--nsf-epochs 50`, overriding the
  v2 python default of 100. So Job B actually ran with `profile=wide, ctx=2 (z²),
  epochs=50, cosine LR`. The wide-profile + z² parts of Fix 4b were applied
  correctly; only the epoch count was short. Job script now patched to omit the
  flag so the python default (100) governs. Diag will tell us whether 50 epochs
  of wide+z² already shifted the ranking.

## Verification checklist for v2

- [x] Display loader produces (5, 160, 160) tensors with per-band medians in
  the 0.001–0.05 range (verified: 0.0007–0.009 on random sample).
- [x] `props/{logmstar, tage_mw, z_mw, avg_sfr}` will be written to the v2
  latents H5 (encode script updated).
- [x] 5-panel property plot renders cleanly with the v1 score data (smoke
  posted to Discord).
- [ ] After resubmission: 7 figures + 3 diagnostic artifacts land on Discord.
  Anomaly tiles look like the paper figure (not washed out). Middle-rank tiles
  look TYPICAL relative to top tiles. Diagnostic D2 verdict states whether the
  wide cond flow uses its context.
