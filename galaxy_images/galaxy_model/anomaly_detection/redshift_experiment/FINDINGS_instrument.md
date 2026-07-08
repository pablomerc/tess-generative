# FINDINGS — Instrument-space redshift anomaly experiment

Port of the redshift anomaly experiment (see `FINDINGS.md`) to the **instrument latent**
(`encoder_2(img).mean(1)`, 16-D) instead of the physics latent (`encoder_1`). Same base
checkpoint (`checkpoints/base/snapshot.ckpt`), same 9,210 finite-`desi_z`>0 HSC galaxies, same
redshift-conditioning, same NSF profiles / splits. Only the encoded latent changed
(`--encoder instrument`). Outputs in `outputs_instrument/`; physics `outputs/` untouched.

Run: 4 SLURM jobs on `mi2101x` (encode→A&B→diag), ~4 min total. Comparison:
`compare_physics_vs_instrument.py` → `outputs_instrument/compare_physics_vs_instrument.png` + `compare_metrics.txt`.

## Metrics (physics vs instrument)

| metric | physics (encoder_1) | instrument (encoder_2) |
|---|---|---|
| Ridge R²(z \| latent) | 0.591 | **0.373** |
| RandomForest R²(z \| latent) | 0.640 | **0.453** |
| Spearman(uncond, cond) | 0.949 | 0.905 |
| top-32 overlap (uncond ∩ cond) | 28/32 | 12/32 |
| **cross-space top-32 overlap (physics-uncond ∩ instrument-uncond)** | — | **2/32** |

## Findings

**1. The two encoders flag almost entirely different anomalies (headline).** Only **2 of 32**
top unconditional anomalies are shared between the physics and instrument spaces. The physics
encoder surfaces galaxies with unusual intrinsic structure; the instrument encoder surfaces
galaxies with unusual *observational* character (PSF/seeing/depth/noise). This near-disjoint
anomaly set is the clearest evidence that the dual-encoder learned genuinely distinct
representations — the whole point of the physics/instrument split.

**2. Redshift is decodable from the physics latent much more than the instrument latent, but
the instrument latent is NOT redshift-free.** R²(z|latent) drops from 0.59/0.64 (physics) to
0.37/0.45 (instrument) — disentanglement in the expected direction (physics carries most of
the redshift signal). But 0.37–0.45 is well above zero: the instrument latent retains moderate
redshift information. This is most likely **genuine observational correlates of redshift**
(apparent size, brightness, S/N and color all vary with z and legitimately live in instrument
space) rather than pure leakage — but it could be partly residual leakage. Follow-up: partial
out apparent magnitude / SHAPE_R and re-probe; connects to the reverse-direction diagnosis in
`hierarchical_attention/INSTRUMENT_LEAK_OPTIONS.md`.

**3. Redshift-conditioning reshuffles the instrument-space anomaly *tail* more than physics.**
Global rank stays similar (Spearman 0.905), but the top-32 uncond∩cond overlap is 12/32
(instrument) vs 28/32 (physics). Interpretation is cautious — this is a small-N tail metric —
but it is consistent with the modest redshift content in the instrument latent (point 2) being
concentrated enough that conditioning on z re-ranks the extreme instrument outliers. Worth a
larger-K check before over-reading.

## Takeaways
- **Disentanglement works at the anomaly level** (2/32 cross-space overlap) even though it is
  only *partial* at the linear-probe level (instrument still decodes z at R²≈0.4).
- Instrument-space anomaly detection is a distinct, complementary tool: it finds observationally
  weird galaxies, not physically weird ones — useful for data-quality / artifact flagging.
- Open question: is the residual R²(z|instrument) real (apparent-property correlates) or leak?
  The clean test is to re-probe after partialling apparent magnitude.

## Reproduce
```
bash anomaly_detection/redshift_experiment/submit_all_instrument.sh      # encode→A&B→diag on mi2101x
python anomaly_detection/redshift_experiment/compare_physics_vs_instrument.py --webhook <url>
```
Code delta from the physics experiment: `common.encode_instrument_mean` (encoder_2), the
`--encoder {physics,instrument}` flag in `encode_redshift_subset.py`, and the
`*_instrument.sh` SLURM wrappers. The A/B/diagnostic scripts are reused unchanged
(`--latents`/`--out-dir`).
