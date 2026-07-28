We thank the reviewer for pushing on the data-scale question, which was raised in Q2, Q4, the §4.1 weakness and the Limitations. We ran the ablation; the result is a clean dissociation between the two latents, and we think it strengthens the paper.

**What we varied, and why.** In multi-instrument astronomy the scarce resource is not single-survey data — it is the *overlap*. A new survey may hold 10⁶ galaxies while crossmatching to only 10³ HSC sources. Our architecture is built for exactly that asymmetry: only the anchor path requires a crossmatch, while the same-instrument conditioning neighbors need no counterpart in the other survey. Concretely, the model in the paper trains on ~104k crossmatched anchors but draws its neighbors from a pool of 468k sources, and we verified that the sky-kNN graph is strictly bipartite anchors→pool (zero anchor→anchor edges) — so single-survey sources genuinely enter training without ever being crossmatched. We therefore ablate **the number of crossmatched anchors**, holding single-survey coverage at full density: the realistic deployment scenario for a new survey.

**Setup.** Four nested arms, 1,000 ⊂ 3,162 ⊂ 10,000 ⊂ 31,622 crossmatches (1.1% / 3.4% / 10.7% / 33.9% of the training anchors), each trained for the same 75k steps as the full-data control, balanced 50/50 over which survey is the target. Subsets are nested so the curve is not dominated by draw variance, and cyclically tiled so that steps-per-epoch and the learning-rate trajectory are identical in every arm — only the number of distinct crossmatches changes. The four arms also exclude the downstream evaluation galaxies at load time, whereas the full-data references did not; the full-data numbers below are therefore mildly optimistic, which biases the comparison *against* our scaling claim.

**Result 1 — physics saturates early, instrument identification does not.** Downstream R² on 5,469 held-out crossmatches, using the same probes as Table 5. The `untrained` row is the random-init encoder, which also answers the reviewer's general comment about the R² floor in Fig. 4:

| N crossmatches | mean physics (7) | mean Legacy instr. (5) | mean HSC instr. (2) |
|---|---|---|---|
| 1,000 | 0.593 | 0.289 | 0.219 |
| 3,162 | 0.612 | 0.390 | 0.282 |
| 10,000 | 0.625 | 0.471 | 0.353 |
| 31,622 | 0.634 | 0.499 | 0.375 |
| full (93k, step-matched control) | 0.640 | 0.611 | 0.478 |
| full (paper checkpoint) | 0.639 | 0.591 | 0.492 |
| untrained (random-init floor) | 0.394 | 0.135 | 0.119 |

Measured as the fraction of attainable R² recovered (full-data control minus random-init floor), **1,000 crossmatches — 1% of our training set — already recover 81% of the physics signal but only 28–32% of the instrument signal**; by 31,622 the physics latent is at 98% while the instrument latents are still at 71–77% and climbing.

The per-property breakdown explains why, and is the practically useful part: ellipticity is essentially free (e1 = 0.918 at N=1,000 vs 0.924 at full data), while the properties that keep improving are those not determinable from a single image — stellar age (0.109 → 0.283), redshift (0.753 → 0.804), stellar mass (0.709 → 0.763) — and the instrument properties improve most of all (Legacy # Observations 0.175 → 0.509, HSC PSF FWHM 0.100 → 0.408). Instrument signature is only identifiable by contrasting the *same* source through two instruments, which is precisely what a crossmatch provides. Morphology is not.

**Practical answer to "when does this become applicable to a new survey?"** For morphology- and geometry-driven science, and for cross-survey retrieval, ~10³ crossmatches suffice. For accurate instrument-property read-out and for the noise-model application of §5, plan for 10⁴–10⁵.

**Result 2 — reconstruction.** Intact (C0) reconstruction MSE on 256 held-out anchors, 250 Euler steps:

| N crossmatches | MSE HSC | MSE Legacy |
|---|---|---|
| 1,000 | 0.1069 ± 0.0147 | 0.1752 ± 0.0099 |
| 3,162 | 0.0746 ± 0.0084 | 0.1538 ± 0.0063 |
| 10,000 | 0.0591 ± 0.0058 | 0.1611 ± 0.0062 |
| 31,622 | 0.0534 ± 0.0055 | 0.1876 ± 0.0072 |

HSC improves monotonically and by exactly 2× across the ladder: more crossmatches buy sharper generation for the higher-resolution target, and unlike the physics R² this metric has not saturated by N=31,622. Legacy shows no monotone trend, and all four arms sit at or below the three full-data references we evaluated (0.181–0.200) despite never having seen the eval galaxies, whereas those references saw ~95% of them. We read the Legacy column as saturating against that survey's shallower noise floor: once the residual is dominated by photon noise rather than by conditioning quality, further crossmatches cannot reduce it.

**Result 3 — rare-object retrieval (§4.1 and Q2).** We repeated the Fig. 5 lens query for every training-size variant and counted how many promising lens candidates appear among the top 8 nearest neighbors in the physics latent space. Panels from all variants were pooled in random order and scored by eye, blind to which model produced each one.

Crucially, the Fig. 5 lens is a single-survey source rather than a cross-match, so no model ever saw it as a training target — and we verified that the four reduced-overlap variants never saw it at all, not even as a conditioning neighbor, unlike the full-data models.

In the **physics** space, N=1,000 already returns 5–6 lens-like objects in the top 8 (the range spans the final 75k-step and the best-validation checkpoints), and the full-data models return 7/8. In the **instrument** space, both the paper model and the N=31,622 variant return **0/8**. These two results are the significant ones: with only 1,000 cross-matches, and with the lens absent from training in any role, the physics latent still surfaces rare morphology it has never seen — while the instrument latent surfaces none of it, confirming that the retrieval is driven by source structure rather than by image or noise similarity.

This is a single query, so we do not read a scaling curve from the variant-to-variant differences. For the revision we will report leave-one-out mean recall@k over all known lenses in the gallery, together with what each variant saw of each lens.

**Limitation we will add.** This ablation varies the *overlap* at fixed single-survey density; it does not independently vary per-instrument catalog size, and the two are not fully separable in our design — because neighbors are drawn per anchor, the number of distinct single-survey images the model touches also grows with N (6,942 at N=1,000 vs 176,610 at N=31,622). The instrument-property scaling may therefore reflect the number of distinct instrument realizations seen as much as the number of crossmatches. We will state this, and the untested extension to different spectral ranges the reviewer raises, as explicit limitations. We note that the physics result is robust to this confound in the useful direction: near-ceiling physics R² is reached having touched fewer than 7,000 distinct images in total.
