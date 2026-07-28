# Suggested response — reviewer JcH2, data-scale / rare-object questions

Draft for Pablo to merge into the official response. **Not** an official file.
Rewrite of `reviews/review_2_response.md`, grounded in
`galaxy_images/galaxy_model/scale_ablation/`.

---

## What the reviewer actually asked — this is *four* separate asks

The current draft answers one of them. Worth answering all four explicitly and saying so,
because JcH2 raised this in four places and rated the paper **4 (borderline accept)**:

| where | ask |
|---|---|
| **Q4** | "the amount of data needed to train the architecture … how big of a dataset **on each instrument** before we could expect to fit a set of reasonably performant physics / instrument encoders?" |
| **Q2** | "quantification of *how rare of phenomena can be detected by the physics encoder* … motivate **seed survey size** requirements" |
| **weakness (minor), §4.1** | "how *common* does a rare event need to be to be trained into the latent space? Does the architecture have a **'resolution' for rare phenomena** in terms of training data? … how big of a source-instrument catalog is needed, **and how much overlap** with the target instrument?" |
| **Limitations** | "the amount of source data required needs to be ablated for application to smaller surveys (*'when does this technique become applicable for a new survey?'*)" |

Note the asymmetry: **three** of the four ask about *per-instrument catalog size*, and only
the §4.1 weakness asks about *overlap*. Our ablation varies **overlap** and holds
single-survey density at full. That is the right choice (see below) but it means we are
answering a related question, not literally the one in Q4 — and we should say so in one
sentence rather than let the reviewer discover it. See **Decision 4**.

---

## Problems with the current draft

| line | problem |
|---|---|
| 4 | Ungrammatical run-on ("…does that make sense so we decided to ablating is the number of…"). Cannot ship. |
| 5 | **In Spanish.** ("Porque me refiero, si tienes una nueva survey…") The content is good and belongs in the response — it is the clearest statement of *why* overlap is the scarce resource — but it needs to be in English. |
| 14 | Internal run bookkeeping ("plus smoke / checkpoint snapshots", "paper footing"). Cut. |
| 16–47 | Table pasted as one column per line — unreadable. Also mislabels the rungs: prose says "3.3k / 33k", table says 3,162 / 31,622. Use the exact values. |
| 36, 40 | `fm-control` and `base` appear with no explanation. A reviewer cannot know these are (a) a step-matched full-data control and (b) the paper's own checkpoint, or why both are shown. |
| 64 | "Legacy is non-monotonic (best around 3k–10k, worse at 31k)" — true but incomplete, and stated in the most damaging possible way. See **Legacy MSE** below: every scale arm is at or *below* all three full-data references, so this is not a scaling failure. |
| 71–82 | Recall reported as ranges ("5–6 / 8", "5–7 / 8"). Those are **two different checkpoints of the same arm**, not error bars, and will be read as error bars. |
| 86 | "Planned LOO recall-vs-N … was **blocked / not finished** (needs the AMD `lens_indices.txt`); lens33 deck **exists but isn't scored yet**." Never tell a reviewer about a cluster file-transfer problem or an unscored deck. Reframe as a revision commitment. |
| 89, 91 | The conclusion claims "**Both** overlap and single-survey catalog size matter." Single-survey catalog size was **not ablated** — it was held at full density by design. As written this is an unsupported claim on exactly the axis Q4 asked about, and it is the sentence most likely to draw a follow-up. There *is* a defensible version of it (the distinct-image confound, below) — but it has to be argued, not asserted. |
| — | **Buries the finding.** The headline result is a dissociation: the physics latent is ~saturated at 10³ crossmatches while the instrument latents keep climbing to 10⁵. That is the answer to "what does overlap buy you," and it is currently three tables deep. |
| — | Missing the free win: the `untrained` column is the R²-floor JcH2 asked for in their **general comments** ("do you have an idea of *random guessing* for R² in Fig 4?"). Point at it. |

---

## The number that makes the answer quotable

Fraction of the achievable R² recovered, where achievable = `fm-control` (step-matched,
full data) minus `untrained` (random-init floor). Computed from `results/r2_table.md`:

| crossmatches N | % of train | mean physics (7) | mean Legacy instr. (5) | mean HSC instr. (2) |
|---|---|---|---|---|
| 1,000 | 1.1% | **81%** | 32% | 28% |
| 3,162 | 3.4% | 89% | 54% | 45% |
| 10,000 | 10.7% | 94% | 71% | 65% |
| 31,622 | 33.9% | 98% | 77% | 71% |

**One percent of the crossmatches recovers four-fifths of the physics signal and less than
a third of the instrument signal.** That single sentence answers the reviewer.

Mechanism, from the per-property rows — this is worth stating because it makes the result
predictive rather than empirical:

- **Ellipticity is free**: e1 0.918 → 0.924, e2 0.924 → 0.917 from N=1,000 to full data. Flat.
  Morphology needs essentially no crossmatch. *This also closes the loop on the reviewer's own
  general comment* about the untrained ResNet's ellipticity inductive bias — the properties
  that need no overlap are exactly the geometric ones.
- **What still improves with overlap is what is *not* visible in one image**: t_age 0.109 →
  0.283 (2.6×), redshift 0.753 → 0.804, log M\* 0.709 → 0.763 — i.e. the SED/colour-inferred
  quantities, which are the ones that need the cross-survey pairing to be pinned down.
- **Instrument properties need the most overlap**: `# Observations (L)` 0.175 → 0.509,
  `PSF FWHM (H)` 0.100 → 0.408. Makes sense — instrument signature is only identifiable by
  contrast against the *same* source seen through another instrument, which is exactly what a
  crossmatch is.

---

## Legacy reconstruction MSE — reframe, don't concede

From `results/mse_table.md`. Full range across all four scale arms is 0.154–0.188; the three
full-data references are 0.181 / 0.196 / 0.200. **Every scale arm, including N=1,000, is at or
below all three full-data references** — and the full-data references had seen ~95% of the
eval galaxies during training while the scale arms had seen none. So the honest reading is
"Legacy recon MSE is insensitive to N," not "it degrades."

The 3,162 → 31,622 rise (0.1538 ± 0.0063 → 0.1876 ± 0.0072) is larger than the SEMs, so don't
call it noise. Suggested wording: report HSC as the monotone 2× improvement it is, say Legacy
shows no monotone trend and sits within the full-data band across every arm, and offer the
noise-floor interpretation without over-claiming a mechanism.

---

## Lens retrieval — n = 1 query, label it as such

Decoded `lens_recall/decks/deck_lens48_KEY.json` against `lens_recall/scores_lens48.json`
(blinded, 12 decks, seed 20260726, crossmatched counterpart excluded from the gallery).
Full mapping, which the draft collapses into misleading ranges:

| arm | space | 75k ckpt | best-MSE ckpt |
|---|---|---|---|
| N=1,000 | physics | 5/8 (deck H) | 6/8 (deck B) |
| N=3,162 | physics | 5/8 (C) | 7/8 (I) |
| N=10,000 | physics | 4/8 (E) | 5/8 (L) |
| N=31,622 | physics | 6/8 (G) | 5/8 (K) |
| full data (fm-control 75k) | physics | 7/8 (F) | — |
| full data (paper ckpt 87k) | physics | 7/8 (D) | — |
| paper ckpt 87k | **instrument** | **0/8** (A) | — |
| N=31,622 | **instrument** | **0/8** (J) | — |

Mean over the eight scale-arm decks: 5.4/8. Full data: 7/8 twice. Instrument space: 0/8 twice.

Two things are genuinely strong here and should be kept: the **0/8 instrument-space control**
(retrieval is a property of the physics latent, not of image similarity), and the fact that
**N=1,000 already retrieves 5–6 of 8**. But `lens_recall/PLAN.md:22` is right that one query
with ~7 relevant items cannot support a curve — the arm-to-arm spread here is draw noise, and
the non-monotonicity (10,000 scoring lowest) proves it. Present it as one worked example, give
the 0/8 control the weight, and commit to LOO in the revision. Do **not** present the column
as a scaling curve.

### ⚠ The highest-value missing number in the whole response

Q2 asks for a *rarity threshold*. The strongest available answer is:

> the N=1,000 arm trained on approximately **zero** known lenses and still retrieves them —
> so a rare class need not be represented in the crossmatch set to be retrievable.

`lens_recall/PLAN.md:44` puts the known-lens list at ~70 objects. If those are anchors, the
expected count inside a 1,000-anchor draw is 70 × 1000/103,741 ≈ **0.7**, versus ≈21 at
N=31,622. If that holds, it converts a qualitative claim into the quantitative rarity answer
the reviewer asked for, and it is *one histogram over `lens_indices.txt` ∩ each rung's subset
JSON* — the subsets are already on disk. **This is worth chasing the file for.** I have not
verified that the lens rows are anchors rather than pool rows, so the 0.7 is an estimate, not
a result — do not put it in the response until it is computed.

---

## Decisions I need from you before this is final

1. **Disclose the LR-schedule finding?** `PROGRESS.md:27-44` documents that
   `CosineAnnealingLR(T_max=-1)` makes the LR alternate 1e-4 / 0 per epoch, that the *published*
   model was trained this way, and that naive subsetting would have made the LR waveform
   dataset-size-dependent — which we fixed with index tiling (every arm: 364 steps/epoch,
   identical waveform). **My recommendation: mention the control, not the pathology.** One
   clause — "every arm shares an identical optimization schedule; subsets are cyclically tiled
   so steps-per-epoch and the LR trajectory are identical across arms" — pre-empts the obvious
   "did you match optimization budget?" question without opening a discussion of the base
   model's schedule in a rebuttal where we cannot control the follow-up. Your call; if you'd
   rather be maximally transparent I can write the fuller version.

2. **Lens48 in or out?** In, with the n=1 caveat and the 0/8 control — the reviewer called
   Fig. 5 "extremely motivating," so extending it along the scale axis is responsive to *their*
   stated interest. But it is an anecdote and must read as one.

3. **Does `lens_indices.txt` exist here yet?** Drives whether the rarity-threshold sentence
   above can go in.

4. **How hard to concede the per-instrument-size axis?** Q4 literally asks about single-survey
   catalog size, which we did not vary. But there *is* a real, defensible observation in the
   manifest: distinct images touched scales with N — 6,942 at N=1,000 → 176,610 at N=31,622
   (`subsets/manifest.json`). So the arms co-vary "number of crossmatches" with "number of
   distinct single-survey images actually touched," and the instrument-R² growth could be driven
   by either. That is (a) the honest confound a sharp reviewer will find, and (b) the legitimate
   basis for the draft's "both matter" line. I've written it as a limitation with the
   6,942-images point turned into a *strength* for the physics claim. Tell me if that's too much
   ceding of ground for a rebuttal.

---

## Suggested response

> **Q2 / Q4 / §4.1 / Limitations — how much data, and how much overlap.**
> We ran the ablation. Thank you for pushing on it; the result turned out to be a clean
> dissociation and we think it strengthens the paper.
>
> **What we varied, and why.** In multi-instrument astronomy the scarce resource is not
> single-survey data — it is the *overlap*. A new survey may hold 10⁶ galaxies while
> crossmatching to only 10³ HSC sources. Our architecture is built for exactly that
> asymmetry: only the anchor path requires a crossmatch, while the same-instrument
> conditioning neighbours need no counterpart in the other survey. Concretely, the paper's
> model trains on ~104k crossmatched anchors but draws its neighbours from a pool of 468k
> sources, and we verified the sky-kNN graph is strictly bipartite anchors→pool (zero
> anchor→anchor edges) — so single-survey sources genuinely enter training without ever being
> crossmatched. We therefore ablate **the number of crossmatched anchors**, holding
> single-survey coverage at full density: the realistic deployment scenario for a new survey.
>
> **Setup.** Four nested arms, 1,000 ⊂ 3,162 ⊂ 10,000 ⊂ 31,622 crossmatches (1.1% / 3.4% /
> 10.7% / 33.9% of the training anchors), each trained for the same 75k steps as the full-data
> control, 50/50 balanced over which survey is the target. Subsets are nested so the curve is
> not dominated by draw variance, and cyclically tiled so that steps-per-epoch and the
> learning-rate trajectory are identical in every arm — only the number of distinct
> crossmatches changes. Crucially, the four arms exclude the downstream evaluation galaxies at
> load time, whereas the full-data references did not; the full-data numbers below are
> therefore mildly optimistic, which biases the comparison *against* our scaling claim.
>
> **Result 1 — physics saturates early, instrument identification does not.** Downstream R²
> on 5,469 held-out crossmatches (same probes as Table 5; `untrained` is the random-init
> encoder, which also answers your general comment about the R² floor):
>
> | N crossmatches | mean physics (7) | mean Legacy instr. (5) | mean HSC instr. (2) |
> |---|---|---|---|
> | 1,000 | 0.593 | 0.289 | 0.219 |
> | 3,162 | 0.612 | 0.390 | 0.282 |
> | 10,000 | 0.625 | 0.471 | 0.353 |
> | 31,622 | 0.634 | 0.499 | 0.375 |
> | full (93k, step-matched control) | 0.640 | 0.611 | 0.478 |
> | full (paper checkpoint) | 0.639 | 0.591 | 0.492 |
> | untrained (random-init floor) | 0.394 | 0.135 | 0.119 |
>
> Measured as the fraction of the attainable R² recovered (full-data control minus
> random-init floor), **1,000 crossmatches — 1% of our training set — already recover 81% of
> the physics signal but only 28–32% of the instrument signal**; by 31,622 the physics latent
> is at 98% while the instrument latents are still at 71–77% and climbing.
>
> The per-property breakdown explains why, and we think it is the practically useful part:
> ellipticity is essentially free (e1 0.918 at N=1,000 vs 0.924 at full data), while the
> properties that keep improving are the ones not determinable from a single image — stellar
> age (0.109 → 0.283), redshift (0.753 → 0.804), stellar mass (0.709 → 0.763) — and the
> instrument properties improve most of all (Legacy `# Observations` 0.175 → 0.509, HSC
> `PSF FWHM` 0.100 → 0.408). Instrument signature is only identifiable by contrasting the
> *same* source through two instruments, so it is the component whose supervision *is* the
> crossmatch. Morphology is not.
>
> **Practical answer to "when does this become applicable to a new survey?"** For
> morphology/geometry-driven science and for cross-survey retrieval, ~10³ crossmatches suffice.
> For accurate instrument-property read-out and for the noise-model application of §5, plan
> for 10⁴–10⁵.
>
> **Result 2 — reconstruction.** C0/intact reconstruction MSE on 256 held-out anchors: HSC
> improves monotonically and by 2× with N (0.107 → 0.053), i.e. more crossmatches buy sharper
> generation for the higher-resolution target. Legacy shows no monotone trend (0.154–0.188
> across arms) and every arm sits at or below all three full-data references (0.181–0.200)
> despite the arms never having seen the eval galaxies — consistent with Legacy reconstruction
> residual being dominated by its shallower noise floor rather than by conditioning quality.
>
> **Result 3 — rare-object retrieval (your §4.1 and Q2).** We repeated the Fig. 5 lens query
> for every arm, with the retrieved panels blinded and scored by eye. In the **physics** space,
> N=1,000 already returns 5–6 lens-like objects in the top 8, and the full-data models return
> 7/8. In the **instrument** space, both the paper model and the N=31,622 arm return **0/8** —
> retrieval is a property of the physics latent, not of generic image similarity. We stress
> this is a single query (~7 relevant objects in the gallery), so the arm-to-arm differences
> are within draw noise and we do not read a curve off it; the 0/8 instrument control and the
> N=1,000 result are the robust statements. For the revision we will report leave-one-out mean
> recall@k over all known lenses in the gallery, together with the number of known lenses each
> arm actually trained on — which turns "how rare can a phenomenon be" into a measured
> threshold rather than an anecdote.
>
> **Limitation we will add.** This ablation varies the *overlap* at fixed single-survey
> density; it does not independently vary per-instrument catalog size, and the two are not
> fully separable in our design — because neighbours are drawn per anchor, the number of
> distinct single-survey images the model touches also grows with N (6,942 at N=1,000 vs
> 176,610 at N=31,622). So the instrument-property scaling may reflect the number of distinct
> instrument realizations seen as much as the number of crossmatches. We will state both this
> and the untested cross-spectral-range extension you raise as explicit limitations. Notably,
> the physics result is robust to this confound in the useful direction: near-ceiling physics
> R² is reached having touched fewer than 7,000 distinct images in total.

---

## Assets ready to attach

- `results/r2_vs_n.png`, `results/mse_vs_n.png` — curves with the random-init floor.
- `results/r2_table.md` (self-checks against the paper's own CSV, max |Δ| = 5.6e-17),
  `results/mse_table.md`, `results/r2_per_target.csv`.
- `lens_recall/decks/deck_lens48_page{1,2}.png` + `deck_lens48_KEY.json` — the blinded deck,
  if a reviewer wants to see the scoring was not post-hoc.
