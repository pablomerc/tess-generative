# Blinded lens-retrieval scores — query `lens48` (HSC row 183713)

> 4-row contrastive-vs-FM 2x2. base-87k physics was previously shown as row D of scale_ablation/lens_recall/decks/deck_lens48 (scored 7/8), so its score here is NOT independent of that session.

Scored by eye, blind to arm. Rows shuffled with seed 20260727. k=8; query's own cross-survey counterpart excluded.

| row | arm | family | lens exposure | space | lens-like / 8 | ranks | best rank |
|---|---|---|---|---|---|---|---|
| B | `base-5NB-DWNVAL` | flow matching | zero-shot | physics | **8/8** | 1,2,3,4,5,6,7,8 | 1 |
| A | `base-87k` | flow matching (paper) | saw lenses (unlabeled) | physics | **7/8** | 1,2,3,4,5,7,8 | 1 |
| C | `contrastive-spatial-conv1x1-DWNVAL` | contrastive conv1x1 | zero-shot | physics | **7/8** | 1,2,3,4,5,6,8 | 1 |
| D | `contrastive-spatial-conv1x1` | contrastive conv1x1 | saw lenses (unlabeled) | physics | **6/8** | 1,2,3,4,5,6 | 1 |

**Physics-latent mean:** contrastive 6.5/8 (n=2) · flow matching 7.5/8 (n=2) · difference -1.0

With n=2 and n=2 single-query rows this is a direction, not a measurement — no error bar is claimable from one query.

---

## Interpretation

**No large difference between the architectures on this query.** One panel out of eight. The
zero-shot contrastive arm ties the paper model at 7/8, and the ranges overlap.

**The scoring was stable across sessions, which is what licenses the comparison below.**
`base-87k` scored **exactly 7/8 again** — the same row, from the same latents, judged the same way
it was in the scale-ablation session. The deck builder was also verified to reproduce that row
bit-identically (same neighbour rows, same distances to 5 dp), so the two sessions are directly
comparable rather than merely similar.

**Cross-session context — contrastive sits at the top of the FM distribution, not below it.**
The same query, same k, same rendering, previous session, 10 flow-matching physics rows:

| | range | mean |
|---|---|---|
| previous session, 10 FM physics rows (scale arms + base-87k + fm-control) | 4-7 / 8 | 5.70 |
| previous session, 2 instrument-space rows (the floor) | **0 / 8** | 0.00 |
| this session, 2 contrastive conv1x1 physics rows | 6-7 / 8 | 6.50 |
| this session, 2 FM physics rows | 7-8 / 8 | 7.50 |

Contrastive conv1x1 beats 8 of those 10 FM arms and ties the best of them.

**The load-bearing finding is structural, not the scores.** The two families retrieve
**near-disjoint** neighbourhoods — 6/8 shared galaxies *within* a family, only **1/8 across** them,
19 unique galaxies filling 32 row-slots. So the contrastive and generative physics latents surface
almost completely different candidates, and *both* are judged lens-like at similar rates. Rare-object
retrieval is therefore **not specific to the flow-matching objective**, and the two latents are
neither redundant nor rankable on this evidence.

**Exposure axis.** Both zero-shot arms outscored their full-data twins (8>7 for FM, 7>6 for
contrastive) — consistent in direction across 2/2 families, though noise-level at n=1 query. The
direction supports the claim in `reviews/claude-suggestions/response_2_JcH2_q4_PASTE.md` that
retrieval does not depend on having seen the lens in training.

## Caveats that bound all of the above

1. **The metric saturated.** All four rows scored 6-8/8 against a ceiling of 8, leaving ~2 panels of
   usable dynamic range. A 1-panel gap sits inside it.
2. **No floor row in this deck** (deliberate scope cut, PLAN.md §8). Last session's two
   instrument-space rows scored 0/8, and that is what demonstrated the scoring was discriminative
   rather than "bright early-type with a halo -> lens". The `base-87k` anchor argues the criterion
   held today, but it is a single row, so a looser criterion cannot be fully excluded.
3. **`base-87k` is not an independent observation** — its panels are pixel-identical to a row already
   scored 7/8, so its score may be memory-anchored. It agrees with last session either way.
4. **n=1 query, n=1 scorer.** No error bar is claimable.

## What this does and does not license

- **Not supported:** that the generative model retrieves rare morphology *better* than the
  contrastive baseline.
- **Supported:** that the retrieval property is *robust* — both architectures, with and without lens
  exposure, all >= 6/8 on a query that 73/74 catalog lenses' holdout excludes from two of the arms.

## Cheapest paths to a defensible number

1. **LOO recall@k over all 74 catalog lenses — zero GPU** (latents already on disk). Removes both the
   saturation and the single-query problem at once. This is what the draft response already promises
   for the revision.
2. **Add the floor rows** (2 instrument-space + random-init) to this deck, ~10 GPU-min, so a null
   result becomes interpretable.
