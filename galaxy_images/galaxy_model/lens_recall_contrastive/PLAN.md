# Blinded Fig-5 Reproduction for the Contrastive conv1x1 Baseline

**Status:** reviewed and **executed** — see §10 for what was actually run and §11 for follow-ups
that were deliberately left out. No existing file was touched.
**Goal:** the same blinded, Discord-scored lens-retrieval decks we ran for the scale ablation
(`scale_ablation/lens_recall`), but for the contrastive dual-encoder — so we can say whether a
*contrastive* dual encoder surfaces rare morphology the way the *generative* one does.

---

## 0. TL;DR

Reuse the `scale_ablation/lens_recall` machinery unchanged (import it, don't edit it), add a
model-agnostic loader so it can also load contrastive checkpoints, and emit blinded A/B/C… decks
for Discord exactly as before.

The one thing I want your call on before submitting: **which lens queries to use**, because two of
the candidate rows have already been shown to you in the `lens48` deck, and you have seen that key.
See §4.

---

## 1. What I verified first (all of it holds)

| Claim | Status |
|---|---|
| Contrastive checkpoints load and expose a physics/instrument split | ✅ tested on CPU, 3 of them |
| Physics latent is **dimension-matched** to the FM model | ✅ e1 = e2 = **64** for all three |
| Same image store, so pixels are identical to the FM arms' inputs | ✅ both trained on `efficient_neighs_48` |
| Lens ground truth is available on Engaging | ✅ `/orcd/pool/007/pablomer/lens_indices.txt`, 74 lenses |
| All 74 catalog lenses are in the retrieval gallery | ✅ all 74 present, all `source_type==0` anchors |
| The Fig-5 query is one of the 74 | ✅ `42692125246129290` → row 183713 → legacy `3528p005-10026` |

**Dimension match, in detail.** `conv1x1` at 48×48 gives a 2×2 ResNet18 feature map, and
`token_dim = embedding_dim // 4 = 16`, so `encoder_galaxy` emits `(B,16,2,2) → (B,64)`. The FM
model's `encoder_1` emits `(B,4,16) → (B,64)`. Same width, so kNN geometry is compared on equal
footing rather than across latents of different size. This is luck we should state, not engineer.

**Loader mapping** (the canonical adapter already exists in
`downstream_evaluation/engaging/prepare_combined.py:164-177` — I will reuse its convention exactly):

| Arm family | Class | Module |
|---|---|---|
| the 2×2 sweep (`contrastive-{spatial,random}-{avg,conv1x1}`) | `DualEncoderContrastiveModule` | `contrastive_baseline/dual_encoder_contrastive.py` |
| the `-DWNVAL` arms | `ContrastiveNegativeAblationModule` | `contrastive_ablation/dual_encoder_contrastive_variants.py` |

`e1 = encoder_galaxy` (physics), `e2 = encoder_instrument`. **Raw encoders, not the projection
heads** — same as the downstream probes, so the deck and the R² table describe the same vectors.
The `-DWNVAL` checkpoints carry three extra hparams (`galaxy_negatives`, `instrument_negatives`,
`use_projection_head`), which is why they need the subclass; loading them with the base class is
the one way to get this silently wrong.

---

## 2. The finding that should shape the design: a matched 2×2 is sitting right there

I checked whether the contrastive arms ever saw the lenses, and the answer is cleaner than expected.

`contrastive_ablation/holdout_legacy_ids.txt` is **byte-identical** to
`downstream_evaluation/engaging/outputs/index/holdout_legacy_ids.txt` (5,361 legacy ids). So the
contrastive `-DWNVAL` arms and the FM `-DWNVAL`/scale arms excluded *exactly the same galaxies*.
And:

> **73 of the 74 catalog lenses — including the Fig-5 query — are in that holdout.**

So the `-DWNVAL` arms are genuinely zero-shot on essentially the entire lens catalog, and the
non-DWNVAL arms saw those same lenses as unlabeled images. That gives a fully matched 2×2 with no
extra training:

|  | **saw the lenses** (unlabeled) | **never saw the lenses** (zero-shot) |
|---|---|---|
| **generative (FM)** | `base-87k` — the paper model | `base-5NB-DWNVAL` |
| **contrastive** | `contrastive-spatial-conv1x1` | `contrastive-spatial-conv1x1-DWNVAL` |

All four checkpoints exist on Engaging. This is a much stronger exhibit than contrastive-alone,
because it separates *architecture* from *exposure* in one blinded deck, and it answers JcH2's
"how rare can a phenomenon be" on the architecture axis too.

I'd build the deck around this 2×2 rather than around the conv1x1 arm in isolation. Same cost.

---

## 3. Deck design

Identical mechanics to `make_scoring_deck.py`: top-8 neighbours per row, query's own cross-survey
counterpart excluded, rows shuffled and relabelled A/B/C…, key written to a separate file I don't
post, blank scoresheet generated, pages pushed to the Discord webhook.

**Recommended core deck — 8 rows / query** (2 pages of 4, so the stamps stay large on Discord):

| # | Arm | Space | Role |
|---|---|---|---|
| 1 | `contrastive-spatial-conv1x1-DWNVAL` | physics | **the ask**, zero-shot |
| 2 | `contrastive-spatial-conv1x1` | physics | **the ask**, saw lenses |
| 3 | `base-5NB-DWNVAL` | physics | matched FM, zero-shot |
| 4 | `base-87k` | physics | matched FM, the paper model |
| 5 | `contrastive-spatial-avg` | physics | is `conv1x1` the reason? (pooling contrast) |
| 6 | `contrastive-spatial-conv1x1-DWNVAL` | instrument | disentanglement control, expect ~0 |
| 7 | `base-5NB-DWNVAL` | instrument | disentanglement control, expect ~0 |
| 8 | `untrained-contrastive-conv1x1` | physics | random-init floor, expect ~0 |

Rows 6-8 are the reason the deck is interpretable: three rows that *should* score ~0. If they do,
your scoring is discriminative and the physics-row numbers mean something. If they don't, we learn
that before writing any of it into a rebuttal. (In the `lens48` deck both instrument rows scored
0/8, so this control has already worked once.)

**Optional +4 rows** if you want the full mechanism sweep in the same session:
`contrastive-random-conv1x1` physics (neighbour-semantics ablation), `contrastive-random-avg`
physics, `contrastive-nohead-DWNVAL` physics (projection-head ablation), and
`contrastive-spatial-conv1x1` instrument. Costs ~10 extra GPU-minutes and 4 more rows of your time.

---

## 4. ⚠️ The one real problem: blinding is already partly spent on `lens48`

You scored the `lens48` deck and the key is decoded, so you know `base-87k` physics scored 7/8
there. If I put `base-87k` physics into a new `lens48` deck, **the panel images are pixel-identical
to the row you already scored** — you'd recognise it, and every other row would get anchored to a
number you already know. That is exactly the bias the blinding exists to remove.

Three ways out, in the order I'd pick them:

**(a) Fresh queries — recommended.** 73 of the 74 catalog lenses have never been used as a query.
Pick 2 of them, and every row in the deck is material you have never seen. Blinding fully intact,
and the FM-vs-contrastive comparison happens *within* one deck and one scoring session, which is
the comparison that actually matters. Cost: it is no longer literally "the Fig. 5 query".

**(b) `lens48` with new arms only.** Drop `base-87k` and use `base-5NB-DWNVAL` as the sole FM
reference (never shown). Keeps the paper query for continuity; the FM-vs-contrastive contrast is
still within-deck. Residual risk: you have seen 12 rows' worth of `lens48` neighbours already, so
some individual *stamps* will look familiar even in new rows.

**(c) Both.** `lens48` under option (b) for continuity with the paper, plus 2 fresh queries as the
clean measurement. This is what I'd actually ship if you have the scoring patience — the fresh
decks carry the claim, `lens48` ties it to Fig. 5.

Note the previously-built `lens33` deck was **never scored**, so `lens33` is still clean and is a
free third query under any option.

---

## 5. Step 0: you pick the queries from a contact sheet

The paper used `lens48` and `lens33` presumably because they are visually striking. A randomly
drawn catalog lens may be unremarkable at 48×48, and a query you can't confidently judge makes the
whole deck unscoreable.

So before any GPU work I'd send you a **single contact sheet of all 74 catalog lenses** (HSC and
Legacy stamps, numbered), and you pick the 2-3 you're willing to score against. Costs no GPU and
~2 minutes of Ceph reads. It also doubles as the figure we'd need anyway to state what the lens
ground truth actually is.

---

## 6. Code — five new files, zero edits to existing code

Everything lives in `galaxy_model/lens_recall_contrastive/` and imports the working machinery from
`scale_ablation/lens_recall/` rather than copying it. I deliberately do **not** edit
`encode_many.py` / `make_scoring_deck.py`, so the already-posted scale results stay reproducible
byte-for-byte.

| File | What |
|---|---|
| `lens_catalog.py` | joins `lens_indices.txt` → `neighbours_v2` rows via `object_id_hsc`, tags holdout membership, renders the §5 contact sheet. Ignores the `absolute_index` column — it does not index `neighbours_v2` (max 99,695 vs row 183,713), and the id join is unambiguous. |
| `encode_arms.py` | imports `_Ram48Gallery`, `_collate`, `anchor_rows_and_mapping`, `check_not_degenerate` from `lens_recall/encode_gallery.py`; adds `--arm LABEL:MODULE:CLASS:CKPT` (importlib, per `prepare_combined.py`) and the `encoder_galaxy`/`encoder_1` adapter. `--random-init` builds the floor arm from a real checkpoint's hparams with a fixed seed and no weights. Store loads **once** for all arms. |
| `make_deck.py` | imports `topk` and `rgb` from `lens_recall/make_scoring_deck.py`; takes an explicit `--arm LABEL:SPACE` list and an arbitrary `--query-row`, so composition is declared rather than "every `.h5` in the directory". Writes `KEY.json` / pages / scoresheet under a `--out-prefix`. |
| `run_decks.slurm` | `pg_mki_aryeh`, 1×H200, 96 G, ~1 h. Encodes, builds decks, posts pages to the Discord webhook with start/finish pings. Modelled on `run_scoring_sweep.slurm`. |
| `decode_scores.py` | you reply `A: 1,3,4` / `B: none` on Discord; this maps letters → arms and emits the results table. The scale run decoded this by hand; worth having as a script this time. |

**Correctness guards carried over, because they each already caught a real bug:**
- identity stays in `neighbours_v2` row space; the RAM48 catalog is used *only* to fetch pixels
  (the two stores agree on only 4.8% of rows — the row-space trap in `encode_gallery.py`'s docstring);
- `check_not_degenerate` on every arm's `hsc_physics`;
- the 64-sample blank-frame guard in `_Ram48Gallery`;
- new: assert every arm's `e1` width is 64, so an `avg`-vs-`conv1x1` mixup can't pass silently.

---

## 7. Compute and storage

| Stage | Cost |
|---|---|
| Contact sheet (§5) | no GPU, ~2 min |
| RAM48 store load, once per job | 4-8 min, ~35 G RAM |
| Encode per arm (103,741 anchors × 2 surveys, 48×48 ResNet18) | 2-5 min |
| 6 new arms core / 10 extended | ~30 min / ~50 min |
| kNN + deck render per query | < 2 min |
| **Total, one job, one GPU** | **~45-60 min** |

`base-87k` latents are already cached in `scale_ablation/lens_recall/latents/`, so that arm is free.

**Storage — this needs care.** POOL is at **92%** (941/1024 G) and SCRATCH is **100%** full. Each
gallery is 108 MB, so 6-10 new arms is 0.65-1.1 G. I'd put latents on
`/orcd/data/mki_aryeh/001/pablomer/lens_recall_contrastive/` (200 T, 1% used) and symlink, exactly
as `scale_ablation/snapshots` does. Only the decks (~2 MB/query) land on POOL.

---

## 8. Decisions taken

Pablo's call, 2026-07-27 — scope cut hard in favour of a fast read on whether the gap is large:

| Question | Decision |
|---|---|
| Query set (§4) | **`lens48` only.** No fresh queries, no contact sheet. |
| Deck size (§3) | **4 rows**: 2 contrastive conv1x1 + their 2 matched FM counterparts. No instrument controls, no random-init floor. |
| LOO recall (§9) | **No.** Decks only. |

Rationale in his words: *"i just wanna see quickly if there is large differences."* So this is a
direction-finder, not the rebuttal exhibit. Two consequences to hold onto when reading the result:

- **No control rows.** The scale deck's interpretability came from three rows that should score ~0
  (two instrument-space, one random-init). Without them, a null result ("all four rows look
  similar") is ambiguous: it could mean the architectures really are similar, or that this query is
  too easy to separate anything. A *large* difference, which is what we are looking for, is still
  readable. If the four rows come out close, the controls are the first thing to add.
- **`base-87k` is not independent.** It was row D of the earlier `deck_lens48` and scored 7/8, and
  its panels here are pixel-identical to the ones already seen (verified — see §10). Its score in
  this deck may be anchored by memory. Recorded in the deck's KEY `note` field so it cannot quietly
  become a reported number.

---

## 9. Bonus now unblocked (separate from the ask — tell me yes or no)

`scale_ablation/lens_recall/PLAN.md` §1 listed `lens_indices.txt` as a ⛔ blocker for the
*quantitative* metric. **It is here** (`/orcd/pool/007/pablomer/lens_indices.txt`, 74 lenses, all
verified in-gallery). So the leave-one-out `recall@k` curve that plan wanted is now computable —
and the scale-arm latents are already on disk, so it costs **no GPU at all**, just brute-force kNN
on 207,482×64.

That would give the reviewer a real number with an error bar instead of one anecdote, and it
directly addresses the caveat your current draft response already concedes:

> *"This is a single query, so we do not read a scaling curve from the variant-to-variant
> differences. For the revision we will report leave-one-out mean recall@k over all known lenses."*

With the holdout join from §2 we can also report, per arm, exactly how many known lenses it
trained on — the confound-as-finding row. Two caveats to keep it honest: the catalog lenses are
Euclid-Q1 *candidates*, and none of the recorded Fig-5 neighbours are in the catalog, so
`recall@k` against the catalog measures something narrower than the visual "lens-like" judgement
the decks measure. The two metrics are complements, not substitutes.

Declined for now (§8). It stays available at any time — the latents it needs are already on disk.

---

## 10. What was actually run

**Job 19017525**, `run_decks.slurm`. Four rows, all **physics** latent, on query `lens48`
(`neighbours_v2` row 183713):

| Arm | Family | Saw the lenses? | Checkpoint |
|---|---|---|---|
| `contrastive-spatial-conv1x1` | contrastive | yes, unlabeled | `best-epoch=53-step=83000` |
| `contrastive-spatial-conv1x1-DWNVAL` | contrastive | **no — zero-shot** | `best-epoch=48-step=70000` |
| `base-87k` | flow matching (paper) | yes, unlabeled | `base/snapshot.ckpt` (cached latents) |
| `base-5NB-DWNVAL` | flow matching | **no — zero-shot** | `best-epoch=197-step=75000` |

Checkpoint choices are the canonical ones recorded in each arm's downstream-embedding `attrs`, so
the deck describes the same weights as the R² tables. `base-87k` reuses the scale sweep's cached
latents, which is what keeps its number comparable to the earlier session.

**Verification done before submitting:**

1. All three new checkpoints load on CPU and emit `e1 = e2 = 64` dims — matching the FM physics
   latent width, so no arm is silently compared at a different dimensionality. `encode_arms.py`
   also asserts width 64 at write time.
2. `make_deck.py` was run against the cached `base-87k` latents and its top-8 came out
   **bit-identical** (same rows, same distances to 5 dp) to row D of
   `scale_ablation/lens_recall/decks/deck_lens48_KEY.json`. So the new deck builder's retrieval and
   rendering match the already-scored decks exactly, and scores are comparable across sessions.

**Where things live:** latents on `/orcd/data/mki_aryeh/001/pablomer/lens_recall_contrastive/`
(symlinked as `./latents`) — POOL is at 92% and SCRATCH is full, so only the ~1 MB deck PNGs land
on POOL. The 17 GB RAM48 store is read from the group-storage copy staged by job 19010433.

**Partition:** `mit_normal_gpu`, one L40S, not `pg_mki_aryeh` — node4900's 8 H200s are fully
allocated to `cf-fm-long` and `scale-100`, which are not worth evicting three days out from the
deadline. This job needs ~20 s of GPU per arm; it is entirely I/O-bound.

**Outcome:** COMPLETED in **17:19** on node4105, 34.9 GB peak RSS. The I/O-bound diagnosis held —
6.3 min to load the store from group storage, 6.9 min for the single 2.2 GB FM checkpoint, and
**~10 s of actual GPU work per arm** (10-12k img/s). All six degeneracy checks passed; the
row-mapping guard passed 64/64. The group-storage mount is present on `mit_normal_gpu` nodes, so
the fallback to POOL was not needed.

**Post-render integrity check** (`decks/deck_lens48_contrastive_KEY.json`): 4 distinct arms, all
physics space, no two rows sharing a neighbour list, 19 unique galaxies surfaced across the 32
row-slots. Structural note held back until scoring is done, because it would reveal the pairing:
the four rows fall into **two tight clusters of two** — 6/8 shared galaxies within a cluster,
**1/8 across** them. So the two model families retrieve nearly disjoint neighbourhoods, while the
zero-shot/full-data variants of one family largely agree. That is why this query can show a large
difference at all, and it should be reported alongside whatever the scores say.

---

## 11. Follow-ups this deliberately does not do

In rough order of value, if the 4-row read says the gap is worth pursuing:

1. **Add the control rows** (2 instrument-space + random-init floor). Without them a null result is
   uninterpretable; ~10 GPU-min.
2. **Fresh queries** from the 72 unused catalog lenses, so blinding is fully intact and `base-87k`
   is no longer a memory-anchored row.
3. **LOO recall@k** over all 74 lenses (§9) — zero GPU, turns one anecdote into a number with an
   error bar, and is what the draft response already promises for the revision.
4. **The mechanism arms** (`contrastive-random-*`, `contrastive-nohead-DWNVAL`,
   `contrastive-spatial-avg`) if the conv1x1-vs-FM gap turns out to be real and we want to know
   which part of the contrastive recipe causes it.
