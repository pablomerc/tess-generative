# Lens-Retrieval Recall vs Data Scale — Implementation Plan

**Status:** for Pablo's review. Fig-5 reproduction submitted (job 18933414); the sweep below is
NOT submitted and is blocked on one file (§1).

**Purpose:** answer reviewer **JcH2** on rare-object discovery as a function of survey size. Their
words, verbatim:

- Q2: *"A quantification of 'how rare of phenomena can be detected by the physics encoder' would
  strengthen the discussion and motivating usecase for followup observations on better instruments,
  and also motivate seed survey size requirements (for sampling rare phenomena)"*
- S4.1 weakness: *"how 'common' does a rare astrophysical event / source need to be to be trained
  into the encoder's latent space? Can you quantify this? Does the architecture have a 'resolution'
  for rare phenomena in terms of training data?"*

This extends the paper's own most-praised demo (Fig. 5, which JcH2 called *"extremely motivating"*)
along the scale axis. That is a better answer than anything R²-based, because lenses are a genuine
rare class rather than a property regression.

---

## 0. Is recall@k the right metric? Yes, with three corrections

**Correction 1 — a single query cannot carry a curve.** With one query and ~7 relevant items, the
difference between 5/7 and 7/7 across arms is within draw noise. Fix: **leave-one-out over all known
lenses in the gallery.** Every known lens takes a turn as query; we score retrieval of the
*remaining* lenses. Same latents, ~L× more signal for zero extra encoding. LOO-mean-recall@k becomes
the curve; the Fig-5 query stays as the headline anecdote panel that ties back to the paper.

**Correction 2 — precision vs recall bookkeeping.** The paper's "7 of top 8" is *precision@8*. Your
"7/7" phrasing is *recall* against the other known lenses. These coincide only if the gallery holds
exactly 8 known lenses. It almost certainly holds more (`search_lens_neighbors.py`'s default index
list runs to 70), so the two numbers will differ and must be reported separately. I will emit both.

**Correction 3 — report the confound, don't fight it.** Subsets are nested and random, so each arm
trained on some number of known lenses (possibly zero at 1k). That is not noise; it is the answer to
*"how common does a rare event need to be."* Report it as a second row under the curve.

Two outcomes, both publishable:
- Small arms saw **0** lenses yet still retrieve them → *rare classes need not appear in training to
  be retrievable; the physics space generalises to unseen morphology.* Strongest possible answer.
- Recall tracks lenses-seen → you have measured the rarity threshold directly.

**Also report median rank**, because recall@k saturates: once several arms hit 7/7 at k=20, recall
can no longer separate them, while median rank still can.

---

## 1. ⛔ BLOCKER — one file, and it is small

The ground-truth lens list lives only on the AMD cluster:

| File | Needed for | Status |
|---|---|---|
| `/work1/jeroenaudenaert/pablomer/data/lens_indices.txt` | **the lens ground truth** (`hsc_object_id → absolute_index`) | ❌ AMD only |
| `lense_reconstruction/lens_reconstruction_dataset.h5` | convenience (query stamps); rebuildable from the txt | ❌ AMD only |
| `/orcd/pool/007/pablomer/neighbours_v2.h5` | gallery images + ids | ✅ here (624 GB) |
| `/orcd/pool/007/pablomer/efficient_neighs_48/` | pre-baked 48×48 store | ✅ here (17 GB) |

**Please copy `lens_indices.txt` over** (`create_lens_dataset.py:6` is the reference: a TSV with
columns `hsc_object_id`, `absolute_index`). It is a few KB. Everything else is reconstructible: the
lens h5 was itself built by joining that txt against `neighbours_v2.h5`, which we have.

**Without it** we can still do the Fig-5 *reproduction* (job 18933414) — because
`check_objects/objects.json` records the exact query and neighbour rows — but we cannot label which
retrieved galaxies *are* lenses, so no recall number.

**The `lens_indices_zero_based: [4, 7, 11, 17, 19, 28, 31, 32]` in the `*_dwnval` configs does not
substitute.** Those are 0-based indices *into the lens h5*, not object ids, and are meaningless
without it. They do tell us 8 lenses were used for lens-validation, which is likely a curated subset
of the ~70.

---

## 2. Verified ground truth for the reproduction

The paper query **`42692125246129290`** is **"Lens 48"**, raw row **183713** of `neighbours_v2.h5`
(RA 352.9561, Dec +0.5864 — I verified id, row, and coordinates all agree with
`check_objects/objects.json`).

Recorded neighbours from that file (displayed rank → row):

| #1 | #3 | #4 | #5 | #6 | #8 |
|---|---|---|---|---|---|
| legacy 183713 | hsc 187190 | hsc 15375 | hsc 181039 | hsc 40202 | hsc 23615 |

⚠️ **A discrepancy to settle.** You remembered *"7 of top 8 were lenses, ranks #1,#2,#3,#4,#5,#7,#8"*.
The recorded figure config (`lens_final_figure.py:53-56`) shows lens 48 displaying six panels at
ranks **#1,#3,#4,#5,#6,#8**, and lens 33 displaying **#1,#2,#3,#6,#7,#8**. Neither is your
seven-item list. Possibilities: the "7 of 8" refers to a different/earlier run, or to lens 33, or the
figure showed a subset of a larger hit list. The reproduction will settle it empirically — but note
**the paper text never states "7 of 8"**; tex L464 only says *"The top eight nearest neighbors
successfully retrieve multiple HSC images that exhibit..."*. So the rebuttal should quote whatever we
measure, not the remembered number.

**Rank-convention subtlety** (already handled in the code): the original encoded the query from the
lens h5 while the gallery came from `neighbours_v2.h5`, so the query's own gallery entry returned at
a small but nonzero distance and survived the `dist < 1e-6` self-filter as `all_nns[0]`. We take the
query embedding straight from the gallery, so self-distance is exactly 0 and filters cleanly —
our index `i` corresponds to their displayed `#(i+1)`.

---

## 3. Leakage audit (already done — this is what makes the metric credible)

| Galaxy | in holdout? | in val? | 1k | 3.2k | 10k | 31.6k |
|---|---|---|---|---|---|---|
| **query 42692125246129290** | **YES** | no | — | — | — | — |
| NN#3 42692675001945884 | no | no | — | — | — | **IN** |
| NN#4 37489227633550294 | no | **YES** | — | — | — | — |
| NN#5 42687864638564440 | no | no | — | — | — | **IN** |
| NN#6 38558640130523161 | no | no | — | — | — | — |
| NN#8 37490078037069717 | no | no | — | — | — | — |

The query sits in the **downstream holdout**, so it was excluded from *all four* arms' training —
retrieval of it is genuinely zero-shot for every arm. Two of its neighbours entered training only at
31.6k. I will extend this audit to the full lens list once the txt arrives.

---

## 4. Design

**Latent space.** `encoder_1` (physics), flattened `(B,4,16) → (B,64)`. Gallery is the combined
`[hsc_physics ; legacy_physics]` = 207,482 rows, L2/euclidean via brute force (`np.argsort` — exact,
seconds at this size; no need for sklearn/faiss). Self-matches filtered at `dist < 1e-6`.

**Instrument-space control (you asked for it, and it is free).** `encode_gallery.py` computes
`encoder_2` in the same pass. Physics space should retrieve lenses; instrument space should not. If
instrument-space recall ≈ chance while physics-space recall is high, that is a direct, quantitative
restatement of the paper's central disentanglement claim on a rare class — a strong rebuttal exhibit.

**Checkpoints (10, as you specified).**

| # | Label | Path |
|---|---|---|
| 1-4 | `scale-{1000,3162,10000,31622}-75k` | `checkpoints_new/scale_ablation/frozen/scale_N/…-75k.ckpt` |
| 5-8 | `scale-N-bestmse` | `snapshots/scale_N/bestmse_step*.ckpt` |
| 9 | `base-87k` (paper) | `checkpoints_new/base/snapshot.ckpt` |
| 10 | `fm-control-75k` | `diffusion_ablation/frozen/fm-control-75k-fixed.ckpt` |

Note 9 and 10 trained **without** the holdout, so they saw the eval lenses as unlabeled images. That
biases them *upward*, which is conservative for the scaling claim (it can only widen the gap between
the small arms and full data). Label the rows.

**Floor.** `untrained-resnet-matched` (dim-matched random init) gives the chance level. Recall@k for
a random ranking is ≈ k·(L−1)/N, i.e. ~0.1% at k=20 — so any real signal will be enormous relative
to it, and the floor mostly serves to make that explicit.

**Metrics per (checkpoint, space):**
- `recall@{8,10,20}` — LOO mean over all in-gallery lenses, ± SEM across queries
- `precision@8` — comparable to the paper's framing
- `median rank` of relevant items — does not saturate
- `fig5_query_recall@k` — the single-query anecdote for continuity with the paper

---

## 5. The efficiency decision that makes this cheap

`make_latents_all.py` (the original path) is unusable here: it drives `NeighborsSimpleDataset`, which
loads **~60 metadata scalars per item** — ~6.2M extra random HDF5 reads over a full gallery pass. I
wrote a lean `encode_gallery.py` that reads only the two image datasets.

Two image sources, and the choice matters for cost:

| Source | What | Per-checkpoint cost | Fidelity |
|---|---|---|---|
| `h5` | raw 160×160 from `neighbours_v2.h5`, preprocessed live | ~96 GB of reads | bit-faithful to the paper |
| `ram48` | pre-baked 48×48 float16 store | 17 GB **loaded once**, then pure GPU | fp16 rounding of the same pixels |

The RAM48 store already *is* the preprocessed gallery — the same one training used. So the sweep
should use `ram48` and load it **once** for all 10 checkpoints, rather than re-reading ~1 TB.

Job 18933414 runs base-87k through **both** sources precisely to check they agree on ranks. If they
do, the sweep is cheap and its fidelity is established. If they disagree, we fall back to `h5` and
pay more.

---

## 6. Runtime estimates

Honest confidence levels — the encode throughput is the one number I have not yet measured, which is
why job 18933414 prints img/s for both sources.

| Stage | Estimate | Confidence |
|---|---|---|
| RAM48 store load (once per job) | 4-8 min | high — measured before |
| Encode 103,741 anchors × 2 surveys, `ram48` | **2-5 min/ckpt** (GPU-bound, ResNet18 on 48×48) | medium |
| Encode, `h5` source | **20-45 min/ckpt** (~96 GB Ceph reads) | low — the number 18933414 will pin down |
| kNN + all metrics, per ckpt | < 1 min (207k × 64, brute force) | high |
| Figures + Discord | ~1 min | high |

**Sweep total, `ram48`, all 10 checkpoints in ONE job** (store loaded once):
> **~35-60 min wall on one GPU.** ≈ 8 min load + 10 × ~4 min encode + 10 × 1 min metrics.

**If forced onto `h5`:** ~4-8 h, better split into 2-3 jobs.

**Storage:** 10 × 207,482 × 128 dims × 4 B ≈ **1.1 GB** total. Goes on `/orcd/data/mki_aryeh/001`
(200 TB, 1% used) — no POOL pressure.

So the whole sweep is roughly one hour of one GPU. The expensive part was never the compute; it was
getting the ground truth.

---

## 7. Deliverables

1. **Curve** — LOO mean recall@{10,20} vs N (log x), physics vs instrument space, with the RandInit
   floor and the base-87k reference. Second panel: median rank vs N.
2. **Table** — per arm: recall@8/10/20, precision@8, median rank, **plus "known lenses in this arm's
   training set"** (the confound-as-finding row from §0).
3. **Anecdote panel** — the Fig-5 query retrieved by each arm, side by side. Visually shows the
   result degrading (or not) with scale; the most reviewer-legible artifact.
4. **Rebuttal paragraph** → `reviews/claude-suggestions/` (never into `reviews/` directly).

---

## 8. Open questions for you

1. **Can you copy `lens_indices.txt`?** Everything in §4-§7 is blocked on it. (§1)
2. **Which lens set is the ground truth** — all ~70 in the lens h5, or the curated 8 from the
   `*_dwnval` configs? I would use all ~70 for the denominator and note the 8 separately, but you
   know their provenance and whether all 70 are confidently lenses vs candidates. This directly sets
   what "recall" means.
3. **The "7 of 8" discrepancy in §2** — do you recall whether that came from lens 33, lens 48, or an
   earlier run? Not blocking (we quote what we measure), but it affects how we phrase continuity with
   the paper.
4. **Should the sweep include the dense step-ladder** (15 snapshots/arm) for a recall-vs-*training-
   step* curve, not just recall-vs-*data*? Cheap on `ram48` (~4 min each, so ~4 h for all 60), and it
   would show whether the 1k arm's retrieval peaks early and then degrades — the same overfitting
   question your val/loss-vs-val/mse observation raised. My default: **no**, keep scope tight, decide
   after seeing the 10-checkpoint curve.
