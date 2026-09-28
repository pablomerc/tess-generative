# Hyrax (arXiv:2605.18959) — what to read, and what to steal

Ghosh, Oldag, Tauraso, Connolly et al., *Hyrax: An Extensible Framework for Rapid ML Experimentation and
Unsupervised Discovery in the Era of Rubin, Roman, and Euclid*, 18 May 2026, 28 pp.
Repo: https://github.com/lincc-frameworks/hyrax (MIT, `pip install hyrax`, Python 3.11-3.13).

Notes written 2026-09-16 for the HSC x Legacy x LSST DP2 crossmatch and the physics/instrument
disentanglement work. Every idea below was checked against both the paper text and this repo's own files;
scores and caveats come from an adversarial verification pass, so the caveats are the load-bearing part.

**Bottom line: take the ideas, do not adopt the framework.** See the verdict at the end.

**Version warning.** The paper freezes v0.8.1; PyPI is at 0.12 with materially different internals
(DataProvider joins, `[balance]`, `save_to_database`, a `to_onnx` verb). Read the *package* for capability
questions and cite only the *paper* for ideas — several of the most attractive features are code-only.

---

## Section ranking (most useful first)


### 1. §3.1 Unsupervised Discovery & Representation Learning in Rubin DP1, especially §3.1.2 Latent-Space Exploration Results (L975-1186)
*pp.12-16 (Figs 11-14)*

This is the only part of the paper whose findings land directly on your two live problems. (a)
Instrument artifacts self-segregate into a latent island, and removing them and retraining
reveals a second generation — your own DP2 follow-up already found NSF/IForest tops are near-
empty n_input_i=1 stamps, i.e. the same phenomenon without the loop. (b) Validation loss does
not track representation quality, which is exactly how your 7-run AE/FM-AE/JEPA matrix is about
to be ranked (val MSE 0.114 COSMOS vs 0.341 DEEP2-3). It also supplies the crossmatch-overlay-
then-expand pattern, which you can run with instrument scalars in place of morphology labels.

**Read:** L975-995 plus Fig. 11 (p.13) for the peel-and-retrain loop. L1054-1065 plus the Fig. 12 caption
at L996-1002 (pp.13-14) for the non-monotonic loss result — look at the three panel titles, they
carry the losses the text never states. L1023-1053 (p.14) for overlay-an-external-catalog-then-
expand-by-similarity. The single sentence at L967-971 (p.12) on magnitude-stratified subsamples,
which is a cheap depth-confound control. L1037-1042 + footnote 13 for the greedy chi^2 matcher
(photometry-aware, with the LSST meas_astrom module named). Skip the merger/LSB narrative at
L1086-1135 unless you want the seed-and-expand recipe verbatim.

*Feeds ideas:* mask-ivar-leak, leak-as-training-metric, tri-view-consensus-anomaly, metadata-manifold


### 2. §3.5 Semi-resolved dwarf galaxies (L1538-1622)
*pp.20-21 (Fig. 20)*

Source of the single highest-value idea on this list. It is the paper's only true causal
intervention: wrap a pixel-level generator in a dataloader and manufacture ground truth instead
of waiting for labels. Equally important is how the result is reported — a 2-D detection-
fraction surface over physical parameters rather than a scalar accuracy — which is precisely the
shape your depth story needs (n_input_i x psf_fwhm_i instead of size x luminosity). One detail
nobody flags: they run HSC coadds through the LSST Science Pipelines, which is a free ablation
on whether your 'instrument' factor is optics or reduction software.

**Read:** L1576-1591 (p.21) for the injection-wrapped-in-a-dataloader construction and the HSC-through-
Rubin-pipelines detail. L1596-1608 plus the right panel of Fig. 20 (p.21) for the detection-
fraction map with constant-surface-brightness lines and a literature limit overlaid. L1606-1608
for framing false-positive rate as a human-inspection budget. L1613-1617 for the one clause
about differentiable diffusion injectors — read it as aspiration, not a roadmap.

*Feeds ideas:* ivar-ladder, depth-recall-map, depth-counterfactual-render


### 3. §2.5 Interactive Latent-Space Exploration (L670-834)
*pp.9-11 (Figs 9-10)*

The paper states, as the design rationale for a tool, the exact judgement you currently make by
eye from static UMAP PNGs: is this cluster or outlier island a real population or an instrument
systematic. Read it as much for what it cannot do — no linked brushing across two latent spaces,
six random stamps of one modality per selection, scalar colour-by only — because those gaps are
precisely where your three-view crossmatch beats the framework.

**Read:** L690-695 (p.9) for the rationale sentence, which is quotable in your own write-up. L713-762 for
the input contract (embeddings + persistent IDs + metadata, UMAP-reduced) and the region-select
-> object IDs -> catalog table -> six stamps loop, including 'any cross-matched properties
provided by the user'. L777-787 (p.10) for the DataProvider coupling (display any modality the
provider serves) and DataFrame export. Skip the three.js/3D material at L794-834.

*Feeds ideas:* three-view-brush, metadata-manifold


### 4. §2.4 Vector Database and Similarity Search (L557-669)
*pp.7-9 (Fig. 8)*

search_by_vector() plus the Fig. 8 layout (query column, neighbour columns) is the template for
both your cheapest quantitative disentanglement metric and your most legible figure: physics-
latent neighbours in one row, instrument-latent neighbours in the row below, same query galaxy.
Read it for the workflow and ignore the infrastructure — the paper's own numbers (milliseconds
at tens of millions of records) show your 65k sample sits ~150x below the regime HNSW is built
for, so a chunked numpy matmul is the right implementation.

**Read:** L602-624 (pp.7-8) for the analog-finding framing and the one query verb. L634-642 (p.8) for the
user-defined metric (cosine is named — do not claim the paper is silent on this) and the scale
claim. The Fig. 8 caption L649-667 and the panel on p.9 for the gallery layout. Skip the
LanceDB/ChromaDB/Qdrant backend comparison entirely.

*Feeds ideas:* depth-recall-map, tri-view-consensus-anomaly, three-view-brush


### 5. §3.4 Distant TNOs / KBMOD (L1415-1537)
*pp.19-20*

Three transferable moves, all aimed at a stage you have and has not finished: a deliberately over-
triggering high-recall detector followed by a separate learned purity gate; a negative class
manufactured by breaking a physical prior rather than by labelling; and results reported as an
explicit filtering budget. Your anomaly stage (flow NLL + IsolationForest + recon MSE) is all
recall, has no purity gate, and its current top-100 is depth artifacts.

**Read:** L1457-1460 (p.19) for the 10:1-20:1 false-positive framing. L1494-1507 (p.20) for the
85-degrees-off-ecliptic negative-class trick and the augmentation list — note the one-pixel
origin shift, which is directly relevant to your pixel-parity preprocessing. L1508-1515 for the
budget sentence (72.2% rejected, 87,490 survive, 95.2% of injected synthetics retained) — copy
that reporting form, and note the recall is conditional on synthetics, not end-to-end.

*Feeds ideas:* tri-view-consensus-anomaly, ivar-ladder


### 6. §2.2 Data Acquisition and Definition (L389-490)
*pp.5-7, mainly p.6 (Figs 5-6)*

Two things matter here. The DataProvider's in-memory preload is the paper's single largest
measured training win (order-of-magnitude epoch speedup) and it is exactly the next build step
in your own notes (lsst48 RAM store/loader). And 'a dataset class may wrap a generator, not just
storage' is the plumbing pattern for the depth ladder. Read it knowing the paper NEVER claims
multi-survey paired views of one object — the composition story is strictly multimodal
(images/spectra/light curves), and the actual left-outer-join semantics exist only in the 0.12
package, not in these pages.

**Read:** Fig. 6 caption L442-447 (p.6) for the preload/cache claim and the literal data_a/data_b/data_c
composition. L466-477 for the dataset-class contract (one modality, keyed by a uniform object
identifier) and the DataProvider assembly. L420-429 for on-the-fly multi-threaded cutouts from
cached patches. L461-463 for the bring-your-own-data bypass. Fig. 5 (p.6) only if you ever go
back to the RSP.

*Feeds ideas:* ivar-ladder, mask-ivar-leak


### 7. §4 Summary & Future Direction (L1623-1726) + the useful half of §2.3 (L491-556)
*p.22 and p.7*

§4 is the two-paragraph, quotable restatement of the two findings in rank 1 — useful when you
writes this up, not for discovery. §2.3 contributes exactly two things: the clause that lets you
turn 'val loss is the wrong criterion' into an implementation (log custom quantities during
training), and the config-provenance convention your LSST run dirs lack — those dirs currently
hold checkpoints, val_object_ids.txt and latents.h5 but no record of
--field/--steps/--batch/norm_key.

**Read:** L1675-1690 (p.22) — both methodological findings in one place. L545-547 (p.7) for the custom-
metric clause. L497-504 (p.7) for config capture plus structured on-disk outputs. L1696-1703 for
the roadmap (foundation models, contribution templates) as a watch item only. Skip the Optuna
and MLflow/TensorBoard material — it duplicates WandB and hyrax 0.12 has no wandb support at
all.

*Feeds ideas:* leak-as-training-metric


### 8. §3.3 Multimodal ZTF early classification / AppleCiDEr (L1281-1414)
*pp.17-18 (Figs 16-17)*

Worth one pass for the experiment design, not the architecture: the headline result is a
modality ablation reported as 'adding spectra broke these two specific degeneracies', not as an
accuracy delta. Your analogue is obvious — ablate the DP2 third view in and out and report the
drop in instrument R^2 from the physics latent. Fig. 17 (one misclassified and one correct
object with every modality displayed side by side) is the template for your latent-weird-vs-
pixel-weird forensics panel. Read it also as a warning: fusing metadata into a shared
representation is the exact opposite of what you want for n_input/psf_fwhm.

**Read:** L1296-1301 and L1325-1342 (p.17) for the modality list and which degeneracies the fourth
modality broke. L1388-1402 plus the Fig. 17 caption (pp.18-19) for the per-object forensics
panel. Skip the fusion-architecture description — no MoE routing, no expert count, no
dimensions, no missing-modality handling, all deferred to Junell et al.

*Feeds ideas:* three-view-brush, leak-as-training-metric


### 9. §3.2 Cluster-scale lensed galaxies (L1187-1280) — skim two passages only
*pp.16-17*

Almost all anecdote (one candidate, no denominator, confirmation deferred), but two paragraphs
are genuinely useful: they build a 'latent space' from tabular columns alone (RA, Dec, z_phot)
and cluster it, which is the nuisance-only reference manifold idea; and they let an independent,
sharper survey adjudicate a flagged candidate, which is exactly what your three views can do for
each other. The two-angular-scale cutout habit is a free improvement to your anomaly galleries.

**Read:** L1232-1239 (p.16) for the tabular-only UMAP + DBSCAN. L1245-1259 (p.16) for the two cutout
scales and Euclid VIS confirming the arc (and revealing a second, blended candidate). L1247-1251
for the contaminant taxonomy written down before inspection. Nothing else in the section.

*Feeds ideas:* metadata-manifold, depth-counterfactual-render, tri-view-consensus-anomaly


---

## Sections to skip


- **§1 INTRODUCTION (L81-265, pp.1-3) and Figures 1-3** — Pure motivation: survey grasp-vs-sensitivity, ADS publication counts, and a Sculley-style box diagram arguing that infrastructure not model design is the bottleneck. No technique. If you want five minutes of value, use Fig. 3 (p.4) as a self-audit checklist of which non-model boxes your own repo already fills — loaders and WandB yes, interactive latent exploration and similarity search no.

- **§2 preamble + §2.1 Hyrax Overview (L266-388, pp.3-5) and Figure 4** — The four verbs (download/train/infer/visualize) and a feature list. You already has an equivalent of every stage, so this only sets adoption expectations. Two sentences are worth grepping for and nothing else: L336-343 (the downloader supports only Rubin-LSST and HSC-SSP, everything else uses the bypass — so your Legacy/DECam view has no supported path) and L367-370 ('reverse-image/spectra searches', a component name worth grepping in the repo since the paper never explains it).

- **§2.3's MLOps material — Optuna, MLflow, TensorBoard, run comparison (L505-544, p.7) and Figure 7** — Duplicates your WandB tess-ml/lsst-disentanglement setup exactly, with no astronomy-specific objective, no search spaces and no numbers. Adopting it means running two trackers, and the current package has zero wandb integration. Keep only L545-547, already ranked above.

- **§3.1.1 Rubin DP1 ECDFS Dataset (L909-974, p.12)** — One reusable convention (extendedness=1 in >=1 band AND imaging in all six bands) and a depth yardstick. Otherwise it describes a field you are not using, deliberately chosen for near-uniform six-band coverage — i.e. the opposite of the COSMOS/DEEP2-3 depth heterogeneity that defines your sample. Cutout size, pixel scale, latent dims, backbones and augmentations are never stated, so there is no recipe to copy.

- **§4's roadmap items (L1696-1703, p.22)** — Explicit future work: deploying community foundation models and contribution templates. No API, no named model, no timeline, nothing in the release. A watch item at most, not something to plan around.

- **Appendix A Code, Docs, Tutorials (L1994-2021, p.25) and the software-citation list** — Repo URL, readthedocs, pip install, CI/ruff/Codecov, and the frozen v0.8.1 tag. The tag is the problem: PyPI is now at 0.12 with materially different internals (DataProvider joins, save_to_database, to_onnx verb, a [balance] sampler), so for any adoption question read the package, not the appendix. Note also that no license is stated anywhere in 28 pages — the wheel metadata says MIT.


---

## Ideas that survived verification


Ordered by verified value. `effort` and `adopt_level` are as scored; the caveats killed several
sibling ideas outright and are where the real information is.


### 1. Rebuild the dropped ivar plane, then synthesize a depth ladder (the one causal experiment available to you)

`ivar-ladder` · **effort:** days for the rebuild plus the ladder; weeks for the full two-latent test · **needs:** steal-the-idea (§3.5's injection-as-label-generator, not the package)


**What:** The 39 shards under cross_matching/outputs/rsp_results/run2_full/dp2_cutouts/ carry ivar
(list<float>, 160x160) and mask (list<bool>) per object per band — verified non-null, ~0% NaN,
~0% zeros — and both were dropped when lsst_image.bin was built. One sequential pass (~45-90 min
per the build README) writes an lsst_ivar.bin sidecar. Then degrade COSMOS stamps down a ladder
of instrument states: add noise with variance (1/ivar)*(N/N_target - 1) to emulate n_input_i =
60/30/15/7, and convolve psf_fwhm_i from ~1.38 toward DEEP2-3's median. The same galaxy now
exists at a ladder of instrument conditions with physics held fixed BY CONSTRUCTION. Three uses:
(a) a ground-truth disentanglement test — z_phys flat along the ladder, z_inst monotonic —
replacing the proxy 'instrument R^2 from the physics latent ~ 0'; (b) completeness curves for
every other diagnostic here (at what depth does counterpart recall collapse, at what depth does
an injected anomaly stop being anomalous); (c) training augmentation, the mechanism most likely
to actually close the 0.114/0.341 val-MSE gap. Run degraded stamps through the identical
lsst/preprocessing.py path (crop 48, clamp, arcsinh, x0.84 zoom, scalar grizy norm) or you
measure preprocessing, not depth.


**Why now:** Every other diagnostic you have is correlational: COSMOS and DEEP2-3 differ in depth AND in sky,
selection and sample, so 'the latent is field-separable' has no clean causal reading. This is
the only intervention where physics is provably constant, it uses data you already downloaded and
discarded, and it needs no RSP session, no Butler and no re-crossmatch. It is also what converts
agreed strategy item (3) — continuous depth conditioning — from a design preference into a
measured response curve.


**Caveats:** Two number traps that will silently ruin it: shard fluxes are nJy while the store is nanomaggies
(njy_per_nanomaggy 3630.78), so the ivar conversion carries a 3630.78^2 factor; and DEEP2-3's
psf_fwhm_i median is 1.56, not 2.0 (that is the field maximum), so aiming at 2.0 overshoots into
a regime almost no real object occupies. Coadd noise is spatially correlated (warp, resample,
outlier rejection) and a shallow coadd's PSF is drawn from a different seeing distribution
rather than merely broadened — a referee will raise both. Run the calibration check FIRST:
degrade COSMOS to n_input~12 and show it lands on top of the 4,637 real DEEP2-3 objects at
n_input_i>=10 in the instrument latent, and run psd_astronomy/ on degraded vs real DEEP2-3. That
check is cheap and is what makes the whole thing a measurement rather than an assumption. Note
the stated payoff (physics stays, instrument moves) needs the two-latent COSMOS HSC x LSST model
that does not exist yet; today it reduces to how far a single z moves relative to the local kNN
scale. Budget ~13-20 GB against 180 GB free on /work1.


*From:* §3.5 pp.20-21 (L1583-1591 dataloader wrapping source_injection; Fig. 20 at L1596-1600); §2.2 p.6 (L420-424, dataset class as generator); §3.4 p.20 (L1494-1499, manufacturing a class by breaking a physical prior)


### 2. Log instrument leak at every validation, then check whether early stopping picks the right checkpoint

`leak-as-training-metric` · **effort:** hours · **needs:** steal-the-idea (§2.3's 'record any custom application-specific quantities during training')


**What:** lsst/ae_baseline/model.py's validation_step already accumulates z, field, ninp and psf on CPU
(lines 81-93) and on_validation_epoch_end already calls self.log for val/mse and per-field
val/mse_<key> — so this is a handful of lines. Fit a ridge or one-layer probe on train-split
latents predicting n_input_i and psf_fwhm_i, evaluate on the frozen val split recorded in each
run dir's val_object_ids.txt, and log val/leak_r2_ninput and val/leak_r2_psf to tess-ml/lsst-
disentanglement next to val/mse_cosmos and val/mse_deep23. Then answer one question directly: is
the step-14k best-val-MSE checkpoint also the best-leak checkpoint? Apply the same
instrumentation to the fm_ae and jepa siblings and plot leak against val MSE across the 7-run
matrix to see whether the ordering is non-monotonic the way Fig. 12 was.


**Why now:** You selects checkpoints by val MSE — the README already pins 'best ckpt at step 14k, use ~15-20k
steps in future runs' — and the matrix will be ranked the same way, which is exactly what §3.1.2
and §4 argue is unreliable for representation quality. You currently measures leak once, post-
hoc, on a final checkpoint, so you have no idea whether the leak curve is flat, falling or rising
across the window where your early stopping fires. Highest value per hour on this list.


**Caveats:** Log it WITHIN field, not jointly. COSMOS and DEEP2-3 n_input_i (median 89 vs 7) and psf_fwhm_i
(5-95% 1.344-1.422 vs 1.433-1.662) are nearly disjoint, so a joint-field leak R^2 is a field
classifier in disguise and just re-measures what umap_by_field_ninp_psf.png already shows.
Within-COSMOS is the informative version (2,838 objects below n_input_i=40, 1,354 below 20, 567
below 10). Drop the counterpart-rank term from the original proposal — it needs a cross-view
model that does not exist. And reword the claim: the LSST AE has a single z, so 'leak' here
means 'the latent encodes depth', which for a plain AE is expected rather than a failure; the
deliverable is the checkpoint-selection answer, which stands on its own. Cheap probes are noisy
— fit on train latents only, keep the val split clean, read curve shape not absolute level.
Hyrax's own version of this claim is eyeballed UMAP compactness, so this is the more rigorous
version, not a reproduction.


*From:* §3.1.2 pp.13-14 (L1054-1065 and Fig. 12 caption L996-1002); §4 p.22 (L1683-1690); §2.3 p.7 (L545-547)


### 3. Counterpart recall@k on an n_input x psf grid: a label-free disentanglement metric that is also a Rubin result

`depth-recall-map` · **effort:** hours of work, but sitting on top of a model that does not exist yet · **needs:** steal-the-idea


**What:** With a shared HSC<->LSST embedding, build two tables over the same 65,163 objects and query the
DP2 table with each galaxy's own HSC physics latent, recording the rank of its true counterpart
(recall@1, recall@10, median rank). The pairing is exact via lsst_row_map.npy — 65,163 mapped
rows, all source_type==0 — so this is ground truth with no labels. At 65,163 x 128 it is one
chunked matmul; skip the vector DB entirely, your sample sits ~150x below the scale §2.4 is
engineered for. Report recall binned on a 2-D grid of n_input_i x psf_fwhm_i in the layout of
Fig. 20, with lines of constant coadd depth (~sqrt N) replacing constant surface brightness. The
grid is genuinely populated, not degenerate: corr(n_input_i, psf_fwhm_i) is -0.475 in COSMOS and
+0.327 in DEEP2-3, 989 COSMOS objects sit below n_input_i=15 and 4,637 DEEP2-3 objects at >=10,
giving a real overlap band. The readable output is a number the Rubin community wants: the visit
count at which a galaxy's morphology becomes as identifiable as in HSC-deep — a DR1/DR2
forecast.


**Why now:** Your only quantitative disentanglement evidence is instrument R^2 ~ 0 measured from the physics
latent, which is a NEGATIVE test that a collapsed or uninformative latent passes trivially.
Counterpart retrieval is the missing positive control: it fails both when physics content was
destroyed and when instrument signal dominates. It is only possible because you have the same
galaxy imaged by three instruments, and it doubles as a standalone survey result rather than a
representation-learning claim.


**Caveats:** It cannot rank the current 7-run matrix — every lsst/ model is single-view, so no cross-survey
retrieval task is even defined; this runs after the COSMOS HSC x LSST dual-encoder (strategy
item 1). The HSC<->Legacy version DOES run today from checkpoints/base/snapshot.ckpt. It is also
already queued: contrastive_ablation/PLAN.md line 199 lists task F 'cross-survey retrieval,
physics space, Ours vs contrastive' as TODO because a reviewer asked, and ANALYSIS.md already
notes InfoNCE trains exactly this task so contrastive should win or tie on top-1 — pitch this as
extending task F with the genuinely new parts: the instrument-latent control arm, the n_input_i-
decile stratification, and the DP2 third view. Keep both controls (instrument-latent recall at
chance; a raw-pixel/PCA floor) or the number conflates invariance with 'this galaxy is
distinctive'. Traps: psf_fwhm_i has 4 NaNs in DEEP2-3 that will silently NaN a whole correlation
or grid cell; neighbor_search/search_neighbors.py hardcodes a dead /data/vision/billf/ path; and
preprocessing parity between the 48x48 HSC path and the DP2 crop-48+x0.84 path is load-bearing —
verify against lsst/fixtures/preprocess_reference.npz before believing any number. Latents from
different objectives are not comparable under one metric, so whiten per model and report a
shuffled-pair null.


*From:* §2.4 pp.7-9 (L588-624 analog-finding + search_by_vector; L640-642 the scale claim; Fig. 8 layout L649-667); §3.5 p.21 (Fig. 20, L1596-1600); §3.1.2/§4 (L1054-1057, L1683-1690)


### 4. Per-pixel artifact scalars as probe targets, then two rounds of peel-and-retrain

`mask-ivar-leak` · **effort:** hours for the mask half; days with ivar plus the retrain · **needs:** steal-the-idea


**What:** Column-prune the 39 shards for hsc_object_id/band/mask/ivar and reduce each stamp to four
scalars per object per band: masked-pixel fraction, zero-ivar fraction, median ivar, border-
truncation flag. Do the mask half first — mask is 1.9 MB compressed per row group against 1,225
MB for ivar, so masked fraction and the truncation flag cost roughly 75 MB of I/O across all 39
shards (seconds), while the ivar scalars are a ~48 GB pass. Then use them twice: add them to the
MLP-R^2 target list to get an ARTIFACT R^2 alongside the existing depth R^2, and recolour
umap_all_latents.npz by them. Then run the paper's loop: cut objects above an artifact
threshold, retrain the frozen run-1 recipe through lsst/run_matrix.sbatch (~30 min on mi2101x,
best val at step 14k), and check whether the COSMOS/DEEP2-3 val-MSE gap survives the peel and
what second-generation island appears.


**Why now:** The DP1 finding is that filter artifacts, not astrophysics, dominate the first latent structure
— and your own 2026-09-02 follow-up already records that NSF/IForest top-3 are near-empty stamps
at n_input_i=1 and suggests an n_input floor cut. But n_input is a coadd summary statistic;
masked and zero-ivar pixel fraction is the actual per-pixel pathology. Right now you literally
cannot distinguish 'the latent encodes depth' from 'the latent encodes bad pixels', because the
plane that would say so was discarded at build time.


**Caveats:** rsp_full_cutouts.py ORs NO_DATA|SATURATED|COSMIC_RAY|INTERPOLATED into one bool (line 39, line
127), so bit semantics are already collapsed — separating a coverage hole from a cosmic ray
needs an RSP re-cut, and the paper never promises LSSTDataset returns ivar or mask, so verify in
code before betting on that route. downstream_evaluation/predict_neighbors.py cannot be reused
as-is: its DEFAULT_TARGETS are read from an H5 labels/ group and the LSST latents.h5 has none,
so write a small adapter. Word the result 'artifact R^2 from the AE latent', not 'from the
physics latent', until a two-latent LSST model exists. The paper shows two peel rounds with no
stopping criterion and no count of what was removed — cap it at two rounds, keep the peeled
objects, and route outputs to a new dir per the no-overwrite policy, because a peel can discard
genuinely rare objects along with artifacts.


*From:* §3.1.2 pp.12-13 (L978-995, Fig. 11); §4 p.22 (L1675-1682); §2.2 p.6 (L413-441, the Butler re-cut path)


### 5. A nuisance-only manifold, and kNN overlap as a capacity-free leak metric

`metadata-manifold` · **effort:** hours · **needs:** steal-the-idea


**What:** Standardize the instrument columns of lsst_meta.parquet (n_input_{g,r,i,z,y},
psf_fwhm_{g,r,i,z,y}) into a pure-nuisance reference space — and on the HSC x Legacy side the
analogous legacy_PSFSIZE_*/NOBS_*/GALDEPTH_* plus the derived hsc_*_psf_fwhm you already lists as
DEFAULT_TARGETS. For each object, compute the overlap between its k nearest neighbours in that
space and its k nearest neighbours in the physics latent, and separately in the instrument
latent; report mean overlap@k against a shuffled-id chance floor. Physics should sit at the
floor, instrument far above it, and the gap is a single probe-free leak number. The same space
gives depth-matched control sampling for free: pairs matched in nuisance coordinates but far
apart in physics, and vice versa.


**Why now:** MLP-R^2 only measures what a small MLP can decode — a latent can carry depth multimodally, non-
monotonically, or entangled with morphology and still score R^2 ~ 0. Neighbourhood overlap is
nonparametric and capacity-free, costs minutes on 65k x 128, and runs today on latents you
already has. It also exploits something DP1 ECDFS structurally cannot have: twelve continuous
per-object instrument coordinates per galaxy, where ECDFS was deliberately chosen for near-
uniform coverage.


**Caveats:** Drop the u band — n_input_u and psf_fwhm_u are NaN for all 31,031 DEEP2-3 objects (psf_fwhm_y is
NaN for 452 COSMOS, n_input_y for 245) — or handle it per field. The physics-vs-instrument
contrast only exists on the HSC x Legacy dual-encoder latents; on LSST it degrades to a single
'does the latent track the nuisance manifold' number. The HSC/Legacy metadata columns are NOT in
neighbors_efficient/catalog.parquet — they live in data/neighbours_v2.h5, and
neighbor_latents_103k.h5 has no labels group, so it is a row-index join: minutes, but not the
one-liner it sounds like. Soften the within-COSMOS pitch: n_input_i IQR is 78-95 and only 2,838
of 34,132 objects fall below 40, so the depth-matched-control claim rests on a modest tail and
the within-field overlap number will be noisy. Compute the overlap in standardized raw metadata
space, never in UMAP space (unstable across seeds and hyperparameters); use UMAP only for the
figure. Depth and PSF correlate with sky position and therefore with field, so quote the within-
COSMOS number alongside the joint one.


*From:* §3.2 p.16 (L1232-1239, UMAP from tabular columns alone + DBSCAN); §2.4 p.8 (L620-624); §2.5 p.9 (L747-755, colour by user-supplied crossmatched properties)


### 6. Three-view consensus anomalies: a candidate list instrument artifacts cannot enter

`tri-view-consensus-anomaly` · **effort:** about a week · **needs:** steal-the-idea


**What:** Run the encode-all + NSF + IsolationForest stage three times over the SAME 65,163 objects — once
on the DP2 store, once on the HSC rows and once on the Legacy rows reached through
lsst_row_map.npy — giving three latents/scores triples. Fuse them into a consensus score (rank-
product of the three per-survey NSF ranks, each already standardised on its own field's train
latents) and keep only objects that are outliers in all three views. A DP2 cosmic ray, a one-
exposure coverage hole or an unmasked hot pixel is not present in the 2016 HSC coadd, so the
consensus requirement is a purity gate whose negative class is manufactured by construction
rather than labelled — §3.4's trick without needing an impossible orbit. Report it as a
filtering budget ('X% of the current top-100 rejected, N survive, of which M are in the DESI
spec-z subset') and extend gallery() from one row to a three-row HSC/Legacy/DP2 panel per
survivor.


**Why now:** Your README records that the top NSF/IForest anomalies are near-empty stamps at n_input_i=1 and
that the recon-MSE tail is shallow-coverage objects even inside COSMOS — the anomaly stage is
currently measuring depth, not astrophysics. With NSF-IForest overlap of 84 in the top 100 and
recon overlapping neither, you have two disagreeing diagnostics and no adjudicator. Cross-survey
consensus is the cheapest adjudicator you own and turns the stage from an internal sanity check
into a list with a defensible purity argument.


**Caveats:** Not three invocations of anomaly.py — encode_all hardcodes LsstGrizyDataset('all'), and the
existing HSC/Legacy latents are keyed by index_mmu (raw-MMU row space) which does NOT align to
neighbors_efficient rows (only 13,906 of 65,163 coincide, and that is coincidence); re-encode or
join on object_id_hsc. The cheap part: efficient_neighs_48/{hsc48,legacy48}.bin already hold
468,197 x 4 x 48 x 48 preprocessed stamps row-aligned to the same catalog. Two real risks. DP2
deep_coadds are outlier-rejected at coadd time, so the artifacts this gate targets survive
mainly in the n_input=1-to-few stacks — which is exactly where your current top-100 lives, so
expect the gate to DELETE your list rather than surface new science; that is still a publishable
filtering-budget result, but sell it as quality control, not discovery. And the three latents
come from different preprocessing paths and different models, so this is rank fusion, not a
shared metric, and a reviewer can attack it — run it on the existing AE run-1 latents first, and
if the three-way intersection is empty, weaken to two-of-three and report the pairwise matrix.
DP2 pixels are internal-use-only (REPORT.md §8), so a three-row survivor gallery cannot be
published as-is.


*From:* §3.1.2 pp.12-13 (L978-995); §3.4 pp.19-20 (L1457-1460, L1494-1515); §2.4 p.9 (Fig. 8, L649-667)


### 7. A linked latent brush that shows each galaxy in all three surveys

`three-view-brush` · **effort:** days · **needs:** use-one-component (HoloViews/Datashader directly; Hyrax's viewer optional)


**What:** Port galaxy_model/umap_datashader_viz.py (HoloViews + Bokeh, lasso_select, base64 thumbnails)
into a brush over the run-dir UMAPs, colourable by any lsst_meta.parquet column and by
scores.parquet's nsf_nll / iforest / recon_mse and their rank columns. On tap, render that
galaxy's HSC, Legacy and DP2 stamps in one row with n_input_i, psf_fwhm_i and its three anomaly
ranks printed underneath. Ship it as a self-contained HTML written into the run dir — no server,
no port-forwarding off the AMD login node — one per checkpoint, under the no-overwrite
convention.


**Why now:** §2.5 exists explicitly to judge whether latent structure is astrophysics or instrument
systematics, which is the call you now make from static PNGs. The paper concedes two gaps your
data closes: it has no linked view across two latent spaces, and it shows six random stamps of a
single modality. A lobe is leak if its members are a depth family and real if the three views
show the same galaxy under three PSFs. It also answers the question umap_by_field_ninp_psf.png
cannot — which objects are in the detached low-n_input arm, and what they look like elsewhere.


**Caveats:** Scope it down: the two-panel physics|instrument layout is not available for LSST (single z), so
ship either a one-panel LSST brush with three-view stamps on tap, or a two-panel HSC x Legacy
brush with DP2 stamps shown alongside by id. Budget a port, not a config change —
umap_datashader_viz.py is pinned to dead /data/vision/billf/ paths, one old FM checkpoint and
4,048 points, and despite its filename it imports zero datashader, so rasterizing to 65k points
is new work. Hyrax's own visualize_v2 is not drop-in either: it calls setup_dataset, requires a
data_request.visualize group with a primary_id_field, and expects the primary dataset to already
carry 2-D coordinates with get_<field> getters — pointing it at latents.h5 means writing an
adapter dataset class. This is tooling, not a result; rank it below the diagnostics and timebox
it.


*From:* §2.5 pp.9-11 (L690-695 the rationale; L713-787 the select->ids->table->stamps loop and DataProvider coupling); §3.3 pp.18-19 (Fig. 17 per-object all-modalities forensics panel, L1388-1402)


### 8. Render DEEP2-3 at COSMOS depth and let HSC referee the result

`depth-counterfactual-render` · **effort:** weeks, strictly after the ladder lands and passes its calibration check · **needs:** steal-the-idea


**What:** Condition the existing lsst/fm_ae decoder (ResNet spatial tokens -> all-CrossAttn UNet, 4 tokens
x 32 = 128-d, rectified-flow loss) on n_input_<band> and psf_fwhm_<band> as continuous scalars;
train on COSMOS where the ladder supplies matched (degraded input -> real deep target) pairs;
then run the counterfactual on the 31,031 DEEP2-3 galaxies — encode the shallow image, decode at
COSMOS instrument conditions. Validation is exact on held-out COSMOS (degrade, render, compare
to the real deep coadd), which is the ground truth §3.5's injection experiment never had since
they never ran on LSST data at all. Then borrow §3.2's adjudication move: a feature recovered in
the render counts as real only if it is independently present in the HSC deep view of the same
galaxy, which was never used to train the depth conversion. Publish the rendered-vs-HSC
agreement rate binned on the same n_input x psf grid.


**Why now:** This is the only place your three-survey pairing gives a genuinely held-out referee for a
generative model — astronomical image-restoration papers almost never have one, which is exactly
why they get hammered on hallucination. It also makes you the learned injector §3.5 gestures at,
turning a tool contribution and a science result into the same artifact.


**Caveats:** Riskiest item here and entirely downstream of idea 1 — the matched degraded-to-deep pairs do not
exist until lsst_ivar.bin is rebuilt, and fm_ae has no scalar conditioning pathway today. Reuse
latent_traversal/quality_conditioning_traversal.py (and the snr_/psf_ variants): conditioning-
and-render across quality buckets already works on HSC/Legacy, so this is a port plus a new
referee protocol — lower novelty and lower build cost than the pitch implies. Aim the referee
claim at DEEP2-3 (i-maglim ~24.5) where HSC deep/UD at i~26 is a genuine ~1.5 mag adjudicator; a
COSMOS demo has no referee, since DP2 COSMOS already reaches ~25.6. Never present rendered
images as detections: publish the agreement rate as the measurement, include a negative control
(render at COSMOS conditions from an input with no feature, confirm nothing appears) and a
blur/deconvolution baseline, and if agreement is not clearly above that baseline, report it and
stop. DP2 pixels are internal-use-only, so a public 'forecast image set' is not a deliverable.
Soften the Hyrax framing — L1616-1617 is one aspirational clause, not a stated need the §3.5
team is waiting on.


*From:* §3.5 p.21 (L1613-1617 differentiable diffusion for image-level injection); §2.2 p.6 (dataset class wrapping a generator, L420-424); §3.2 p.16 (L1255-1259, Euclid VIS adjudicates the Rubin arc)


---

## Adopt / don't adopt


Take the ideas; do not adopt the framework. The model contract Hyrax describes is one backbone
plus the logic of a single train/validation step, which is precisely the shape your dual-encoder
asymmetry does not have — and nothing in 28 pages touches disentanglement, paired multi-survey
views, two-encoder architectures or conditioning on continuous instrument metadata, so wholesale
migration would cost a model rewrite plus a loader rewrite for zero stated benefit to the input
asymmetry that produces your disentanglement in the first place; your k-of-20 same-survey
neighbour sampler is not even expressible in their join. The AMD question is genuinely open
rather than closed: AMD, ROCm and CUDA appear nowhere in the paper, the only hardware
acknowledged is NCSA Delta, and there is no SLURM or distributed-backend detail anywhere — but
torch is an unpinned dependency, the package is MIT and requires-python >=3.11 (your torchenv is
3.11.14 with torch 2.12 + rocm7.1), and training runs on pytorch-ignite whose idist.device()
resolves through torch.cuda, which a ROCm build reports, so it should work and nobody has tested
it. Spend one timeboxed day in a throwaway venv (never torchenv — 0.12 drags in mlflow,
chromadb, lancedb, qdrant-client, confluent-kafka and onnxruntime, and never pass -U): the only
component plausibly worth keeping is the 2D explorer loop, and even there expect to write an
adapter, since visualize_v2 wants a hyrax-shaped dataset with 2-D coordinates and get_<field>
getters rather than an h5 path. Skip the rest on evidence: the vector DB is drop-in (public
insert/search_by_vector) but pointless at 65,163 x 128 where a numpy matmul is instant, and its
Qdrant payload is id-only so 'neighbours restricted to n_input_i<10' is your own code regardless;
MLflow/TensorBoard would fragment tracking because 0.12 has zero wandb support. One caution for
anything you build on this: the paper froze v0.8.1 while PyPI is at 0.12 with materially
different internals (DataProvider joins, [balance], save_to_database, a to_onnx verb), so read
the package for capability questions and cite only the paper for ideas — several of the most
attractive features are code-only and citing §2.2 for them will not survive a careful reader.


### Verified locally on this cluster (2026-09-16)


`uv pip install hyrax` into a throwaway venv succeeds on hpcfund: **hyrax 0.12**, Python 3.11. What ships:

- **Verbs:** `train`, `infer`, `infer_stream`, `test`, `umap`, `reduce_dimensions`, `visualize`,
  `visualize_v2`, `save_to_database`, `database_connection`, `lookup`, `create_splits`, `to_onnx`, `engine`,
  `model`, `prepare`.
- **Dataset classes:** `LSSTDataset` (Butler, RSP-only), `DownloadedLSSTDataset`, `RangeReadLSSTDataset`,
  `HSCDataset`, `MultimodalUniverseDataset` (loads MMU sets over `hf://` — the parent of raw_mmu),
  `HyraxHATSDataset` (`lsdb.open_catalog` — the same route as the DP2 crossmatch), `NestedPandasDataset`
  (the format of DP2 `objectForcedSource` light curves), `FitsImageDataset`, `LanceDBDataset`, `KafkaStreamDataset`.
- **Models:** `HSCAutoencoder`, `HSCDCAE`, `HyraxAutoencoder(V2)`, `HyraxCNN`, `ImageDCAE`, `SimCLR`.
- **DataProvider** docstring: *"acts like a GraphQL gateway that fetches data from multiple datasets"* — this
  is the multi-view composition mechanism, though the paper only ever claims multimodal, never multi-survey-paired.

Caution: 0.12 pulls in mlflow, chromadb, lancedb, qdrant-client, confluent-kafka and onnxruntime. Never
install into `torchenv`, and never pass `-U`.


---

## Proposed and rejected


Five proposals were killed and should not come back. (1) "Three surveys as three epochs" — a
latent-residual variability/proper-motion search over the 6-10 year HSC-to-DP2 baseline — failed
on all three legs: its Hyrax citation was fabricated ("cross-collection retrieval" appears
nowhere; the only two hits for "collection" are unrelated prose), 64,899 of your 65,163 objects
have i_extendedness_value==1 with just 253 point-like, so an extended-galaxy selection has
essentially no proper-motion population to be enriched in, and the variability half is dominated
by data your own REPORT.md already flags — the same crossmatch yields DP2 forced-photometry light
curves (nested objectForcedSource) and DIA products, so inferring variability from two averaged
coadds is strictly the worse instrument; if you want the time domain, pull those light curves as
a fourth modality in one RSP notebook run. (2) Adopting Hyrax's Butler patch cache to "make
cutout geometry a parameter again" died because three of its four payoffs are already free on
disk — preprocess_lsst takes crop_size as an argument (you have already produced fullstamp160
galleries from the same store), ivar and mask are sitting in the local 70 GB of shards, and XMM-
LSS is gated by Rubin releasing no deep coadds there (0/36,273 matched), not by tooling — while
the fourth (FoV beyond 32", sub-pixel re-registration) is RSP-only, data-rights restricted,
splits the pipeline across two environments and helps nothing on AMD. (3) Seeding a value-added
catalogue from the 4,579 DESI/PROVABGS labels and measuring how the propagation degrades with
depth is unmeasurable as stated: all 4,579 anchors are COSMOS and essentially all sit at
n_input_i>60 (zero below 15), so there is no shallow-regime ground truth to check a propagated
mass against — the distinctive claim only exists once the ivar ladder does, at which point it is
a use of idea 1, and the regression-probe half is already open thread 3. (4) ONNX-exporting the
encoders is harmless but mis-sold — downstream_evaluation/engaging/ already runs on Engaging
with completed job logs, so the honest benefit is only that the eval host stops needing to
import the model class and match diffusers/Lightning versions, and ONNX appears in the paper
solely as three Appendix package citations. (5) "Declare depth/PSF as a joined stream with
stratified sampling" largely restates your own agreed strategy item 3 and open thread 2 — your
notes already specify routing it "into the instrument pathway" — and the one new piece, hyrax's
[balance], stratifies on a categorical class field, so continuous n_input_i needs binning first,
which is the per-arm sampler you were trying to avoid. Three further items survived verification
but lost a top-eight slot and are worth an afternoon each rather than a plan: dump the argparse
Namespace, preprocessing constants and git SHA to run_dir/run_config.json and write the
train/val split as an .npz artifact instead of recomputing sha1 in two codebases (skip MLflow
and numbered config migrations entirely — and rotate the WandB API key, which sits in plaintext
in six tracked .slurm files); copy the shape of hyrax 0.12's ID-joined data_request into your own
config so a new arm is a config block rather than a new module, remembering that
lsst_row_map.npy already IS that join; and the bounded Hyrax spike itself, covered in the
verdict above.

