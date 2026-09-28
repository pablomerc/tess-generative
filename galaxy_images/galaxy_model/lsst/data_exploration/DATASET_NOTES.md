# Understanding the LSST DP2 data in our dataset

Companion to `nvisits_maps.py` / `figures/nvisits_maps.png`. Written 2026-09-02.

## Where this data comes from

Our stamps are cutouts from **Rubin Observatory Data Preview 2 (DP2)** — the
second preview release, built from **LSSTCam commissioning / science-validation
observations** (28,698 visits over 131 nights, including the June 2025 "First
Look"). This is *pre-survey* data: the camera and pipelines being exercised
before the 10-year LSST proper. The catalogs were distributed to data-rights
holders as "EDP2" HATS files (what we crossmatched on the RSP); the images we
cut come from DP2 `deep_coadd`s.

Each of our images is a **coadd**: a stack of `n_input_<band>` individual
exposures at that position (stored per object in `lsst_meta.parquet`).

## The two fields

| | COSMOS | DEEP2-3 |
|---|---|---|
| position | RA 150.1°, Dec +2.1° | RA ~352°, Dec ~0° |
| name origin | famous multi-wavelength survey field | HSC-SSP deep field (from the DEEP2 redshift survey, field 3) |
| Rubin status | **official Deep Drilling Field** (one of 5) | ordinary sky inside the wide-survey (WFD) footprint |
| our objects | 34,132 | 31,031 |
| median exposures (u/g/r/i/z/y) | 54 / 55 / 94 / 89 / 57 / 32 | — / 2 / 5 / 7 / 8 / 4 |
| DP2 i-band coadd depth | maglim ≳ 25.6 | maglim ≈ 24.5 |
| bands | all 6 (245 objs lack y at patch edges) | 5 — **no u-band coadds at all** in DP2 |

The heatmaps show *why* they look different: COSMOS is a single telescope
pointing revisited hundreds of times — the dither pattern makes a smooth
radial depth profile (~100 exposures in the core, ~10–20 at the rim). DEEP2-3
is a patchwork of overlapping wide-survey pointings — depth jumps 2→13
between overlap regions, and the per-band coverage is thin and uneven
(g-band median: **2 exposures**).

## Will these same sources get better in the future? Yes — on four axes

1. **More visits (deeper stacks).** COSMOS, as a Deep Drilling Field, gets an
   intensive repeat cadence for the whole decade — order 10⁴ visits total by
   year 10 (thousands per band vs our ~30–90). DEEP2-3 gets standard WFD
   cadence: roughly ~20 visits/band/year in season → ~2–4× today's depth by
   the first annual Data Release (DR1), ~200/band by year 10. Coadd noise
   drops roughly as 1/√N, so DEEP2-3's transformation is the more dramatic
   one in relative terms (g: 2 → hundreds), though COSMOS stays deeper in
   absolute terms forever (final i ≈ 28 vs ≈ 26.9).
2. **Reprocessing of everything.** Each Data Release reprocesses *all* raw
   images from scratch with improved pipelines and calibration (PSF models,
   deblending, background subtraction, astrometric/photometric reference
   frames). So even the exposures already in our stacks contribute better
   pixels next time.
3. **Missing coverage appears.** Our third crossmatched field, XMM-LSS
   (36,273 objects), was *visited* in DP2 — 132 visits (u:42 g:15 r:8 i:47
   z:10 y:10), so the visit images exist — but DP2 built deep coadds only
   inside a pre-defined tract set with sufficient visits, and XMM-LSS wasn't
   in it: no coadds → no Object catalog → nothing to crossmatch or cut.
   "Visited, coadds not ready yet" is the one-line status; a future release
   should coadd it (the runbook then reruns unchanged). DEEP2-3's u-band
   likewise appears once WFD u visits accumulate there.
4. **New data products.** Better DIA templates, photo-z, per-band metadata.

**The catch:** nothing updates in place. A new release means new coadds, new
tract/patch layouts, and — importantly — **new object IDs** (detection and
deblending are re-run). Our stamps stay valid as "DP2 data" but to upgrade we
re-run the crossmatch + cutout pipeline against the new release; the
repeatable recipe is `cross_matching/RUNBOOK.md`. The `lsst_efficient` store
records its provenance (DP2, build date) in `meta.json` for exactly this
reason.

## Where do the names "COSMOS" and "DEEP2-3" come from? What is a "field"?

A **field** is just an informal name for a patch of sky a survey deliberately
targets and revisits — a social/scheduling concept, not a data structure.
(Rubin's formal sky units are *tracts* and *patches*; a *visit* is one
exposure's footprint; a "field" is the named region people agree to point at.)

Neither name is LSST's or HSC's originally — both are inherited:

- **COSMOS** = the *Cosmic Evolution Survey*, an HST program from the 2000s
  that surveyed this ~2 deg² equatorial patch. Because that made it the
  best-studied extragalactic region of sky, every subsequent observatory
  piled on (Chandra, Spitzer, Subaru/HSC, JWST — and now Rubin, which chose
  it as a Deep Drilling Field precisely *because* of that legacy). The name
  travels with the sky region.
- **DEEP2-3** = field 3 of the *DEEP2 Galaxy Redshift Survey* (Keck, 2000s).
  HSC-SSP adopted it as one of its Deep-layer fields and kept the name; our
  labels follow HSC-SSP because the raw_mmu sample is HSC-selected. Rubin has
  no special program there — to LSST it is anonymous wide-survey sky that
  happens to contain a famous patch.

So the chain is: earlier surveys named the regions → HSC-SSP built its deep
fields on them (that's where our galaxies come from) → Rubin overlaps them,
deliberately (COSMOS) or incidentally (DEEP2-3).

## Why can one band be missing where another has hundreds of exposures?

Because Rubin (like HSC and DECam) observes through **one filter at a time**.
LSSTCam has a filter carousel holding 5 of the 6 ugrizy filters (the sixth is
swapped in during daytime); each ~30 s "visit" exposes through exactly one of
them. Multi-band coverage of a sky position is therefore built up across
*separate visits on different nights* — the color channels of our stamps were
never taken simultaneously. The scheduler decides each visit's filter from
moonlight (u is scheduled in dark time because moonlight swamps the blue),
seeing, and program priorities — so per-band visit counts at any sky position
are essentially independent numbers. During commissioning, DEEP2-3's patch of
the wide survey simply never got u-band visits (while the COSMOS DDF ran
dedicated u sequences), hence: no u images to coadd there at all.

**Carousel details.** The camera owns six filters but the carousel physically
holds five; the sixth is stored off-camera and exchanged during daytime
maintenance. There is no permanently excluded filter — the out-filter rotates
with the **lunar cycle**: u is mounted around new moon (dark time — moonlight
scatters strongly in the blue and ruins u), and typically swapped out around
full moon in favor of the red bands (y and z barely care about moonlight).
Rotating the carousel to change filters *during* the night takes ~2 minutes,
so the scheduler batches visits by filter.

**Visit arithmetic.** One visit = a standard ~30 s exposure (+ ~2 s readout,
~5 s slew to a neighboring pointing) → roughly 40 s per visit, so a full
night yields on the order of **~800–1000 visits** covering thousands of
square degrees (9.6 deg² FoV). The whole visible sky gets touched every few
nights; any given wide-survey spot gets revisited in some band every few days.

**Coadds combine many standard visits — nobody exposes longer.** Every LSST
exposure is ~30 s, DDF or not. COSMOS's depth comes from *count*, not
exposure time: DDF scheduling points at the same field in long same-night
sequences (dozens of consecutive visits, cycling filters) plus revisits
across nights and seasons. Our COSMOS median `n_input_i = 89` means 89
separate 30 s exposures — ~45 minutes of total i-band integration accumulated
over many nights — aligned, outlier-rejected (cosmic rays, satellites), and
averaged into one coadd. Why not one 45-minute exposure? Saturation of bright
stars, cosmic-ray buildup, changing atmosphere, and — the real point — each
visit is a **time sample**: the DDF cadence exists to produce dense light
curves (supernovae etc.), and deep coadds are almost a by-product. This is
also why each matched galaxy has a forced-photometry light curve with one
epoch per visit on the Rubin side.

Consequences baked into our dataset: per-band `n_input` and `psf_fwhm` differ
because each band's stack comes from different nights with different seeing;
the bands of one galaxy are independent noise realizations (the DEEP2-3
"chroma speckle" in previews is exactly this); and band coverage is a
property of the *schedule*, not of the sources.

## Legacy Surveys: three telescopes, one catalog (see survey_footprints.png)

"Legacy Surveys" is a federation of sub-surveys on different telescopes:
DECaLS+DES on **DECam** (Blanco 4m, Chile) for Dec ≲ +32, and, for the north
galactic cap, **BASS** (Bok 2.3m; g,r) + **MzLS** (Mayall 4m; z). Mid-IR
W1–W4 fluxes come from space (**WISE** forced photometry — the FLUX_W*
columns in raw_mmu). All three of our patches are in the DECam zone, which is
why raw_mmu's legacy bands are literally prefixed `des-`: our Legacy images
are single-instrument (DECam), no cross-instrument heterogeneity there.

Note on M49 (Virgo): the public HSC-SSP release (wide/deep/ultradeep) does
NOT cover M49 — it sits ~1.5° north of the spring wide strip. "HSC observed
M49" claims refer to the *instrument*, not the survey: deep Virgo/M49 halo
work used Suprime-Cam (HSC's predecessor; e.g. the planetary-nebula surveys),
plus HSC PI-time pointings; the Virgo I/II/III ultra-faint dwarfs were found
in the SSP strip crossing the Virgo *constellation*, south of M49. That
non-SSP data is reachable in principle: Subaru raw frames go public in the
SMOKA archive after ~18–24 months (self-reduction with hscPipe required), or
ask the PI groups for their reduced mosaics. Rubin DP2 meanwhile made M49 its
most-visited field (1,105 visits, ugri only — First Look target), and M49 IS
inside the public Legacy/DECam footprint, so a Legacy×LSST pairing there is
possible today with the runbook — but for the disentanglement work M49 is a
distraction: crowded-cluster environment, different selection, no HSC-SSP
counterpart. At most a future anomaly stress-test field.

## "I thought we didn't overlap COSMOS?" — the definitive map

`sky_overlap_map.py` → `figures/sky_overlap_map.png` plots **all 101,725**
raw_mmu objects against the DP2 footprint, with per-patch zooms. Verdicts:
COSMOS ✔ (on the isolated DDF blob), DEEP2-3 ✔ (inside the wide band),
XMM-LSS ✘ (no DP2 coadds — the band ends just east of it). Two known sources
of the earlier "no overlap" impression: (1) the old
`debugging_contrastive/plot_radec.ipynb` plotted a *contiguous ~8k-row slice*
(START_INDEX=70400) of neighbours_v2.h5 — a field-ordered file, so the slice
shows only one patch, not all three; (2) any check against **DP1** was
correct at the time — DP1's seven fields (gray squares on the map) overlap
none of our patches; the COSMOS DDF and the wide band are new in DP2.

## Expansion options beyond the three patches

- **Autumn-strip triple overlap (available today):** the HSC-SSP *wide*
  autumn strip (RA 330→40, Dec −7..+6.5) crosses the DP2 band's eastern end
  around RA 0–40. HSC-wide depth (i≈26, shallower than our deep patches) but
  real three-survey sky — potentially tens of thousands more
  HSC×Legacy×LSST galaxies via the same runbook, if scale ever beats
  selection cleanliness.
- **XMM-LSS (~36k objects)** joins automatically when a future Rubin release
  coadds it — at *intermediate* depth (132 visits), between DEEP2-3 and
  COSMOS.
- **M49**: Legacy×LSST only (see above) — stress-test material, not a core
  field.

## Agreed research strategy (2026-09-02)

Framing: COSMOS and DEEP2-3 are not two instruments — they are two **depth
regimes of one instrument**, and the per-object metadata (`n_input_<band>`,
`psf_fwhm_<band>`) parameterizes the systematics *continuously*. Prefer
conditioning on those over a binary field label (a future XMM-LSS at
intermediate depth has no slot in a binary label).

1. **COSMOS-only HSC↔LSST** (34k pairs, full ugrizy, HSC-comparable
   quality): the core disentanglement experiment — isolates the *survey*
   difference with depth heterogeneity removed.
2. **DEEP2-3 as a domain-shift eval before it is ever a training set**:
   train COSMOS, test DEEP2-3 — measures whether the physics latent survives
   a 13× depth drop. Reference scale of the effect from the AE baseline:
   val recon MSE 0.114 (COSMOS) vs 0.341 (DEEP2-3).
3. **Joint training with continuous depth conditioning** (grizy, both
   fields, n_input/psf into the instrument pathway); diagnostic = physics
   latent's field-separability vs step 1 (UMAP colored by n_input_i).
4. **XMM-LSS stays a sealed external test set when it arrives** — never
   trained on until deliberately unsealed.

## Figure index (this folder)

- `figures/nvisits_maps.png` — all-sky context + per-field depth heatmaps,
  histograms, per-band medians (`nvisits_maps.py`)
- `figures/sky_overlap_map.png` — ALL 101,725 objects vs DP2 + DP1 history +
  per-patch zoom verdicts (`sky_overlap_map.py`)
- `figures/survey_footprints.png` — Legacy (split by instrument) vs HSC PDR3
  vs DP2 with official field names (`survey_footprints.py`)
- `figures/three_survey_<FIELD>.png`, `figures/preprocessing_preview_<FIELD>.png`
  — per-field example galleries (`field_galleries.py`)

## What this means for modeling (short version)

- `n_input_<band>` and `psf_fwhm_<band>` are per-object *instrument*
  conditioning variables — our two fields sample two very different depth
  regimes of the same instrument (the AE baseline already measured the
  consequence: val recon MSE 0.114 COSMOS vs 0.341 DEEP2-3).
- DEEP2-3 g-band (median 2 exposures) is the noisiest channel in the whole
  dataset — the chroma speckle in previews comes mostly from there.
- Anything trained now is "commissioning-depth LSST"; the same galaxies at
  DR1+ depths will make a natural controlled study of depth robustness (same
  sources, deeper truth).
