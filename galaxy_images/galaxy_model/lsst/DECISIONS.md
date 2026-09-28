# HSC × LSST dual-encoder: decisions log

Running record of the choices behind the first cross-survey disentanglement
run that uses LSST DP2 as a view. Each entry gives the decision, the numbers
it rests on, and where in the code it is enforced. Append, don't rewrite;
if a decision is reversed, add a dated entry that says so.

Background primer: `data_exploration/DATASET_NOTES.md`. Strategy it slots
into: the four-step plan at the end of that file (COSMOS core experiment →
DEEP2-3 as domain-shift test → joint training with depth conditioning →
XMM-LSS sealed).

---

## 2026-09-26 — scope of the first run

### D1. Field: COSMOS only
DEEP2-3 is held out entirely as the domain-shift test set; it is not
trained on. COSMOS is the only field whose LSST depth is in the same regime
as HSC (i-band limiting magnitude ≳ 25.6 vs HSC Deep ~26.9; DEEP2-3 ≈ 24.5).
Object counts in the LSST store: COSMOS 34,132, DEEP2-3 31,031.
*Enforced in:* the dataset's anchor selection (step 3), not in any index or
store, so the same artefacts serve later joint-field runs.

### D2. Bands: g, r, i, z
The dual encoder is `in_channels=4`. The LSST store's channel order is
g,r,i,z,u,y, so `[:4]` is griz, matching `hsc_image.bin[:4]` and Legacy.
No architecture change. u is absent in DEEP2-3 anyway; y is deferred.
*Enforced in:* `preprocessing.preprocess_lsst(bands=("g","r","i","z"))`.

### D3. Quality cut: `n_input ≥ 5` in every one of g, r, i, z
Applied to anchors AND to neighbour slots (a near-empty stamp fed to the
instrument encoder is worse than no stamp). Counts on 2026-09-26:

| requirement | COSMOS objects kept |
|---|---|
| none | 34,132 |
| griz ≥ 5 | **33,346** (97.7%) |
| grizy ≥ 5 | 31,543 (y is what fails: 95% of drops) |

Neighbour survival under the cut: 99.9% of passing COSMOS anchors keep ≥ 5
passing neighbours among their 20; 99.0% keep ≥ 15.
Why 5: the AE baseline's anomaly rankings were dominated by `n_input ≤ 1`
pedestal-broken stamps; the LSST matrix froze its recipe *without* a cut for
comparability, so this run deliberately departs from that recipe.
A uniform 5 is impossible in DEEP2-3 (g median 2 exposures; 99% fail); if
DEEP2-3 is ever trained on, use per-band thresholds, e.g. r,i,z ≥ 3 and g ≥ 1.
*Enforced in:* the dataset at load time from `lsst_meta.parquet`
(`n_input_<band>`), as a boolean mask over rows. NOT baked into the
neighbour index or the RAM store, so the threshold can change without a rebuild.

### D4. Same-instrument neighbours for LSST come from a k-NN inside the LSST store
The parent catalogue's `neighbor_idx_hsc/_legacy` cannot be reused: all
65,163 LSST rows are parent *anchor* rows and the 364,456 neighbour-pool
rows were never crossmatched to DP2, so zero of an anchor's HSC neighbours
have an LSST stamp (measured). Instead: 20 nearest other store rows in the
same field, exact great-circle distance, self excluded, geometry only.

| field | 1st neighbour | 5th | 20th |
|---|---|---|---|
| COSMOS | 19″ | 58″ | 139″ |
| DEEP2-3 | 19″ | 58″ | 135″ |

At these separations a neighbour shares the anchor's coadd depth and PSF
(DP2 depth varies on the dither scale, degrees).
*Built by:* `data_preprocessing/build_lsst_neighbors.py` →
`data/lsst_efficient/lsst_neighbors.{npz,json}` (store-row space, arcsec,
plus `parent_row` = inverse of `lsst_row_map.npy`). Existing store files are
never modified.

### D5. Data path: RAM-resident 48 px store, mmap as fallback
Measured on the login node, anchor + 5 neighbours, griz 160 px:
LSST mmap 13.6 ms median / 25 ms p90 cold, 1.9 ms warm; parent HSC store
5.0 ms / 6.2 ms. Both are far below the ~1 s training step, so the old HDF5
problem (4.5 s per batch) cannot recur. Still, following
`contrastive_baseline/neighbors_ram48.py`, the run will preprocess once to
48 px griz float16 and hold everything in RAM so `__getitem__` is pure
indexing. Size for COSMOS: the 34,132 anchors reference 165,472 distinct
parent rows for HSC/Legacy neighbours (the neighbour pool is mostly
non-anchor rows), so

    HSC + Legacy 48 px griz float16: 6.1 GB  +  LSST: 0.6 GB  ≈ 6.7 GB

(5.8 GB with 5 neighbours; 13.5 GB in float32). Fits any node.
**The HSC/Legacy half already exists:** `data/efficient_neighs_48/`
(`hsc48.bin`, `legacy48.bin`, float16 (468197, 4, 48, 48), 8.6 GB each,
built 2026-07-25 by `contrastive_baseline/build_ram48.py`, catalogue
row-aligned with `neighbors_efficient`, neighbour indices preserved). Only
the LSST counterpart needs building: 65,163 rows × 4 × 48 × 48 float16 =
1.2 GB (COSMOS alone 0.6 GB), a few minutes of sequential reading.
Before relying on the July store, spot-check a handful of rows against a
fresh `NeighborsEfficientDataset._preprocess` of the 160 px store, in case
the HSC/Legacy preprocessing changed after 2026-07-25.
Login-node caveat: `ulimit -v` is 32 GB there, so the 120 GB parent binary
cannot be mmapped on the login node; use seek+read for smoke tests, as
`single_latent/ae_baseline/data.py` does. The 48 px stores are small enough to mmap anywhere.
*Built by:* step 2 script (to be written): `data_preprocessing/build_lsst48.py`,
mirroring `build_ram48.py` but calling `preprocess_lsst`.

### D6. Storage budget
Group quota 1.9 TB, 1.7 TB used (all Pablo), 144 GB free at 2026-09-26.
Everything in this plan is < 10 GB. The 582 GB `data/neighbours_v2.h5`
(float32 160 px HSC + Legacy + Legacy ivar; superseded by
`neighbors_efficient`, and the `.vds` files point at `neighbor_batches/`,
not at it) is the obvious reclaim if space is ever needed — but
`config.py`/`data_factory.py` and the scale-ablation configs still name it
as the HDF5-mode default, so confirm nothing in HDF5 mode is still wanted.

### Open — D7. Train/val split convention
Two conventions exist: the LSST AE baseline froze a 90/10 split by
`sha1(object_id_hsc)` (`single_latent/ae_baseline/data.py`, `val_object_ids.txt`), while
the neighbours work excludes the 5,466 downstream-holdout anchors
(`*-DWNVAL` variants). Comparability with one baseline or the other depends
on which is chosen. Decide before step 3.

### Known risk — resampling confound
Pixel parity zooms LSST by 0.84 to HSC's 8.0″ field; the bilinear resample
introduces correlated pixel noise that HSC lacks, so a model could identify
the survey from noise texture alone. Planned control: pass HSC stamps
through the same resample operation and check whether the instrument
probe's survey accuracy changes.

---

## Step order for the first run

1. `build_lsst_neighbors.py` — neighbour index (seconds). ← current
2. `build_cosmos48.py` — RAM store: 48 px griz float16 for COSMOS anchors,
   their HSC/Legacy pool rows, and their LSST rows; `ok` mask per D3
   stored alongside but not applied.
3. Dataset class (same output contract as `NeighborsEfficientDataset`,
   `collate_neighbors` unchanged), anchor = HSC or LSST at random, samegal =
   the other, sameins = same-survey neighbours; D1–D3 applied here.
4. Config + sbatch (mi2101x), smoke test, launch: 15–20k steps.
5. Probes: MLP R² on frozen embeddings, instrument targets now include
   per-object `n_input_*` and `psf_fwhm_*`; leak check = physics-latent R²
   on `n_input_i` ≈ 0; UMAP of physics latent coloured by survey and n_input.
