# Cross-predict run status (May 6, ~18:34 EDT)

## What we set out to do

Add **cross-predict** results (predict one survey's instrument props from
the *other* survey's image) as a 5th orange "Cross-predict" series on the
publication plot `unified_three_probes_aion_phys-flat_instr-pooled.png`.

Code already existed in `cross_predict_experiment/cross_predict.py` — train
a fresh ResNet18 from scratch, two directions:

- **HSC→Legacy**: predicts `legacy_PSFSIZE/PSFDEPTH/GALDEPTH/NOBS_{G,R,I,Z}` (banded)
- **Legacy→HSC**: predicts HSC variance + derived PSF FWHM


## Round 1 — quick start, then realized targets were incomplete

I edited `cross_predict.py`:
- hardcoded engaging path (`/orcd/pool/007/pablomer/neighbours_v2.h5`)
- swapped to engaging Discord webhook
- skipped the comparison-plot step (it referenced an MIT-only CSV)
- added per-5-epoch progress pings

Wrote `run_engaging.slurm`, submitted both directions in parallel:
**13446880** + **13446881**.

Then you noticed E(B-V) and HSC `a_*` extinction would be missing from
the cross-predict bars. I checked `neighbours_v2.h5`:

| target                       | in H5? |
|------------------------------|--------|
| `legacy_EBV`                 | yes (just never asked for) |
| `legacy_MW_TRANSMISSION_*`   | no |
| `hsc_a_g/r/i/z/y`            | no |

You said: ignore `MW_TRANSMISSION`, crossmatch `/home/pablomer/orcd/scratch/hsc_downstream`
for `hsc_a_*`, and confirmed `legacy_EBV` is already there.

I scancelled both jobs (~10 min sunk).


## Between rounds — sidecar build

- `build_extra_targets.py` reads `object_id_hsc` (decoded from bytes) from
  `neighbours_v2.h5` and joins to `hsc_downstream/catalog.parquet` `object_id`.
  **100% join rate** on all 103,741 MMU rows. RA/DEC agree to 6 decimals as a
  sanity check.
- Wrote `extra_targets_per_h5_row.parquet` (~10 MB), aligned to H5 row indices.
- Edited `cross_predict.py`:
  - appended `legacy_EBV` to LEGACY_TARGETS
  - added `HSC_TARGETS_FROM_SIDECAR = ["hsc_a_g", ..., "hsc_a_y"]`
  - plumbed sidecar reads into `CrossPredictDataset._init_indices_and_targets`
- Smoke-tested both directions on a 1024-sample subset.


## Round 2 — submitted, asymmetric outcome

Resubmitted in parallel:
- **13448060** (hsc→legacy, **17 targets**)
- **13448061** (legacy→hsc, **12 targets**)


### `13448061` — DONE in 53 min ✅

CSV: `cross_predict_experiment/r2_results_legacy_to_hsc.csv`

```
hsc_a_g/r/i/z/y     R² ≈ 0.67  (all five bands)
hsc_z_psf_fwhm      R² = 0.65
hsc_*_psf_fwhm      R² = 0.24-0.31
hsc_*_variance_*    noise as expected
AVERAGE             R² = +0.385
```

Why this worked: 35k train samples (the HSC variance NaN filter dropped 65%
of MMU rows, leaving 35k of 103k) fit comfortably in the OS page cache, so
after epoch 1 every read was nearly free.


### `13448060` — STUCK on epoch 1-2 ⚠️

93k train samples × 5×160×160×float32 = **~46 GB per epoch**. 32 GB RAM means
page cache can't cover the dataset. Combined with `num_workers=0` in the
DataLoader (serial reads in the main process), effective throughput is
**~10-25 MB/s**.

Current state (last poll, 18:34 EDT):
```
elapsed:           2:13:37 of 6:00:00
state:             RUNNING (not stuck — actively reading)
last log line:     Epoch 1/50  train_loss=0.86097 val_loss=0.98289 avg_r2=0.037
                   (printed at the 92-min mark — that's it; no epoch 2 yet)
disk read total:   84.75 GB  (so partway through epoch 2)
CPU time used:     2:47       (the rest is GPU-idle, blocking on disk)
remaining budget:  ~3:46
projected:         ~46 min/epoch from here → SLURM timeout at ~epoch 6,
                   no CSV written (script only saves at the end of all 50 epochs)
```


## Decision

`r2_results_hsc_to_legacy.csv` won't be produced unless we intervene.

**Option A — scancel + restart with `num_workers=8`** (RECOMMENDED)
- 8 CPUs are allocated to the job and currently idle
- Parallelizes HDF5 reads
- Projected: epoch 1 ~12-15 min, subsequent ~3-4 min each, **total ~3 h**
- One-line edit: `cross_predict.py:387,392` (`num_workers=0` → `num_workers=8`
  in both train and val loaders)
- Then: `scancel 13448060` + `sbatch --export=ALL,XPRED_DIRECTION=hsc_to_legacy run_engaging.slurm`

**Option B — let it run out**
- SLURM timeout fires at ~22:20 wall-clock
- Trap fires `❌ failed (rc=…)`
- We get nothing for hsc→legacy direction

I'm holding on Option A — won't `scancel` autonomously per the rule we set.
Just reply "go" (or "do it" / "kill it") and I'll do the kill+restart.
