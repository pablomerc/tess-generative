# Review walkthrough: `build_lsst_efficient.py`

Read this next to the code. Line numbers refer to the current file; each section
says what the block does, *why* it's that way, and what's worth double-checking.

## The one-paragraph version

Stream all DP2 cutout shards once, assemble each galaxy's bands into a
`(6, 160, 160)` float16 cube (nanomaggies, NaN where a band doesn't exist),
append cubes to a flat binary in arrival order, then build the lookup
structures (meta table, parent-catalog row map), validate, and atomically
publish the directory.

## Constants — lines 44–61

- `SHARD_GROUPS` (L50): run2_full first, run1 pilot second. Order matters —
  the first complete version of an object wins, and run2 is the authoritative
  full extraction; the pilot only fills anything run2 lacks.
- `BANDS = g,r,i,z,u,y` (L55): non-standard order on purpose — always-present
  bands form a contiguous prefix so griz training slices `[:4]`, same trick as
  `hsc_image.bin` (grizy). `CORE` = the four bands an object must have.
- `NJY_PER_NMGY = 10^((31.4−22.5)/2.5) ≈ 3630.78`: DP2 fluxes are nJy
  (AB zeropoint 31.4); dividing brings them to nanomaggies (zeropoint 22.5,
  the unit the whole preprocessing stack works in). Stored this way because
  raw nJy overflows float16 (max 65 504) on bright pixels.
- **Check:** `F16_MAX` clip (L59, used L98–101) only guards genuinely
  saturated pixels; the final report prints how many were clipped (expect ~0).

## `Builder` class — lines 63–129

State for the streaming pass. Key fields (L67–74): the open binary file,
`rows` (meta records; list index == compact row), `row_of` (object id →
compact row), `pending` (partially assembled objects), plus counters and the
spot-check reservoir.

### `add_row` — L76–84
One shard row = one (object, band) stamp. Accumulates the 160×160 flux plane
plus `psf_fwhm` / `n_input_images` into `pending[oid]`. Field/ra/dec are taken
from the first row seen for the object.

### `finalize` — L85–121, the heart of the builder
Called when we believe no more band rows are coming for an object.
1. L87–89: **inclusion rule** — no complete griz → count as incomplete and
   drop (patch-edge rarities; DEEP2-3's missing u does *not* trigger this
   because u isn't in `CORE`).
2. L90–103: build the 6-band cube. Missing bands stay NaN (never 0 — zero
   looks like real sky to a model). Per-band: divide by 3630.78, clip anything
   above float16 max (counted).
3. L104–110: reservoir keeps ~20 full-precision cubes in memory so the end of
   the run can verify the file bytes against independently re-derived values.
4. L112–120: append meta row and **write the cube to the binary**. The compact
   row index is simply "how many objects were written before me" — write
   order defines row order, and `lsst_meta.parquet` row *k* describes bin
   row *k*.
- **Check:** nothing is written for excluded objects — `row_of` and the bin
  stay in lockstep because both are only touched in the success path.

### `flush_untouched` / `flush_all` — L122–129
Why not finalize as soon as an object has 6 bands? Because we don't know how
many bands an object *should* have (4, 5, or 6). Instead: an object is
finalized when a whole shard goes by without adding anything to it. The
extraction wrote rows sequentially, so an object's bands can straddle at most
two *consecutive* shards — one shard of look-ahead is provably enough.
`flush_all` drains whatever remains after the last shard.

## `main()` streaming loop — lines 132–160

- L134–137: refuses to run if the output dir exists (no-overwrite policy);
  builds into `lsst_efficient.building/`.
- L145–157: per shard — read only the needed columns, iterate rows, skip
  objects already written (that's how run1 duplicates are ignored), accumulate,
  then `flush_untouched(touched)` implements the look-ahead rule.
- **Check:** memory stays bounded — `pending` holds at most ~2 shards' worth of
  objects (~1 GB worst case) and each shard DataFrame is freed before the next.

## Meta + row map — lines 162–182

- L164–170: match distance to the DP2 object is joined in from the
  cross_matching paired index (best match per id).
- L173–182: **row map**. Parent catalog's `object_id_hsc` is a *string*
  (empty for legacy-only rows); non-empty ids are cast to int64 and looked up
  in `row_of` → int64 array of length 468 197, −1 where a parent row has no
  LSST view. Also computes a sha256 over the parent's id column — the
  fingerprint a loader uses to detect that the parent store was rebuilt and
  the map is stale.
- **Check:** ids in the catalog are unique when non-empty (verified in the
  design phase), so `.map()` is well-defined.

## Validation + publish — lines 184–230

Runs *before* anything is visible at the final path:
1. inf scan over ~200 sampled rows (NaN is fine, inf is not).
2. Spot-check: for the ~20 reservoir objects, read their bytes back from the
   binary and compare to the independently held float32 cubes after float16
   rounding — catches any offset/ordering bug in the write path.
3. Prints per-field counts (expect COSMOS ≈34.1k, `with_u≈with_y≈34k`;
   DEEP2-3 ≈31.0k with `with_u=0`), mapped parent rows (≈65k / 468 197),
   clipped pixels, bin size (≈20 GB).
4. Writes `meta.json` (all stats + provenance) and a `README.md` derived from
   the module docstring, then `os.rename(.building → lsst_efficient)` — the
   publish is atomic; a crashed or failed build leaves only `.building` to
   inspect and can never produce a half-valid store.

## What this deliberately does NOT do

- No clamp / compression / normalization — load-time decisions, kept out of
  the store (the only baked-in transform is the unit conversion, for float16's
  sake).
- No ivar/mask planes (parent stores are flux-only; add `lsst_ivar.bin` later
  if ivar-weighted normalization wins the DEEP2-3 argument).
- No resume logic — a restart costs ~1 h; atomicity was judged more valuable
  than resumability at this runtime.

## Failure modes to keep in mind

- If the parent `neighbors_efficient` store is ever rebuilt, `lsst_row_map.npy`
  is stale — the fingerprint in `meta.json` is the tripwire; the loader
  (dataset.py, next step) must check it at init.
- `iterrows` over ~400k rows is the slow part (~10–20 min CPU); I/O over NFS
  dominates the rest. If it ever needs to be faster, vectorizing the per-shard
  groupby is the lever — not needed at one-off runtime.
