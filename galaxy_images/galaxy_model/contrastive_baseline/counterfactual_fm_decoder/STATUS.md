# CF frozen-embedding FM decoder — status & next steps

Last updated: 2026-07-26 (post-review, stage 1 relaunched)

## What this package is

[`counterfactual_fm_decoder/`](.) trains a flow-matching UNet that reconstructs a galaxy
image from **frozen contrastive embeddings of other images** (never the target):

| Slot | Source image | Frozen tower |
|------|--------------|--------------|
| Physics | `samegal` (cross-survey pair) | `encoder_galaxy` → 64-d |
| Instrument | ≤5 same-survey neighbors | `encoder_instrument` → K×64-d |

Two conditioning styles (same H5):

- **B (primary):** `tokens` + `xattn` — sequence `[gal, inst_n1, …, inst_nK]`
- **A:** `meanpool` + `adagn` — `concat(gal, masked_mean(inst))` → 128-d

Compare recon MSE on the matched diffusion-ablation val protocol against
**fm-control** and the **paper base** checkpoint via `eval_cf_recon_mse.py`.

W&B project: `galaxy-counterfactual-fm-decoder` (online — `~/.netrc` already holds the
`api.wandb.ai` credential, no login needed).
Discord webhook: wired into every SLURM script.

---

## Root cause of the 2026-07-25/26 failure

The earlier diagnosis ("precompute is inherently slow, give it 24 h and take snapshots")
treated a symptom. The actual bug was **reading the wrong store**.

`precompute_cf_embeddings.py` built the H5 from `efficient_neighs` — the 160×160 float16
**Ceph mmap** store. That needs 7 random row reads per anchor (anchor-hsc + anchor-legacy
+ K neighbors, ~256 KB each) and delivered **~1 anchor/s** ⇒ ~29 h for 103,741 anchors.
Job `18875467` was killed at its 4 h wall having done ~10k, leaving an unopenable 394 MB
H5 (`bad object header version number` — the file was never closed, so the `f.attrs[...]`
block never ran).

`efficient_neighs_48` (the ram48 store from `build_ram48.py`) is the fix, and it happens
to correct two silent correctness problems at the same time:

1. **Speed** — sequential 17 GB slurp into RAM, after which the pass is GPU-bound.
2. **Encoder fidelity** — `train_contrastive_engaging.py` defaults to
   `efficient_neighs_48`, so `best-epoch=599-step=240000.ckpt` was *fit* on the float16
   ram48 tensors. The old path fed it float32 tensors from a different store.
3. **Eval comparability** — `eval_cf_recon_mse.py` scores the image baselines against the
   ram48 loader's `x_1` but the CF decoder against the H5 targets. Building from ram48
   makes those the *same pixels*; from the 160×160 store they differed by float16
   rounding (~1e-6 in MSE — never enough to invalidate a result, but unguarded).

Both stores share `catalog.parquet` (`build_ram48.py` copies it), so anchor order, anchor
count and the `idx%2` survey assignment are identical either way. Verified offline that
`make_train_val_split(0.05, 42)` reproduces `data_factory`'s `random_split` **index for
index** at n=103,741 (val=5,187) — so the CF val split is exactly fm-control's, and no
eval galaxy leaks into CF training.

### Other findings from the review

| Severity | Finding |
|---|---|
| Blocker | The corrupt H5 would have been **silently loadable** had the kill landed a moment later: datasets are pre-allocated, so unwritten rows read as zero targets / zero embeddings / all-False masks. `build_catalog_index` would also collapse every unwritten row onto key `(0,0)`. |
| Real | Checkpoint selection was asymmetric — CF selected `best` on `val/loss` (velocity MSE) while fm-control selects on `val/mse`, then the two were compared *on recon MSE*. |
| Real | `val/mse` was computed inside a `if not self.logger: return` guard, so it did not exist under `--no-wandb` and could not drive selection. |
| Real | The conditioning UMAP was recomputed at every validation — but the encoder is **frozen**, so the figure is byte-identical every time. |
| Real | The recon grid ran `n_rows` separate 250-step ODE solves at batch `n_samp` instead of one solve at batch `n_rows*n_samp`. |
| Real | `emb_dim` came from `hparams.embedding_dim`, which only equals the true encoder width for `pool="avg"` or default-`token_dim` `conv1x1`. |
| Minor | `tok_mask` was threaded through dataset → collate → module and then never used; pad tokens still receive attention via the K/V biases. This *matches* the paper model, so it is now an explicit opt-in flag rather than dead code. |
| Minor | float32 targets doubled the H5 (~3.8 GB) although the source data is float16-precision. POOL is at 90%. |
| Minor | The eval manifest recorded indices but not the store, so a store swap with a shared catalog would pass unnoticed. |

---

## What changed (this pass)

**`precompute_cf_embeddings.py`**
- `--loader ram48|efficient` (**default ram48**); `--data-dir` default `efficient_neighs_48`.
- `--target-dtype float16` default: lossless w.r.t. the ram48 store, halves the H5.
- `emb_dim` from `encoder_*.out_dim` (raises if the towers disagree, plus a per-batch
  width assert) instead of `hparams.embedding_dim`.
- `attrs["complete"]=True` written **last**, plus `n_anchors_total`, `source_loader`,
  `target_dtype`, `n_real_neighbors`; refuses to stamp `complete` unless `ptr == n`.
- Kept the atomic `.tmp` → `os.replace`; progress line now reports ETA.
- Milestone snapshots kept but **off by default** — they were insurance against the 29 h
  pass, and on a 90%-full pool they'd cost ~1.3 GB for nothing.

**`cf_embedding_dataset.py`**
- `require_complete=True`: refuses an H5 without `complete`, naming the zero-row failure mode.
- `is_full_store` property — distinguishes a full corpus (split matches data_factory)
  from a snapshot (it does not).
- Targets stay float16 in RAM, cast per item → ~1.9 GB resident instead of ~3.8 GB.
- `build_catalog_index` raises on duplicate `(row_idx, survey)` keys.

**`cf_fm_module.py`**
- Recon MSE hoisted **above** the logger guard, so `val/mse` always reaches
  `callback_metrics` and can drive checkpoint selection (works under `--no-wandb` too).
- UMAP computed **once** (`_umap_logged`); collection stops after that.
- Recon grid uses one batched ODE solve (`repeat_interleave`) for all rows × samples.
- New `attn_mask_pads` hparam (**default False = paper parity**); when True, passes
  `encoder_attention_mask` so pad neighbor tokens are masked out of cross-attention.

**`train_cf_fm.py`**
- `--monitor` (**default `val/mse`**) so "best" means the same thing as fm-control's
  `monitor_metric`.
- Refuses a partial H5 unless `--allow-partial-h5`, with an explicit leakage explanation.
- `--attn-mask-pads` passthrough; `auto_insert_metric_name=False` on the periodic callback
  (filenames were coming out as `latest-step=step=1234.ckpt`).

**`eval_cf_recon_mse.py`**
- First-batch guard: `max|H5 target − loader x_1|` must be under `--target-tol` (1e-2) or
  it raises. This is the one comparison-invalidating failure mode the manifest can't catch.
- Manifest records and compares `data_mode` + `data_dir`; documented why `num_steps` is
  recorded but deliberately not compared.

**SLURM**
- `precompute_cf_embeddings.slurm`: ram48 store, 3 h wall (was 24 h), `mem=96G`, snapshots
  off, single `--gres`.
- `train_cf_fm_long.slurm` is now the **aryeh trainer**: `gpu:h200:1`, `mem=64G`, 2-day
  wall, `WANDB_MODE` guard, Discord pings, `--monitor val/mse`, env-overridable
  `COND_STYLE`/`INJECTION`/`RUN_DIR`/`MAX_STEPS`.

**Verification** — 17/17 offline checks pass (no cluster data, no GPU): complete-flag
rejection, full-vs-snapshot detection, duplicate-key detection, float16 round-trip, cond
shapes for both styles, zero-neighbor anchors zeroed, finite loss + sample for
tokens/xattn, meanpool/adagn and tokens/xattn+`attn_mask_pads`, and split parity with
`data_factory`.

---

## Live jobs

| JobID | Name | Partition | State |
|-------|------|-----------|-------|
| `18930807` | `cf-precompute` | `pg_mki_aryeh` | ✅ **COMPLETED** 01:18:09 |
| `18931378` | `cf-fm-long` (tokens/xattn) | `pg_mki_aryeh` | ▶ RUNNING (2-day wall) |
| ~~`18930808`~~ | `cf-fm-tokens` | `mit_normal_gpu` | cancelled — replaced by the aryeh job |

### Stage 1 result (2026-07-26 17:00)

`cf_embeddings_all.h5` — 2.0 GB, `complete=True`, `num_examples = n_anchors_total = 103741`,
`source_loader=ram48`, float16 targets.

- **Encode throughput: 4,260 anchors/s** — the whole 103,741-anchor pass took **0.4 min**,
  against ~1 anchor/s (~29 h projected) on the old store. The 78 min of job wall was
  essentially all the 17 GB Ceph slurp, which was running ~40× below normal today.
- Neighbor slots filled: 518,286 / 518,705 = **99.92%** (so pad tokens are ~0.08% — the
  `attn_mask_pads` question is confirmed near-irrelevant in practice).
- Sanity: `target[hsc] mean=+0.039 std=1.031`, `target[legacy] mean=-0.001 std=1.003` —
  both ≈0/≈1 as expected after per-survey normalization.
- `survey[:6] = [0 1 0 1 0 1]`, confirming the `idx%2` anchor-survey assignment survived.

### Split parity confirmed live

The training job reports **train=98,554 / val=5,187** — exactly the `val=5187` the offline
parity test predicted from `data_factory`'s `random_split`. The CF val split *is*
fm-control's, so the matched recon-MSE comparison is sound.

W&B: <https://wandb.ai/tess-ml/galaxy-counterfactual-fm-decoder/runs/cf-fm-tokens-xattn-long>
(161 M-param UNet, `monitor=val/mse`).

Cosmetic only: the banner prints `H100=False` because `is_h100_gpu()` greps for "h100" and
these are H200s. The flag is used for nothing but that log line.

**Why aryeh, not `mit_normal_gpu`:** that partition sits at the 2-GPU QOS cap with several
jobs queued, while aryeh scheduled instantly. Also `train_cf_fm_tokens.slurm` self-
resubmits via `sbatch $0`, so its own `#SBATCH --partition=mit_normal_gpu` wins on every
segment after the first — a CLI `-p pg_mki_aryeh` only moves segment 1. The single 2-day
`train_cf_fm_long.slurm` job sidesteps both problems.

**Stage-1 timing note:** Ceph was badly contended, so the 17 GB slurp ran at ~6 MB/s
(~65 min) rather than the ~1 min the ram48 docstring quotes; encoding itself took 0.4 min. If the FS is this slow again, `--no-ram` (mmap,
~18 KB random reads instead of 256 KB) is the fallback — the precompute makes a single
in-order pass, so it doesn't actually need random access.

---

## What to do next

### 1. Confirm stage 1 landed

```bash
python - <<'PY'
import h5py
f = h5py.File("/orcd/pool/007/pablomer/outputs/counterfactual_fm/cf_embeddings_all.h5")
print(dict(f.attrs))          # complete=True, num_examples == n_anchors_total == 103741
print({k: f[k].shape for k in f})
PY
```
The log's per-survey `target mean/std` should read ≈0 / ≈1.

### 2. Tokens (B) trains automatically

`18931378` releases on `afterok`. Watch W&B `cf-fm-tokens-xattn-long` or Discord.

### 3. Then meanpool (A) + matched eval

```bash
COND_STYLE=meanpool INJECTION=adagn sbatch train_cf_fm_long.slurm
sbatch eval_cf_recon_mse.slurm   # resolves best-*.ckpt per run dir
```

✅ `eval_cf_recon_mse.slurm` now searches both the `-long` (aryeh) and chain run dirs per
arm and warns loudly if an arm has no `best-*.ckpt`, so a CF row can no longer go missing
silently.


### Open decisions

- **Run A and B in parallel?** aryeh had capacity; launching meanpool now would roughly
  halve wall-clock to a complete table. Held pending a look at aryeh contention.
- **`--n 256` in the eval** gives ~128 anchors/survey, so the SEM is loose. Cost is
  250 ODE steps × n per model, so 512–1024 is affordable and tightens the headline gap.
- **`attn_mask_pads` as a third arm?** Default-off is right for the matched comparison,
  and only ~0.4% of neighbor slots are pads (638–640 of 640 real), so the expected effect
  is tiny. Probably not worth a run.
- **fm-paper caveat stands:** that checkpoint may have trained on these val galaxies; the
  eval already annotates the row. fm-control is the honest baseline.

---

## Success checklist

- [x] `cf_embeddings_all.h5` opens, `complete=True`, `num_examples=103741`
- [x] Target mean/std ≈ 0 / 1 per survey in the stage-1 log
- [~] `cf-fm-tokens-xattn-long` logging to W&B with `val/mse` falling (running)
- [ ] `best-*.ckpt` under `runs/cf-fm-tokens-xattn-long/checkpoints/` (selected on `val/mse`)
- [ ] meanpool run complete
- [ ] `eval_cf_recon_mse` CSV with fm-control + fm-paper + both CF arms, target-tol guard passing

---

## Related paths

| Item | Path |
|------|------|
| Frozen contrastive ckpt | `/orcd/pool/007/pablomer/outputs/contrastive/best-epoch=599-step=240000.ckpt` (pool=avg, out_dim=64) |
| **ram48 store (now the source)** | `/orcd/pool/007/pablomer/efficient_neighs_48` |
| 160×160 store (slow; provenance only) | `/orcd/pool/007/pablomer/efficient_neighs` |
| CF outputs | `/orcd/pool/007/pablomer/outputs/counterfactual_fm/` |
| Paper base | `galaxy_images/galaxy_model/checkpoints/base/snapshot.ckpt` ✔ present |
| fm-control best | `checkpoints_new/diffusion_ablation/best/fm-control-ram48-h200-75k/…-step=104000.ckpt` ✔ present |
| Non-CF probe (for contrast) | `../embedding_fm_decoder/` |
