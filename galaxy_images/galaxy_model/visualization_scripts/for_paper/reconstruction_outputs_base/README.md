# Reconstruction outputs — `base` checkpoint

Each subfolder here is one run of `../reconstruction_base.py` against
`checkpoints/base/snapshot.ckpt` (ConditionalFlowMatchingModule). One folder per `--tag`
— folders are append-only, the script refuses to overwrite an existing tag.

## What's in a tag folder

| File                       | What it is                                                                 |
|----------------------------|----------------------------------------------------------------------------|
| `reconstruction_data.h5`   | All sampled images + inputs + metadata. Schema matches `replot_reconstruction.py`. |
| `reconstruction_plot.png`  | Raw layout: `SameGal | SameIns(1st) | Target | Sample×N | Mean`, one row per anchor. |
| `reconstruction_all.png`   | Styled "paper" layout: Inputs / Target / Output groups, colored panels, `→` row labels. Shows `Sample 1 | Sample 2 | Mean` (only the first two samples). |
| `manifest.json`            | `tag`, ckpt path, data dir, `num_examples`, `num_samples`, `seed`, chosen anchor positions, host, slurm job id. |

## How it was generated

```bash
sbatch --export=ALL,TAG=<tag>,NUM_EXAMPLES=16,NUM_SAMPLES=5,SEED=42 \
  galaxy_images/galaxy_model/visualization_scripts/for_paper/run_reconstruction_base.slurm
```

`SEED` controls the random anchor pick (passed to `numpy.random.default_rng`). The job:

1. Loads `checkpoints/base/snapshot.ckpt` with `ConditionalFlowMatchingModule.load_from_checkpoint`.
2. Picks `NUM_EXAMPLES` random anchors from `NeighborsEfficientDataset` at
   `/work1/jeroenaudenaert/pablomer/data/neighbors_efficient`.
3. For each anchor, generates `NUM_SAMPLES` flow-matching reconstructions via
   `model.sample(samegal, sameins, masks)`.
4. Saves the H5, renders both PNGs, posts both to `$DISCORD_WEBHOOK`.

## Replotting from a saved H5

The styled plot uses `replot_reconstruction.py` under the hood, so once the H5 is on disk
you can regenerate per-row plots without re-running the model:

```bash
cd galaxy_images/galaxy_model/visualization_scripts/for_paper

# All rows
python replot_reconstruction.py \
  --data reconstruction_outputs_base/<tag>/reconstruction_data.h5 \
  --all

# Specific rows (--index takes one or more integers)
python replot_reconstruction.py \
  --data reconstruction_outputs_base/<tag>/reconstruction_data.h5 \
  --index 0 3 7
```

By default the rendered PNG lands next to the H5 (e.g. `reconstruction_all.png`,
`reconstruction_0_3_7.png`).

## Adding a new variant

Just resubmit with a fresh `TAG` (and a different `SEED` if you want a different anchor pick):

```bash
sbatch --export=ALL,TAG=base_v2,NUM_EXAMPLES=16,NUM_SAMPLES=5,SEED=7 \
  galaxy_images/galaxy_model/visualization_scripts/for_paper/run_reconstruction_base.slurm
```

The first run that tries to write into an existing `<tag>/` will hard-fail before
generating anything — that's intentional.

## Tweakables (env vars on `sbatch --export`)

| Var            | Default | Effect                                              |
|----------------|---------|-----------------------------------------------------|
| `TAG`          | —       | Required; subfolder name.                           |
| `NUM_EXAMPLES` | 16      | Random anchors to sample.                           |
| `NUM_SAMPLES`  | 5       | Stochastic samples per anchor (raw plot shows all). |
| `SEED`         | 42      | Anchor pick seed. Sampling itself uses model RNG.   |

For other knobs (`--max-neighbors`, `--crop-size`, `--output-root`) edit the slurm script
or invoke `reconstruction_base.py` directly.
