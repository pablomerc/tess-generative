# Xi-Squared Calibration Experiment

This experiment tests whether the flow matching model's posterior is well-calibrated
by checking if z-scores computed from samples follow N(0,1).

## Scripts

| Script | Purpose |
|---|---|
| `generate_samples.py` | Load pretrained model, generate N galaxies × M samples, save with ground truth (4 channels) |
| `calibration_test.py` | Compute z-scores, KS test, Q-Q plot, histogram vs N(0,1) |
| `compile_benchmark.py` | Benchmark `torch.compile` vs uncompiled inference speed |

## Model

- **Checkpoint:** `galaxy-flow-matching-neighbours/g2g9kvr4/checkpoints/latest-step=step=75000.ckpt`
- **Data shard:** `neighbor_batches/neighbors_shard_0000.h5`
- **Architecture:** Flow matching model conditioned on neighbour galaxies

## Calibration Test Logic

For each test galaxy:
1. Generate M posterior samples from the model
2. Compute per-pixel mean μ and std σ across samples
3. Compute z-scores: `z = (true_image - μ) / σ`
4. Flatten and pool all z-scores across all galaxies and pixels
5. Test if the pooled distribution is N(0,1): mean≈0, std≈1, KS test, Q-Q plot, histogram

Results are split by anchor survey (`hsc` vs `legacy`).

## Job Scripts

| Script | Galaxies | Samples | Steps | Purpose |
|---|---|---|---|---|
| `run_test.slurm` | 2 | 2 | 50 | End-to-end smoke test |
| `run_full.slurm` | 100 | 20 | 50 | Full run (fast, indicative) |
| `run_full_250steps.slurm` | 100 | 20 | 250 | Publication quality |
| `run_full_300g_250steps.slurm` | 300 | 20 | 250 | Larger publication run |

## Outputs

Results go in `outputs_<N>steps/`:
- `full_samples_<N>steps.h5` — generated samples + ground truth
- `full_calibration/calibration_{hsc,legacy}.png` — figures
- `full_calibration/calibration_report_{hsc,legacy}.txt` — text report
- `full_calibration/calibration_stats_{hsc,legacy}.json` — machine-readable stats
