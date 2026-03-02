# Contrastive Downstream Evaluation (2 Stages)

This folder provides an easy two-stage downstream pipeline for the contrastive baseline.

Objectives covered:
- `mmu`
- `neighbors`
- `legacy_provabgs`
- `hsc_provabgs`

## Stage 1: Prepare embeddings H5s

Script:
- `prepare_all_contrastive.py`

Outputs (in this folder by default):
- `downstream_mmu_{suffix}.h5`
- `downstream_legacy_provabgs_{suffix}.h5`
- `downstream_neighbors_{suffix}.h5`
- `downstream_hsc_provabgs_{suffix}.h5`

Each H5 contains:
- real embeddings
- untrained embeddings
- random embeddings
- labels

## Stage 2: Predict + plot

Script:
- `predict_all_contrastive.py`

Outputs:
- `predict_all_contrastive_{suffix}.csv`
- one plot per objective:
  - `predict_all_contrastive_{suffix}_mmu.png`
  - `predict_all_contrastive_{suffix}_neighbors.png`
  - `predict_all_contrastive_{suffix}_legacy_provabgs.png`
  - `predict_all_contrastive_{suffix}_hsc_provabgs.png`

## Easy runners

- `run_stage1_prepare.sh`
- `run_stage2_predict_plot.sh`
- `run_all_downstream.sh`

Edit `CHECKPOINT_PATH` and `SUFFIX` in `run_stage1_prepare.sh`, and keep the same `SUFFIX` in `run_stage2_predict_plot.sh`.

Then run:

```bash
cd /data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy_images/galaxy_model
./contrastive_baseline/downstream_evaluation/run_all_downstream.sh
```
