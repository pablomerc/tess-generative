from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import analyze_delta_r2 as analysis
import run_delta_r2 as probe


def _synthetic_inputs(tmp_path: Path, n: int = 50) -> tuple[Path, Path]:
    rng = np.random.default_rng(17)
    signal = rng.normal(size=n).astype(np.float32)
    hsc = np.column_stack([signal, rng.normal(size=(n, 3))]).astype(np.float32)
    legacy = np.column_stack([0.5 * signal + rng.normal(scale=0.5, size=n), rng.normal(size=(n, 3))]).astype(np.float32)
    combined = np.concatenate([hsc, legacy], axis=1)
    y = (1.7 * signal + rng.normal(scale=0.1, size=n)).astype(np.float32)
    y[[3, 21]] = np.nan

    h5_path = tmp_path / "synthetic.h5"
    with h5py.File(h5_path, "w") as handle:
        handle.attrs["checkpoint_name"] = "synthetic-model"
        group = handle.create_group("overlap")
        group.create_dataset("hsc_e1", data=hsc)
        group.create_dataset("legacy_e1", data=legacy)
        group.create_dataset("combined_e1", data=combined)
        group.create_dataset("labels/hsc_desi_z", data=y)

    parquet_path = tmp_path / "overlap.parquet"
    pd.DataFrame({
        "hsc_object_id": np.arange(10_000, 10_000 + n),
        "legacy_object_id": np.arange(20_000, 20_000 + n),
        "not_an_id": np.arange(n),
    }).to_parquet(parquet_path, index=False)
    return h5_path, parquet_path


class DeltaR2Tests(unittest.TestCase):
    def test_folds_are_reproducible_disjoint_and_balanced(self):
        first = probe.make_folds(53, 5, 9)
        second = probe.make_folds(53, 5, 9)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(set(first), set(range(5)))
        self.assertLessEqual(np.bincount(first).max() - np.bincount(first).min(), 1)

    def test_metrics_are_normalized_by_test_standard_deviation(self):
        truth = np.array([0.0, 1.0, 2.0])
        predicted = np.array([0.0, 0.0, 2.0])
        result = probe.metrics(truth, predicted)
        self.assertTrue(np.isclose(result["normalized_rmse"], np.sqrt(1 / 3) / np.std(truth)))
        self.assertTrue(np.isclose(result["normalized_mae"], (1 / 3) / np.std(truth)))

    def test_synthetic_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            h5_path, parquet_path = _synthetic_inputs(tmp_path)
            model_out = tmp_path / "model"
            self.assertEqual(probe.main([
                "--embeddings", str(h5_path),
                "--overlap-parquet", str(parquet_path),
                "--out-dir", str(model_out),
                "--targets", "desi_z",
                "--device", "cpu",
                "--hidden", "8",
                "--max-epochs", "2",
                "--patience", "2",
                "--batch-size", "32",
                "--min-samples", "10",
                "--skip-input-file-hashes",
            ]), 0)

            scores = pd.read_csv(model_out / "fold_scores.csv")
            predictions = pd.read_csv(model_out / "oof_predictions.csv")
            self.assertEqual(len(scores), 5 * 3)
            self.assertEqual(set(scores.latent_variant), set(probe.VARIANTS))
            self.assertLessEqual({"r2", "normalized_rmse", "normalized_mae"}, set(scores))
            self.assertEqual(predictions.groupby(["latent_variant", "row_index"]).size().max(), 1)
            self.assertEqual(predictions.groupby("latent_variant").size().nunique(), 1)
            self.assertLessEqual({"row_index", "hsc_object_id", "legacy_object_id"}, set(predictions))
            provenance = json.loads((model_out / "provenance.json").read_text())
            self.assertTrue(provenance["sanity_checks"]["oof_once_per_variant"])
            self.assertEqual(len(set(scores.finite_mask_sha256)), 1)

            analysis_out = tmp_path / "analysis"
            self.assertEqual(analysis.main([
                "--pair", "self-control",
                str(model_out / "fold_scores.csv"), str(model_out / "oof_predictions.csv"),
                str(model_out / "fold_scores.csv"), str(model_out / "oof_predictions.csv"),
                "--out-dir", str(analysis_out),
                "--bootstrap-samples", "20",
            ]), 0)
            fold_values = pd.read_csv(analysis_out / "paired_fold_values.csv")
            bootstrap = pd.read_csv(analysis_out / "paired_object_bootstrap.csv")
            self.assertTrue(np.allclose(fold_values.interaction, 0.0))
            interaction = bootstrap[bootstrap.quantity == "interaction"]
            self.assertTrue(np.allclose(interaction[["estimate", "ci_low", "ci_high"]], 0.0))


if __name__ == "__main__":
    unittest.main()
