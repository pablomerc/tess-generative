from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from evaluate_adversarial import (
    load_prediction,
    make_matched_controls,
    metrics,
    paired_bootstrap,
    survey_interaction_bootstrap,
)
from freeze_manifest import build_manifest
from make_gallery import robust_asinh_rgb
from quality_table import build_quality_table


def synthetic_catalogs(n: int = 20):
    overlap = pd.DataFrame(
        {
            "hsc_image_idx": np.arange(n)[::-1],
            "legacy_image_idx": np.arange(n),
            "hsc_object_id": 1000 + np.arange(n)[::-1],
            "legacy_object_id": [f"L{i}" for i in range(n)],
            "ra": np.linspace(10, 11, n),
            "dec": np.linspace(-1, 1, n),
        }
    )
    hsc = pd.DataFrame(
        {"image_idx": np.arange(n), "object_id": 1000 + np.arange(n)}
    )
    legacy = pd.DataFrame(
        {"image_idx": np.arange(n), "object_id": [f"L{i}" for i in range(n)]}
    )
    for bi, band in enumerate("griz"):
        hsc[f"shape_hsm_moments_11_{band}"] = 9.0 + np.arange(n) / 10
        hsc[f"shape_hsm_moments_22_{band}"] = 8.0 + np.arange(n) / 10
        hsc[f"shape_hsm_moments_12_{band}"] = 0.2
        hsc[f"shape_hsm_psf_moments_11_{band}"] = 1.0
        hsc[f"shape_hsm_psf_moments_22_{band}"] = 1.1
        hsc[f"shape_hsm_psf_moments_12_{band}"] = 0.0
        hsc[f"cmodel_flux_{band}"] = 100.0 + np.arange(n)
        hsc[f"cmodel_flux_err_{band}"] = 5.0
        hsc[f"psf_fwhm_{band}"] = 0.8
        legacy[f"PSFSIZE_{band.upper()}"] = 1.6
        legacy[f"FLUX_{band.upper()}"] = 100.0
        legacy[f"FLUX_IVAR_{band.upper()}"] = 0.01
    return overlap, hsc, legacy


class QualityTableTests(unittest.TestCase):
    def test_build_joins_by_image_idx_and_scores(self):
        overlap, hsc, legacy = synthetic_catalogs()
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            paths = [td / f"{name}.parquet" for name in ("overlap", "hsc", "legacy")]
            for frame, path in zip((overlap, hsc, legacy), paths):
                frame.to_parquet(path, index=False)
            table, meta = build_quality_table(
                *paths,
                pixel_scale=0.2,
                moments_unit="pixel2",
                metric_version="test-v1",
            )
        self.assertEqual(len(table), len(overlap))
        self.assertTrue(table.coverage_joint.all())
        self.assertTrue((table.resolution_gap > 0).all())
        self.assertTrue((table.snr_gap > 0).all())
        self.assertTrue(table.joint_score.between(0, 1).all())
        self.assertEqual(meta["metric_version"], "test-v1")
        row = table.set_index("hsc_object_id").loc["1019"]
        self.assertEqual(int(row.hsc_image_idx), 19)

    def test_invalid_psf_correction_sets_coverage_false(self):
        overlap, hsc, legacy = synthetic_catalogs(3)
        for band in "griz":
            hsc.loc[:, f"shape_hsm_moments_11_{band}"] = 0.5
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            paths = [td / f"{name}.parquet" for name in ("o", "h", "l")]
            for frame, path in zip((overlap, hsc, legacy), paths):
                frame.to_parquet(path, index=False)
            table, _ = build_quality_table(
                *paths, pixel_scale=0.2, moments_unit="pixel2", metric_version="test"
            )
        self.assertFalse(table.coverage_resolution.any())
        self.assertTrue(table.joint_score.isna().all())


class ManifestAndEvaluationTests(unittest.TestCase):
    def setUp(self):
        n = 50
        self.quality = pd.DataFrame(
            {
                "hsc_object_id": [f"H{i}" for i in range(n)],
                "legacy_object_id": [f"L{i}" for i in range(n)],
                "hsc_image_idx": np.arange(n),
                "legacy_image_idx": np.arange(n),
                "resolution_gap": np.linspace(-1, 2, n),
                "snr_gap": np.sin(np.linspace(0, 4, n)),
                "joint_score": np.linspace(0, 1, n),
                "metric_version": "frozen-v1",
            }
        )

    def test_manifest_is_quality_only_and_control_is_disjoint(self):
        manifest = build_manifest(self.quality, top_n=10, n_bins=5)
        mask, meta = make_matched_controls(manifest, "joint", seed=4)
        self.assertEqual(int(manifest.joint_top_n.sum()), 10)
        self.assertFalse((mask & manifest.joint_top_n).any())
        self.assertEqual(meta["selected"], 10)

    def test_metrics_and_paired_bootstrap(self):
        manifest = build_manifest(self.quality, top_n=10, n_bins=5)
        rows = []
        for model, offset in (("ours", 0.1), ("contrastive", 0.4)):
            for i in range(len(manifest)):
                rows.append(
                    {
                        "pair_id": f"H{i}",
                        "model": model,
                        "view": "hsc",
                        "target": "z",
                        "y_true": i / 10,
                        "y_pred": i / 10 + offset,
                        **{
                            c: manifest.loc[i, c]
                            for c in manifest
                            if c.endswith("_top_n") or c.endswith("_quantile_bin")
                        },
                    }
                )
        data = pd.DataFrame(rows)
        result = metrics(data[data.model == "ours"].y_true, data[data.model == "ours"].y_pred)
        self.assertLess(result["nrmse"], 0.1)
        boot = paired_bootstrap(
            data,
            {axis: set() for axis in ("resolution", "snr", "joint")},
            replicates=30,
            seed=2,
        )
        row = boot[(boot.axis == "joint") & (boot.stratum == "top_n") & (boot.metric == "nrmse")]
        self.assertEqual(len(row), 1)
        self.assertLess(float(row.iloc[0].a_minus_b), 0)

    def test_standard_oof_file_filters_view_and_accepts_both_ids(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "oof.csv"
            pd.DataFrame(
                {
                    "hsc_object_id": ["H1", "H1"],
                    "legacy_object_id": ["L1", "L1"],
                    "target": ["z", "z"],
                    "latent_variant": ["hsc_e1", "legacy_e1"],
                    "y_true": [1.0, 1.0],
                    "y_pred": [1.1, 1.2],
                }
            ).to_csv(path, index=False)
            loaded = load_prediction(f"ours:hsc={path}")
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded.iloc[0].pair_id_kind, "hsc_object_id")
        self.assertAlmostEqual(float(loaded.iloc[0].y_pred), 1.1)

    def test_survey_interaction_uses_all_four_model_views(self):
        manifest = build_manifest(self.quality, top_n=10, n_bins=5)
        rows = []
        offsets = {
            ("ours", "hsc"): 0.05,
            ("ours", "legacy"): 0.30,
            ("contrastive", "hsc"): 0.25,
            ("contrastive", "legacy"): 0.30,
        }
        for (model, view), offset in offsets.items():
            for i in range(len(manifest)):
                rows.append(
                    {
                        "pair_id": f"H{i}",
                        "model": model,
                        "view": view,
                        "target": "z",
                        "y_true": i / 10,
                        "y_pred": i / 10 + offset,
                        **{
                            c: manifest.loc[i, c]
                            for c in manifest
                            if c.endswith("_top_n") or c.endswith("_quantile_bin")
                        },
                    }
                )
        result = survey_interaction_bootstrap(
            pd.DataFrame(rows),
            {axis: set() for axis in ("resolution", "snr", "joint")},
            replicates=20,
            seed=3,
        )
        row = result[
            (result.axis == "joint")
            & (result.stratum == "all")
            & (result.metric == "r2")
        ]
        self.assertEqual(len(row), 1)
        self.assertGreater(float(row.iloc[0].interaction), 0)


class GalleryTests(unittest.TestCase):
    def test_robust_asinh_is_finite_and_bounded(self):
        x = np.zeros((4, 48, 48), dtype=np.float32)
        x[:, 20:28, 20:28] = 10
        x[0, 0, 0] = np.nan
        rgb = robust_asinh_rgb(x)
        self.assertEqual(rgb.shape, (48, 48, 3))
        self.assertTrue(np.isfinite(rgb).all())
        self.assertGreaterEqual(rgb.min(), 0)
        self.assertLessEqual(rgb.max(), 1)


if __name__ == "__main__":
    unittest.main()
