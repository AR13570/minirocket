"""
test_transform_service.py - tests for the Fast transform and Reproduction pages' logic

Author : Shreyas Shinde (Student ID 26104692)

Run from the repository root:  python -B -m unittest discover -s ui -p "test_*.py" -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transform_service as ts   # noqa: E402


class TransformServiceTests(unittest.TestCase):
    def test_joint_ppv_gives_identical_features(self):
        rng = np.random.default_rng(0)
        X_train = rng.normal(size=(12, 60)).astype(np.float32)
        X_test = rng.normal(size=(6, 60)).astype(np.float32)
        result = ts.compare_transforms(X_train, X_test, num_features=168, repetitions=1)
        self.assertTrue(result["identical"])
        self.assertEqual(result["features"], 168)
        self.assertGreater(result["original_seconds"], 0)
        self.assertAlmostEqual(result["ratio"], result["joint_seconds"] / result["original_seconds"])

    def test_published_benchmark_is_read(self):
        rows = ts.published_joint_ppv()
        if not rows:
            self.skipTest("results/joint_ppv_2026-10-26/summary.csv not present")
        self.assertEqual([r["features"] for r in rows], [2436, 4956, 9996])
        largest = rows[-1]
        self.assertTrue(largest["identical"])
        self.assertAlmostEqual(largest["train_reduction_percent"], 48.7, places=1)
        self.assertAlmostEqual(largest["test_reduction_percent"], 46.9, places=1)

    def test_reproduction_summary(self):
        summary, rows = ts.published_reproduction()
        if summary is None:
            self.skipTest("reproduction results not present")
        self.assertEqual(summary["datasets"], 109)
        self.assertAlmostEqual(summary["ours_percent"], 85.69, places=2)
        self.assertAlmostEqual(summary["paper_percent"], 85.67, places=2)
        self.assertGreater(summary["wilcoxon_p"], 0.05)       # no systematic difference


if __name__ == "__main__":
    unittest.main()
