"""
Tests for ui/pruning_service.py (the layer between the group-pruning code and the UI).

Run from the repository root:  python -B -m unittest discover -s ui -p "test_*.py"

Small synthetic data and 168 features keep them fast. The key test (test_matches_experiment_runner)
checks that the UI layer gives exactly the numbers experiments/group_pruning.py gives.
"""
import sys
import unittest
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))      # ui/

import pruning_service as ps  # noqa: E402  (sets up paths)
import numpy as np            # noqa: E402


def synthetic(n=30, length=64, classes=3, seed=0):
    """Noisy sine waves whose frequency depends on the class (unbalanced on purpose)."""
    rng = np.random.default_rng(seed)
    y = np.array([i % classes if i % 5 else 0 for i in range(n)])
    t = np.linspace(0, 2 * np.pi, length)
    X = np.sin((y[:, None] + 1) * t[None, :]) + 0.3 * rng.normal(size=(n, length))
    return np.ascontiguousarray(X, dtype=np.float32), y


class PruningServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X_train, cls.y_train = synthetic(seed=0)
        cls.X_test, cls.y_test = synthetic(seed=1)

    def run_small(self, **options):
        settings = dict(keep_fractions=(1.0, 0.5, 0.25), num_features=168, repetitions=1)
        settings.update(options)
        return ps.run_pruning(self.X_train, self.y_train, self.X_test, self.y_test, **settings)

    def test_rows_are_consistent_and_exact(self):
        result = self.run_small()
        fractions = [r["keep_fraction"] for r in result.rows]
        self.assertEqual(fractions, [1.0, 0.5, 0.25])                      # full model first
        kept = [r["groups_kept"] for r in result.rows]
        self.assertEqual(kept, sorted(kept, reverse=True))
        self.assertTrue(all(r["exact"] for r in result.rows))               # pruned == full[:, kept]
        self.assertEqual(result.rows[0]["accuracy_change_pp"], 0.0)
        self.assertAlmostEqual(result.rows[0]["time_ratio"], 1.0)
        # size = (weight rows + scaler mean/scale) x features x 8 bytes; 3 classes -> 3 weight rows
        self.assertAlmostEqual(result.rows[0]["model_kb"], (3 + 2) * 168 * 8 / 1024)
        for fraction, keep in result.masks.items():
            self.assertEqual(keep.shape, (len(result.dilations), 84))
            self.assertEqual(int(keep.sum()), max(1, round(fraction * keep.size)))

    def test_every_scorer_and_bias_strategy_runs(self):
        for scorer in ps.SCORERS:
            for strategy in ps.BIAS_STRATEGIES:
                result = self.run_small(scorer=scorer, bias_strategy=strategy, keep_fractions=(0.5,))
                self.assertEqual(len(result.rows), 2)                       # 1.0 is always added
                self.assertTrue(all(r["exact"] for r in result.rows))
        with self.assertRaises(ValueError):
            self.run_small(scorer="lasso")

    def test_matches_experiment_runner(self):
        """Same data, seed and settings -> same features and accuracy as experiments/group_pruning.py."""
        from experiments import group_pruning as runner
        dataset = Namespace(name="synthetic", fingerprint="-",
                            X_train=self.X_train[:, None, :], y_train=self.y_train,
                            X_test=self.X_test[:, None, :], y_test=self.y_test)
        args = Namespace(feature_count=168, max_dilations_per_kernel=32, examples_per_group=1,
                         keep_fractions=[1.0, 0.5, 0.25], scorers=["ridge"], drop_fraction=0.2,
                         transform_repetitions=1)
        for strategy, scorer in (("original", "ridge"), ("stratified", "filter")):
            args.scorers = [scorer]
            theirs = [r for r in runner.run_case(dataset, 0, strategy, args) if r["implementation"] == "joint"]
            ours = self.run_small(bias_strategy=strategy, scorer=scorer).rows
            self.assertEqual([r["features_kept"] for r in theirs], [r["features_kept"] for r in ours])
            self.assertEqual([r["accuracy"] for r in theirs], [r["accuracy"] for r in ours])
            # same kept groups and same feature values, bit for bit
            self.assertEqual([r["test_features_sha256"] for r in theirs], [r["features_sha256"] for r in ours])

    def test_threshold_sources_are_balanced_when_stratified(self):
        sources = ps.threshold_sources(self.y_train, num_groups=84 * 3, seed=0)
        counts = list(sources["stratified"].values())
        self.assertEqual(sum(counts), 84 * 3)
        self.assertLessEqual(max(counts) - min(counts), 1)
        self.assertEqual(sum(sources["uniform"].values()), 84 * 3)

    def test_published_results_are_readable(self):
        rows = ps.published_pruning_summary()
        self.assertEqual(len(rows), 11)                                      # full + 2 scorers x 5 fractions
        full = [r for r in rows if r["scorer"] == "none"][0]
        self.assertAlmostEqual(full["accuracy_percent"], 86.08, places=2)   # headline number of the 113-dataset run
        self.assertTrue(all(path.exists() for path in ps.published_files().values()))


if __name__ == "__main__":
    unittest.main()
