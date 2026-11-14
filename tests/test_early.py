"""
Tests for code/minirocket_early.py

Author : Shreyas Shinde (Student ID 26104692)

Run from the repository root:
    python -B -m unittest tests.test_early -v

What is tested and why:
 1. Prefix lengths are at least 9, sorted, unique and end at the full length.
 2. The threshold rule picks the lowest margin that is still reliable enough.
 3. With an impossible target (accuracy > 100%) the model never stops early, so it
    behaves exactly like the full-length model.
 4. On an easy problem (classes differ at the START of the series) it stops early
    and stays accurate; earliness and harmonic mean are in range.
 5. stream() (point-by-point simulation) agrees with predict().
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))     # code/ (same as the other tests)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))               # repository root (benchmarks/)
from minirocket_early import EarlyClassifier, PrefixModel, lowest_reliable_threshold, prediction_and_margin


def early_signal_dataset(n_per_class, length=60, seed=0):
    """Classes differ by a bump in the first 20% of the series; the rest is noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    X, y = [], []
    for c in (0, 1):
        for _ in range(n_per_class):
            bump = (1.0 if c == 1 else -1.0) * 2.0 * np.exp(-0.5 * ((t - 6) / 2) ** 2)
            X.append(bump + 0.3 * rng.standard_normal(length))
            y.append(c)
    order = rng.permutation(len(y))
    return np.array(X, dtype=np.float32)[order], np.array(y)[order]


class EarlyClassificationTests(unittest.TestCase):

    def test_1_prefix_lengths(self):
        model = EarlyClassifier(prefix_fractions=(0.05, 0.1, 0.5, 0.5, 1.0))
        self.assertEqual(model._prefix_lengths(100), [9, 10, 50, 100])
        self.assertEqual(model._prefix_lengths(12), [9, 12])

    def test_2_threshold_rule(self):
        margins = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        correct = np.array([True, True, True, False, False])
        # top-3 are 100% correct, top-4 only 75% -> lowest reliable threshold is 3.0
        self.assertEqual(lowest_reliable_threshold(margins, correct, 1.0, 1), 3.0)
        # target 75% -> top-4 still OK -> threshold 2.0
        self.assertEqual(lowest_reliable_threshold(margins, correct, 0.75, 1), 2.0)
        # need at least 4 examples at 100% -> impossible -> never stop
        self.assertEqual(lowest_reliable_threshold(margins, correct, 1.0, 4), np.inf)

    def test_3_impossible_target_means_full_length(self):
        X, y = early_signal_dataset(8)
        X_test, _ = early_signal_dataset(4, seed=5)
        model = EarlyClassifier(prefix_fractions=(0.2, 0.5, 1.0), target_accuracy=1.01,
                                num_features=840).fit(X, y)
        predictions, used = model.predict(X_test)
        self.assertTrue(np.all(used == X.shape[1]))
        full = PrefixModel(X.shape[1], 840, 0).fit(X, y)
        expected, _ = prediction_and_margin(full.scores(X_test), full.classes_)
        np.testing.assert_array_equal(predictions, expected)

    def test_4_stops_early_when_the_signal_is_early(self):
        X, y = early_signal_dataset(12)
        X_test, y_test = early_signal_dataset(10, seed=7)
        model = EarlyClassifier(prefix_fractions=(0.2, 0.5, 1.0), num_features=840).fit(X, y)
        result = model.evaluate(X_test, y_test)
        self.assertLess(result["earliness"], 1.0)
        self.assertGreaterEqual(result["accuracy"], 0.9)
        self.assertTrue(0.0 <= result["harmonic_mean"] <= 1.0)

    def test_5_stream_agrees_with_predict(self):
        X, y = early_signal_dataset(10)
        X_test, _ = early_signal_dataset(3, seed=11)
        model = EarlyClassifier(prefix_fractions=(0.2, 0.5, 1.0), num_features=840).fit(X, y)
        predictions, used = model.predict(X_test)
        for i, x in enumerate(X_test):
            steps = model.stream(x)
            self.assertTrue(steps[-1]["stop"])
            self.assertEqual(steps[-1]["points_seen"], used[i])
            self.assertEqual(steps[-1]["prediction"], predictions[i])


if __name__ == "__main__":
    unittest.main()
