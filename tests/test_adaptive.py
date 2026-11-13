"""
Tests for the final model (code/minirocket_adaptive.py) and the faster PPV+MPV
pooling path in code/minirocket_multipool.py

Author : Shreyas Shinde (Student ID 26104692)

Run from the repository root:
    python -B -m unittest tests.test_adaptive -v

What is tested and why:
 1. The fast PPV+MPV path gives exactly the same numbers as the first two blocks
    of the full four-pool transform (so the faster code is not a different method).
 2. Only the two supported pooling sets are accepted, and the block names follow them.
 3. When the original features already score perfectly in cross-validation (a tie),
    the model keeps the original "PPV" features.
 4. With fewer than 2 examples per class (no cross-validation possible) it keeps "PPV".
 5. When the model chooses "PPV", its predictions are IDENTICAL to the plain original
    MiniRocket pipeline (seeded fit + original transform + same classifier).
 7. Both prediction paths (fast PPV-only and PPV+MPV) give the right features.
 8. ridge_classifier (code/) is configured exactly like benchmarks.common.make_classifier.
 6. Same seed -> same choice and same predictions; outputs have the right shape.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))     # code/ (same as the other tests)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))               # repository root (benchmarks/)
import minirocket_multipool as mp
from benchmarks.common import make_classifier
from minirocket_adaptive import AdaptivePoolingMiniRocket, CANDIDATES
from minirocket_baseline import fit_univariate, transform_univariate


def easy_dataset(n_per_class, length=60, seed=0):
    """Two very different classes (sine vs square wave) with a little noise."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, length)
    X = np.vstack([np.sin(t) + 0.1 * rng.standard_normal((n_per_class, length)),
                   np.sign(np.sin(t)) + 0.1 * rng.standard_normal((n_per_class, length))])
    y = np.array([0] * n_per_class + [1] * n_per_class)
    order = rng.permutation(len(y))
    return X[order].astype(np.float32), y[order]


def harder_dataset(n_per_class, length=50, seed=1):
    """Two classes that differ only by a weak bump position, with strong noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    X, y = [], []
    for label, centre in ((0, 18), (1, 30)):
        for _ in range(n_per_class):
            X.append(0.8 * np.exp(-0.5 * ((t - centre) / 4) ** 2) + 0.7 * rng.standard_normal(length))
            y.append(label)
    return np.array(X, dtype=np.float32), np.array(y)


class AdaptivePoolingTests(unittest.TestCase):

    def test_1_fast_ppv_mpv_path_matches_four_pool_transform(self):
        rng = np.random.default_rng(0)
        train = rng.normal(size=(5, 70)).astype(np.float32)
        test = rng.normal(size=(4, 70)).astype(np.float32)
        for use_difference in (False, True):
            with self.subTest(use_difference=use_difference):
                four = mp.fit(train, num_features=840, use_difference=use_difference, pools=mp.ALL_POOLS)
                two = mp.fit(train, num_features=840, use_difference=use_difference, pools=mp.PPV_MPV)
                F4, F2 = mp.transform(test, four), mp.transform(test, two)
                for name in ("raw_PPV", "raw_MPV") + (("diff_PPV", "diff_MPV") if use_difference else ()):
                    a, b = F2[:, mp.feature_blocks(two)[name]], F4[:, mp.feature_blocks(four)[name]]
                    if name.endswith("PPV"):
                        np.testing.assert_array_equal(a, b)        # PPV: exactly equal
                    else:
                        # MPV: the fast version may add the numbers in a different order
                        # (SIMD), so allow a tiny floating-point difference.
                        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
                self.assertEqual(F2.shape[1], mp.num_output_features(two))

    def test_2_pool_sets_and_block_names(self):
        X = np.random.default_rng(1).normal(size=(4, 30)).astype(np.float32)
        with self.assertRaises(ValueError):
            mp.fit(X, pools=("PPV", "LSPV"))          # not a supported combination
        params = mp.fit(X, num_features=840, pools=mp.PPV_MPV)
        self.assertEqual(list(mp.feature_blocks(params)), ["raw_PPV", "raw_MPV"])
        self.assertIsNone(params["diff"])            # difference series is off by default now

    def test_3_tie_keeps_original_features(self):
        X, y = easy_dataset(10)
        model = AdaptivePoolingMiniRocket(num_features=840, random_state=0).fit(X, y)
        self.assertEqual(model.cv_scores_["PPV"], 1.0)
        self.assertEqual(model.cv_scores_["PPV+MPV"], 1.0)
        self.assertEqual(model.chosen_feature_set_, "PPV")

    def test_4_too_few_examples_keeps_original_features(self):
        X, y = easy_dataset(1)                       # one example per class
        model = AdaptivePoolingMiniRocket(num_features=840).fit(X, y)
        self.assertEqual(model.chosen_feature_set_, "PPV")
        self.assertEqual(model.cv_scores_, {})

    def test_5_choosing_ppv_is_exactly_the_original_pipeline(self):
        X, y = easy_dataset(8, seed=3)
        X_test, _ = easy_dataset(5, seed=4)
        model = AdaptivePoolingMiniRocket(num_features=840, random_state=2).fit(X, y)
        self.assertEqual(model.chosen_feature_set_, "PPV")
        # plain original MiniRocket with the same seed and the same classifier
        params = fit_univariate(X, num_features=840, random_state=2)
        classifier = make_classifier(len(X), "ridge", 2).fit(transform_univariate(X, params), y)
        np.testing.assert_array_equal(model.predict(X_test),
                                      classifier.predict(transform_univariate(X_test, params)))
        # and the features themselves are bit-identical
        np.testing.assert_array_equal(model.transform(X_test), transform_univariate(X_test, params))

    def test_6_repeatable_and_well_formed(self):
        X, y = harder_dataset(12)
        a = AdaptivePoolingMiniRocket(num_features=840, random_state=5).fit(X, y)
        b = AdaptivePoolingMiniRocket(num_features=840, random_state=5).fit(X, y)
        self.assertIn(a.chosen_feature_set_, [name for name, _ in CANDIDATES])
        self.assertEqual(a.chosen_feature_set_, b.chosen_feature_set_)
        self.assertEqual(a.cv_scores_, b.cv_scores_)
        np.testing.assert_array_equal(a.predict(X), b.predict(X))
        self.assertEqual(a.predict(X).shape, (len(X),))
        self.assertTrue(0.0 <= a.score(X, y) <= 1.0)

    def test_7_both_prediction_paths_give_the_right_features(self):
        X, y = harder_dataset(12)
        model = AdaptivePoolingMiniRocket(num_features=840, random_state=0).fit(X, y)
        both = mp.transform(X, model.parameters_)            # PPV + MPV for every series
        # Path 1 (fast path, joint-PPV transform) must equal the PPV block exactly.
        model.chosen_feature_set_, model.chosen_blocks_ = "PPV", ["raw_PPV"]
        np.testing.assert_array_equal(model.transform(X), mp.select_blocks(both, model.parameters_, ["raw_PPV"]))
        # Path 2 (PPV + MPV) must equal both blocks.
        model.chosen_feature_set_, model.chosen_blocks_ = "PPV+MPV", ["raw_PPV", "raw_MPV"]
        np.testing.assert_array_equal(model.transform(X), both)

    def test_8_ridge_classifier_matches_the_benchmark_classifier(self):
        for n_train in (10, 57, 1000):
            ours, theirs = mp.ridge_classifier(n_train, 0), make_classifier(n_train, "ridge", 0)
            self.assertEqual([type(step) for _, step in ours.steps], [type(step) for _, step in theirs.steps])
            np.testing.assert_array_equal(ours.steps[-1][1].alphas, theirs.steps[-1][1].alphas)
            self.assertEqual(ours.steps[0][1].get_params(), theirs.steps[0][1].get_params())


if __name__ == "__main__":
    unittest.main()
