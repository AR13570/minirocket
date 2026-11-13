"""
Correctness tests for code/minirocket_multipool.py

Author : Shreyas Shinde (Student ID 26104692)

Run from the repository root:
    python -B -m unittest tests.test_multipool -v

What is tested and why:
 1. A tiny example worked out BY HAND gives the expected PPV, MPV, MIPV and LSPV.
 2. The raw PPV block is BIT-FOR-BIT identical to the joint-PPV transform (and therefore
    to the original MiniRocket). This proves the copied convolution code is correct
    and that we did not change the original features.
 3. MPV, MIPV and LSPV agree with a very simple, slow reference implementation
    written directly from the definitions (no shared code).
 4. The difference-series block equals the transform of np.diff(X).
 5. Shapes, block positions and feature counts are what the documentation says.
 6. The same random_state always gives the same features.
 7. Bad input is rejected with a clear error.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))     # code/ (same as the other tests)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))               # repository root (benchmarks/)
import minirocket_joint_ppv_simd as joint                  # fast PPV transform
import minirocket_multipool as mp
from minirocket_baseline import fit_univariate              # seeded fit


# ---------------------------------------------------------------------------
# Slow reference implementation, written straight from the definitions.
# ---------------------------------------------------------------------------
def reference_convolution(x, kernel_index, dilation, use_padding):
    """
    Convolution of one series with one MiniRocket kernel, from the definition:
    9 weights, all -1 except three weights of 2 (positions from combinations(range(9), 3)),
    spread out by `dilation`, with zeros outside the series.
    With use_padding=False only the 'valid' middle part is kept.
    """
    from itertools import combinations
    two_positions = list(combinations(range(9), 3))[kernel_index]
    weights = np.full(9, -1.0)
    weights[list(two_positions)] = 2.0

    L = len(x)
    out = np.zeros(L)
    for t in range(L):
        total = 0.0
        for i in range(9):
            position = t + (i - 4) * dilation      # centre of the kernel is weight 4
            if 0 <= position < L:
                total += weights[i] * x[position]
        out[t] = total
    if not use_padding:
        padding = 4 * dilation
        out = out[padding:L - padding]
    return out


def reference_pool(C, bias):
    """PPV, MPV, MIPV, LSPV of one convolution output for one bias, the slow obvious way."""
    n = len(C)
    positive = C > bias
    positions = np.nonzero(positive)[0]
    ppv = positive.sum() / n
    mpv = (C[positive] - bias).mean() if positive.any() else 0.0
    mipv = positions.mean() / n if positive.any() else -1.0
    longest, run = 0, 0
    for is_positive in positive:
        run = run + 1 if is_positive else 0
        longest = max(longest, run)
    return ppv, mpv, mipv, longest / n


def reference_transform(X, representation_parameters):
    """Full multi-pooling transform of one representation using the slow functions above."""
    dilations, per_dilation, biases = representation_parameters
    n_base = len(biases)
    out = np.zeros((len(X), 4 * n_base))
    for e, x in enumerate(X.astype(np.float64)):
        f = 0
        for d_index, dilation in enumerate(dilations):
            for kernel_index in range(84):
                # Same padding rule as the original: alternate between padded and valid.
                use_padding = (d_index % 2 + kernel_index) % 2 == 0
                C = reference_convolution(x, kernel_index, int(dilation), use_padding)
                for _ in range(per_dilation[d_index]):
                    values = reference_pool(C, float(biases[f]))
                    for p in range(4):
                        out[e, p * n_base + f] = values[p]
                    f += 1
    return out


# ---------------------------------------------------------------------------
# The tests
# ---------------------------------------------------------------------------
class MultiPoolTests(unittest.TestCase):

    def test_1_hand_computed_example(self):
        # C has 8 values. With bias 0.5 the positive positions are 1, 2, 4, 5, 6:
        #   PPV  = 5/8                                   = 0.625
        #   MPV  = mean(1.5, 2.5, 3.5, 4.5, 5.5)         = 3.5
        #   MIPV = mean(1, 2, 4, 5, 6) / 8 = 3.6 / 8     = 0.45
        #   LSPV = longest run (positions 4, 5, 6) = 3/8 = 0.375
        # With bias 10 nothing is positive: PPV 0, MPV 0, MIPV -1, LSPV 0.
        C = np.array([-1, 2, 3, -2, 4, 5, 6, 0], dtype=np.float32)
        biases = np.array([0.5, 10.0], dtype=np.float32)
        row = np.zeros(4 * 2, dtype=np.float32)
        scratch = [np.zeros(2, dtype=np.int64), np.zeros(2, dtype=np.float64),
                   np.zeros(2, dtype=np.int64), np.zeros(2, dtype=np.int64),
                   np.zeros(2, dtype=np.int64)]
        mp._pool_four(C, biases, 0, 2, row, 2, *scratch)
        # layout: [PPV b0, PPV b1, MPV b0, MPV b1, MIPV b0, MIPV b1, LSPV b0, LSPV b1]
        expected = np.array([0.625, 0.0, 3.5, 0.0, 0.45, -1.0, 0.375, 0.0], dtype=np.float32)
        np.testing.assert_allclose(row, expected, rtol=1e-6)

    def test_2_raw_ppv_block_is_identical_to_joint_ppv(self):
        rng = np.random.default_rng(0)
        for length, budget, seed in [(9, 84, 0), (30, 840, 1), (150, 2500, 2), (301, 10_000, 0)]:
            with self.subTest(length=length, budget=budget, seed=seed):
                train = rng.normal(size=(5, length)).astype(np.float32)
                test = rng.normal(size=(4, length)).astype(np.float32)
                params = mp.fit(train, num_features=budget, use_difference=False, random_state=seed)
                ours = mp.transform(test, params)
                ppv = ours[:, mp.feature_blocks(params)["raw_PPV"]]
                # Joint PPV fitted with the same seed -> same biases -> must give the same PPV.
                joint_params = fit_univariate(train, num_features=budget, random_state=seed)
                reference = joint.transform(test, joint_params)
                # Compare the raw bits, not just "close": they must be exactly equal.
                np.testing.assert_array_equal(ppv.view(np.uint32), reference.view(np.uint32))

    def test_3_all_four_pools_match_slow_reference(self):
        rng = np.random.default_rng(1)
        for length, budget in [(12, 84), (40, 336)]:
            with self.subTest(length=length, budget=budget):
                train = rng.normal(size=(4, length)).astype(np.float32)
                test = rng.normal(size=(3, length)).astype(np.float32)
                params = mp.fit(train, num_features=budget, use_difference=False, random_state=0)
                ours = mp.transform(test, params)
                slow = reference_transform(test, params["raw"])
                # Small tolerance: the reference sums in a different order (float rounding).
                np.testing.assert_allclose(ours, slow, rtol=1e-4, atol=1e-4)

    def test_4_difference_block_equals_transform_of_diff(self):
        rng = np.random.default_rng(2)
        train = rng.normal(size=(6, 50)).astype(np.float32)
        test = rng.normal(size=(3, 50)).astype(np.float32)
        params = mp.fit(train, num_features=840, use_difference=True, random_state=3)
        features = mp.transform(test, params)
        blocks = mp.feature_blocks(params)
        diff_part = features[:, blocks["diff_PPV"].start:]
        expected = mp._transform_one(mp.difference(test), params["diff"])
        np.testing.assert_array_equal(diff_part, expected)
        # And the difference series itself is x[t+1] - x[t]
        np.testing.assert_allclose(mp.difference(test)[0, :3], test[0, 1:4] - test[0, :3])

    def test_5_shapes_and_blocks(self):
        X = np.random.default_rng(3).normal(size=(5, 60)).astype(np.float32)
        params = mp.fit(X, num_features=10_000, use_difference=True)
        features = mp.transform(X, params)
        # 10,000 rounds down to 9,996 (multiple of 84); 4 pools x 2 representations
        self.assertEqual(features.shape, (5, 9_996 * 8))
        self.assertEqual(mp.num_output_features(params), 9_996 * 8)
        blocks = mp.feature_blocks(params)
        self.assertEqual(list(blocks), ["raw_PPV", "raw_MPV", "raw_MIPV", "raw_LSPV",
                                        "diff_PPV", "diff_MPV", "diff_MIPV", "diff_LSPV"])
        self.assertEqual(blocks["diff_LSPV"].stop, features.shape[1])
        # PPV and LSPV are fractions, MIPV is a fraction or -1
        self.assertTrue(np.all((features[:, blocks["raw_PPV"]] >= 0) & (features[:, blocks["raw_PPV"]] <= 1)))
        self.assertTrue(np.all((features[:, blocks["raw_LSPV"]] >= 0) & (features[:, blocks["raw_LSPV"]] <= 1)))
        mipv = features[:, blocks["raw_MIPV"]]
        self.assertTrue(np.all((mipv == -1) | ((mipv >= 0) & (mipv < 1))))
        # select_blocks keeps only the chosen columns
        self.assertEqual(mp.select_blocks(features, params, ["raw_PPV", "diff_MPV"]).shape, (5, 2 * 9_996))
        # empty input
        self.assertEqual(mp.transform(X[:0], params).shape, (0, 9_996 * 8))

    def test_6_same_seed_same_features(self):
        X = np.random.default_rng(4).normal(size=(6, 40)).astype(np.float32)
        a = mp.transform(X, mp.fit(X, num_features=840, random_state=7))
        b = mp.transform(X, mp.fit(X, num_features=840, random_state=7))
        np.testing.assert_array_equal(a, b)

    def test_7_bad_input_is_rejected(self):
        with self.assertRaises(ValueError):
            mp.fit(np.zeros((3, 9), dtype=np.float32), use_difference=True)   # diff would be length 8
        with self.assertRaises(ValueError):
            mp.fit(np.full((3, 20), np.nan, dtype=np.float32))                 # missing values
        with self.assertRaises(ValueError):
            mp.fit(np.zeros(20, dtype=np.float32))                             # 1D instead of 2D


if __name__ == "__main__":
    unittest.main()
