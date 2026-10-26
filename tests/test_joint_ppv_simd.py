"""Exactness and compiler checks for the isolated indexed joint-PPV variant."""
import importlib.util
import platform
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))


class JointSIMDTests(unittest.TestCase):
    def candidate(self):
        self.assertIsNotNone(importlib.util.find_spec('minirocket_joint_ppv_simd'),
                             'The separately measurable SIMD implementation is missing')
        import minirocket_joint_ppv_simd
        return minirocket_joint_ppv_simd

    def test_exact_full_features_and_parameters(self):
        candidate = self.candidate()
        from minirocket_baseline import fit_univariate, transform_univariate
        rng = np.random.default_rng(492)
        for length, budget in [(9, 84), (31, 2500), (150, 5000), (512, 10000), (1639, 2500)]:
            train = rng.normal(size=(4, length)).astype(np.float32)
            X = rng.normal(size=(3, length)).astype(np.float32)
            p = fit_univariate(train, num_features=budget, random_state=1)
            saved = [a.copy() for a in p]
            actual = candidate.transform(X, p)
            for expected in (transform_univariate(X, p),):
                np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
            for a, b in zip(p, saved):
                np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(candidate.transform(X[::-1], p), actual[::-1])
            np.testing.assert_array_equal(candidate.transform(X[:1], p), actual[:1])
            self.assertEqual(candidate.transform(X[:0], p).shape, (0, actual.shape[1]))

    def test_pooling_ties_subnormals_strides_and_block_remainders(self):
        candidate = self.candidate()
        tiny = np.nextafter(np.float32(0), np.float32(1))
        responses = np.array([-3, -tiny, -0., 0., tiny, 1., 1., 3.], np.float32)
        biases = np.array([1, 0, tiny, -tiny, 3, -3, 1], np.float32)
        for count in range(1, 34):
            b = np.resize(biases, count)
            for x in (responses, responses[::-1], responses[::2]):
                a = candidate.ppv_joint(x, b)
                expected = np.array([(x > bias).astype(np.float32).mean() for bias in b], np.float32)
                np.testing.assert_array_equal(a.view(np.uint32), expected.view(np.uint32))

    def test_constant_extreme_inputs_and_custom_threshold_counts(self):
        candidate = self.candidate()
        from minirocket_baseline import transform_univariate
        tiny = np.nextafter(np.float32(0), np.float32(1))
        X = np.vstack([np.zeros(33), np.ones(33), np.full(33, tiny),
                       np.resize([1e38, -1e38, 0.], 33)]).astype(np.float32)
        for counts in ([1, 4], [2, 3], [8, 17]):
            p = (np.array([1, 3], np.int32), np.array(counts, np.int32),
                 np.resize(np.array([0., 1., -1., tiny], np.float32), 84 * sum(counts)))
            np.testing.assert_array_equal(candidate.transform(X, p).view(np.uint32),
                                          transform_univariate(X, p).view(np.uint32))

    def test_long_input_dispatch_preserves_original_fallback(self):
        candidate = self.candidate()
        from unittest.mock import patch
        class LongInput:
            shape = (1, 2 ** 24 + 1)
        x, parameters = LongInput(), object()
        sentinel = object()
        with patch.object(candidate, 'validate_input', return_value=x), \
             patch.object(candidate, 'transform_univariate', return_value=sentinel) as fallback:
            self.assertIs(candidate.transform(x, parameters), sentinel)
            fallback.assert_called_once_with(x, parameters)

    def test_small_width_helpers_vectorize_on_audited_runtime(self):
        self.candidate()
        import numba
        from numba import njit
        import _joint_ppv_simd_kernels as kernels
        if platform.machine() != 'arm64' or numba.__version__ != '0.63.1':
            self.skipTest('Compiler regression check is specific to the audited ARM64/Numba runtime')
        for width in range(1, 5):
            # Fresh compilation avoids the empty inspection result of disk-cached overloads.
            function = njit(fastmath=True)(getattr(kernels, f'_pool_{width}').py_func)
            function(np.arange(512, dtype=np.float32), np.zeros(width, np.float32),
                     0, np.empty(width, np.float32))
            llvm = function.inspect_llvm(function.signatures[0])
            self.assertRegex(llvm, r'fcmp[^\n]*<\d+ x float>', f'width {width} lost SIMD')


if __name__ == '__main__':
    unittest.main()
