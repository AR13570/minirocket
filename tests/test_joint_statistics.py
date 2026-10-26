"""Hand-check ratio aggregation and multiplicity correction."""
import importlib.util
import unittest


class JointStatisticsTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('benchmarks.joint_ppv_statistics'))
        from benchmarks import joint_ppv_statistics
        return joint_ppv_statistics

    def test_mean_of_ratios_differs_from_ratio_of_means(self):
        r=self.module().ratio_summary([1.,9.],[2.,3.],samples=100)
        self.assertEqual(r['ratio_of_arithmetic_means'],2.)
        self.assertEqual(r['mean_per_dataset_ratio'],1.75)
        self.assertLessEqual(r['ratio_of_means_ci95_low'],2.)
        self.assertGreaterEqual(r['ratio_of_means_ci95_high'],2.)
        r=self.module().ratio_summary([1.],[2.],samples=10)
        self.assertIsNone(r['ratio_of_means_ci95_low'])
        self.assertIsNone(r['ratio_of_means_ci95_high'])

    def test_holm_and_zero_effect_signflip(self):
        m=self.module()
        for actual,want in zip(m.holm_adjust([.01,.04,.03]),[.03,.06,.06]):
            self.assertAlmostEqual(actual,want)
        self.assertEqual(m.sign_flip_pvalue([0.,0.,0.],samples=100),1.)
        with self.assertRaises(ValueError):m.ratio_summary([0.],[1.])
        with self.assertRaises(ValueError):m.holm_adjust([2.])


if __name__=='__main__':unittest.main()
