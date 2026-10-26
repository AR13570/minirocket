"""Hand-calculated checks for paired, dataset-weighted CSV analysis."""
import csv
import tempfile
import unittest
from pathlib import Path

from benchmarks import analyze_results as analysis


def row(dataset, model, seed, accuracy, seconds=1., fingerprint='split'):
    return dict(dataset=dataset, model=model, seed=seed, split_sha256=fingerprint,
                classifier='ridge', accuracy=accuracy, training_seconds=seconds,
                test_seconds=seconds, train_transform_seconds=seconds / 2,
                test_transform_seconds=seconds / 2, channels=1, length=100,
                n_train=50, n_test=20, features=9996)


class AnalysisTests(unittest.TestCase):
    def test_optional_pipeline_memory_metrics_average_seeds_then_datasets(self):
        values = []
        for dataset, seed, baseline, candidate in [('A', 0, 100., 50.), ('A', 1, 300., 150.),
                                                ('B', 0, 1000., 1200.)]:
            for model, peak in [('original', baseline), ('candidate', candidate)]:
                values.append(dict(row(dataset, model, seed, .8), peak_memory_bytes=peak,
                                   peak_memory_metric='peak_rss', pipeline_seconds=peak / 100,
                                   pooling_prepare_seconds=peak / 1000,
                                   pooling_metadata_bytes=peak,
                                   after_fit_peak_memory_bytes=peak,
                                   extractor_parameter_array_bytes=40.,
                                   total_parameter_array_bytes=40. + peak))
        report = analysis.analyze(values, bootstrap_samples=100)
        model = next(r for r in report['models'] if r['model'] == 'candidate')
        self.assertIn('peak_memory_bytes', model)
        self.assertEqual(model['peak_memory_bytes'], 650.)
        self.assertEqual(model['max_peak_memory_bytes'], 1200.)
        self.assertEqual(model['pooling_prepare_seconds'], .65)
        self.assertEqual(model['pooling_metadata_bytes'], 650.)
        self.assertEqual(model['extractor_parameter_array_bytes'], 40.)
        self.assertEqual(model['total_parameter_array_bytes'], 690.)
        self.assertEqual(model['after_fit_peak_memory_bytes'], 650.)
        self.assertAlmostEqual(report['comparisons'][0]['peak_memory_ratio'], .85)
        self.assertAlmostEqual(report['comparisons'][0]['pipeline_ratio'], .85)

    def test_seed_means_then_equal_dataset_weighting_and_transform_ratios(self):
        rows = [row('A', 'original', 0, .5), row('A', 'original', 1, .7),
                row('A', 'candidate', 0, .7, 2), row('A', 'candidate', 1, .9, 2),
                row('B', 'original', 0, .8), row('B', 'candidate', 0, .6, .5)]
        report = analysis.analyze(rows, baseline='original', bootstrap_samples=100)
        summary = report['comparisons'][0]
        self.assertEqual(summary['datasets'], 2)
        self.assertAlmostEqual(summary['accuracy_gain_pp'], 0.)
        self.assertEqual((summary['wins'], summary['ties'], summary['losses']), (1, 0, 1))
        self.assertAlmostEqual(summary['test_transform_ratio'], 1.25)
        self.assertAlmostEqual(summary['geomean_test_transform_ratio'], 1.)
        self.assertAlmostEqual(summary['median_test_transform_ratio'], 1.25)
        means = next(r for r in report['models'] if r['model'] == 'candidate')
        self.assertAlmostEqual(means['accuracy_percent'], 70.)
        self.assertEqual(means['min_seeds'], 1)
        self.assertEqual(means['max_seeds'], 2)

    def test_only_shared_seeds_and_matching_fingerprints_are_compared(self):
        rows = [row('A', 'original', 0, .5), row('A', 'candidate', 0, .7),
                row('A', 'candidate', 1, .1), row('B', 'original', 0, .5),
                row('B', 'candidate', 0, .9, fingerprint='different')]
        report = analysis.analyze(rows, baseline='original', bootstrap_samples=100)
        pairs = report['per_dataset']
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]['paired_seeds'], 1)
        self.assertAlmostEqual(pairs[0]['accuracy_gain_pp'], 20.)
        self.assertEqual(report['coverage']['unmatched_candidate_runs'], 2)

    def test_reference_same_model_and_baseline_comparisons_are_distinct(self):
        current = [row('A', 'original', 0, .5), row('A', 'candidate', 0, .7)]
        reference = [row('A', 'original', 0, .6), row('A', 'candidate', 0, .8)]
        report = analysis.analyze(current, reference=reference, baseline='original',
                                  bootstrap_samples=100)
        gains = {(r['model'], r['comparison']): r['accuracy_gain_pp']
                 for r in report['comparisons']}
        self.assertAlmostEqual(gains[('candidate', 'current:original')], 20.)
        self.assertAlmostEqual(gains[('candidate', 'reference:candidate')], -10.)
        self.assertAlmostEqual(gains[('candidate', 'reference:original')], 10.)
        self.assertAlmostEqual(gains[('original', 'reference:original')], -10.)

    def test_single_dataset_groups_do_not_claim_an_uncertainty_estimate(self):
        rows = [row('A', 'original', 0, .5), row('A', 'candidate', 0, .5),
                dict(row('B', 'original', 0, .8), length=1000),
                dict(row('B', 'candidate', 0, .8), length=1000)]
        report = analysis.analyze(rows, baseline='original', group_by=['length_group'],
                                  bootstrap_samples=100)
        self.assertEqual(len(report['comparisons']), 2)
        for result in report['comparisons']:
            self.assertEqual(result['ties'], 1)
            self.assertIsNone(result['gain_ci95_low_pp'])
            self.assertIsNone(result['gain_ci95_high_pp'])

    def test_loader_rejects_duplicate_runs_and_nonfinite_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'runs.csv'
            def write(rows):
                with path.open('w', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            value = row('A', 'original', 0, .8)
            write([value, value])
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                analysis.load_rows(path)
            write([dict(value, accuracy='nan')])
            with self.assertRaisesRegex(ValueError, 'finite'):
                analysis.load_rows(path)


if __name__ == '__main__':
    unittest.main()
