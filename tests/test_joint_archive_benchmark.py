"""Fair execution, strict provenance and training-only learning contracts."""
import csv
import fcntl
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


class JointArchiveBenchmarkTests(unittest.TestCase):
    def controller(self):
        self.assertIsNotNone(importlib.util.find_spec('benchmarks.benchmark_joint_ppv'),
                             'The focused original/joint archive runner is missing')
        from benchmarks import benchmark_joint_ppv
        return benchmark_joint_ppv

    def test_fresh_workers_exact_predictions_and_no_test_learning(self):
        runner = self.controller()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); data = root/'Tiny.npz'
            rng = np.random.default_rng(411)
            train = rng.normal(size=(8, 33)).astype(np.float32)
            test = rng.normal(size=(3, 33)).astype(np.float32)
            def save(test_values, labels):
                np.savez(data, X_train=train, y_train=['a','b']*4,
                         X_test=test_values, y_test=labels)
            save(test, ['a','b','a'])
            output = root/'first.csv'
            runner.main(['--data',str(data),'--seeds','0','--budgets','2500',
                         '--threads','1','--batch-size','3','--output',str(output)])
            with output.open() as f: first = list(csv.DictReader(f))
            self.assertEqual(len(first), 2)
            runner.validate_pairs(first)
            for key in runner.EXACT_KEYS:
                self.assertEqual(first[0][key], first[1][key], key)
            for row in first:
                self.assertEqual(row['features'], '2436')
                self.assertEqual(row['numba_threads'], '1')
                self.assertIn('transform_cache_hits',row,'Disk-cache provenance is missing')
                self.assertGreater(int(row['transform_cache_hits']),0)
                self.assertEqual(int(row['transform_cache_misses']),0)
                self.assertEqual(int(row['fit_cache_misses']),0)
                self.assertEqual(int(row['total_parameter_array_bytes']),
                                 sum(int(row[k]) for k in ('extractor_parameter_array_bytes',
                                     'scaler_parameter_array_bytes','classifier_parameter_array_bytes')))
                self.assertGreater(int(row['total_parameter_array_bytes']),int(row['parameter_array_bytes']))
                for split in ('train','test'):
                    times = json.loads(row[f'{split}_transform_repetition_seconds'])
                    hashes = json.loads(row[f'{split}_feature_repetition_sha256'])
                    self.assertEqual(len(times), 3)
                    self.assertGreater(min(times), 0)
                    self.assertEqual(len(set(hashes)), 1)
            # Changing all external test data/labels cannot change learned state.
            save(test*1000+200, ['b','a','b'])
            output2 = root/'changed_test.csv'
            runner.main(['--data',str(data),'--seeds','0','--budgets','2500',
                         '--threads','1','--batch-size','3','--output',str(output2)])
            with output2.open() as f: second = {r['model']:r for r in csv.DictReader(f)}
            for a in first:
                b = second[a['model']]
                for key in ('fit_input_sha256','extractor_parameters_sha256',
                            'train_feature_sha256','scaler_parameters_sha256',
                            'classifier_parameters_sha256','classifier_alpha'):
                    self.assertEqual(a[key], b[key], key)
                self.assertNotEqual(a['test_feature_sha256'], b['test_feature_sha256'])
            # Test changes and source changes are forbidden when resuming.
            with self.assertRaisesRegex(ValueError,'split|fingerprint'):
                runner.main(['--data',str(data),'--seeds','0','--budgets','2500',
                             '--threads','1','--batch-size','3','--output',str(output),'--resume'])

    def test_pair_guard_rejects_any_divergent_learned_state_or_features(self):
        runner = self.controller()
        for budget in (2500,5000,10000):
            control = dict.fromkeys(runner.EXACT_KEYS, 'same')
            control.update(dataset='D',model=f'original_{budget}',seed=0,accuracy=.5,classifier_alpha=1.0)
            candidate = dict(control,model=f'joint_ppv_{budget}')
            runner.validate_pairs([control,candidate])
            for key in runner.EXACT_KEYS:
                changed = dict(candidate,**{key: .25 if key=='accuracy' else 'changed'})
                with self.assertRaises(ValueError,msg=key):
                    runner.validate_pairs([control,changed])

    def test_lock_and_resume_guards_prevent_mutation(self):
        runner = self.controller()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'out.csv'
            with Path(str(output)+'.lock').open('a') as f:
                fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                with self.assertRaisesRegex(RuntimeError,'lock|running'):
                    runner.main(['--data','unused','--output',str(output)])
            self.assertFalse(output.exists())

    def test_order_keeps_budget_pairs_adjacent_and_ignores_test_contents(self):
        runner = self.controller()
        variants = runner.build_variants(runner.build_parser().parse_args(['--data','unused']))
        self.assertEqual(len(variants),6)
        for seed in range(3):
            order = runner.case_order('Dataset',seed,variants,20261026)
            self.assertEqual({v['name'] for v in order},{v['name'] for v in variants})
            for i in range(0,6,2):
                self.assertEqual(order[i]['num_features'],order[i+1]['num_features'])
                self.assertNotEqual(order[i]['implementation'],order[i+1]['implementation'])

    def test_frozen_inventory_rejects_changed_split_before_measuring(self):
        runner=self.controller()
        from benchmarks.inventory_joint_ppv import inspect_archive
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'Tiny.npz';inventory=root/'inventory.json'
            np.savez(source,X_train=np.zeros((4,9),np.float32),y_train=['a','b']*2,
                     X_test=np.ones((2,9),np.float32),y_test=['a','b'])
            frozen=inspect_archive(source);frozen['datasets'][0]['fingerprint']='changed'
            inventory.write_text(json.dumps(frozen))
            with self.assertRaisesRegex(ValueError,'inventory|fingerprint'):
                runner.main(['--data',str(source),'--inventory',str(inventory),
                             '--budgets','2500','--seeds','0','--output',str(root/'out.csv')])


if __name__ == '__main__':
    unittest.main()
