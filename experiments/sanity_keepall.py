"""Keep-all sanity check: the masked joint transform must match, and not be slower than, joint PPV.

  NUMBA_NUM_THREADS=1 python -B -m experiments.sanity_keepall --data ./data/Univariate_ts/ \
      --datasets GunPoint Wafer --repetitions 3

For each dataset: fit once (9,996 features, seed 0), warm up each transform once, then time
TEST transforms ``repetitions`` times for
  (a) minirocket_joint_ppv_simd.transform              (existing joint PPV)
  (b) minirocket_pruned.transform(..., keep_all, 'joint')
and also the original-pooling pair (c) transform_univariate vs (d) pruned 'original'.
Reports min and median seconds, ratio b/a and d/c of medians, and bitwise equality.
"""
import os
os.environ.setdefault('NUMBA_NUM_THREADS', '1')

import argparse
import csv
from pathlib import Path
from time import perf_counter

import numpy as np

from benchmarks.common import _dataset_sources, load_benchmark_dataset
import minirocket_joint_ppv_simd as joint
import minirocket_pruned as pruned
from minirocket_baseline import fit_univariate, transform_univariate


def time_it(function, repetitions):
    function()                                         # warm-up (compilation, caches)
    times, value = [], None
    for _ in range(repetitions):
        start = perf_counter()
        value = function()
        times.append(perf_counter() - start)
    return value, np.array(times)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', required=True)
    p.add_argument('--datasets', nargs='+', default=['GunPoint', 'Wafer'])
    p.add_argument('--repetitions', type=int, default=3)
    p.add_argument('--output', type=Path, default=None, help='optional CSV')
    args = p.parse_args(argv)

    rows = []
    for name, train, test in _dataset_sources(args):
        dataset = load_benchmark_dataset(args, name, train, test, [])
        X_train = np.ascontiguousarray(dataset.X_train[:, 0])
        X_test = np.ascontiguousarray(dataset.X_test[:, 0])
        parameters = fit_univariate(X_train, num_features=10_000, random_state=0)
        ones = pruned.keep_all(parameters)
        pairs = [('joint', lambda: joint.transform(X_test, parameters),
                  lambda: pruned.transform(X_test, parameters, ones, 'joint')),
                 ('original', lambda: transform_univariate(X_test, parameters),
                  lambda: pruned.transform(X_test, parameters, ones, 'original'))]
        for label, reference, candidate in pairs:
            a, ta = time_it(reference, args.repetitions)
            b, tb = time_it(candidate, args.repetitions)
            equal = bool(np.array_equal(a.view(np.uint32), b.view(np.uint32)))
            ratio = float(np.median(tb) / np.median(ta))
            rows.append(dict(dataset=name, pooling=label, n_test=len(X_test), length=X_test.shape[1],
                             reference_median_s=float(np.median(ta)), reference_min_s=float(ta.min()),
                             pruned_keepall_median_s=float(np.median(tb)), pruned_keepall_min_s=float(tb.min()),
                             ratio_median=ratio, ratio_min=float(tb.min() / ta.min()), identical=equal))
            print(f'{name:12s} {label:8s} n_test={len(X_test):5d} ref median {np.median(ta):.4f}s '
                  f'| keep-all median {np.median(tb):.4f}s | ratio {ratio:.3f} | identical={equal}', flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
