"""Stratified versus uniform biases: accuracy and across-seed spread, from measurements.csv.

  python -B -m experiments.analyse_bias ./results/group_pruning_bias/ --data ./data/Univariate_ts/

The comparison that counts is ``stratified`` against ``uniform`` (same code path and RNG policy).
``original`` against ``uniform`` only shows the effect of the random-number policy and is
reported separately for reference.

Per configuration (scorer, keep_fraction) and dataset: mean and standard deviation (ddof=1) of
accuracy across seeds per strategy; the paired difference stratified - uniform; a Wilcoxon
signed-rank test across datasets on the mean difference and on the standard-deviation difference.
Accuracy is identical for both transform implementations, so only ``implementation == joint``
rows are used. Only (dataset, seed) pairs present for both strategies are compared.

Subgroups (from TRAIN labels, read from --data):
  imbalanced : smallest-class share of the training set < 20%
  small      : fewer than 100 training series
  other      : neither imbalanced nor small
(imbalanced and small can overlap; their counts are reported.)

Writes ``bias_analysis.md`` and ``bias_per_dataset.csv`` into the folder. By default only the
full model (keep_fraction 1.0) is analysed; ``--all-fractions`` adds every pruned configuration
(used for the combined run).
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

STRATEGIES = ('uniform', 'stratified')


def read_rows(folder):
    with (folder / 'measurements.csv').open(newline='') as handle:
        rows = [r for r in csv.DictReader(handle) if r['implementation'] == 'joint']
    for r in rows:
        r['keep_fraction'] = float(r['keep_fraction'])
        r['accuracy'] = float(r['accuracy'])
        r['seed'] = int(r['seed'])
        r['n_train'] = int(r['n_train'])
    return rows


def class_shares(data, names):
    """Smallest-class share of TRAIN per dataset, from the local archive."""
    from types import SimpleNamespace
    from benchmarks.common import _dataset_sources, load_benchmark_dataset
    args = SimpleNamespace(data=str(data), datasets=sorted(names))
    shares = {}
    for name, train, test in _dataset_sources(args):
        y = load_benchmark_dataset(args, name, train, test, []).y_train
        _, counts = np.unique(y, return_counts=True)
        shares[name] = float(counts.min() / counts.sum())
    return shares


def wilcoxon_p(values):
    values = np.asarray(values, dtype=float)
    values = values[values != 0]
    if len(values) < 10:
        return float('nan')
    from scipy.stats import wilcoxon
    return float(wilcoxon(values).pvalue)


def compare(rows, a, b, scorer, fraction):
    """Per-dataset stats of strategy b vs a for one configuration."""
    acc = defaultdict(dict)
    n_train = {}
    for r in rows:
        if r['scorer'] == scorer and r['keep_fraction'] == fraction and r['bias_strategy'] in (a, b):
            acc[(r['dataset'], r['bias_strategy'])][r['seed']] = r['accuracy']
            n_train[r['dataset']] = r['n_train']
    out = []
    for dataset in sorted({d for d, _ in acc}):
        sa, sb = acc.get((dataset, a), {}), acc.get((dataset, b), {})
        seeds = sorted(set(sa) & set(sb))
        if not seeds:
            continue
        xa, xb = np.array([sa[s] for s in seeds]), np.array([sb[s] for s in seeds])
        std = (lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else float('nan'))
        out.append(dict(dataset=dataset, n_train=n_train[dataset], seeds=len(seeds),
                        **{f'{a}_mean': float(xa.mean()), f'{a}_std': std(xa), f'{a}_min': float(xa.min()),
                           f'{a}_max': float(xa.max()), f'{b}_mean': float(xb.mean()), f'{b}_std': std(xb),
                           f'{b}_min': float(xb.min()), f'{b}_max': float(xb.max())},
                        diff_mean=float(xb.mean() - xa.mean()), diff_std=std(xb) - std(xa)))
    return out


def summary_lines(per_dataset, a, b, label):
    if not per_dataset:
        return [f'| {label} | 0 | | | | | | | |']
    d = np.array([r['diff_mean'] for r in per_dataset])
    s = np.array([r['diff_std'] for r in per_dataset])
    s_ok = s[~np.isnan(s)]
    mean_a = 100 * np.mean([r[f'{a}_mean'] for r in per_dataset])
    mean_b = 100 * np.mean([r[f'{b}_mean'] for r in per_dataset])
    sd_a = 100 * np.nanmean([r[f'{a}_std'] for r in per_dataset])
    sd_b = 100 * np.nanmean([r[f'{b}_std'] for r in per_dataset])
    wins, losses = int((d > 0).sum()), int((d < 0).sum())
    lower_sd = int((s_ok < 0).sum())
    return [f'| {label} | {len(d)} | {mean_a:.2f} | {mean_b:.2f} | {100 * d.mean():+.3f} | {100 * np.median(d):+.3f} '
            f'| {wins}/{int((d == 0).sum())}/{losses} | {wilcoxon_p(d):.3f} '
            f'| {sd_a:.3f} | {sd_b:.3f} | {lower_sd}/{len(s_ok)} | {wilcoxon_p(s_ok):.3f} |']


HEADER = ['| subset | datasets | {a} mean % | {b} mean % | mean diff pp | median diff pp | {b} better/tie/worse '
          '| Wilcoxon p (mean) | {a} mean SD pp | {b} mean SD pp | {b} SD lower | Wilcoxon p (SD) |',
          '|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---|---:|']


def table(per_dataset, a, b, shares):
    imbalanced = {d for d, v in shares.items() if v < 0.2}
    small = {r['dataset'] for r in per_dataset if r['n_train'] < 100}
    subsets = [('all', per_dataset),
               ('imbalanced (min class < 20%)', [r for r in per_dataset if r['dataset'] in imbalanced]),
               ('small (n_train < 100)', [r for r in per_dataset if r['dataset'] in small]),
               ('other', [r for r in per_dataset if r['dataset'] not in imbalanced | small])]
    lines = [h.format(a=a, b=b) for h in HEADER]
    for label, subset in subsets:
        lines += summary_lines(subset, a, b, label)
    overlap = len({r['dataset'] for r in per_dataset} & imbalanced & small)
    return lines + ['', f'Imbalanced and small overlap on {overlap} datasets.', '']


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('folder', type=Path)
    p.add_argument('--data', required=True, help='Local archive, to read TRAIN class counts')
    p.add_argument('--all-fractions', action='store_true')
    args = p.parse_args(argv)
    rows = read_rows(args.folder)
    shares = class_shares(args.data, {r['dataset'] for r in rows})
    configs = sorted({(r['scorer'], r['keep_fraction']) for r in rows}, key=lambda c: (-c[1], c[0]))
    if not args.all_fractions:
        configs = [c for c in configs if c[1] == 1.0]
    seeds = sorted({r['seed'] for r in rows})
    lines = ['# Stratified versus uniform biases', '',
             f'Source: `{args.folder / "measurements.csv"}` (joint rows; accuracy is identical for both implementations). '
             f'Seeds: {seeds}. Datasets get equal weight; SD is the across-seed standard deviation (ddof = 1) '
             'within a dataset, averaged over datasets. Differences are stratified minus uniform. Wilcoxon '
             'signed-rank tests run across datasets on per-dataset differences (zeros dropped; not computed '
             'with fewer than 10 non-zero differences).', '']
    csv_rows = []
    for scorer, fraction in configs:
        lines += [f'## keep_fraction {fraction:.2f}, scorer {scorer}', '', '### stratified vs uniform (the comparison that counts)', '']
        per = compare(rows, 'uniform', 'stratified', scorer, fraction)
        lines += table(per, 'uniform', 'stratified', shares)
        csv_rows += [dict(scorer=scorer, keep_fraction=fraction, comparison='stratified-uniform',
                          min_class_share=round(shares[r['dataset']], 4), **r) for r in per]
        if any(r['bias_strategy'] == 'original' for r in rows):
            lines += ['### uniform vs original (effect of the random-number policy only)', '']
            ref = compare(rows, 'original', 'uniform', scorer, fraction)
            lines += table(ref, 'original', 'uniform', shares)
    (args.folder / 'bias_analysis.md').write_text('\n'.join(lines) + '\n')
    if csv_rows:
        with (args.folder / 'bias_per_dataset.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0])); writer.writeheader(); writer.writerows(csv_rows)
    print(f'Wrote {args.folder / "bias_analysis.md"} and {args.folder / "bias_per_dataset.csv"}')


if __name__ == '__main__':
    main()
