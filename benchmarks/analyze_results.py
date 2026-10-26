"""Summarize saved benchmark CSVs without fitting or transforming any models."""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import geometric_mean, mean, median

METRICS = ('accuracy', 'training_seconds', 'test_seconds',
           'train_transform_seconds', 'test_transform_seconds')
OPTIONAL_METRICS = ('fit_seconds', 'classifier_fit_seconds', 'predict_seconds', 'load_seconds',
                    'pipeline_seconds', 'worker_wall_seconds', 'peak_memory_bytes', 'peak_rss_bytes',
                    'peak_physical_footprint_bytes', 'parameter_array_bytes', 'train_feature_bytes',
                    'test_feature_bytes', 'pooling_prepare_seconds', 'pooling_metadata_bytes',
                    'extractor_parameter_array_bytes', 'total_parameter_array_bytes',
                    'after_fit_peak_memory_bytes', 'after_train_peak_memory_bytes')
GROUPS = ('classifier', 'channels', 'length', 'n_train', 'n_test', 'features',
          'length_group', 'train_size_group')
CASE = ('dataset', 'split_sha256', 'classifier')


def ratio_name(metric):
    return metric.removesuffix('_seconds').removesuffix('_bytes') + '_ratio'


def run_key(row):
    return tuple(row[key] for key in CASE) + (row['model'], row['seed'])


def load_rows(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        required = set(CASE + ('model', 'seed') + METRICS)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'{path}: missing columns {sorted(missing)}')
        rows, seen = [], set()
        for line, row in enumerate(reader, 2):
            for key in CASE + ('model',):
                if not row[key].strip():
                    raise ValueError(f'{path}:{line}: empty {key}')
                row[key] = row[key].strip()
            row['seed'] = int(row['seed'])
            for key in METRICS:
                row[key] = float(row[key])
                if not math.isfinite(row[key]) or row[key] < 0:
                    raise ValueError(f'{path}:{line}: {key} must be finite and nonnegative')
            for key in OPTIONAL_METRICS:
                if row.get(key) not in ('', None):
                    row[key] = float(row[key])
                    if not math.isfinite(row[key]) or row[key] < 0:
                        raise ValueError(f'{path}:{line}: {key} must be finite and nonnegative')
            if row['accuracy'] > 1:
                raise ValueError(f'{path}:{line}: accuracy must be a fraction in [0, 1]')
            for key in ('channels', 'length', 'n_train', 'n_test', 'features'):
                if row.get(key):
                    row[key] = int(row[key])
            key = run_key(row)
            if key in seen:
                raise ValueError(f'Duplicate dataset/split/classifier/model/seed in {path}:{line}')
            seen.add(key)
            rows.append(row)
    if not rows:
        raise ValueError(f'{path}: no result rows')
    return rows


def group_label(row, group_by):
    parts = []
    for key in group_by:
        if key == 'length_group':
            length = int(row['length'])
            value = '<=100' if length <= 100 else '101-500' if length <= 500 else '>500'
        elif key == 'train_size_group':
            size = int(row['n_train'])
            value = '<=100' if size <= 100 else '101-1000' if size <= 1000 else '>1000'
        else:
            value = row[key]
        parts.append(f'{key}={value}')
    return ', '.join(parts) or 'all'


def bootstrap_interval(values, samples):
    """Dataset bootstrap; seeds have already been averaged within each split."""
    if len(values) < 2:
        return None, None
    import numpy as np
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(0)
    means = np.empty(samples)
    for start in range(0, samples, 1000):
        stop = min(start + 1000, samples)
        indices = rng.integers(len(data), size=(stop - start, len(data)))
        means[start:stop] = data[indices].mean(axis=1)
    low, high = np.quantile(means, [.025, .975])
    return float(low), float(high)


def analyze(rows, reference=None, baseline=None, group_by=None, bootstrap_samples=10000):
    if bootstrap_samples < 1:
        raise ValueError('bootstrap_samples must be positive')
    group_by = ['classifier'] if group_by is None else group_by
    if set(group_by) - set(GROUPS):
        raise ValueError(f'Unknown grouping columns: {set(group_by) - set(GROUPS)}')
    names = {row['model'] for row in rows}
    if baseline is None:
        baseline = 'original' if 'original' in names else None
        if baseline is None:
            raise ValueError('Specify --baseline: no original model found')
    if baseline not in names:
        raise ValueError(f'Baseline {baseline!r} is absent from input')
    metrics = METRICS + tuple(key for key in OPTIONAL_METRICS
                             if all(row.get(key) not in ('', None) for row in rows))
    if 'peak_memory_bytes' in metrics:
        kinds = {row.get('peak_memory_metric') for row in rows}
        if len(kinds) != 1 or None in kinds:
            raise ValueError('Cannot aggregate different peak memory metrics')
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in CASE) + (row['model'],)].append(row)
    dataset_means = []
    for values in grouped.values():
        result = {key: values[0][key] for key in CASE + ('model',)}
        result.update(group=group_label(values[0], group_by), seeds=len(values))
        result.update({key: mean(float(row[key]) for row in values) for key in metrics})
        if 'peak_memory_bytes' in metrics:
            result['max_peak_memory_bytes'] = max(float(row['peak_memory_bytes']) for row in values)
            result['peak_memory_metric'] = values[0]['peak_memory_metric']
        dataset_means.append(result)
    model_groups = defaultdict(list)
    for row in dataset_means:
        model_groups[row['group'], row['model']].append(row)
    models = []
    for (group, model), values in sorted(model_groups.items()):
        result = dict(group=group, model=model, datasets=len(values),
                      min_seeds=min(row['seeds'] for row in values),
                      max_seeds=max(row['seeds'] for row in values),
                      accuracy_percent=100 * mean(row['accuracy'] for row in values))
        result.update({key: mean(row[key] for row in values) for key in metrics if key != 'accuracy'})
        if 'peak_memory_bytes' in metrics:
            result['max_peak_memory_bytes'] = max(row['max_peak_memory_bytes'] for row in values)
            result['peak_memory_metric'] = values[0]['peak_memory_metric']
        models.append(result)

    current_index = {run_key(row): row for row in rows}
    reference_index = {run_key(row): row for row in reference} if reference is not None else {}
    paired = defaultdict(list)
    unmatched = 0
    for row in rows:
        targets = []
        if row['model'] != baseline:
            targets.append(('current', baseline, current_index))
        if reference is not None:
            targets.append(('reference', row['model'], reference_index))
            if row['model'] != baseline:
                targets.append(('reference', baseline, reference_index))
        for source, other_model, index in targets:
            key = tuple(row[key] for key in CASE) + (other_model, row['seed'])
            other = index.get(key)
            if other is None:
                unmatched += 1
                continue
            pair_key = tuple(row[key] for key in CASE) + (row['model'], f'{source}:{other_model}')
            paired[pair_key].append((row, other))
    per_dataset = []
    for key, pairs in sorted(paired.items()):
        candidate = pairs[0][0]
        result = dict(zip(CASE + ('model', 'comparison'), key))
        result.update(group=group_label(candidate, group_by), paired_seeds=len(pairs),
                      accuracy_gain_pp=100 * mean(a['accuracy'] - b['accuracy'] for a, b in pairs))
        for metric in metrics[1:]:
            available = [(a, b) for a, b in pairs if a.get(metric) not in ('', None)
                         and b.get(metric) not in ('', None)]
            if metric == 'peak_memory_bytes':
                available = [(a, b) for a, b in available
                             if a.get('peak_memory_metric') == b.get('peak_memory_metric')]
            numerator = mean(float(a[metric]) for a, _ in available) if available else None
            denominator = mean(float(b[metric]) for _, b in available) if available else None
            result[ratio_name(metric)] = numerator / denominator if denominator and denominator > 0 else None
        per_dataset.append(result)
    comparison_groups = defaultdict(list)
    for row in per_dataset:
        comparison_groups[row['group'], row['model'], row['comparison']].append(row)
    comparisons = []
    for (group, model, comparison), values in sorted(comparison_groups.items()):
        gains = [row['accuracy_gain_pp'] for row in values]
        low, high = bootstrap_interval(gains, bootstrap_samples)
        result = dict(group=group, model=model, comparison=comparison, datasets=len(values),
                      min_paired_seeds=min(row['paired_seeds'] for row in values),
                      max_paired_seeds=max(row['paired_seeds'] for row in values),
                      accuracy_gain_pp=mean(gains), median_gain_pp=median(gains),
                      wins=sum(gain > 1e-6 for gain in gains),
                      ties=sum(abs(gain) <= 1e-6 for gain in gains),
                      losses=sum(gain < -1e-6 for gain in gains),
                      gain_ci95_low_pp=low, gain_ci95_high_pp=high)
        for metric in metrics[1:]:
            ratio = ratio_name(metric)
            defined = [row[ratio] for row in values if row[ratio] is not None]
            result[ratio] = mean(defined) if defined else None
            result['median_' + ratio] = median(defined) if defined else None
            result['geomean_' + ratio] = geometric_mean(defined) if defined and min(defined) > 0 else None
            result[ratio + '_datasets'] = len(defined)
        result['faster_transform_without_accuracy_loss'] = sum(
            row['accuracy_gain_pp'] >= -1e-6 and row['test_transform_ratio'] is not None
            and row['test_transform_ratio'] < 1 for row in values)
        comparisons.append(result)
    return dict(models=models, comparisons=comparisons, per_dataset=per_dataset,
                dataset_means=dataset_means, baseline=baseline, metrics=metrics,
                coverage=dict(input_runs=len(rows), reference_runs=len(reference or []),
                              matched_seed_comparisons=sum(len(values) for values in paired.values()),
                              unmatched_candidate_runs=unmatched),
                policies=dict(aggregation='Average seeds within dataset/split/classifier, then equal dataset weights',
                              timing_ratio='Mean of per-dataset candidate/reference ratios; <1 is faster',
                              confidence='95% percentile bootstrap across paired datasets, seed 0; exploratory',
                              bootstrap_samples=bootstrap_samples))


def print_table(headers, rows):
    def display(value):
        if value is None:
            return 'n/a'
        return f'{value:.3f}' if isinstance(value, float) else str(value)
    cells = [[display(value) for value in row] for row in rows]
    widths = [min(28, max(len(header), *(len(row[index]) for row in cells)))
              for index, header in enumerate(headers)]
    def line(values):
        shortened = [value if len(value) <= width else value[:width - 3] + '...'
                     for value, width in zip(values, widths)]
        return '| ' + ' | '.join(value.ljust(width) if index < 2 else value.rjust(width)
                                  for index, (value, width) in enumerate(zip(shortened, widths))) + ' |'
    print(line(headers))
    print('|' + '|'.join('-' * (width + 2) for width in widths) + '|')
    for row in cells:
        print(line(row))


def print_report(report):
    print('Dataset averages (seeds averaged first; accuracy in percent)')
    for group in sorted({row['group'] for row in report['models']}):
        print(f'\nGroup: {group}')
        values = [row for row in report['models'] if row['group'] == group]
        print_table(['Model', 'Datasets', 'Seeds', 'Accuracy %', 'Train s', 'Test s', 'Train TF s', 'Test TF s'],
                    [[r['model'], r['datasets'], f'{r["min_seeds"]}-{r["max_seeds"]}', r['accuracy_percent'],
                      r['training_seconds'], r['test_seconds'], r['train_transform_seconds'],
                      r['test_transform_seconds']] for r in values])
        if 'peak_memory_bytes' in report.get('metrics', ()):
            print('\nFull-process memory peaks (mean seeds, then datasets; maximum is worst individual run)')
            print_table(['Model', 'Memory metric', 'Mean peak MiB', 'Max peak MiB', 'Pipeline s', 'Fit s'],
                        [[r['model'], r['peak_memory_metric'], r['peak_memory_bytes'] / 2**20,
                          r['max_peak_memory_bytes'] / 2**20, r.get('pipeline_seconds'), r.get('fit_seconds')]
                         for r in values])
    print('\nPaired accuracy comparisons (positive gain favors the input file)')
    for group in sorted({row['group'] for row in report['comparisons']}):
        print(f'\nGroup: {group}')
        values = [row for row in report['comparisons'] if row['group'] == group]
        print_table(['Model', 'Compared with', 'N', 'Gain pp', 'Median pp', 'W/T/L', '95% low', '95% high'],
                    [[r['model'], r['comparison'], r['datasets'], r['accuracy_gain_pp'], r['median_gain_pp'],
                      f'{r["wins"]}/{r["ties"]}/{r["losses"]}', r['gain_ci95_low_pp'], r['gain_ci95_high_pp']]
                     for r in values])
        print('\nTime ratios (below 1 is faster; TF = transform only)')
        print_table(['Model', 'Compared with', 'Train x', 'Test x', 'Train TF x', 'Test TF x'],
                    [[r['model'], r['comparison'], r['training_ratio'], r['test_ratio'],
                      r['train_transform_ratio'], r['test_transform_ratio']] for r in values])
        if 'peak_memory_bytes' in report.get('metrics', ()):
            print('\nPaired process-peak ratios (below1 is lower memory)')
            print_table(['Model', 'Compared with', 'Mean peak x', 'Median peak x'],
                        [[r['model'], r['comparison'], r['peak_memory_ratio'], r['median_peak_memory_ratio']]
                         for r in values])
    print(f'\nCoverage: {report["coverage"]}')
    print('Intervals resample datasets, not seeds; they are exploratory across this dataset suite.')
    if report['coverage']['reference_runs']:
        print('Reference-file timings are historical measurements and may reflect machine/load differences.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--reference', type=Path, help='Also compare with the same models and baseline in an older CSV')
    parser.add_argument('--baseline', help='Default: original')
    parser.add_argument('--group-by', nargs='+', choices=GROUPS, default=['classifier'])
    parser.add_argument('--bootstrap-samples', type=int, default=10000)
    parser.add_argument('--output-dir', type=Path, help='Save model, paired, and per-dataset summaries as CSV plus report JSON')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args(argv)
    rows = load_rows(args.csv)
    reference = load_rows(args.reference) if args.reference else None
    report = analyze(rows, reference, args.baseline, args.group_by, args.bootstrap_samples)
    report['input'] = str(args.csv.resolve())
    report['reference'] = str(args.reference.resolve()) if args.reference else None
    report['configurations'] = {}
    for label, path in [('input', args.csv), ('reference', args.reference)]:
        if path:
            sidecar = path.with_suffix(path.suffix + '.json')
            if sidecar.exists():
                metadata = json.loads(sidecar.read_text())
                report['configurations'][label] = {key: metadata.get(key) for key in
                                                    ('configuration', 'platform', 'versions', 'skipped_datasets')}
    print_report(report)
    if args.output_dir:
        names = ('models', 'comparisons', 'per_dataset', 'dataset_means')
        paths = [args.output_dir / f'{name}.csv' for name in names] + [args.output_dir / 'report.json']
        if not args.overwrite and any(path.exists() for path in paths):
            raise FileExistsError('Summary output exists; choose a new directory or --overwrite')
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, path in zip(names, paths):
            values = report[name]
            with path.open('w', newline='') as handle:
                if values:
                    writer = csv.DictWriter(handle, fieldnames=list(values[0]))
                    writer.writeheader()
                    writer.writerows(values)
        paths[-1].write_text(json.dumps(report, indent=2) + '\n')
        print(f'Summaries: {args.output_dir.resolve()}')
    return report


if __name__ == '__main__':
    main()
