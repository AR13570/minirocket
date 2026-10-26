"""Paired dataset ratio intervals and multiplicity checks for the joint study."""
import numpy as np


def holm_adjust(pvalues):
    values = np.asarray(pvalues, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError('pvalues must be finite probabilities')
    order = np.argsort(values, kind='stable')
    adjusted = np.empty(len(values))
    running = 0.
    for rank, index in enumerate(order):
        running = max(running, min(1., (len(values) - rank) * values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def sign_flip_pvalue(gains, samples=50000):
    data = np.asarray(gains, dtype=float)
    if data.ndim != 1 or not len(data) or not np.isfinite(data).all() or samples < 1:
        raise ValueError('Require finite paired gains and positive samples')
    observed = abs(data.mean())
    if observed == 0:
        return 1.
    rng = np.random.default_rng(0)
    extreme = 0
    for start in range(0, samples, 1000):
        size = min(1000, samples - start)
        signs = 2 * rng.integers(2, size=(size, len(data))) - 1
        extreme += int(np.count_nonzero(np.abs((signs * data).mean(axis=1)) >= observed - 1e-14))
    return (extreme + 1) / (samples + 1)


def ratio_summary(candidate, baseline, samples=10000):
    a, b = np.asarray(candidate, dtype=float), np.asarray(baseline, dtype=float)
    if (a.ndim != 1 or a.shape != b.shape or not len(a) or samples < 1
            or not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(a <= 0) or np.any(b <= 0)):
        raise ValueError('Require matching finite positive dataset means')
    rng = np.random.default_rng(0)
    ratios = np.empty(samples)
    for start in range(0, samples, 1000):
        stop = min(start + 1000, samples)
        indices = rng.integers(len(a), size=(stop - start, len(a)))
        ratios[start:stop] = a[indices].mean(axis=1) / b[indices].mean(axis=1)
    low, high = np.quantile(ratios, [.025, .975]) if len(a) > 1 else (None, None)
    return dict(ratio_of_arithmetic_means=float(a.mean() / b.mean()),
                mean_per_dataset_ratio=float(np.mean(a / b)),
                ratio_of_means_ci95_low=None if low is None else float(low),
                ratio_of_means_ci95_high=None if high is None else float(high))

