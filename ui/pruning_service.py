"""
pruning_service.py - group pruning and class-stratified biases, packaged for the UI

Author : Shreyas Shinde (Student ID 26104692)

It only CALLS the existing code (code/minirocket_pruned.py, code/minirocket_stratified.py,
code/minirocket_baseline.py, benchmarks/common.py), in exactly the way the experiment
runner experiments/group_pruning.py does:

  1. fit the FULL MiniRocket layout once ('original', 'uniform' or 'stratified' biases),
  2. transform TRAIN once with every group, rank groups on TRAIN only,
  3. for each keep fraction: prune, refit the classifier on the kept TRAIN columns,
     time the pruned TEST transform, and check the pruned features equal the
     full model's kept columns bit for bit.

No web-framework code here, so the Streamlit page and the tests use the same functions.

Glossary (also shown in the UI):
  group          one (dilation, kernel) pair: 84 kernels x up to 32 dilations
  keep fraction  share of groups kept, e.g. 0.3 keeps the 30 % most useful groups
  scorer         how usefulness is measured: 'ridge' (classifier weights, iterative)
                 or 'filter' (ANOVA F-score, one shot, no classifier)
  bias strategy  which training series sets each group's thresholds:
                 'original' (one random series, as in the paper), 'uniform' (same, but
                 with a NumPy RNG: the control), 'stratified' (classes take turns)
"""
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import service                                    # import paths (code/ and the repository root)
import numpy as np

import minirocket_pruned as pruned
import minirocket_stratified as stratified
from benchmarks.common import make_classifier     # same classifier everywhere
from minirocket_baseline import fit_univariate, transform_univariate

RESULTS = service.RESULTS_FOLDER
DEFAULT_KEEP_FRACTIONS = (1.0, 0.75, 0.5, 0.3, 0.2, 0.1)
BIAS_STRATEGIES = ("original", "uniform", "stratified")
SCORERS = ("ridge", "filter")


# ---------------------------------------------------------------------------
# 1. Running group pruning (mirrors experiments/group_pruning.py::run_case)
# ---------------------------------------------------------------------------
@dataclass
class PruningResult:
    """Everything the UI shows for one pruning run."""
    rows: list                         # one dict per keep fraction (see run_pruning)
    masks: dict                        # keep fraction -> bool array [num_dilations, 84]
    dilations: list                    # dilation of each mask row
    settings: dict = field(default_factory=dict)


def fit_parameters(X_train, y_train, bias_strategy="original", num_features=10_000, seed=0):
    """Fit MiniRocket's parameters with the chosen bias strategy (same calls as experiments/group_pruning.py)."""
    if bias_strategy == "original":
        return fit_univariate(X_train, num_features=num_features, random_state=seed)
    if bias_strategy not in BIAS_STRATEGIES:
        raise ValueError(f"bias_strategy must be one of {BIAS_STRATEGIES}")
    return stratified.fit(X_train, y_train, num_features=num_features,
                          strategy=bias_strategy, random_state=seed)


def _timed(function, repetitions=1):
    """Return (value of the last call, median seconds over the repetitions)."""
    times, value = [], None
    for _ in range(repetitions):
        start = perf_counter()
        value = function()
        times.append(perf_counter() - start)
    return value, float(np.median(times))


def _model_kilobytes(classifier, n_features):
    """Approximate stored size: ridge weights + scaler mean/scale (float64) for the kept features."""
    rows = np.atleast_2d(classifier.steps[-1][1].coef_).shape[0]   # binary coef_ may be 1-D
    return (rows + 2) * n_features * 8 / 1024


def run_pruning(X_train, y_train, X_test, y_test, keep_fractions=DEFAULT_KEEP_FRACTIONS,
                scorer="ridge", bias_strategy="original", seed=0, num_features=10_000,
                drop_fraction=0.2, repetitions=3):
    """
    Prune at every keep fraction and measure it on the test set.

    Each row: keep_fraction, groups_kept, groups_total, features_kept, accuracy (0-1),
    accuracy_change_pp (vs keep 1.0), test_seconds, time_ratio (vs keep 1.0, same session),
    selection_seconds, model_kb, exact (pruned features == full model's kept columns),
    features_sha256 (fingerprint of the TEST features, same as experiments/group_pruning.py writes).
    Ranking uses TRAIN only; the test set is touched only to report the numbers.
    """
    if scorer not in SCORERS:
        raise ValueError(f"scorer must be one of {SCORERS}")
    X_train = np.ascontiguousarray(X_train, dtype=np.float32)
    X_test = np.ascontiguousarray(X_test, dtype=np.float32)
    y_train, y_test = np.asarray(y_train), np.asarray(y_test)
    fractions = sorted({float(f) for f in keep_fractions} | {1.0}, reverse=True)   # 1.0 = reference
    if not all(0 < f <= 1 for f in fractions):
        raise ValueError("keep fractions must be in (0, 1]")

    parameters = fit_parameters(X_train, y_train, bias_strategy, num_features, seed)
    everything = pruned.keep_all(parameters)
    pruned.transform(X_test[:2], parameters, everything)              # compile/warm up outside timers
    train_full = pruned.transform(X_train, parameters, everything)
    reference_test = transform_univariate(X_test, parameters)        # untimed exactness reference

    rows, masks = [], {}
    for fraction in fractions:
        if fraction >= 1.0:
            keep, selection_seconds = everything, 0.0
        else:
            (keep, _history), selection_seconds = _timed(lambda: pruned.prune_groups(
                train_full, y_train, parameters, fraction, scorer=scorer, drop_fraction=drop_fraction,
                classifier_factory=lambda n: make_classifier(n, "ridge", seed)))
        columns = pruned.kept_columns(parameters, keep)
        classifier = make_classifier(len(X_train), "ridge", seed).fit(train_full[:, columns], y_train)
        features, test_seconds = _timed(lambda: pruned.transform(X_test, parameters, keep), repetitions)
        accuracy = float(np.mean(classifier.predict(features) == y_test))
        exact = bool(np.array_equal(features.view(np.uint32),
                                    np.ascontiguousarray(reference_test[:, columns]).view(np.uint32)))
        masks[fraction] = keep.copy()
        rows.append(dict(keep_fraction=fraction, groups_kept=int(keep.sum()), groups_total=int(everything.sum()),
                         features_kept=int(len(columns)), accuracy=accuracy, test_seconds=test_seconds,
                         selection_seconds=selection_seconds,
                         model_kb=_model_kilobytes(classifier, len(columns)), exact=exact,
                         features_sha256=hashlib.sha256(np.ascontiguousarray(features).tobytes()).hexdigest()))

    full = rows[0]
    for row in rows:
        row["accuracy_change_pp"] = 100 * (row["accuracy"] - full["accuracy"])
        row["time_ratio"] = row["test_seconds"] / full["test_seconds"] if full["test_seconds"] > 0 else float("nan")
    settings = dict(scorer=scorer, bias_strategy=bias_strategy, seed=seed, num_features=num_features,
                    drop_fraction=drop_fraction, n_train=int(len(X_train)), n_test=int(len(X_test)),
                    length=int(X_train.shape[1]))
    return PruningResult(rows, masks, [int(d) for d in parameters[0]], settings)


def compare_bias_strategies(X_train, y_train, X_test, y_test, seeds=(0, 1, 2), num_features=10_000):
    """Test accuracy of the full (unpruned) model for each bias strategy and seed: {strategy: [acc per seed]}."""
    X_train = np.ascontiguousarray(X_train, dtype=np.float32)
    X_test = np.ascontiguousarray(X_test, dtype=np.float32)
    out = {}
    for strategy in BIAS_STRATEGIES:
        accuracies = []
        for seed in seeds:
            parameters = fit_parameters(X_train, y_train, strategy, num_features, seed)
            everything = pruned.keep_all(parameters)
            classifier = make_classifier(len(X_train), "ridge", seed).fit(
                pruned.transform(X_train, parameters, everything), y_train)
            accuracies.append(float(np.mean(classifier.predict(
                pruned.transform(X_test, parameters, everything)) == np.asarray(y_test))))
        out[strategy] = accuracies
    return out


def class_balance(y):
    """Number of training series per class (for explaining stratified biases)."""
    labels, counts = np.unique(np.asarray(y), return_counts=True)
    return {str(label): int(count) for label, count in zip(labels, counts)}


def threshold_sources(y_train, num_groups, seed=0):
    """
    Which class supplies each group's thresholds, for 'uniform' and 'stratified'
    (draw_example_table in code/minirocket_stratified.py): {strategy: {class: number of groups}}.
    """
    y_train = np.asarray(y_train)
    out = {}
    for strategy in ("uniform", "stratified"):
        table = stratified.draw_example_table(y_train, num_groups, 1, strategy, seed)[:, 0]
        labels, counts = np.unique(y_train[table], return_counts=True)
        out[strategy] = {str(label): int(count) for label, count in zip(labels, counts)}
    return out


def format_table(result):
    """Plain-text table of a PruningResult (for a quick look in a terminal)."""
    s = result.settings
    lines = [f"Group pruning: scorer={s['scorer']}, biases={s['bias_strategy']}, seed={s['seed']}, "
             f"{s['n_train']} train / {s['n_test']} test series of length {s['length']}",
             f"{'keep':>5} {'groups':>11} {'features':>8} {'accuracy':>9} {'change':>8} "
             f"{'time':>6} {'size KB':>8} exact"]
    for r in result.rows:
        lines.append(f"{r['keep_fraction']:>5.2f} {r['groups_kept']:>5}/{r['groups_total']:<5} "
                     f"{r['features_kept']:>8} {100 * r['accuracy']:>8.2f}% {r['accuracy_change_pp']:>+7.2f}pp "
                     f"{r['time_ratio']:>5.2f}x {r['model_kb']:>8.0f} {'yes' if r['exact'] else 'NO'}")
    lines.append("time = TEST transform time relative to the unpruned model, measured in this session")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. Published results (read-only), for the "Benchmark" part of the page
# ---------------------------------------------------------------------------
def published_pruning_summary():
    """Rows of results/group_pruning_full/summary.csv (113 datasets x 3 seeds), joint PPV, original biases."""
    import csv
    path = RESULTS / "group_pruning_full" / "summary.csv"
    if not path.exists():
        return []
    rows = []
    with path.open(newline="") as handle:
        for r in csv.DictReader(handle):
            if r["implementation"] != "joint" or r["bias_strategy"] != "original":
                continue
            rows.append(dict(scorer=r["scorer"], keep_fraction=float(r["keep_fraction"]),
                             datasets=int(r["datasets"]), features_kept=float(r["features_kept"]),
                             accuracy_percent=float(r["accuracy_percent"]),
                             accuracy_change_pp=float(r["accuracy_delta_pp_vs_full"]),
                             time_ratio=float(r["test_transform_ratio_vs_full_same_impl"])))
    return sorted(rows, key=lambda r: (r["scorer"] != "none", r["scorer"], -r["keep_fraction"]))


def published_files():
    """Paths of the published write-ups the page links to (only those that exist)."""
    candidates = {
        "Controls (random groups, smaller models)": RESULTS / "controls" / "controls_analysis.md",
        "Accuracy vs. time plot (controls)": RESULTS / "controls" / "pareto.png",
        "Stratified biases, 113 datasets x 20 seeds": RESULTS / "group_pruning_bias_all_20seeds" / "bias_analysis.md",
        "Final paired tests": RESULTS / "final_analysis_tables.md",
        "Full technical write-up": service.PROJECT_ROOT / "README_GROUP_PRUNING.md",
    }
    return {label: path for label, path in candidates.items() if path.exists()}
