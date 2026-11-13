"""
Multi-pooling experiment: original MiniRocket vs the joint-PPV transform vs adaptive multi-pooling.

Author : Shreyas Shinde (Student ID 26104692)

Run from the repository root with code/ on the path (same as the other experiments):
  export PYTHONPATH="$PWD/code"            # Windows: see README_GROUP_PRUNING.md (compat_windows)

  # local UCR archive (benchmarks/README.md), chosen datasets
  python -B -m experiments.multipool --data ./data/Univariate_ts/ --datasets GunPoint ECG200 Coffee

  # every dataset in the local archive
  python -B -m experiments.multipool --data ./data/Univariate_ts/

  # no local archive: aeon downloads the named datasets (GunPoint, ItalyPowerDemand, ArrowHead ship with aeon)
  python -B -m experiments.multipool --datasets GunPoint ItalyPowerDemand ArrowHead

  # also measure what each pooling feature adds on its own / run the earlier variants
  python -B -m experiments.multipool --data ./data/Univariate_ts/ --datasets GunPoint --ablation --methods all

By default it compares original_10k, joint_10k and adaptive_10k (the final model).
For every dataset, seed and method it measures accuracy, fit time, TRAIN and TEST transform
time and classifier time, prints a summary and writes CSVs + metadata.json to --output-dir
(default results/local/multipool_<time>/, which git ignores).
"""
import argparse
import csv
import json
import platform
import time
from argparse import Namespace
from datetime import datetime
from pathlib import Path

import numpy as np

import minirocket_joint_ppv_simd as joint                      # fast PPV transform (bit-identical)
import minirocket_multipool as mp
from benchmarks.common import _dataset_sources, load_benchmark_dataset, make_classifier
from minirocket_adaptive import AdaptivePoolingMiniRocket
from minirocket_baseline import fit_univariate, transform_univariate   # seeded fit + ORIGINAL transform


# ---------------------------------------------------------------------------
# 1. The methods we compare
# ---------------------------------------------------------------------------
# "num_features" = number of ORIGINAL MiniRocket features (one per kernel/dilation/bias).
# Multi-pooling turns each of them into 4 features (and x2 with the difference series),
# so we use smaller numbers for multi-pooling to keep the TOTAL size comparable (~10k).
# That way any accuracy gain comes from BETTER features, not just MORE features.
METHODS = [
    {"name": "original_10k",       "kind": "original",  "num_features": 10_000,
     "description": "Original MiniRocket transform, PPV only (~10k features)"},
    {"name": "joint_10k",          "kind": "joint",     "num_features": 10_000,
     "description": "Joint PPV transform (same features as original, faster)"},
    {"name": "multipool_10k",      "kind": "multipool", "num_features": 2_500, "use_difference": False,
     "description": "4 pooling features on the raw series (~10k features)"},
    {"name": "multipool_diff_10k", "kind": "multipool", "num_features": 1_250, "use_difference": True,
     "description": "4 pooling features on raw + difference series (~10k features)"},
    {"name": "multipool_diff_50k", "kind": "multipool", "num_features": 6_250, "use_difference": True,
     "description": "4 pooling features on raw + difference series (~50k features, MultiRocket size)"},
    {"name": "adaptive_10k",       "kind": "adaptive",  "num_features": 10_000,
     "description": "FINAL MODEL: the original ~10k features, plus MPV only when cross-validation "
                    "on the TRAINING set says it helps (code/minirocket_adaptive.py)"},
]

# Methods run when --methods is not given: the baseline, the joint-PPV version and the final model.
DEFAULT_METHODS = ["original_10k", "joint_10k", "adaptive_10k"]

# Ablation: which feature blocks to keep. Uses one multi-pooling model with the difference series.
ABLATION_SETS = [
    ("raw PPV only (= original features)", ["raw_PPV"]),
    ("raw PPV + MPV",                      ["raw_PPV", "raw_MPV"]),
    ("raw PPV + MIPV",                     ["raw_PPV", "raw_MIPV"]),
    ("raw PPV + LSPV",                     ["raw_PPV", "raw_LSPV"]),
    ("raw all 4 pools",                    ["raw_PPV", "raw_MPV", "raw_MIPV", "raw_LSPV"]),
    ("raw + diff PPV only",                ["raw_PPV", "diff_PPV"]),
    ("raw + diff all 4 pools",             ["raw_PPV", "raw_MPV", "raw_MIPV", "raw_LSPV",
                                            "diff_PPV", "diff_MPV", "diff_MIPV", "diff_LSPV"]),
]

DEFAULT_DATASETS = ["GunPoint", "ItalyPowerDemand", "ArrowHead"]   # bundled with aeon, work offline


# ---------------------------------------------------------------------------
# 2. Loading data
# ---------------------------------------------------------------------------
def to_2d_float32(X):
    """aeon gives (examples, channels, length). We need (examples, length) as float32."""
    if isinstance(X, list):
        raise ValueError("series have different lengths (not supported by this transform)")
    X = np.asarray(X)
    if X.ndim == 3:
        if X.shape[1] != 1:
            raise ValueError("multivariate dataset (more than one channel)")
        X = X[:, 0, :]
    return np.ascontiguousarray(X, dtype=np.float32)


def load_dataset(name, data=None):
    """
    Return X_train, y_train, X_test, y_test (2-D float32) for one UCR dataset.
    data = local archive folder: read with the benchmark loader (benchmarks/common.py), the same
           files and checks as every other experiment in this repository.
    data = None: aeon's bundled copy, or a download.
    """
    if data is not None:
        sources = _dataset_sources(Namespace(data=data, datasets=[name]))
        dataset = load_benchmark_dataset(Namespace(data=data, datasets=[name]), *sources[0], [])
        X_train, y_train, X_test, y_test = dataset.X_train, dataset.y_train, dataset.X_test, dataset.y_test
    else:
        from aeon.datasets import load_classification
        X_train, y_train = load_classification(name, split="train")
        X_test, y_test = load_classification(name, split="test")
    X_train, X_test = to_2d_float32(X_train), to_2d_float32(X_test)
    if not (np.isfinite(X_train).all() and np.isfinite(X_test).all()):
        raise ValueError("dataset has missing values")
    return X_train, np.asarray(y_train), X_test, np.asarray(y_test)


def all_dataset_names(data):
    """Every dataset with complete TRAIN/TEST files under the local archive folder."""
    return [name for name, _train, _test in _dataset_sources(Namespace(data=data, datasets=None))]


# ---------------------------------------------------------------------------
# 3. Fit + transform for each kind of method
# ---------------------------------------------------------------------------
def fit_method(method, X_train, seed):
    if method["kind"] in ("original", "joint"):
        return fit_univariate(X_train, num_features=method["num_features"], random_state=seed)
    return mp.fit(X_train, num_features=method["num_features"],
                  use_difference=method["use_difference"], random_state=seed)


def transform_method(method, X, parameters):
    if method["kind"] == "original":
        return transform_univariate(X, parameters)       # original minirocket.transform
    if method["kind"] == "joint":
        return joint.transform(X, parameters)            # joint PPV
    return mp.transform(X, parameters)                    # multi-pooling


def warm_up(methods):
    """
    Numba compiles each function the first time it is called. Compilation can take
    several seconds and is NOT part of the real running time, so we call every method
    once on tiny random data before timing anything.
    """
    X = np.random.default_rng(0).normal(size=(4, 30)).astype(np.float32)
    for method in methods:
        if method["kind"] == "adaptive":
            # the adaptive model needs labels because it trains classifiers inside fit()
            AdaptivePoolingMiniRocket(num_features=method["num_features"]).fit(X, np.array([0, 1, 0, 1])).predict(X)
        else:
            transform_method(method, X, fit_method(method, X, seed=0))

    # The very first scikit-learn fit is also slow (libraries load, worker threads start).
    # Without this, that one-off cost (about 1 s) would be charged to whichever method runs
    # first - the original - and make it look slower than it really is.
    F = transform_method(methods[0], X, fit_method(methods[0], X, seed=0))
    make_classifier(len(X), "ridge", 0).fit(F, np.array([0, 1, 0, 1]))


# ---------------------------------------------------------------------------
# 4. One experiment = one dataset, one seed, one method
# ---------------------------------------------------------------------------
def run_adaptive(method, X_train, y_train, X_test, y_test, seed):
    """The final model does its own fitting, feature selection and classifier training."""
    model = AdaptivePoolingMiniRocket(num_features=method["num_features"], random_state=seed)
    model.fit(X_train, y_train)

    t = time.perf_counter()
    predictions = model.predict(X_test)
    predict_s = time.perf_counter() - t
    test_transform_s = model.last_transform_time_

    # "classifier_s" here = choosing the feature set by cross-validation + final training + predicting
    classifier_s = model.selection_time_ + model.classifier_time_ + (predict_s - test_transform_s)
    num_features_used = len(model.chosen_blocks_) * mp.num_base_features(model.parameters_["raw"])
    return {
        "accuracy": float(np.mean(predictions == y_test)),
        "num_output_features": int(num_features_used),
        "fit_s": model.fit_time_,
        "transform_train_s": model.transform_time_,
        "transform_test_s": test_transform_s,
        "classifier_s": classifier_s,
        "total_s": model.fit_time_ + model.transform_time_ + test_transform_s + classifier_s,
        "chosen": model.chosen_feature_set_,
    }, None, None, model


def run_one(method, X_train, y_train, X_test, y_test, seed):
    if method["kind"] == "adaptive":
        return run_adaptive(method, X_train, y_train, X_test, y_test, seed)

    clock = time.perf_counter

    t = clock(); parameters = fit_method(method, X_train, seed); fit_s = clock() - t
    t = clock(); F_train = transform_method(method, X_train, parameters); train_s = clock() - t
    t = clock(); F_test = transform_method(method, X_test, parameters); test_s = clock() - t

    # Same classifier as the benchmark: StandardScaler + RidgeClassifierCV
    t = clock()
    classifier = make_classifier(len(X_train), "ridge", seed)
    classifier.fit(F_train, y_train)
    predictions = classifier.predict(F_test)
    classifier_s = clock() - t

    return {
        "accuracy": float(np.mean(predictions == y_test)),
        "num_output_features": int(F_train.shape[1]),
        "fit_s": fit_s,
        "transform_train_s": train_s,
        "transform_test_s": test_s,
        "classifier_s": classifier_s,
        "total_s": fit_s + train_s + test_s + classifier_s,
        "chosen": "",
    }, F_train, F_test, parameters


# ---------------------------------------------------------------------------
# 5. Summary statistics
# ---------------------------------------------------------------------------
def per_dataset_means(rows, method_name, column):
    """Average over seeds inside each dataset -> {dataset: mean value}."""
    values = {}
    for row in rows:
        if row["method"] == method_name:
            values.setdefault(row["dataset"], []).append(row[column])
    return {dataset: float(np.mean(v)) for dataset, v in values.items()}


def summarise(rows, methods, baseline="original_10k"):
    """One summary line per method: mean accuracy/times, wins vs baseline, Wilcoxon p, mean rank."""
    datasets = sorted({row["dataset"] for row in rows})
    accuracy = {m["name"]: per_dataset_means(rows, m["name"], "accuracy") for m in methods}

    # Average rank of each method on each dataset (1 = most accurate; ties share the rank).
    ranks = {m["name"]: [] for m in methods}
    for dataset in datasets:
        names = [m["name"] for m in methods]
        scores = np.array([accuracy[name][dataset] for name in names])
        for name, score in zip(names, scores):
            better = np.sum(scores > score)
            equal = np.sum(scores == score)
            ranks[name].append(better + (equal + 1) / 2)

    summary = []
    for method in methods:
        name = method["name"]
        line = {"method": name, "datasets": len(datasets),
                "mean_accuracy": float(np.mean([accuracy[name][d] for d in datasets])),
                "mean_rank": float(np.mean(ranks[name]))}
        for column in ("num_output_features", "fit_s", "transform_train_s", "transform_test_s",
                       "classifier_s", "total_s"):
            line["mean_" + column] = float(np.mean(list(per_dataset_means(rows, name, column).values())))

        # Compare with the baseline dataset by dataset.
        differences = np.array([accuracy[name][d] - accuracy[baseline][d] for d in datasets])
        line["wins_vs_original"] = int(np.sum(differences > 1e-12))
        line["ties_vs_original"] = int(np.sum(np.abs(differences) <= 1e-12))
        line["losses_vs_original"] = int(np.sum(differences < -1e-12))

        # Wilcoxon signed-rank test: is the accuracy difference systematic or just noise?
        # Needs a handful of datasets with non-zero differences to mean anything.
        line["wilcoxon_p_vs_original"] = ""
        if name != baseline and np.count_nonzero(np.abs(differences) > 1e-12) >= 5:
            from scipy.stats import wilcoxon
            line["wilcoxon_p_vs_original"] = float(wilcoxon(differences, zero_method="wilcox").pvalue)
        summary.append(line)
    return summary


def print_summary(summary):
    print("\n" + "=" * 118)
    print(f"{'method':<20}{'features':>9}{'accuracy':>10}{'rank':>6}{'W/T/L vs orig':>15}"
          f"{'Wilcoxon p':>12}{'fit s':>8}{'train tr s':>11}{'test tr s':>10}{'clf s':>8}{'total s':>9}")
    print("-" * 118)
    for s in summary:
        p = s["wilcoxon_p_vs_original"]
        p_text = f"{p:.4f}" if p != "" else "n/a"
        wtl = f"{s['wins_vs_original']}/{s['ties_vs_original']}/{s['losses_vs_original']}"
        print(f"{s['method']:<20}{s['mean_num_output_features']:>9.0f}{100 * s['mean_accuracy']:>9.2f}%"
              f"{s['mean_rank']:>6.2f}{wtl:>15}{p_text:>12}{s['mean_fit_s']:>8.3f}"
              f"{s['mean_transform_train_s']:>11.3f}{s['mean_transform_test_s']:>10.3f}"
              f"{s['mean_classifier_s']:>8.3f}{s['mean_total_s']:>9.3f}")
    print("=" * 118)
    print("accuracy = mean over datasets; rank = average rank (1 = best); W/T/L = datasets where the method")
    print("was more accurate / equal / less accurate than the original; times are means per dataset in seconds.")
    print("Wilcoxon p < 0.05 means the accuracy difference to the original is statistically significant.")


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# 6. Main program
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Original vs joint PPV vs adaptive multi-pooling MiniRocket")
    parser.add_argument("--data", default=None, help="local UCR archive folder, e.g. ./data/Univariate_ts/ "
                                                       "(default: aeon download)")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="UCR dataset names (default: every dataset in --data, or GunPoint "
                             "ItalyPowerDemand ArrowHead without --data)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="fitting seeds")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS,
                        choices=[m["name"] for m in METHODS] + ["all"],
                        help="which methods to run ('all' = every method in the METHODS list)")
    parser.add_argument("--threads", type=int, default=None, help="number of CPU threads for numba (default: all)")
    parser.add_argument("--ablation", action="store_true", help="also run the pooling-feature ablation")
    parser.add_argument("--ablation-features", type=int, default=2_500,
                        help="original features per representation for the ablation model (default 2500)")
    parser.add_argument("--output-dir", default=None, help="where to save CSVs (default: results/local/multipool_<time>)")
    args = parser.parse_args()

    # Which datasets
    names = args.datasets or (all_dataset_names(args.data) if args.data else DEFAULT_DATASETS)

    # Which methods (the original is always included because everything is compared to it)
    chosen = set(args.methods) | {"original_10k"}
    if "all" in chosen:
        chosen = {m["name"] for m in METHODS}
    methods = [m for m in METHODS if m["name"] in chosen]

    # Threads (same for every method, so the comparison is fair)
    import numba
    if args.threads is not None:
        numba.set_num_threads(args.threads)
    threads = numba.get_num_threads()

    output_dir = Path(args.output_dir) if args.output_dir else \
        Path("results") / "local" / ("multipool_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Datasets: {len(names)} | seeds: {args.seeds} | numba threads: {threads}")
    print("Compiling (warm-up, not timed)...")
    warm_up(methods)

    rows, ablation_rows, skipped = [], [], []
    for dataset_number, name in enumerate(names, start=1):
        try:
            X_train, y_train, X_test, y_test = load_dataset(name, args.data)
            if X_train.shape[1] < 10:
                raise ValueError("series shorter than 10 points")
        except Exception as error:      # skip datasets this transform cannot handle
            print(f"[{dataset_number}/{len(names)}] {name}: skipped ({error})")
            skipped.append({"dataset": name, "reason": str(error)})
            continue

        print(f"[{dataset_number}/{len(names)}] {name}: train {X_train.shape}, test {X_test.shape}")
        for seed in args.seeds:
            for method in methods:
                result, _, _, _ = run_one(method, X_train, y_train, X_test, y_test, seed)
                rows.append({"dataset": name, "seed": seed, "method": method["name"], **result})
                print(f"    seed {seed}  {method['name']:<20} accuracy {100 * result['accuracy']:6.2f}%"
                      f"   transform train+test {result['transform_train_s'] + result['transform_test_s']:.3f}s"
                      + (f"   chose {result['chosen']}" if result["chosen"] else ""))

            if args.ablation:
                # One multi-pooling model with the difference series; then train the
                # classifier on different subsets of its feature blocks.
                ablation_method = {"name": "ablation", "kind": "multipool",
                                   "num_features": args.ablation_features, "use_difference": True}
                _, F_train, F_test, parameters = run_one(ablation_method, X_train, y_train, X_test, y_test, seed)
                for label, blocks in ABLATION_SETS:
                    classifier = make_classifier(len(X_train), "ridge", seed)
                    classifier.fit(mp.select_blocks(F_train, parameters, blocks), y_train)
                    accuracy = float(np.mean(classifier.predict(mp.select_blocks(F_test, parameters, blocks)) == y_test))
                    ablation_rows.append({"dataset": name, "seed": seed, "feature_set": label,
                                          "num_features": int(mp.select_blocks(F_test[:1], parameters, blocks).shape[1]),
                                          "accuracy": accuracy})

    if not rows:
        print("No dataset could be run.")
        return

    summary = summarise(rows, methods)
    print_summary(summary)

    # Which feature set did the final model choose? (per dataset, over the seeds)
    adaptive_rows = [r for r in rows if r["method"] == "adaptive_10k"]
    if adaptive_rows:
        print("\nFinal model (adaptive_10k) - feature set chosen from TRAINING data only:")
        for name in sorted({r["dataset"] for r in adaptive_rows}):
            picks = [r["chosen"] for r in adaptive_rows if r["dataset"] == name]
            print(f"    {name:<20} {', '.join(picks)}")

    write_csv(output_dir / "measurements.csv", rows)
    write_csv(output_dir / "summary.csv", summary)
    write_csv(output_dir / "skipped.csv", skipped)

    if ablation_rows:
        print("\nAblation (mean accuracy over datasets and seeds):")
        table = []
        for label, _ in ABLATION_SETS:
            values = [r["accuracy"] for r in ablation_rows if r["feature_set"] == label]
            sizes = [r["num_features"] for r in ablation_rows if r["feature_set"] == label]
            table.append({"feature_set": label, "num_features": int(np.mean(sizes)),
                          "mean_accuracy": float(np.mean(values))})
            print(f"    {label:<38}{int(np.mean(sizes)):>8} features   {100 * np.mean(values):6.2f}%")
        write_csv(output_dir / "ablation_measurements.csv", ablation_rows)
        write_csv(output_dir / "ablation_summary.csv", table)

    # Record the settings and environment, so the results can be reproduced.
    with open(output_dir / "metadata.json", "w") as handle:
        json.dump({"datasets": names, "seeds": args.seeds, "methods": methods, "threads": threads,
                   "python": platform.python_version(), "numba": numba.__version__,
                   "numpy": np.__version__, "machine": platform.machine(), "system": platform.system(),
                   "date": datetime.now().isoformat(timespec="seconds")}, handle, indent=2)

    print(f"\nSaved results to {output_dir}")


if __name__ == "__main__":
    main()
