"""
Early classification experiment: how early can MiniRocket decide, and how accurately?

Author : Shreyas Shinde (Student ID 26104692)

Run from the repository root:
    python -B -m experiments.early_classification --data ./data/Univariate_ts/ --datasets GunPoint ECG200 ItalyPowerDemand Trace SyntheticControl
    python -B -m experiments.early_classification --datasets GunPoint --seeds 0 1 2     # aeon download, no --data
    python -B -m experiments.early_classification --datasets GunPoint --target-accuracy 0.9

For every dataset and seed it compares:
    full-length MiniRocket  (waits for the whole series)
    early classifier        (stops as soon as it is confident enough)
and reports accuracy, earliness (fraction of the series used), harmonic mean and time.
Saves CSV files and, for the first dataset, two plots in --output-dir (default results/local/early_<time>/).
"""
import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from experiments.multipool import load_dataset, write_csv
from minirocket_early import EarlyClassifier, PrefixModel, prediction_and_margin


def plot_accuracy_by_prefix(early, X_test, y_test, save_path):
    """Test accuracy of each prefix model (if forced to answer there) + where the early model stops."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fractions = [length / early.series_length_ for length in early.lengths_]
    accuracy = []
    for model in early.models_:
        predicted, _ = prediction_and_margin(model.scores(X_test), model.classes_)
        accuracy.append(np.mean(predicted == y_test))
    _, used = early.predict(X_test)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].plot([100 * f for f in fractions], [100 * a for a in accuracy], "o-", label="test accuracy at this prefix")
    if not np.all(np.isnan(early.cv_accuracy_)):
        axes[0].plot([100 * f for f in fractions], 100 * early.cv_accuracy_, "s--", label="cross-validated (training)")
    axes[0].axhline(100 * early.target_, color="red", linestyle=":", label="target accuracy")
    axes[0].set_xlabel("% of series seen")
    axes[0].set_ylabel("accuracy %")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Accuracy vs. how much of the series is seen")
    axes[1].hist(100 * used / early.series_length_, bins=[100 * f - 5 for f in fractions] + [105], color="#55A868")
    axes[1].set_xlabel("% of series used before deciding")
    axes[1].set_ylabel("test series")
    axes[1].set_title("When did the early classifier stop?")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Early classification experiment")
    parser.add_argument("--datasets", nargs="+", default=["GunPoint", "ECG200", "Coffee", "ItalyPowerDemand", "ArrowHead"])
    parser.add_argument("--data", default=None, help="local UCR archive folder (default: aeon download)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--prefix-fractions", nargs="+", type=float,
                        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    parser.add_argument("--target-accuracy", type=float, default=None,
                        help="stop early only if at least this accurate (default: match full-length CV accuracy)")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    out = Path(args.output_dir) if args.output_dir else \
        Path("results") / "local" / ("early_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    clock = time.perf_counter

    # Warm-up (numba compiles on first use; not timed)
    X0 = np.random.default_rng(0).normal(size=(6, 30)).astype(np.float32)
    EarlyClassifier(prefix_fractions=(0.5, 1.0), num_features=840).fit(X0, np.array([0, 1] * 3)).predict(X0)

    rows = []
    for number, name in enumerate(args.datasets, start=1):
        try:
            X_train, y_train, X_test, y_test = load_dataset(name, args.data)
        except Exception as error:
            print(f"[{number}/{len(args.datasets)}] {name}: skipped ({error})")
            continue
        print(f"[{number}/{len(args.datasets)}] {name}: train {X_train.shape}, test {X_test.shape}")
        for seed in args.seeds:
            # Full-length MiniRocket (the normal way)
            t = clock()
            full = PrefixModel(X_train.shape[1], 10_000, seed).fit(X_train, y_train)
            fit_full = clock() - t
            t = clock()
            predicted, _ = prediction_and_margin(full.scores(X_test), full.classes_)
            predict_full = clock() - t
            full_accuracy = float(np.mean(predicted == y_test))

            # Early classifier
            t = clock()
            early = EarlyClassifier(prefix_fractions=tuple(args.prefix_fractions),
                                    target_accuracy=args.target_accuracy, random_state=seed).fit(X_train, y_train)
            fit_early = clock() - t
            t = clock()
            result = early.evaluate(X_test, y_test)
            predict_early = clock() - t

            rows.append({"dataset": name, "seed": seed, "full_accuracy": full_accuracy,
                         "early_accuracy": result["accuracy"], "earliness": result["earliness"],
                         "harmonic_mean": result["harmonic_mean"], "target_accuracy": early.target_,
                         "fit_full_s": fit_full, "fit_early_s": fit_early,
                         "predict_full_s": predict_full, "predict_early_s": predict_early})
            print(f"    seed {seed}: full {100 * full_accuracy:6.2f}% | early {100 * result['accuracy']:6.2f}% "
                  f"using {100 * result['earliness']:5.1f}% of the series | HM {result['harmonic_mean']:.3f}")

            if number == 1 and seed == args.seeds[0]:
                plot_accuracy_by_prefix(early, X_test, y_test, out / f"accuracy_by_prefix_{name}.png")
                early.plot_stream(X_test[0], save_path=out / f"stream_{name}.png", true_class=y_test[0])

    if not rows:
        print("No dataset could be run.")
        return

    # Summary per dataset (mean over seeds)
    print("\n" + "=" * 92)
    print(f"{'dataset':<22}{'full acc':>10}{'early acc':>11}{'series used':>13}{'HM':>8}{'fit full s':>12}{'fit early s':>13}")
    print("-" * 92)
    summary = []
    for name in dict.fromkeys(r["dataset"] for r in rows):
        r = [x for x in rows if x["dataset"] == name]
        line = {k: float(np.mean([x[k] for x in r])) for k in
                ("full_accuracy", "early_accuracy", "earliness", "harmonic_mean", "fit_full_s", "fit_early_s")}
        line = {"dataset": name, **line}
        summary.append(line)
        print(f"{name:<22}{100 * line['full_accuracy']:>9.2f}%{100 * line['early_accuracy']:>10.2f}%"
              f"{100 * line['earliness']:>12.1f}%{line['harmonic_mean']:>8.3f}{line['fit_full_s']:>12.2f}{line['fit_early_s']:>13.2f}")
    mean = {k: np.mean([s[k] for s in summary]) for k in summary[0] if k != "dataset"}
    print("-" * 92)
    print(f"{'MEAN':<22}{100 * mean['full_accuracy']:>9.2f}%{100 * mean['early_accuracy']:>10.2f}%"
          f"{100 * mean['earliness']:>12.1f}%{mean['harmonic_mean']:>8.3f}{mean['fit_full_s']:>12.2f}{mean['fit_early_s']:>13.2f}")
    print("=" * 92)
    print("series used = average % of each test series seen before deciding (lower = earlier).")
    print("HM = harmonic mean of accuracy and (1 - earliness): high only if accurate AND early.")

    write_csv(out / "measurements.csv", rows)
    write_csv(out / "summary.csv", summary)
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()
