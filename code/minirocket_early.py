"""
minirocket_early.py
===================

Author : Shreyas Shinde (Student ID 26104692)
Feature: Early classification with MiniRocket (decide before the series ends)

THE PROBLEM
-----------
Normal classification waits for the WHOLE series. In real-time uses (fault
detection, a heartbeat monitor, a gesture sensor) we want an answer as early as
possible, but only when we can trust it.

THE IDEA (three steps)
----------------------
1. PREFIX MODELS. Train one MiniRocket classifier per prefix length, e.g. on
   the first 10%, 20%, ..., 100% of every training series.

2. CONFIDENCE THRESHOLDS, learned from the TRAINING data only.
   The ridge classifier gives a score per class. Its confidence is the MARGIN:
       margin = best score - second best score   (2 classes: |score|)
   For each prefix length we estimate, with k-fold cross-validation on the
   training set, how accurate the predictions are at each margin. We then pick
   the smallest margin threshold for which "predictions at least this confident"
   are as accurate as the TARGET accuracy (by default: the cross-validated
   accuracy of the full-length model, i.e. "as reliable as waiting until the end").

3. EARLY STOPPING AT PREDICTION TIME. A new series arrives point by point. At each
   prefix length we predict; if the margin passes that prefix's threshold we STOP
   and answer. If no prefix is confident enough, we answer at full length.

MEASURES
--------
* accuracy  : fraction of correct answers
* earliness : average fraction of the series that was needed (lower = earlier)
* harmonic mean HM = 2 * accuracy * (1 - earliness) / (accuracy + 1 - earliness)
  (standard single score in the early-classification literature: high only if
   the method is BOTH accurate AND early)

The prefix models use plain MiniRocket with the fast joint-PPV transform (fast to fit
many times). Everything is fitted on training data only.
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold

import minirocket_joint_ppv_simd as joint                # fast PPV transform (bit-identical)
from minirocket_baseline import fit_univariate            # seeded MiniRocket fit
from minirocket_multipool import ridge_classifier         # = benchmarks.common.make_classifier(n, 'ridge', seed)

MIN_LENGTH = 9   # MiniRocket kernels have 9 weights, so a prefix needs at least 9 points


# ---------------------------------------------------------------------------
# 1. A small MiniRocket classifier for ONE prefix length
# ---------------------------------------------------------------------------
class PrefixModel:
    """MiniRocket (fast joint-PPV transform) + ridge classifier on the first `length` points."""

    def __init__(self, length, num_features, random_state):
        self.length = length
        self.num_features = num_features
        self.random_state = random_state

    def fit(self, X, y):
        Xp = np.ascontiguousarray(X[:, :self.length], dtype=np.float32)
        self.parameters_ = fit_univariate(Xp, num_features=self.num_features, random_state=self.random_state)
        self.classifier_ = ridge_classifier(len(Xp), self.random_state)
        self.classifier_.fit(joint.transform(Xp, self.parameters_), y)
        return self

    def scores(self, X):
        """Class scores (n_examples x n_classes) from the first `length` points of X."""
        Xp = np.ascontiguousarray(X[:, :self.length], dtype=np.float32)
        s = self.classifier_.decision_function(joint.transform(Xp, self.parameters_))
        if s.ndim == 1:                       # two classes: one score, > 0 means classes_[1]
            s = np.column_stack([-s, s])
        return s

    @property
    def classes_(self):
        return self.classifier_.classes_


def prediction_and_margin(scores, classes):
    """Predicted class and confidence margin (best score minus second best)."""
    order = np.sort(scores, axis=1)
    margin = order[:, -1] - order[:, -2]
    return classes[np.argmax(scores, axis=1)], margin


def lowest_reliable_threshold(margins, correct, target_accuracy, min_examples):
    """
    Smallest margin threshold such that the predictions with margin >= threshold are
    at least `target_accuracy` accurate (and there are at least `min_examples` of them).
    Returns +infinity if no threshold is reliable enough (= never stop at this prefix).
    """
    order = np.argsort(-margins)                    # most confident first
    sorted_margins = margins[order]
    cumulative_correct = np.cumsum(correct[order])
    counts = np.arange(1, len(order) + 1)
    accuracy_above = cumulative_correct / counts    # accuracy of the top-k most confident
    best = np.inf
    for i in range(len(order)):
        # Only use a cut between DIFFERENT margins (ties must stay together).
        if i + 1 < len(order) and sorted_margins[i + 1] == sorted_margins[i]:
            continue
        if counts[i] >= min_examples and accuracy_above[i] >= target_accuracy:
            best = sorted_margins[i]                # keep lowering the threshold while still reliable
    return best


# ---------------------------------------------------------------------------
# 2. The early classifier
# ---------------------------------------------------------------------------
class EarlyClassifier:
    """
    Usage:
        early = EarlyClassifier().fit(X_train, y_train)
        predictions, lengths_used = early.predict(X_test)
        print(early.evaluate(X_test, y_test))
        steps = early.stream(X_test[0])     # point-by-point simulation of one series
    """

    def __init__(self, prefix_fractions=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
                 target_accuracy=None, cv_folds=5, num_features=10_000, min_examples=None,
                 random_state=0):
        self.prefix_fractions = prefix_fractions
        self.target_accuracy = target_accuracy      # None = match the full-length cross-validated accuracy
        self.cv_folds = cv_folds
        self.num_features = num_features
        self.min_examples = min_examples            # None = 10% of the training set (at least 3)
        self.random_state = random_state

    def _prefix_lengths(self, L):
        """Prefix lengths in time steps: at least 9, unique, and always ending with the full length."""
        lengths = {max(MIN_LENGTH, int(round(f * L))) for f in self.prefix_fractions}
        lengths = sorted(length for length in lengths if length <= L)
        if not lengths or lengths[-1] != L:
            lengths.append(L)
        return lengths

    def fit(self, X, y):
        X = np.ascontiguousarray(X, dtype=np.float32)
        y = np.asarray(y)
        n, L = X.shape
        if L < MIN_LENGTH:
            raise ValueError("series must have at least 9 points")
        self.series_length_ = L
        self.lengths_ = self._prefix_lengths(L)
        min_examples = self.min_examples or max(3, int(0.1 * n))

        # Step 1: one model per prefix length, trained on the whole training set.
        self.models_ = [PrefixModel(length, self.num_features, self.random_state).fit(X, y)
                        for length in self.lengths_]

        # Step 2: out-of-fold predictions and margins for every prefix (cross-validation).
        smallest_class = int(np.min(np.unique(y, return_counts=True)[1]))
        folds = min(self.cv_folds, smallest_class)
        oof_correct = np.zeros((len(self.lengths_), n), dtype=bool)
        oof_margin = np.zeros((len(self.lengths_), n))
        if folds >= 2:
            splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.random_state)
            for train_index, valid_index in splitter.split(X, y):
                for p, length in enumerate(self.lengths_):
                    fold_model = PrefixModel(length, self.num_features, self.random_state).fit(X[train_index], y[train_index])
                    predicted, margin = prediction_and_margin(fold_model.scores(X[valid_index]), fold_model.classes_)
                    oof_correct[p, valid_index] = predicted == y[valid_index]
                    oof_margin[p, valid_index] = margin
            self.cv_accuracy_ = oof_correct.mean(axis=1)            # accuracy per prefix length
        else:
            self.cv_accuracy_ = np.full(len(self.lengths_), np.nan)

        # Step 3: the target accuracy and one confidence threshold per prefix.
        if self.target_accuracy is not None:
            self.target_ = float(self.target_accuracy)
        else:
            self.target_ = float(self.cv_accuracy_[-1]) if folds >= 2 else 1.0
        self.thresholds_ = []
        for p in range(len(self.lengths_)):
            if p == len(self.lengths_) - 1:
                self.thresholds_.append(-np.inf)     # full length: always answer
            elif folds < 2:
                self.thresholds_.append(np.inf)      # cannot validate -> never stop early
            else:
                self.thresholds_.append(lowest_reliable_threshold(
                    oof_margin[p], oof_correct[p], self.target_, min_examples))
        return self

    def predict(self, X):
        """
        Predict every series using as few points as possible.
        Returns (predictions, number of points used for each series).
        """
        X = np.ascontiguousarray(X, dtype=np.float32)
        n = len(X)
        predictions = np.empty(n, dtype=self.models_[0].classes_.dtype)
        used = np.full(n, self.series_length_)
        undecided = np.arange(n)
        for model, length, threshold in zip(self.models_, self.lengths_, self.thresholds_):
            if len(undecided) == 0:
                break
            predicted, margin = prediction_and_margin(model.scores(X[undecided]), model.classes_)
            stop = margin >= threshold
            predictions[undecided[stop]] = predicted[stop]
            used[undecided[stop]] = length
            undecided = undecided[~stop]
        return predictions, used

    def evaluate(self, X, y):
        """Accuracy, earliness and their harmonic mean on (X, y)."""
        predictions, used = self.predict(X)
        accuracy = float(np.mean(predictions == np.asarray(y)))
        earliness = float(np.mean(used / self.series_length_))
        harmonic_mean = 0.0 if accuracy + (1 - earliness) == 0 else \
            2 * accuracy * (1 - earliness) / (accuracy + (1 - earliness))
        return {"accuracy": accuracy, "earliness": earliness, "harmonic_mean": harmonic_mean,
                "mean_points_used": float(np.mean(used))}

    def stream(self, x):
        """
        Simulate one series arriving in real time. For every prefix length: prediction,
        margin, threshold and whether the model would stop there.
        """
        x = np.asarray(x, dtype=np.float32)[None, :]
        steps = []
        for model, length, threshold in zip(self.models_, self.lengths_, self.thresholds_):
            predicted, margin = prediction_and_margin(model.scores(x), model.classes_)
            steps.append({"points_seen": length, "fraction_seen": length / self.series_length_,
                          "prediction": predicted[0], "margin": float(margin[0]),
                          "threshold": float(threshold), "stop": bool(margin[0] >= threshold)})
            if margin[0] >= threshold:
                break
        return steps

    def plot_stream(self, x, save_path=None, true_class=None):
        """Series with the stopping point marked, plus confidence vs threshold over time."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = np.asarray(x, dtype=np.float32)
        # Evaluate every prefix (not only until the stop) to draw the full confidence curve.
        rows = []
        for model, length, threshold in zip(self.models_, self.lengths_, self.thresholds_):
            predicted, margin = prediction_and_margin(model.scores(x[None, :]), model.classes_)
            rows.append((length, predicted[0], float(margin[0]), float(threshold)))
        stop_row = next(r for r in rows if r[2] >= r[3])

        fig, axes = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True)
        axes[0].plot(np.arange(len(x)), x, color="lightgray", linewidth=1.4, label="not yet seen")
        axes[0].plot(np.arange(stop_row[0]), x[:stop_row[0]], color="black", linewidth=1.6, label="seen before deciding")
        axes[0].axvline(stop_row[0] - 1, color="green", linewidth=2,
                        label=f"decision: {stop_row[1]} at {100 * stop_row[0] / len(x):.0f}% of the series")
        title = "Early classification"
        if true_class is not None:
            title += f" (true class: {true_class})"
        axes[0].set_title(title)
        axes[0].legend(loc="upper right", fontsize=8)
        lengths = [r[0] for r in rows]
        axes[1].plot(lengths, [r[2] for r in rows], "o-", label="confidence (margin)")
        finite = [(r[0], r[3]) for r in rows if np.isfinite(r[3])]
        if finite:
            axes[1].plot([f[0] for f in finite], [f[1] for f in finite], "s--", color="red", label="threshold")
        axes[1].set_xlabel("points seen")
        axes[1].set_ylabel("margin")
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
        return fig
