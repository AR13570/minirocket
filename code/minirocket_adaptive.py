"""
minirocket_adaptive.py
======================

Author : Shreyas Shinde (Student ID 26104692)
Feature: Adaptive pooling MiniRocket (final model of the multi-pooling feature)

WHY THIS FILE EXISTS
--------------------
The 10-dataset experiment showed that the extra pooling feature MPV helps on
some datasets (e.g. Lightning7, ArrowHead, Wine) but hurts on others (e.g.
Beef, FaceFour). A single fixed choice therefore cannot be best everywhere.

So this model lets EACH DATASET decide, using its TRAINING data only:

    candidate 1: "PPV"      -> exactly the original MiniRocket features (~10k)
    candidate 2: "PPV+MPV"  -> the original features PLUS their MPV values (~20k)

How the decision is made:
    1. Fit MiniRocket once and compute PPV and MPV in ONE pass (minirocket_multipool,
       pools=PPV_MPV). The PPV part is bit-for-bit the original MiniRocket output.
    2. Split the TRAINING set into k folds (stratified = every class in every fold).
       For each candidate, train on k-1 folds and measure accuracy on the remaining
       fold; average over the k folds ("cross-validation accuracy").
    3. Pick the candidate with the higher cross-validation accuracy.
       If they are equal, keep "PPV" (the original): we only add MPV when the
       training data says it helps.
    4. Train the final classifier on the whole training set with the chosen features.

Important: the TEST set is never used for the decision, so the reported test
accuracy is an honest estimate.

SPEED (version 2)
-----------------
* Prediction fast path: when the model keeps the original features ("PPV"), new
  series are transformed with the fast joint-PPV transform (it gives exactly the
  same numbers), so MPV is never computed at prediction time. Only when "PPV+MPV"
  was chosen does prediction use the (also faster) PPV+MPV pooling.
* Tried and NOT adopted (no accuracy gain on the 10 development datasets):
  leave-one-out selection instead of 5-fold, and always averaging both models.

The classifier is exactly the one in the benchmark (benchmarks/common.py,
make_classifier): StandardScaler + RidgeClassifierCV.
"""

import time

import numpy as np
from sklearn.model_selection import StratifiedKFold

import minirocket_joint_ppv_simd as joint             # fast PPV-only transform (bit-identical)
import minirocket_multipool as mp
from minirocket_multipool import ridge_classifier      # = benchmarks.common.make_classifier(n, 'ridge', seed)


# The two feature sets the model chooses between, simplest (= original) first.
CANDIDATES = [
    ("PPV", ["raw_PPV"]),                 # original MiniRocket features
    ("PPV+MPV", ["raw_PPV", "raw_MPV"]),  # original features + mean of positive values
]


class AdaptivePoolingMiniRocket:
    """
    Usage (like a scikit-learn classifier):

        model = AdaptivePoolingMiniRocket(random_state=0)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print(model.chosen_feature_set_, model.cv_scores_)
    """

    def __init__(self, num_features=10_000, cv_folds=5, random_state=0):
        self.num_features = num_features      # original MiniRocket features (PPV block size)
        self.cv_folds = cv_folds              # at most this many folds for the decision
        self.random_state = random_state      # seed for MiniRocket fitting and fold splitting

    # ---------------------------------------------------------------
    # helpers
    # ---------------------------------------------------------------
    def _cross_validation_accuracy(self, F, y, folds):
        """Mean accuracy over stratified folds of the TRAINING features F."""
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.random_state)
        scores = []
        for train_index, valid_index in splitter.split(F, y):
            classifier = ridge_classifier(len(train_index), self.random_state)
            classifier.fit(F[train_index], y[train_index])
            scores.append(np.mean(classifier.predict(F[valid_index]) == y[valid_index]))
        return float(np.mean(scores))

    # ---------------------------------------------------------------
    # training
    # ---------------------------------------------------------------
    def fit(self, X, y):
        y = np.asarray(y)
        clock = time.perf_counter

        # Step 1: fit MiniRocket (raw series only) and compute PPV + MPV in one pass.
        t = clock()
        self.parameters_ = mp.fit(X, num_features=self.num_features, use_difference=False,
                                  pools=mp.PPV_MPV, random_state=self.random_state)
        self.fit_time_ = clock() - t

        t = clock()
        F = mp.transform(X, self.parameters_)
        self.transform_time_ = clock() - t

        # Step 2 + 3: choose the feature set by cross-validation on the training data.
        t = clock()
        # Each fold needs at least one example of every class, so the number of
        # folds cannot be larger than the size of the smallest class.
        smallest_class = int(np.min(np.unique(y, return_counts=True)[1]))
        folds = min(self.cv_folds, smallest_class)

        self.cv_scores_ = {}
        if folds >= 2:
            for name, blocks in CANDIDATES:
                features = mp.select_blocks(F, self.parameters_, blocks)
                self.cv_scores_[name] = self._cross_validation_accuracy(features, y, folds)
            best = max(self.cv_scores_.values())
            # first candidate with the best score -> on a tie the original "PPV" wins
            self.chosen_feature_set_ = next(name for name, _ in CANDIDATES if self.cv_scores_[name] == best)
        else:
            # Too few examples per class to validate anything: keep the original features.
            self.chosen_feature_set_ = "PPV"
        self.selection_time_ = clock() - t
        self.chosen_blocks_ = dict(CANDIDATES)[self.chosen_feature_set_]

        # Step 4: final classifier on the whole training set with the chosen features.
        t = clock()
        self.classifier_ = ridge_classifier(len(X), self.random_state)
        self.classifier_.fit(mp.select_blocks(F, self.parameters_, self.chosen_blocks_), y)
        self.classifier_time_ = clock() - t
        return self

    # ---------------------------------------------------------------
    # prediction
    # ---------------------------------------------------------------
    def transform(self, X):
        """Features of X with the chosen feature set."""
        t = time.perf_counter()
        if self.chosen_feature_set_ == "PPV":
            # Fast path: original features only -> fast joint-PPV transform, no MPV work.
            # (Same dilations and biases, so exactly the same numbers as the PPV block.)
            F = joint.transform(X, self.parameters_["raw"])
        else:
            # PPV + MPV in one pass with the fast _pool_two pooling.
            F = mp.select_blocks(mp.transform(X, self.parameters_), self.parameters_, self.chosen_blocks_)
        self.last_transform_time_ = time.perf_counter() - t
        return F

    def predict(self, X):
        return self.classifier_.predict(self.transform(X))

    def score(self, X, y):
        """Accuracy on (X, y)."""
        return float(np.mean(self.predict(X) == np.asarray(y)))
