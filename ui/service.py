"""
service.py - the "brain" of the UI and the REST API (no user-interface code here)

Author : Shreyas Shinde (Student ID 26104692)

WHY A SEPARATE SERVICE LAYER
----------------------------
The Streamlit app (app.py) and the REST API (api.py) both need the same things:
load a dataset, train or load a model, predict, explain, run early classification.
Putting this logic here, in plain Python, means:
  * one place to change behaviour (app and API never disagree),
  * it can be tested without any web framework (test_service.py),
  * a future frontend (e.g. React) can reuse it through the API.

This file only IMPORTS the existing modules in code/ and experiments/; it never changes
them. Everything it writes goes into ui/artifacts/ (inside this folder, ignored by git).

A "bundle" = everything the UI needs for one dataset:
    data (train/test), final model (adaptive multi-pooling), explainer, early
    classifier, plus a few precomputed numbers.
"""

# ---------------------------------------------------------------------------
# 1. Paths and imports of the existing modules (read-only use)
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

UI_FOLDER = Path(__file__).resolve().parent            # <repository>/ui
PROJECT_ROOT = UI_FOLDER.parent                         # <repository>
RESULTS_FOLDER = PROJECT_ROOT / "results"
ARTIFACTS_FOLDER = UI_FOLDER / "artifacts"              # saved models (created when needed)
# Same import layout as the experiments (PYTHONPATH=code, run from the root), set up here so
# the app also starts with a plain "streamlit run ui/app.py".
for folder in (PROJECT_ROOT, PROJECT_ROOT / "code"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import minirocket_multipool as mp
from experiments.multipool import load_dataset
from minirocket_adaptive import AdaptivePoolingMiniRocket
from minirocket_early import EarlyClassifier, prediction_and_margin
from minirocket_explain import MiniRocketExplainer

import csv
import io
import json
import platform
import time
from dataclasses import dataclass, field
from datetime import datetime

import joblib
import numpy as np


# ---------------------------------------------------------------------------
# 2. Datasets offered in the UI (all univariate, equal length, small = fast demo)
# ---------------------------------------------------------------------------
DEMO_DATASETS = {
    "GunPoint": "Hand movement (x-coordinate): drawing a gun vs. pointing a finger",
    "ECG200": "Heartbeats: normal vs. myocardial infarction",
    "Coffee": "Spectrographs of coffee beans: Arabica vs. Robusta",
    "ItalyPowerDemand": "24 hours of electricity demand: winter vs. summer months",
    "ArrowHead": "Outlines of arrowheads (3 types)",
    "Trace": "Simulated nuclear plant sensor transients (4 classes)",
    "SyntheticControl": "Control-chart patterns (6 classes: normal, cyclic, trends, shifts)",
    "CBF": "Synthetic shapes: cylinder, bell, funnel",
    "Wine": "Spectrographs of wines (2 classes)",
}

ARTIFACT_VERSION = 2   # bump when the bundle format changes (old artifacts are then rebuilt)


@dataclass
class Bundle:
    """Everything the UI shows for one dataset."""
    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    model: AdaptivePoolingMiniRocket
    explainer: MiniRocketExplainer
    early: EarlyClassifier
    metadata: dict = field(default_factory=dict)
    class_summaries: list = field(default_factory=list)   # precomputed (slow to compute)

    @property
    def series_length(self):
        return int(self.X_train.shape[1])

    @property
    def classes(self):
        return [str(c) for c in self.model.classifier_.classes_]


# ---------------------------------------------------------------------------
# 3. Building, saving and loading bundles
# ---------------------------------------------------------------------------
def build_bundle_from_arrays(name, X_train, y_train, X_test, y_test, seed=0,
                             num_features=10_000, max_background=60,
                             prefix_fractions=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)):
    """Train all three modules on the given data and measure them on the test set."""
    X_train = np.ascontiguousarray(X_train, dtype=np.float32)
    X_test = np.ascontiguousarray(X_test, dtype=np.float32)
    y_train, y_test = np.asarray(y_train), np.asarray(y_test)
    clock = time.perf_counter

    t = clock()
    model = AdaptivePoolingMiniRocket(num_features=num_features, random_state=seed).fit(X_train, y_train)
    model_seconds = clock() - t
    accuracy = model.score(X_test, y_test)

    t = clock()
    explainer = MiniRocketExplainer(model, X_train, max_background=max_background, random_state=seed)
    # class summaries explain every training series once -> do it here, not on every page view
    summaries = explainer.class_summaries(X_train, y_train)
    explainer_seconds = clock() - t

    t = clock()
    early = EarlyClassifier(prefix_fractions=prefix_fractions, num_features=num_features,
                            random_state=seed).fit(X_train, y_train)
    early_seconds = clock() - t
    early_metrics = early.evaluate(X_test, y_test)

    metadata = {
        "dataset": name,
        "description": DEMO_DATASETS.get(name, ""),
        "artifact_version": ARTIFACT_VERSION,
        "seed": seed,
        "num_features": num_features,
        "train_examples": int(len(X_train)),
        "test_examples": int(len(X_test)),
        "series_length": int(X_train.shape[1]),
        "classes": [str(c) for c in model.classifier_.classes_],
        "test_accuracy": float(accuracy),
        "chosen_feature_set": model.chosen_feature_set_,
        "cv_scores": {k: float(v) for k, v in model.cv_scores_.items()},
        "early_metrics": early_metrics,
        "seconds": {"model": model_seconds, "explainer": explainer_seconds, "early": early_seconds},
        "created": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    return Bundle(name, X_train, y_train, X_test, y_test, model, explainer, early, metadata, summaries)


def build_bundle(name, data_dir=None, seed=0, **options):
    """Load a UCR dataset (downloaded by aeon if needed) and build its bundle."""
    X_train, y_train, X_test, y_test = load_dataset(name, data_dir)
    return build_bundle_from_arrays(name, X_train, y_train, X_test, y_test, seed=seed, **options)


def artifact_path(name, seed=0, folder=None):
    return Path(folder or ARTIFACTS_FOLDER) / f"{name}_seed{seed}.joblib"


def save_bundle(bundle, folder=None):
    """Save a bundle (model artifact) + a readable metadata JSON next to it."""
    path = artifact_path(bundle.name, bundle.metadata.get("seed", 0), folder)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    path.with_suffix(".json").write_text(json.dumps(bundle.metadata, indent=2, default=str))
    return path


def load_bundle(name, seed=0, data_dir=None, folder=None, build_if_missing=True, save_if_built=True):
    """
    Load a saved bundle; if there is none (or it is from an older format), build it
    and (optionally) save it so the next start is instant.
    """
    path = artifact_path(name, seed, folder)
    if path.exists():
        try:
            bundle = joblib.load(path)
        except Exception:               # saved by an older version of the code: rebuild it
            bundle = None
        if bundle is not None and bundle.metadata.get("artifact_version") == ARTIFACT_VERSION:
            return bundle
    if not build_if_missing:
        raise FileNotFoundError(f"no saved model for {name} (seed {seed}) in {path.parent}")
    bundle = build_bundle(name, data_dir=data_dir, seed=seed)
    if save_if_built:
        save_bundle(bundle, folder)
    return bundle


# ---------------------------------------------------------------------------
# 4. Input handling
# ---------------------------------------------------------------------------
def prepare_series(series, length):
    """
    Turn user input into a float32 series of the length the model expects.
    If the length differs, the series is resampled by linear interpolation.
    Returns (series, note) where note explains any change ('' if none).
    """
    x = np.asarray(series, dtype=np.float64).ravel()
    if x.size < 2:
        raise ValueError("a series needs at least 2 values")
    if not np.all(np.isfinite(x)):
        raise ValueError("the series contains missing or infinite values")
    note = ""
    if x.size != length:
        old_positions = np.linspace(0.0, 1.0, x.size)
        new_positions = np.linspace(0.0, 1.0, length)
        x = np.interp(new_positions, old_positions, x)
        note = f"resampled from {len(old_positions)} to {length} points to match the model"
    return np.ascontiguousarray(x, dtype=np.float32), note


def parse_series_csv(content):
    """
    Read ONE series from CSV text/bytes: either one row of numbers, or one column.
    Non-numeric cells (e.g. a header) are ignored.
    """
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="ignore")
    values = []
    for row in csv.reader(io.StringIO(content)):
        for cell in row:
            cell = cell.strip()
            try:
                values.append(float(cell))
            except ValueError:
                continue
    if len(values) < 2:
        raise ValueError("no numbers found in the file")
    return np.array(values)


# ---------------------------------------------------------------------------
# 5. What the UI / API ask for
# ---------------------------------------------------------------------------
def class_scores(bundle, x):
    """Score per class (higher = more likely), from the final model."""
    scores = bundle.model.classifier_.decision_function(bundle.model.transform(x[None, :]))[0]
    scores = np.atleast_1d(scores)
    if len(bundle.classes) == 2:
        scores = np.array([-scores[0], scores[0]])
    return dict(zip(bundle.classes, (float(s) for s in scores)))


def predict(bundle, series):
    x, note = prepare_series(series, bundle.series_length)
    predicted = str(bundle.model.predict(x[None, :])[0])
    scores = class_scores(bundle, x)
    ordered = sorted(scores.values(), reverse=True)
    return {"predicted_class": predicted, "scores": scores,
            "margin": float(ordered[0] - ordered[1]) if len(ordered) > 1 else 0.0,
            "feature_set": bundle.model.chosen_feature_set_, "note": note}


def strongest_window(heat, fraction=0.1):
    """(start, stop) of the window with the most positive evidence."""
    width = max(3, int(round(fraction * len(heat))))
    smooth = np.convolve(heat, np.ones(width) / width, mode="valid")
    start = int(np.argmax(smooth))
    return start, start + width


def explain(bundle, series, target_class=None):
    x, note = prepare_series(series, bundle.series_length)
    target = None
    if target_class is not None:
        # map a string label back to the model's own label type
        lookup = {str(c): c for c in bundle.model.classifier_.classes_}
        target = lookup[str(target_class)]
    result = bundle.explainer.explain(x, target_class=target)
    start, stop = strongest_window(result["heat"])
    L = len(x)
    return {"predicted_class": str(result["predicted_class"]),
            "explained_class": str(result["explained_class"]),
            "heat": result["heat"].tolist(),
            "decision_score": result["decision_score"],
            "strongest_window": [start, stop],
            "summary": (f"Most evidence for class {result['explained_class']} is at "
                        f"{100 * start / L:.0f}%-{100 * stop / L:.0f}% of the series (t = {start}-{stop - 1})."),
            "note": note}


def early_timeline(bundle, series):
    """
    Prediction, confidence and threshold at EVERY prefix length (for the slider in the UI),
    plus the prefix where the early classifier actually stops.
    """
    x, note = prepare_series(series, bundle.series_length)
    early = bundle.early
    steps, decision_index = [], None
    for index, (model, length, threshold) in enumerate(zip(early.models_, early.lengths_, early.thresholds_)):
        predicted, margin = prediction_and_margin(model.scores(x[None, :]), model.classes_)
        confident = bool(margin[0] >= threshold)
        if confident and decision_index is None:
            decision_index = index
        # Threshold as a JSON-friendly number:
        #   +infinity (never reliable enough at this prefix) -> None ("never stops here")
        #   -infinity (full length: always answers)          -> 0.0 (any margin passes)
        if np.isposinf(threshold):
            shown_threshold = None
        elif np.isneginf(threshold):
            shown_threshold = 0.0
        else:
            shown_threshold = float(threshold)
        steps.append({"points_seen": int(length), "fraction_seen": float(length / early.series_length_),
                      "prediction": str(predicted[0]), "margin": float(margin[0]),
                      "threshold": shown_threshold, "confident": confident})
    return {"steps": steps, "decision_index": decision_index,
            "decision": steps[decision_index], "note": note}


def latest_benchmark_results():
    """
    Read (never write) the adaptive-pooling benchmark summary: the newest local run in
    results/local/multipool_*/ if there is one, otherwise the committed confirmation run on
    20 unseen datasets (results/adaptive_pooling_unseen20/).
    Returns (folder name, list of rows) or (None, []) if neither exists.
    """
    runs = sorted((RESULTS_FOLDER / "local").glob("multipool_*/summary.csv"))
    committed = RESULTS_FOLDER / "adaptive_pooling_unseen20" / "summary.csv"
    path = runs[-1] if runs else (committed if committed.exists() else None)
    if path is None:
        return None, []
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    return path.parent.name, rows
