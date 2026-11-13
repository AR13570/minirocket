"""
transform_service.py - logic for the "Fast transform" and "Reproduction" pages (no web code)

Author : Shreyas Shinde (Student ID 26104692)

Fast transform (joint PPV): the original MiniRocket transform counts positive values once per
threshold; the joint-PPV transform (code/minirocket_joint_ppv_simd.py) counts all thresholds of a
kernel in one pass and gives bit-identical features. This file times both on the same data and
checks that the outputs are identical. It only CALLS the existing code in code/ (never changes it).

It also reads two published result files (read-only):
  results/joint_ppv_2026-10-26/summary.csv            113 datasets x 3 seeds, original vs joint PPV
  results/reference/paper_vs_ours.csv                 our accuracy vs. the paper, 109 datasets
"""
import csv
from pathlib import Path
from time import perf_counter

import service                                   # sets up paths (code/ on sys.path) and caches
import numpy as np

import minirocket as original                    # code/minirocket.py  (the paper's implementation)
import minirocket_joint_ppv_simd as joint        # code/minirocket_joint_ppv_simd.py (fast transform)

JOINT_RESULTS = service.RESULTS_FOLDER / "joint_ppv_2026-10-26" / "summary.csv"
REPRODUCTION_RESULTS = service.RESULTS_FOLDER / "reference" / "paper_vs_ours.csv"


def _median_seconds(function, repetitions):
    times, value = [], None
    for _ in range(repetitions):
        start = perf_counter()
        value = function()
        times.append(perf_counter() - start)
    return value, float(np.median(times))


def compare_transforms(X_train, X_test, num_features=10_000, seed=0, repetitions=5):
    """
    Fit MiniRocket once, then transform the TEST set with the original and the joint-PPV code.
    Returns a dict: features, original_seconds, joint_seconds, ratio (joint / original),
    identical (bit-for-bit equal features), n_test, length.
    """
    X_train = np.ascontiguousarray(X_train, dtype=np.float32)
    X_test = np.ascontiguousarray(X_test, dtype=np.float32)
    parameters = joint.fit(X_train, num_features=num_features, random_state=seed)
    original.transform(X_test[:2], parameters)                  # compile / warm up outside the timers
    joint.transform(X_test[:2], parameters)
    features_original, seconds_original = _median_seconds(lambda: original.transform(X_test, parameters), repetitions)
    features_joint, seconds_joint = _median_seconds(lambda: joint.transform(X_test, parameters), repetitions)
    identical = bool(np.array_equal(np.ascontiguousarray(features_original).view(np.uint32),
                                    np.ascontiguousarray(features_joint).view(np.uint32)))
    return {"features": int(features_joint.shape[1]), "original_seconds": seconds_original,
            "joint_seconds": seconds_joint,
            "ratio": seconds_joint / seconds_original if seconds_original > 0 else float("nan"),
            "identical": identical, "n_test": int(len(X_test)), "length": int(X_test.shape[1])}


def published_joint_ppv():
    """Rows (one per feature count) of the 113-dataset benchmark, or [] if the file is missing."""
    if not JOINT_RESULTS.exists():
        return []
    rows = []
    with JOINT_RESULTS.open(newline="") as handle:
        for r in csv.DictReader(handle):
            rows.append({
                "features": int(r["features"]), "datasets": int(r["datasets"]),
                "accuracy_percent": float(r["joint_accuracy_percent"]),
                "identical": r["exact_equal"] == "True",
                "train_reduction_percent": float(r["train_time_reduction_percent"]),
                "test_reduction_percent": float(r["test_time_reduction_percent"]),
                "pipeline_reduction_percent": 100 * (1 - float(r["pipeline_ratio"])),
                "memory_change_percent": 100 * (float(r["memory_ratio"]) - 1),
            })
    return sorted(rows, key=lambda r: r["features"])


def published_reproduction():
    """
    Our MiniRocket accuracy vs. the paper's published accuracy (same train/test split, 10,000 features).
    Returns (summary dict, per-dataset rows) or (None, []) if the file is missing.
    """
    if not REPRODUCTION_RESULTS.exists():
        return None, []
    with REPRODUCTION_RESULTS.open(newline="") as handle:
        rows = [{"dataset": r["dataset"], "paper": float(r["paper_resample0"]), "ours": float(r["ours"]),
                 "paper_30_mean": float(r["paper_30_resample_mean"]),
                 "difference_pp": float(r["diff_pp"]),
                 "difference_in_stds": float(r["diff_in_paper_stds"]) if r["diff_in_paper_stds"] else 0.0}
                for r in csv.DictReader(handle)]
    ours = np.array([r["ours"] for r in rows])
    paper = np.array([r["paper"] for r in rows])
    difference = 100 * (ours - paper)
    summary = {"datasets": len(rows), "ours_percent": 100 * ours.mean(), "paper_percent": 100 * paper.mean(),
               "paper_30_resamples_percent": 100 * np.mean([r["paper_30_mean"] for r in rows]),
               "identical": int(np.sum(np.abs(difference) < 1e-6)),
               "within_1pp": int(np.sum(np.abs(difference) <= 1.0)),
               "within_2_stds": int(np.sum(np.abs([r["difference_in_stds"] for r in rows]) <= 2))}
    try:
        from scipy.stats import wilcoxon
        summary["wilcoxon_p"] = float(wilcoxon(ours, paper).pvalue)
    except Exception:                       # scipy missing or all differences zero
        summary["wilcoxon_p"] = None
    return summary, rows
