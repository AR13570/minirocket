Name: Shreyas Shinde
Student ID: 26104692
This is my part of the code. (Adaptive Multi-Pooling, Explainability, Early Classification and the MiniRocket Explorer app - three modules built on top of MiniRocket, the joint-PPV transform and group pruning: a more accurate adaptive model, heat-map explanations of every prediction, classification before a time series has finished, and one web app + REST API that shows every feature of the project)

# MiniRocket Extensions: Adaptive Multi-Pooling, Explainability and Early Classification

This README covers my part of the group project. It adds three modules to MiniRocket and a
web app, **without changing any existing file**: the modules sit in `code/`, `experiments/`,
`tests/` and `results/` next to the others, and the app lives in its own folder, `ui/`.

| # | Module | One-line summary | Main file |
|---|---|---|---|
| 1 | **Adaptive Multi-Pooling** | Adds a new feature (MPV, "how strongly a pattern appears") and lets each dataset decide, from its training data, whether to use it; slightly more accurate, never meaningfully worse | `code/minirocket_adaptive.py` |
| 2 | **Explainability** | Shows *where* in a series the model found its evidence (heat maps), which patterns matter, and what separates the classes | `code/minirocket_explain.py` |
| 3 | **Early Classification** | Classifies a series *before it ends*, stopping as soon as it is confident enough | `code/minirocket_early.py` |
| 4 | **MiniRocket Explorer (UI + API)** | One web app with every feature of the project (fast transform, group pruning, stratified thresholds, the three modules above, reproduction results), a REST API and Docker packaging | `ui/` ([ui/README.md](./ui/README.md)) |

---

## Contents

1. [Read this first: which part do you need?](#1-read-this-first-which-part-do-you-need)
2. [Quick start](#2-quick-start)
3. [Background: MiniRocket in plain words](#3-background-minirocket-in-plain-words)
4. [Setup](#4-setup)
5. [Folder structure](#5-folder-structure)
6. [Module 1: Adaptive Multi-Pooling](#6-module-1-adaptive-multi-pooling)
7. [Module 2: Explainability](#7-module-2-explainability)
8. [Module 3: Early Classification](#8-module-3-early-classification)
9. [Using your own data](#9-using-your-own-data)
10. [All options (command line and Python)](#10-all-options-command-line-and-python)
11. [Results summary](#11-results-summary)
12. [How the work was evaluated (and why it is honest)](#12-how-the-work-was-evaluated-and-why-it-is-honest)
13. [Tests](#13-tests)
14. [Relationship to the existing code](#14-relationship-to-the-existing-code)
15. [Troubleshooting / FAQ](#15-troubleshooting--faq)
16. [Glossary](#16-glossary)
17. [Limitations and possible next steps](#17-limitations-and-possible-next-steps)
18. [References](#18-references)

---

## 1. Read this first: which part do you need?

| You are... | Read |
|---|---|
| A teammate who wants to **run** it | Sections 2, 4 and 15 |
| Someone who wants to **understand** the ideas | Sections 3, 6.1, 7.1, 8.1 |
| An examiner / reviewer checking the **results** | Sections 11 and 12 |
| A developer who wants to **use or extend** the code | Sections 9, 10 and 14 |
| Presenting the project | Sections 7 (the plots) and 8 (the stream plot) |

**The 30-second version:**

* **Module 1** made MiniRocket slightly more accurate: on 20 datasets it had never seen,
  89.97% vs 89.73%; better on 5, equal on 14, worse on 1 (by 0.15). Its prediction step is fast.
* **Module 2** turns every prediction into a picture: red where the series supports the
  prediction, blue where it argues against it.
* **Module 3** answers early: on 5 datasets it decided after seeing **37%** of each series
  on average (GunPoint: same 99.33% accuracy using only 30% of the series).

---

## 2. Quick start

Run these from the **repository root**, in the environment described in
[benchmarks/README.md](./benchmarks/README.md#setup-and-dataset-download) (with that guide's
virtual environment, write `.venv/bin/python` instead of `python`). `code/` is put on the import
path automatically; on Windows also add `compat_windows` to `PYTHONPATH`
(see [README_GROUP_PRUNING.md](./README_GROUP_PRUNING.md)).

```bash
# 1. check everything works (all repository tests; the first run compiles code, so allow a minute)
python -B -m unittest discover -s tests

# 2. Module 1: original MiniRocket vs joint PPV vs the adaptive model
python -B -m experiments.multipool --datasets GunPoint ECG200 Coffee Beef Wine Lightning7 ItalyPowerDemand ArrowHead CBF FaceFour

# 3. Module 2: explain the model on GunPoint (saves heat-map images)
python -B -m experiments.explain_demo --dataset GunPoint

# 4. Module 3: early classification on 5 datasets
python -B -m experiments.early_classification --datasets GunPoint ECG200 ItalyPowerDemand Trace SyntheticControl

# 5. the app with every feature of the project
pip install -r ui/requirements.txt
streamlit run ui/app.py
```

Without `--data`, datasets are downloaded by aeon the first time (stored in `~/aeon_data`).
With the local UCR archive (see [benchmarks/README.md](./benchmarks/README.md)) add
`--data ./data/Univariate_ts/`: the files are then read with the same loader as every other
experiment (`benchmarks/common.py`). New outputs go to `results/local/` (ignored by git); the
committed results are listed in [section 11](#11-results-summary).

---

## 3. Background: MiniRocket in plain words

**The task.** *Time series classification*: give a whole sequence of numbers one label.
For example, "this ECG is normal / abnormal" or "this hand movement is drawing a gun / pointing".

**How MiniRocket does it (Dempster, Schmidt & Webb, 2021):**

1. **84 fixed pattern detectors (kernels).** Each kernel is 9 weights, six of −1 and three of 2.
   Sliding a kernel along the series gives a new series `C` (the *convolution output*) that is
   high wherever the series looks like the kernel's pattern.
2. **Different widths (dilations).** Each kernel is also used "stretched out": with dilation `d`,
   its 9 weights are spread `d` steps apart, so it looks for wider patterns.
3. **Thresholds (biases).** For every kernel and dilation, a few thresholds are taken from the
   training data.
4. **One number per threshold: PPV.** PPV = the fraction of `C` above the threshold, i.e.
   *how often* the pattern appears. 84 kernels × dilations × thresholds ≈ **10,000 features**.
5. **A simple linear classifier** (ridge regression) is trained on these features.

It is very fast (seconds for whole datasets) and about as accurate as much slower methods.

**Already in this repository:** a faster way to compute PPV ("joint PPV", `code/minirocket_joint_ppv_simd.py`),
which reads `C` once for all thresholds instead of once per threshold with identical results, and
group pruning / class-stratified biases (`code/minirocket_pruned.py`, `code/minirocket_stratified.py`).

**My contribution:** the three modules and the app described below.

---

## 4. Setup

### 4.1 Requirements

* Python 3.9 or newer
* Libraries: `numpy`, `numba`, `scikit-learn`, `scipy`, `pandas`, `matplotlib`, `aeon`

### 4.2 Install

```bash
pip install numpy numba scikit-learn scipy pandas matplotlib aeon     # the modules
pip install -r ui/requirements.txt                                    # + the app (Streamlit, FastAPI)
```

With Anaconda, everything except `aeon` and the app's packages is already installed.

### 4.3 Check

```bash
python -c "import numpy, numba, sklearn, scipy, matplotlib, aeon; print('all good')"
```

### 4.4 Important: where to run commands from

Always run commands from the **repository root**. The experiments are modules
(`python -B -m experiments.<name>`), like the group-pruning ones, and find `code/` by themselves.

---

## 5. Folder structure

Only the files marked `<-` are added by this part; no existing file is changed.

```
<repository>/
├── README_ADAPTIVE_EXPLAIN_EARLY.md      <- this file
├── code/
│   ├── minirocket_multipool.py           <- Module 1: pooling transform (PPV, MPV, MIPV, LSPV)
│   ├── minirocket_adaptive.py            <- Module 1: the final adaptive model
│   ├── minirocket_explain.py             <- Module 2: MiniRocketExplainer
│   ├── minirocket_early.py               <- Module 3: EarlyClassifier
│   └── minirocket.py, minirocket_baseline.py, minirocket_joint_ppv_simd.py, minirocket_pruned.py, ...
├── experiments/
│   ├── multipool.py                      <- Module 1: comparison experiment (+ ablation)
│   ├── explain_demo.py                   <- Module 2: all plots + report for one dataset
│   ├── early_classification.py           <- Module 3: experiment + plots
│   └── group_pruning.py, controls.py, ...
├── tests/
│   ├── test_multipool.py                 <- 7 tests
│   ├── test_adaptive.py                  <- 8 tests
│   ├── test_explain.py                   <- 5 tests
│   ├── test_early.py                     <- 5 tests
│   └── test_joint_ppv.py, test_pruned.py, ...
├── results/
│   ├── adaptive_pooling_dev10/           <- Module 1: 10 development datasets, final model
│   ├── adaptive_pooling_ablation_dev10/  <- Module 1: pooling ablation + earlier variants
│   ├── adaptive_pooling_unseen20/        <- Module 1: 20 datasets never used during development
│   ├── early_classification_5ds/         <- Module 3: 5 datasets + plots
│   ├── explain_GunPoint/                 <- Module 2: heat maps, importance, class summaries
│   └── joint_ppv_2026-10-26/, group_pruning_full/, controls/, reference/, ...
└── ui/                                   <- the MiniRocket Explorer app + REST API + Docker (ui/README.md)
```

The older result files name the joint-PPV method `arnav_joint_10k`; the experiment now writes
`joint_10k`. Both mean the same method.

---

## 6. Module 1: Adaptive Multi-Pooling

### 6.1 The idea

PPV says **how often** a pattern appears, but not **how strongly**. Two series can have the
same PPV even if the pattern is barely there in one and very pronounced in the other.

I added up to three more numbers per kernel/dilation/threshold, computed from the same `C`:

| Feature | In words | Formula (for convolution output `C` of length `n`, threshold `b`) | If the pattern never appears |
|---|---|---|---|
| **PPV** (original) | how **often** | count of `C[j] > b`, divided by `n` | 0 |
| **MPV** | how **strongly** | average of `C[j] − b` over the positions where `C[j] > b` | 0 |
| **MIPV** | **where** | average position `j` where `C[j] > b`, divided by `n` (0 = start, 1 = end) | −1 |
| **LSPV** | how **long** | longest run of consecutive `C[j] > b`, divided by `n` | 0 |

**Worked example** (also test 1 in `test_multipool.py`): `C = [-1, 2, 3, -2, 4, 5, 6, 0]`, `b = 0.5`.
Positive positions: 1, 2, 4, 5, 6.

* PPV = 5 / 8 = **0.625**
* MPV = mean(1.5, 2.5, 3.5, 4.5, 5.5) = **3.5**
* MIPV = mean(1, 2, 4, 5, 6) / 8 = **0.45**
* LSPV = longest run (positions 4, 5, 6) = 3 / 8 = **0.375**

These operators come from the MultiRocket paper (Tan et al., 2022). What this project adds:

1. **One pass for all thresholds**: all features for all thresholds of a kernel come from reading `C`
   once (the same idea the joint-PPV transform uses for PPV).
2. **Adaptive choice per dataset**: experiments showed MPV helps on some datasets and hurts on
   others, so the final model decides **from its training data** whether to use it.

### 6.2 How the final model works (`AdaptivePoolingMiniRocket`)

1. Fit MiniRocket on the training series (the seeded fit in code/minirocket_baseline.py = the original dilations and thresholds).
2. Compute PPV and MPV in one pass. The PPV part is **bit-for-bit** the original MiniRocket output.
3. 5-fold cross-validation **on the training set**: accuracy with "PPV only" vs. "PPV + MPV".
4. Keep the better one. **On a tie, keep the original** (MPV is added only when it helps).
5. Train the final classifier (StandardScaler + RidgeClassifierCV, same as the benchmark).
6. **Prediction fast path:** if the original features were kept, new series use the fast joint-PPV
   PPV-only transform, so MPV is never computed at prediction time.

### 6.3 How the design was decided

| Step | What was tried | Mean accuracy, 10 datasets | Decision |
|---|---|---:|---|
| Baseline | Original MiniRocket (~10k PPV features) | 92.55% | — |
| Version 1 | 4 pools, same total size | 92.52% (3 big losses) | changed |
| Version 1 | 4 pools + difference series | 90.56% | removed |
| Version 1 | 4 pools + difference series, 50k features | 91.61%, 5x slower | removed |
| Ablation | add one feature to PPV | MPV +0.41, MIPV −0.76, LSPV −0.30, difference series −0.67 | keep MPV only |
| Check | wider ridge regularisation range | original got worse (92.12%) | not adopted |
| **Final** | **original features + MPV only if training CV prefers it** | **92.62%** | **adopted** |
| Version 2 | leave-one-out selection / always averaging two models | 92.45% / 92.57% | not adopted |
| Version 2 | faster pooling + prediction fast path | 92.62%, **37% faster** | adopted |

MIPV, LSPV and the difference series are still in `minirocket_multipool.py` for the ablation
study (`--methods all --ablation`), but the final model does not use them.

### 6.4 How to run

```bash
# final comparison (original vs joint PPV vs adaptive)
python -B -m experiments.multipool --datasets GunPoint ECG200 Coffee Beef Wine Lightning7 ItalyPowerDemand ArrowHead CBF FaceFour

# every version from the design history + the ablation table
python -B -m experiments.multipool --datasets GunPoint ECG200 Coffee Beef Wine Lightning7 ItalyPowerDemand ArrowHead CBF FaceFour --methods all --ablation
```

### 6.5 What you will see

A table, one row per method:

```
method               features  accuracy  rank  W/T/L vs orig  Wilcoxon p   fit s train tr s test tr s   clf s  total s
original_10k             9996    92.55%  2.05         0/10/0         n/a   0.014      0.002     0.007   0.007    0.029
joint_10k                9996    92.55%  2.05         0/10/0         n/a   0.014      0.001     0.004   0.007    0.026
adaptive_10k            11662    92.62%  1.90          2/7/1         n/a   0.014      0.004     0.005   0.059    0.082
```

| Column | Meaning |
|---|---|
| features | average number of features the classifier used |
| accuracy | mean test accuracy over datasets (each averaged over seeds) |
| rank | average rank across datasets (1 = best) |
| W/T/L vs orig | datasets where the method was better / equal / worse than the original |
| Wilcoxon p | statistical test of the differences (needs ≥ 5 datasets with a difference); p < 0.05 = significant |
| fit s, train tr s, test tr s, clf s, total s | seconds per dataset: fitting, TRAIN transform, TEST transform, classifier (for the adaptive model this includes the 5-fold choice), total |

Then the feature set the adaptive model chose for each dataset and seed. Files in
`results/local/multipool_<date_time>/` (or `--output-dir`): `measurements.csv`, `summary.csv`,
`metadata.json` (+ `ablation_*.csv` with `--ablation`).

### 6.6 Use it in Python

```python
from minirocket_adaptive import AdaptivePoolingMiniRocket
model = AdaptivePoolingMiniRocket(random_state=0).fit(X_train, y_train)
print(model.chosen_feature_set_, model.cv_scores_)   # e.g. "PPV+MPV", {"PPV": 0.75, "PPV+MPV": 0.79}
print(model.score(X_test, y_test))                     # accuracy
predictions = model.predict(X_test)
```

---

## 7. Module 2: Explainability

### 7.1 The idea

MiniRocket gives an answer ("class 1") but no reason. This module answers three questions:

| Question | Function | Output |
|---|---|---|
| Where in **this** series did the model find its evidence? | `explain(x)`, `plot_explanation(x)` | heat map over the series |
| Which parts of MiniRocket matter most overall? | `global_importance()`, `plot_global_importance()` | importance per dilation, per kernel, per pooling feature |
| What separates the classes? | `class_summaries(X, y)`, `plot_class_summaries()` | average heat map per class, strongest window, text summary |

### 7.2 How the heat map is computed (exact, not an approximation)

The classifier is linear, so for class `c` its score is a weighted sum of the features:

```
score_c(x) = sum over features f of  w_cf × (F_f(x) − mean_f) / scale_f  +  b_c
```

Each MiniRocket feature is itself a **sum over positions** of the convolution output:

* PPV = sum over positions j of `[C_j > threshold] / len(C)`
* MPV = sum over positions j of `[C_j > threshold] × (C_j − threshold) / count`

So the score splits **exactly** into one piece per position. Each position belongs to a
kernel that covers 9 time steps (spaced `dilation` apart), and the piece is shared equally
between those 9 time steps. Finally, the classifier compares each feature with its training
average, so we subtract the **average map of the training series** (the "background", the same
idea as SHAP with an average baseline).

**Guarantee (checked by the tests):** adding up the heat map over all time steps gives exactly
the model's score minus its intercept. The picture is the model's own calculation, not an
approximation.

**Reading the colours:** red = this part of the series pushes towards the predicted class,
blue = pushes away from it, white = no effect. Darker = stronger.

### 7.3 How to run

```bash
python -B -m experiments.explain_demo --dataset GunPoint
python -B -m experiments.explain_demo --dataset ECG200 --examples 4
```

### 7.4 What you will see

Terminal (GunPoint):

```
GunPoint: train (50, 150), test (150, 150)
Final model: accuracy 99.33%, feature set PPV
  saved example_1.png (test series 0, true class 1)
  ...
Class summaries:
  Class 1: strongest evidence at 12%-22% of the series (t = 18-32). There the series is on average
  LOWER than in other classes (-0.76 vs -0.49) and rising (slope 0.016 vs 0.076).
  Class 2: strongest evidence at 12%-22% of the series (t = 18-32). ... HIGHER ...
```

Files in `results/explain_<dataset>/`:

| File | Contents |
|---|---|
| `example_1.png` … | one test series (black line) over a red/blue heat strip, title with predicted and true class |
| `class_summaries.png` | one panel per class: average series over its average heat map, green box = strongest evidence |
| `global_importance.png` | left: importance per dilation; right: the 10 most important kernels |
| `report.md` | accuracy, chosen features, class summaries, top dilations and kernels in text |

To view images: `open results/local/explain_GunPoint/*.png` (Mac viewer),
`code <file>.png` (VS Code tab), or click them in the VS Code Explorer.

**GunPoint interpretation:** the classes are separated at 12–22% of the series, the moment the hand
leaves the hip: in "Gun" series (class 1) the hand first reaches for the holster (lower and flatter),
in "Point" series (class 2) it is already rising. The model mostly uses short patterns
(dilation 1 = 23% of importance, dilations 1–4 = 60%).

### 7.5 Use it in Python

```python
from minirocket_adaptive import AdaptivePoolingMiniRocket
from minirocket_explain import MiniRocketExplainer

model = AdaptivePoolingMiniRocket().fit(X_train, y_train)
explainer = MiniRocketExplainer(model, X_train)          # training series = background

result = explainer.explain(X_test[0])                     # dict
result["heat"]              # one number per time step
result["predicted_class"]   # the model's prediction
explainer.explain(X_test[0], target_class="2")            # explain a different class

explainer.plot_explanation(X_test[0], "where_it_looked.png", true_class=y_test[0])
info = explainer.global_importance()                      # by_dilation, by_kernel, by_pool, top_kernels
summaries = explainer.class_summaries(X_train, y_train)  # list of dicts with "text", "window", ...
explainer.plot_class_summaries(summaries, "classes.png")
```

### 7.6 Good to know

* With **two classes**, evidence for class A is exactly the negative of evidence for class B,
  so both class panels look the same; with three or more classes they differ.
* Sharing each piece over the kernel's 9 time steps makes the map smooth at the kernel's width.
* Speed: explaining one series takes about 0.1–0.3 s (NumPy); the background is computed once.

---

## 8. Module 3: Early Classification

### 8.1 The idea

Normal classification waits for the whole series. In real-time uses (machine faults, heart
monitoring, gestures) an earlier answer is more valuable, **but only if it can be trusted**.

1. **Prefix models:** one MiniRocket classifier per prefix length: the first 10%, 20%, …, 100%
   of each training series (prefixes are at least 9 points, the kernel length).
2. **Confidence:** `margin = best class score − second best class score` (two classes: `|score|`).
3. **Thresholds from training data only:** for each prefix, 5-fold cross-validation on the training
   set shows how accurate predictions are at each confidence level. The threshold is the **lowest
   margin at which predictions are still as accurate as the target**. The default target is the
   cross-validated accuracy of the full-length model ("only stop early when it is as reliable as
   waiting until the end"). If a prefix never reaches the target, it never stops there.
4. **Prediction:** the series arrives step by step; the model answers at the **first** prefix whose
   confidence passes its threshold, or at full length if none does.

**Measures:**

* **accuracy**: fraction correct
* **earliness**: average fraction of each series used before deciding (lower = earlier)
* **harmonic mean (HM)** = `2 × accuracy × (1 − earliness) / (accuracy + 1 − earliness)`, high only
  when the method is both accurate *and* early (the standard single score in this research area)

### 8.2 How to run

```bash
python -B -m experiments.early_classification --datasets GunPoint ECG200 ItalyPowerDemand Trace SyntheticControl
python -B -m experiments.early_classification --target-accuracy 0.95     # more cautious
python -B -m experiments.early_classification --prefix-fractions 0.25 0.5 0.75 1.0
```

### 8.3 What you will see

```
dataset                 full acc  early acc  series used      HM  fit full s  fit early s
GunPoint                  99.33%     99.33%        30.1%   0.821   ...
ECG200                    91.00%     89.00%        18.1%   0.853   ...
ItalyPowerDemand          96.31%     91.55%        51.7%   0.632   ...
Trace                    100.00%     96.00%        53.5%   0.626   ...
SyntheticControl          98.67%     93.33%        29.5%   0.803   ...
MEAN                      97.06%     93.84%        36.6%   0.747   ...
```

Files in `results/local/early_<date_time>/` (committed run: `results/early_classification_5ds/`):

| File | Contents |
|---|---|
| `stream_<dataset>.png` | top: the series, black up to the decision point, grey after, green line "decision: … at 30%"; bottom: confidence (blue) vs threshold (red) |
| `accuracy_by_prefix_<dataset>.png` | left: accuracy vs % of series seen (GunPoint: ~61% at 10%, 84% at 20%, ~99% from 30%); right: when the test series stopped |
| `summary.csv`, `measurements.csv` | the numbers |

### 8.4 Use it in Python

```python
from minirocket_early import EarlyClassifier

early = EarlyClassifier().fit(X_train, y_train)
predictions, points_used = early.predict(X_test)
print(early.evaluate(X_test, y_test))     # {"accuracy", "earliness", "harmonic_mean", "mean_points_used"}

for step in early.stream(X_test[0]):      # real-time simulation of one series
    print(step)  # points_seen, fraction_seen, prediction, margin, threshold, stop
early.plot_stream(X_test[0], "stream.png", true_class=y_test[0])

early.lengths_       # prefix lengths used
early.thresholds_    # learned confidence threshold per prefix (inf = never stops there)
early.cv_accuracy_   # cross-validated accuracy per prefix
```

### 8.5 Good to know

* Training fits (number of prefixes) × (1 + folds) models: 60 by default. Each is fast
  (joint-PPV transform), but early training takes clearly longer than one normal fit.
* On small training sets the cross-validated estimate is a little optimistic, so test accuracy
  can end slightly below the target. Raise `target_accuracy` for safer (later) decisions.
* The prefix models use plain MiniRocket (PPV) for speed; the adaptive model could be plugged in.

---

## 9. Using your own data

All modules need:

* `X`: a 2-D NumPy array of shape **(number of series, series length)**, `float32`
  (all series the same length, no missing values)
* `y`: a 1-D array of labels (numbers or strings), one per series

**From a CSV file** where each row is one series and the first column is the label:

```python
import sys, numpy as np
sys.path.insert(0, "code")                       # run from the repository root
from minirocket_adaptive import AdaptivePoolingMiniRocket

data = np.loadtxt("my_train.csv", delimiter=",")
y_train = data[:, 0]
X_train = np.ascontiguousarray(data[:, 1:], dtype=np.float32)

model = AdaptivePoolingMiniRocket().fit(X_train, y_train)
```

**From the local UCR archive** (downloaded as described in `benchmarks/README.md`; read with the benchmark loader):

```bash
python -B -m experiments.multipool --data ./data/Univariate_ts/ --seeds 0
```

**Any UCR dataset by name** (downloaded automatically): just pass `--datasets NameOfDataset`.
Univariate, equal-length datasets work; others are skipped with a message.

---

## 10. All options (command line and Python)

### 10.1 `experiments/multipool.py` (Module 1)

| Option | Default | Meaning |
|---|---|---|
| `--data PATH` | (aeon download) | local UCR archive, read with `benchmarks/common.py` |
| `--datasets A B ...` | every dataset in `--data`; without `--data`: GunPoint ItalyPowerDemand ArrowHead | UCR dataset names |
| `--seeds 0 1 2` | 0 1 2 | random seeds (results are averaged) |
| `--methods ...` | original_10k joint_10k adaptive_10k | also: multipool_10k, multipool_diff_10k, multipool_diff_50k, `all` |
| `--threads N` | all cores | numba threads (use 1 to compare with the single-thread benchmark) |
| `--ablation` | off | also measure each pooling feature's contribution |
| `--ablation-features N` | 2500 | size of the ablation model |
| `--output-dir PATH` | results/local/multipool_<time> | where to save |

### 10.2 `experiments/explain_demo.py` (Module 2)

| Option | Default | Meaning |
|---|---|---|
| `--dataset NAME` | GunPoint | dataset to explain |
| `--data PATH` | (aeon download) | local UCR archive |
| `--examples N` | 3 | how many test series to draw heat maps for (one per class first) |
| `--seed N` | 0 | random seed |
| `--output-dir PATH` | results/local/explain_<dataset> | where to save |

### 10.3 `experiments/early_classification.py` (Module 3)

| Option | Default | Meaning |
|---|---|---|
| `--datasets A B ...` | GunPoint ECG200 Coffee ItalyPowerDemand ArrowHead | datasets |
| `--data PATH` | (aeon download) | local UCR archive |
| `--seeds 0 ...` | 0 | random seeds |
| `--prefix-fractions ...` | 0.1 0.2 … 1.0 | prefix lengths as fractions of the series |
| `--target-accuracy X` | full-length CV accuracy | only stop early if at least this accurate (e.g. 0.95) |
| `--output-dir PATH` | results/local/early_<time> | where to save |

### 10.4 Python classes

| Class / function | Parameters (default) |
|---|---|
| `AdaptivePoolingMiniRocket` | `num_features=10_000`, `cv_folds=5`, `random_state=0` |
| `MiniRocketExplainer` | `model`, `X_background`, `max_background=200`, `random_state=0` |
| `EarlyClassifier` | `prefix_fractions=(0.1, …, 1.0)`, `target_accuracy=None`, `cv_folds=5`, `num_features=10_000`, `min_examples=None` (10% of training set, ≥ 3), `random_state=0` |
| `minirocket_multipool.fit` | `num_features=10_000`, `max_dilations_per_kernel=32`, `use_difference=False`, `pools=ALL_POOLS` or `PPV_MPV`, `random_state=0` |
| `minirocket_multipool.transform` | `X`, `parameters` → features `[PPV block, MPV block, ...]` |

---

## 11. Results summary

Committed runs (Mac, Apple ARM64, numba 0.63.1, 3 seeds): `results/adaptive_pooling_dev10/`,
`results/adaptive_pooling_ablation_dev10/`, `results/adaptive_pooling_unseen20/`,
`results/early_classification_5ds/`, `results/explain_GunPoint/`.

### 11.1 Module 1, accuracy

| Data | Original MiniRocket | Adaptive model | Better / equal / worse |
|---|---:|---:|---:|
| 10 development datasets | 92.55% | **92.62%** | 2 / 7 / 1 |
| **20 datasets never used during development** | 89.73% | **89.97%** | **5 / 14 / 1** (Wilcoxon p = 0.075) |
| All 30 | | | **7 / 21 / 2** (losses −0.76, −0.15) |

Largest gains: Herring +3.65, Lightning7 +0.91, ArrowHead +0.57, SyntheticControl +0.44.

**Fair claim:** a consistent small improvement that is almost never worse than MiniRocket
(close to, but not below, the 0.05 significance level).

### 11.2 Module 1, speed (real numba, Mac, mean per dataset)

| | Original | Joint PPV | Adaptive v1 | **Adaptive v2** |
|---|---:|---:|---:|---:|
| TRAIN transform | 0.002 s | 0.001 s | 0.014 s | **0.004 s** |
| TEST transform | 0.007 s | 0.004 s | 0.047 s | **0.005 s** |
| Total | 0.029 s | 0.026 s | 0.131 s | **0.082 s** |

Most of the remaining cost is the 5-fold check that chooses the feature set.

### 11.3 Module 2, explainability

The heat maps add up exactly to the model's scores (tested); on GunPoint they point to the
physically meaningful moment (hand leaving the hip, 12–22% of the series).

### 11.4 Module 3, early classification

On 5 datasets: decisions after **36.6%** of the series on average, accuracy 93.84% vs 97.06%
full-length (HM 0.747). GunPoint: **99.33% using 30%** (no loss at all).

---

## 12. How the work was evaluated (and why it is honest)

* **No test-set tuning.** Every choice the models make (MPV or not, early-stopping thresholds)
  uses **training data only** (cross-validation). Test data is only used to report results.
* **Unseen confirmation.** Module 1's design was fixed on 10 datasets, then checked on **20 new
  datasets**, which were downloaded only after the design was frozen.
* **Same classifier everywhere.** All comparisons use the benchmark's `make_classifier`
  (StandardScaler + RidgeClassifierCV), so differences come from the features, not the classifier.
* **Several seeds** (default 3) and the **Wilcoxon signed-rank test** for significance.
* **Failed ideas are reported** (section 6.3), not hidden.
* All committed results and timings come from runs with real numba on a Mac (Apple ARM64).
  The scripts were moved into `experiments/` without changing their logic: on the same data and
  seeds the moved code gives identical accuracies, feature counts and early-classification numbers.

---

## 13. Tests

```bash
python -B -m unittest tests.test_multipool tests.test_adaptive tests.test_explain tests.test_early -v
python -B -m unittest discover -s tests                         # together with every other test
python -B -m unittest discover -s ui -p "test_*.py"             # the app's logic (12 tests)
```

| File | Tests | Most important check |
|---|---:|---|
| `tests/test_multipool.py` | 7 | PPV block is **bit-identical** to the joint-PPV output; all 4 pools match a slow from-the-definition implementation |
| `tests/test_adaptive.py` | 8 | choosing "PPV" gives **exactly** the original MiniRocket predictions; fast path = full path; same classifier as the benchmark |
| `tests/test_explain.py` | 5 | heat map **adds up exactly** to the decision score (2 and 3 classes, PPV and PPV+MPV) |
| `tests/test_early.py` | 5 | impossible target → never stops early (= normal model); streaming = batch prediction |
| **Total** | **25** | |

---

## 14. Relationship to the existing code

```
                 existing code (unchanged, only imported)                 this part
                 ----------------------------------------                 ---------
seeded fit       code/minirocket_baseline.fit_univariate           ───►  minirocket_multipool.fit, minirocket_early
fast PPV         code/minirocket_joint_ppv_simd.transform          ───►  minirocket_adaptive (fast path), minirocket_early
classifier       benchmarks/common.make_classifier                 ───►  every experiment (code/ uses an identical
                                                                          copy, ridge_classifier, checked by a test)
data loading     benchmarks/common (_dataset_sources, load_...)    ───►  experiments/multipool.load_dataset (--data)
convolution      original lines (inside the transform)             ───►  copied into minirocket_multipool / minirocket_explain
group pruning    code/minirocket_pruned.py, minirocket_stratified  ───►  shown in the app (ui/pruning_service.py)
```

| | Joint PPV | With these modules |
|---|---|---|
| Goal | speed (same results) | accuracy + explanations + early decisions |
| Features | PPV | PPV, plus MPV when the training data says it helps (MIPV, LSPV, difference series available) |
| Accuracy (20 unseen datasets) | 89.73% | 89.97% |
| Explanations | none | heat maps, importance, class summaries |
| Early decisions | no | yes (37% of the series on average in the demo) |

---

## 15. Troubleshooting / FAQ

| Problem | Fix |
|---|---|
| `No module named 'aeon'` | `pip install aeon` (in the same environment you run Python from) |
| `No module named 'numba'` (or sklearn …) | wrong environment: activate the one you installed the packages into |
| `No module named 'benchmarks'` / `minirocket_baseline` | run from the repository root (not from inside `code/` or `experiments/`) |
| First run is slow | numba compiles the code once; later runs use the cache |
| `OMP: Info #276 ... deprecated` | harmless message from a library; ignore it |
| A dataset is "skipped" | it is multivariate, has different lengths or missing values (not supported) |
| Dataset download fails | check the internet connection (some university networks block it); datasets are cached in `~/aeon_data` |
| How do I see the images? | `open results/explain_GunPoint/*.png` (macOS), or click them in VS Code / on GitHub |
| Numbers differ slightly from this README | tiny floating-point differences between machines are possible; timings always differ |
| Out of memory on big datasets | use fewer features, e.g. `AdaptivePoolingMiniRocket(num_features=5000)` |

---

## 16. Glossary

| Term | Meaning |
|---|---|
| Kernel | a small pattern detector (9 weights) slid along the series |
| Dilation | how stretched the kernel is; dilation `d` = its 9 weights are `d` steps apart |
| Convolution output `C` | the series of "how well does the pattern match here" values |
| Bias / threshold | a cut-off value for `C`, learned from training data |
| PPV / MPV / MIPV / LSPV | proportion / mean / mean position / longest stretch of values above the threshold |
| Ridge classifier | a linear classifier with regularisation (penalty on large weights) |
| Cross-validation (k-fold) | split the training set into k parts, train on k−1, test on the remaining one, repeat |
| Seed | number that fixes the randomness, so results can be repeated |
| Wilcoxon test | statistical test of whether paired differences are systematic or chance |
| Heat map | colour per time step showing evidence for (red) or against (blue) the prediction |
| Earliness | fraction of the series used before deciding |
| Harmonic mean (HM) | single score combining accuracy and earliness |
| Ablation | removing/adding one component at a time to see what it contributes |

---

## 17. Limitations and possible next steps

**Limitations**

* Univariate, equal-length series without missing values (same scope as the joint-PPV optimisation).
* Module 1's gain is small and not yet significant at the 0.05 level; the full UCR archive
  (128 datasets) would give a clearer answer.
* The adaptive model trains about 2.8x slower than the original (still a fraction of a second).
* Early classification trains 60 models; its threshold estimate is a little optimistic on small datasets.

**Next steps**

* Run Module 1 on the full UCR archive (`python -B -m experiments.multipool --data ./data/Univariate_ts/`).
* Use the adaptive model inside early classification.
* Show more of the API in the app (pruning and fast-transform endpoints); see `ui/DEPLOYMENT.md` for the deployment plan.

---

## 18. References

* Dempster, A., Schmidt, D. F., & Webb, G. I. (2021). MiniRocket: A very fast (almost)
  deterministic transform for time series classification. *Proceedings of KDD 2021*, 248–257.
* Tan, C. W., Dempster, A., Bergmeir, C., & Webb, G. I. (2022). MultiRocket: multiple pooling
  operators and transformations for fast and effective time series classification.
  *Data Mining and Knowledge Discovery*, 36, 1623–1646.
* Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions
  (SHAP). *Advances in Neural Information Processing Systems 30*.
* Early classification measures (earliness, harmonic mean) follow common practice in the
  early time series classification literature (e.g. Schäfer & Leser, TEASER, 2020).
