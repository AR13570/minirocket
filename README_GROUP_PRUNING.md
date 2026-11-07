[`ROCKET`](https://github.com/angus924/rocket) &middot; [**`MINIROCKET`**](https://github.com/angus924/minirocket) &middot; [`HYDRA`](https://github.com/angus924/hydra)

# MINIROCKET: group pruning and class-stratified biases

***An extension of the joint-PPV work: drop whole (dilation, kernel) groups, and make the thresholds fairer across classes.***

> <div align="justify">Joint PPV made counting thresholds cheap, so the cost of a MiniRocket transform now sits in the convolutions and in the classifier. This extension attacks both. <b>Group pruning</b> removes whole (dilation, kernel) groups: it skips their convolution and their thresholds, and it shrinks the classifier. Surviving groups keep every threshold, so joint PPV's per-group saving is kept. <b>Class-stratified biases</b> choose the example used to set each group's thresholds so that every class is represented, instead of one random example. The two changes are independent and can be combined.</div>

Status: **implemented, tested under real Numba, and run on the UCR univariate archive** (the primary pruning run at the full planned size, the other studies at reduced scope; see [Scope](#scope-of-what-was-run)). Energy was **not** measured (see [Energy](#energy-not-measured)).

This is the single technical document for the new work: results, analysis, every file added, every deviation, the exact commands, and the run log. The existing [README](./README.md) and [benchmark guide](./benchmarks/README.md) are unchanged. A plain-language walkthrough with analogies is in [`PROJECT_EXPLAINED.md`](./PROJECT_EXPLAINED.md).

## Contents

1. [Results](#results)
2. [Scope of what was run](#scope-of-what-was-run)
3. [What changed, file by file](#what-changed-file-by-file)
4. [Concepts and the rules that keep the numbers exact](#concepts-and-the-rules-that-keep-the-numbers-exact)
5. [Examples](#examples)
6. [How to reproduce](#how-to-reproduce)
7. [Experiment grid and accounting](#experiment-grid-and-accounting)
8. [Conflicts with the current repo, and how each is resolved](#conflicts-with-the-current-repo-and-how-each-is-resolved)
9. [Windows compatibility](#windows-compatibility)
10. [What is and is not verified](#what-is-and-is-not-verified)
11. [Energy: not measured](#energy-not-measured)
12. [Limitations](#limitations)
13. [Chronological log](#chronological-log)
14. [Prior work and what is new here](#prior-work-and-what-is-new-here)

## Results

All numbers are copied from files under `results/` (never estimated). Timing: one Windows 11 laptop (AMD Ryzen 9 8940HX, 32 logical CPUs), Python 3.14.7, numba 0.63.1, `NUMBA_NUM_THREADS=1`. Accuracy is the mean over datasets, with seeds averaged within each dataset first. Tests are Wilcoxon signed-rank over datasets (`experiments/final_analysis.py`, output in `results/final_analysis_tables.md`). Each result folder also has its own generated `report.md`.

**Reproduction check.** Original MiniRocket (9,996 features, supplied split) against the authors' published per-dataset numbers (their resample 0), on the 109 datasets both cover (`results/reference/paper_vs_ours.csv`, from [`experiments/compare_to_paper.py`](./experiments/compare_to_paper.py)).

| | Paper resample 0 | This repo (original method) |
|---|---:|---:|
| Mean accuracy, 109 datasets | 85.67% | 85.69% |

Mean paired difference +0.014 pp, median 0.000. Two datasets differ by more than 2 pp (BeetleFly +3.3, Computers +2.5) and none by more than 2 of the paper's own standard deviations. Wilcoxon p = 0.582: no systematic difference. The paper's 30-resample mean (87.24%) uses a different protocol and must not be compared with a single split. The authors' CSVs were fetched from github.com/angus924/minirocket (`results/`).

**Exactness.** Pruned features equal `full[:, kept]` bit for bit in **all 15,744 measured rows** (pilot 300, primary 7,458, drop-0.3 2,486, bias 1,800, combined 3,600, 25k 100; checked from the CSVs). The 13 new tests pass.

### Group pruning, primary run

`results/group_pruning_full/`: 113 datasets, **3 seeds**, bias strategy original, drop fraction 0.2, 10,000 features (the runbook's plan). Time = test transform with joint PPV, relative to the unpruned model with the original transform. Joint PPV alone is 0.811. "p" is a Wilcoxon test of the accuracy change against the full model over 113 datasets.

| Scorer | Groups kept | Features | Accuracy | Delta vs full (86.08%) | p | Time vs full original | Break-even series |
|---|---:|---:|---:|---:|---:|---:|---:|
| none (full), original | 100% (1,598) | 9,996 | 86.08% | 0 | | 1.000 | n/a |
| none (full), joint PPV | 100% | 9,996 | 86.08% | 0 | | 0.811 | n/a |
| filter | 75% | 7,946 | 86.07 | -0.01 pp | 0.91 | 0.657 | 233 |
| filter | 50% | 5,555 | 86.03 | -0.05 pp | 0.50 | 0.463 | 25 |
| filter | 30% | 3,538 | 85.68 | -0.40 pp | 0.048 | 0.311 | 0 |
| filter | 20% | 2,455 | 85.29 | -0.79 pp | 0.002 | 0.232 | 0 |
| filter | 10% | 1,268 | 84.65 | -1.43 pp | <0.0001 | 0.142 | 0 |
| ridge | 75% | 7,133 | 85.97 | -0.11 pp | 0.23 | 0.545 | 1,347 |
| ridge | 50% | 4,527 | 85.81 | -0.27 pp | 0.006 | 0.362 | 1,019 |
| ridge | 30% | 2,582 | 85.29 | -0.79 pp | <0.0001 | 0.233 | 1,017 |
| ridge | 20% | 1,654 | 84.78 | -1.30 pp | <0.0001 | 0.172 | 993 |
| ridge | 10% | 784 | 83.85 | -2.23 pp | <0.0001 | 0.112 | 1,068 |

A break-even of 0 means the pruned model's total training cost is already below the full model's. The full table with the same-implementation ratios, bootstrap intervals and train-cost ratios is in `results/group_pruning_full/report.md`.

Reading:
- **Time falls roughly in proportion to the groups kept**, and the joint-PPV saving carries over to the kept groups.
- **Mild pruning costs nothing detectable:** `filter` at 50% or 75% and `ridge` at 75%. `ridge` at 50% costs 0.27 pp, small but detectable (p = 0.006).
- **Heavy pruning has a real cost:** 20% of groups loses 0.8 to 1.3 pp, and 10% loses 1.4 to 2.2 pp (all p < 0.01).
- **Break-even depends on the scorer.** `ridge` selection costs about 4 to 5 times the full model's training, so it repays only after roughly 1,000 test series. `filter` is nearly free: about 25 series at 50%, and cheaper to train than the full model at 30% or fewer.
- **`filter` minus `ridge` accuracy:** +0.09, +0.23, +0.39, +0.51, +0.80 pp at 75, 50, 30, 20, 10% of groups. The sign is the same at every level, but only the 10% level is nominally significant (p = 0.038; 0.077 at 20%; the rest p > 0.15), uncorrected for five comparisons. `filter` also keeps more features at the same group count, so it is not a like-for-like comparison. "`filter` is at least as good as `ridge` here" is supported; "`filter` is better" is only suggestive.
- **Drop 0.2 vs 0.3:** ridge accuracy deltas are similar (20%: -1.30 vs -1.55 pp); the two runs differ in drop fraction, seeds and session, so this is not a controlled comparison.

### Group pruning, secondary run

`results/group_pruning_full_drop30_1seed/`: same datasets, **1 seed (0), drop fraction 0.3**, 2,486 rows. Run first as a faster substitute (about 1 h 20 min), kept as a robustness check. Baseline 86.07%; joint PPV alone 0.806 of original.

| Scorer | Groups kept | Features | Accuracy % | Delta pp | Time vs full original | Median break-even |
|---|---:|---:|---:|---:|---:|---:|
| filter | 75% | 7,952 | 86.10 | +0.03 | 0.654 | 237 |
| filter | 50% | 5,558 | 85.94 | -0.14 | 0.460 | 23 |
| filter | 30% | 3,536 | 85.51 | -0.57 | 0.306 | 0 |
| filter | 20% | 2,457 | 85.20 | -0.88 | 0.230 | 0 |
| filter | 10% | 1,271 | 84.51 | -1.57 | 0.140 | 0 |
| ridge | 75% | 7,060 | 85.92 | -0.15 | 0.530 | 656 |
| ridge | 50% | 4,508 | 85.70 | -0.38 | 0.359 | 530 |
| ridge | 30% | 2,575 | 85.14 | -0.93 | 0.230 | 594 |
| ridge | 20% | 1,654 | 84.53 | -1.55 | 0.170 | 578 |
| ridge | 10% | 776 | 83.72 | -2.35 | 0.112 | 642 |

Same conclusions as the primary run. Ridge break-even is lower here (about 530-650) because a larger drop fraction means fewer elimination rounds, so selection is cheaper.

### Stratified vs uniform biases

`results/group_pruning_bias_30ds_10seeds/`: 30 datasets, 10 seeds, full model, 1,800 rows. Accuracy difference stratified minus uniform (paired over datasets; details in `bias_analysis.md`):

| Subgroup | Datasets | Mean diff pp | Wins/ties/losses | Wilcoxon p (n non-zero) | Seed std, stratified vs uniform (pp) |
|---|---:|---:|---|---|---|
| all | 30 | -0.032 | 8/8/14 | 0.353 (22) | 0.727 vs 0.652 |
| imbalanced | 10 | -0.097 | 1/1/8 | 0.012 (9) | 0.619 vs 0.541 |
| small | 10 | -0.039 | 2/5/3 | 0.812 (5) | 0.826 vs 0.787 |
| rest | 10 | +0.041 | 5/2/3 | 0.461 (8) | 0.736 vs 0.627 |

Control, original minus uniform (RNG policy only): +0.061 pp, p = 0.290. **The average effect is null**, and stratified did not lower the seed-to-seed spread. The one nominal hit (imbalanced, p = 0.012) is in the wrong direction and unadjusted.

### Combined: stratified/uniform x pruning

`results/group_pruning_combined_30ds_5seeds/`: the same 30 datasets, 5 seeds, `ridge` only, drop 0.3, 3,600 rows. Mean over the 30 datasets:

| Groups kept | Acc % stratified | Acc % uniform | Delta vs full, stratified | Delta vs full, uniform | Time ratio (same impl) |
|---:|---:|---:|---:|---:|---:|
| 100% | 81.56 | 81.58 | 0 | 0 | 1.000 |
| 75% | 81.60 | 81.54 | +0.03 | -0.04 | 0.69 |
| 50% | 81.45 | 81.45 | -0.12 | -0.14 | 0.475 |
| 30% | 81.13 | 80.96 | -0.43 | -0.62 | 0.31 |
| 20% | 80.60 | 80.45 | -0.97 | -1.14 | 0.22 |
| 10% | 79.76 | 79.27 | -1.81 | -2.32 | 0.14 |

Stratified minus uniform over all 30 datasets: -0.02 pp (full), +0.06 (75%), 0.00 (50%), +0.17 (30%), +0.15 (20%), +0.49 (10%); Wilcoxon p = 0.77, 0.31, 0.98, 0.44, 0.41, 0.046. Seed std (stratified / uniform) at 10%: 1.36 / 1.59 pp, not significant.

**Multiplicity check** (`final_analysis.py`; 24 testable slices = 6 keep levels x 4 subgroups; Holm over all 24). Rows shown are the all-dataset slices and anything with raw p < 0.1:

| Keep | Subgroup | Datasets | Mean diff pp | Wins/ties/losses | Raw p | Holm p |
|---:|---|---:|---:|---|---:|---:|
| 1.0 | all | 30 | -0.019 | 11/8/11 | 0.770 | 1.000 |
| 0.75 | all | 30 | +0.057 | 14/6/10 | 0.310 | 1.000 |
| 0.5 | all | 30 | -0.001 | 9/8/13 | 0.975 | 1.000 |
| 0.3 | all | 30 | +0.171 | 14/3/13 | 0.441 | 1.000 |
| 0.2 | all | 30 | +0.149 | 15/6/9 | 0.406 | 1.000 |
| 0.1 | all | 30 | +0.491 | 21/3/6 | 0.046 | 1.000 |
| 0.1 | small | 10 | +1.354 | 9/1/0 | 0.004 | 0.094 |

2 of 24 raw p-values are below 0.05 (about 1.2 expected by chance under no effect); **0 of 24 survive Holm** (the slices are correlated, so Holm is conservative). The only candidate for a follow-up is "small datasets under heavy ridge pruning", which would need new data.

### 25,000 features

`results/group_pruning_25k/`: 24,948 features (1,579 groups), 10 datasets (GunPoint ECG200 Beef Computers Wafer ECG5000 FacesUCR Trace Yoga EthanolLevel, chosen for mixed size, not tuned on results), 1 seed, drop 0.3, 100 rows.

| Variant | Groups | Features | Accuracy % | Delta pp | Time vs full (same impl) | Break-even |
|---|---:|---:|---:|---:|---:|---:|
| full | 1,579 | 24,948 | 88.90 | 0 | 1.000 | n/a |
| ridge 50% | 790 | 11,559 | 89.04 | +0.14 | 0.457 | 398 |
| ridge 20% | 316 | 4,314 | 89.02 | +0.13 | 0.183 | 475 |
| filter 50% | 790 | 14,729 | 88.59 | -0.31 | 0.600 | 34 |
| filter 20% | 316 | 7,017 | 88.57 | -0.33 | 0.304 | 0 |

Descriptive only (10 datasets, 1 seed): with more features there is more redundancy, and `ridge` lost nothing. Joint PPV was not faster than original pooling in this sample (ratio 1.006, against 0.806-0.811 over 113 datasets at 10k); the dataset mix differs, so do not compare those numbers. Test-time classifier work is a visible share on larger datasets (Wafer: 0.33 s predict vs 2.2 s transform).

### Pilot

`results/group_pruning_pilot/`: 5 datasets (GunPoint ECG200 Beef Computers Wafer), 2 seeds, all three bias strategies, keep 1.0/0.5/0.2, drop 0.2; 300 rows, 0 mismatches. A smoke test, not a finding: strategy `original`, keep 50% gave -0.68 pp (`filter`) / -0.91 pp (`ridge`) at about 0.54 of the time; keep 20% gave -1.62 / -1.42 pp at about 0.22. Ridge selection took 0.1-2.3 s per case, so the full run was far from the runbook's 48-hour threshold and `--drop-fraction` stayed at 0.2.

### Recommendation

1. Use group pruning with the cheap `filter` scorer at 50% of groups: about 2.2 times faster test transform (0.463 of original) for -0.05 pp (not significant), repaid within about 25 series.
2. Use `ridge` selection only if you will classify more than roughly 1,000 series per trained model, or at very large feature counts where it lost no accuracy (25k run, indicative).
3. Expect a 1 to 2 pp loss below 20% of groups.
4. Skip stratified biases: no benefit shown.

## Scope of what was run

| Phase | Planned | Run |
|---|---|---|
| 6 pruning | all datasets, 3 seeds, drop 0.2 | **as planned** (`group_pruning_full/`; stopped once after 53 cases because it was slow, then resumed to completion with the same settings and `--resume`; about 9 h wall time in total, Crop 18-24 min and ElectricDevices about 45 min per seed). Also run: a faster 1-seed, drop-0.3 version (`group_pruning_full_drop30_1seed/`) |
| 7 stratified study | all datasets, 20 seeds | **30 datasets, 10 seeds** (rule below) |
| 8 combined | all datasets, 5 seeds, default scorer | **same 30 datasets, 5 seeds, `ridge` only, drop 0.3** (the all-dataset version would take about 10 h) |
| 9 energy | 5 datasets | **skipped**: no valid power measurement on this machine |
| 10 25k features | 10 datasets, reduced Phase 6 | 10 datasets, 1 seed, keep fractions 1.0/0.5/0.2, drop 0.3 |

15 of the 128 archive datasets are skipped by the loader: 12 with variable-length series (AllGestureWiimoteX/Y/Z, GestureMidAirD1-3, GesturePebbleZ1/Z2, MelbournePedestrian, PLAID, PickupGestureWiimoteZ, ShakeGestureWiimoteZ) and 3 with missing values (DodgerLoopDay/Game/Weekend). No dataset was excluded for memory or time; the run covers 113.

**30-dataset subset rule** (fixed before running, not hand-picked): all 113 usable datasets were classed as *imbalanced* (smallest TRAIN class share < 20%: 57 datasets; this also catches balanced datasets with more than 5 classes), *small* (< 100 TRAIN series, not imbalanced: 26) or *rest* (30); within each class, 10 evenly spaced in alphabetical order. The list is in `results/bias_study_subset.csv`; dataset sizes and class shares are in `results/dataset_stats.csv` (generated once from the archive by a throwaway script that was not kept).

Consequences: dataset-bootstrap intervals describe dataset mix, not seed noise; the first 53 cases of the primary run were timed in an earlier session than the other 286 (same machine and code); drop 0.3 changes which groups survive compared with 0.2; the 30-dataset subgroups have little power; the 25k run and drop-0.3 run have one seed.

## What changed, file by file

**No existing tracked file was modified** (`git diff --name-status 8aa4ae2 HEAD` has no `M` entries). The ten protected files are untouched. Origin: **Supplied** = written in the documents you provided, copied in unchanged; **Scratch** = written during this work; **Adapted** = supplied text extended with results. Every file is new; the supplied code builds on the existing code by importing it, not editing it.

| File | Origin | Purpose |
|---|---|---|
| [`code/minirocket_pruned.py`](./code/minirocket_pruned.py) | Supplied | Group layout and keep mask; masked transform (joint and original pooling); group scoring; iterative pruning |
| [`code/minirocket_stratified.py`](./code/minirocket_stratified.py) | Supplied | Fit with an explicit example table: `uniform` (control) or `stratified`; optional pooling of m examples per group |
| [`tests/test_pruned.py`](./tests/test_pruned.py) | Supplied | 13 tests, including a direct-convolution oracle that does not use the C-alpha / C-gamma trick |
| [`experiments/group_pruning.py`](./experiments/group_pruning.py) | Supplied | The experiment grid; own CSV and metadata |
| [`experiments/summarise.py`](./experiments/summarise.py) | Supplied | Pareto tables, dataset-bootstrap intervals, training-cost break-even |
| [`experiments/compare_to_paper.py`](./experiments/compare_to_paper.py) | Supplied | Paper-vs-ours table |
| [`experiments/sanity_keepall.py`](./experiments/sanity_keepall.py) | Scratch | Keep-all pruned transform vs the existing joint transform: identical output; ratios 1.007 (GunPoint), 1.030 (Wafer), within the 10% limit |
| [`experiments/analyse_bias.py`](./experiments/analyse_bias.py) | Scratch | Stratified-vs-uniform analysis: per-dataset mean/std, Wilcoxon, subgroups, `--by-keep-fraction` |
| [`experiments/final_analysis.py`](./experiments/final_analysis.py) | Scratch | Pruned-vs-full and `filter`-vs-`ridge` Wilcoxon tests; Holm-corrected stratification tests |
| [`experiments/energy.py`](./experiments/energy.py) | Scratch | CodeCarbon CPU energy minus idle (written and tested; no valid results here, see [Energy](#energy-not-measured)) |
| [`compat_windows/`](./compat_windows) | Scratch | Windows stand-ins for `fcntl`, `resource`, `os.sysconf` (see [Windows compatibility](#windows-compatibility)) |
| `README_GROUP_PRUNING.md` (this file), `PROJECT_EXPLAINED.md` | Adapted / Scratch | The only two documents for this work |
| `results/reference/` | Fetched + generated | Authors' published CSVs (`results_ucr109_mean.csv`, `accuracy_ucr109_resamples.csv`) and `paper_vs_ours.csv` |
| `results/group_pruning_*/`, `results/*.log` | Generated | Per-run `measurements.csv`, `summary.csv`, `report.md`, `metadata.json`, plus analysis markdown and console logs |
| `results/final_analysis_tables.md`, `results/dataset_stats.csv`, `results/bias_study_subset.csv`, `results/environment_freeze.txt` | Generated | Final tests; dataset properties and subset; `pip freeze` (179 lines, the whole user-site environment, not a clean venv) |

Folder `data/` and `results/local/` are not committed. The untracked nested clone `minirocket/` is unrelated to this work and left in place.

**Do not edit the protected files** while a benchmark run is in progress. `benchmarks/controller.py::source_hashes()` hashes them, and `--resume` refuses to continue if they change: `code/minirocket.py`, `code/minirocket_baseline.py`, `code/minirocket_joint_ppv_simd.py`, `code/_joint_ppv_simd_kernels.py`, `benchmarks/common.py`, `benchmarks/memory.py`, `benchmarks/controller.py`, `benchmarks/worker.py`, `benchmarks/inventory.py`, `tools/generate_joint_ppv.py`. The new runner only imports from `benchmarks/common.py`.

## Concepts and the rules that keep the numbers exact

A **group** is the block of consecutive features that share one (dilation `j`, kernel `k`) pair. It holds `num_features_per_dilation[j]` thresholds. The original feature order is dilation-major, kernel-minor, so group `(j, k)` is one contiguous slice. A **keep mask** is a boolean array of shape `(len(dilations), 84)`.

Cost model (why groups, not features):

| Part of the transform | Cost scales with |
|---|---|
| Convolution of one kernel at one dilation | number of **groups** kept |
| Threshold counting (joint PPV) | number of features, but cheap |
| Classifier fit and predict | number of **features** kept |

Pruning a group saves all three. Pruning one threshold saves only the cheapest.

The contract: `pruned.transform(X, p, keep)` equals `full_transform(X, p)[:, kept_columns(p, keep)]` bit for bit. Four rules guarantee it:

1. **Fit once on the full layout and keep the full `biases` array.** Biases come from `_quantiles(global feature index)`. Refitting for a smaller layout changes them.
2. **Never renumber dilation or kernel indices.** Padding alternates with `(dilation_index + kernel_index) % 2`. In the toy test, compacting the parameters after dropping dilation 0 changed 1,931 of 2,016 feature values.
3. **Rank on TRAIN features only**, and refit the classifier on the kept TRAIN columns.
4. **Keep-fractions count groups.** Groups hold unequal numbers of features (larger at small dilations), so always report both groups kept and features kept.

**Scorers.** `ridge`: iterative backward elimination. Fit a ridge classifier on the current kept columns, score each group by `||W_g||_F / sqrt(p_g)` (coefficient norm normalised by group width), drop the lowest `drop_fraction` of the remaining groups, repeat until the target is reached. `filter`: one-shot ANOVA F per feature on TRAIN, averaged per group (constant features score 0).

## Examples

Set up as in the main README (`export PYTHONPATH="$PWD/code${PYTHONPATH:+:$PYTHONPATH}"`; on Windows also add `compat_windows`), with `X_training`, `X_test` as finite 2D `np.float32` arrays and labels as 1D arrays.

```python
import numpy as np
from minirocket_stratified import fit
from minirocket_pruned import transform, keep_all, prune_groups, kept_columns
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

parameters = fit(X_training, Y_training, num_features=10_000,
                 strategy='stratified', random_state=0)             # or 'uniform' (control)
full_training = transform(X_training, parameters, keep_all(parameters))

keep, history = prune_groups(full_training, Y_training, parameters, keep_fraction=0.3)   # TRAIN only
columns = kept_columns(parameters, keep)

classifier = make_pipeline(StandardScaler(),
                           RidgeClassifierCV(alphas=X_training.shape[0] * np.logspace(-3, 3, 10)))
classifier.fit(full_training[:, columns], Y_training)

X_test_transform = transform(X_test, parameters, keep)               # computes only the kept groups
predictions = classifier.predict(X_test_transform)
```

`transform(..., implementation='original')` uses the original one-pass-per-threshold pooling with the same mask.

## How to reproduce

Install the packages (versions in `results/environment_freeze.txt`), download the UCR archive as in the benchmark guide, then:

```
export PYTHONPATH="$PWD/code:$PWD/compat_windows"     # Windows; use "$PWD/code" on Linux/macOS
python -B -m unittest discover -s tests               # 64 tests OK, 1 skipped (ARM64-only)
python tools/generate_joint_ppv.py --check            # kernels current
```

Use `NUMBA_NUM_THREADS=1`, a quiet machine, and the same machine for every timing you compare. All runs use `python -B -m experiments.group_pruning --data ./data/Univariate_ts/ --output-dir <folder> ...` and then `python -B -m experiments.summarise <folder>`. `DS30` is the 30 names in `results/bias_study_subset.csv`. Code under test did not change after commit `1dabf25` (`minirocket_pruned.py`, `minirocket_stratified.py`, `group_pruning.py`, `summarise.py`); later commits only added scripts and documents.

| Folder | Arguments after `--output-dir <folder>` |
|---|---|
| `reference/` | `python -B -m experiments.compare_to_paper` (CSVs fetched from github.com/angus924/minirocket `results/`) |
| `group_pruning_pilot/` | `--datasets GunPoint ECG200 Beef Computers Wafer --seeds 0 1 --keep-fractions 1.0 0.5 0.2 --scorers ridge filter` (all three bias strategies, drop 0.2) |
| `group_pruning_full/` | `--bias-strategies original --scorers ridge filter` (seeds 0 1 2, drop 0.2 default); the second session added `--resume` |
| `group_pruning_full_drop30_1seed/` | `--bias-strategies original --scorers ridge filter --drop-fraction 0.3 --seeds 0` |
| `group_pruning_bias_30ds_10seeds/` | `--datasets DS30 --bias-strategies original uniform stratified --keep-fractions 1.0 --seeds 0 1 2 3 4 5 6 7 8 9`, then `python -B -m experiments.analyse_bias <folder>` |
| `group_pruning_combined_30ds_5seeds/` | `--datasets DS30 --bias-strategies uniform stratified --seeds 0 1 2 3 4 --drop-fraction 0.3 --scorers ridge`, then `experiments.analyse_bias <folder> --by-keep-fraction` |
| `group_pruning_25k/` | `--datasets GunPoint ECG200 Beef Computers Wafer ECG5000 FacesUCR Trace Yoga EthanolLevel --bias-strategies original --scorers ridge filter --keep-fractions 1.0 0.5 0.2 --seeds 0 --drop-fraction 0.3 --feature-count 25000` |
| final tests | `python -B -m experiments.final_analysis` (reads the committed CSVs; no timing) |

## Experiment grid and accounting

| Factor | Values | Why |
|---|---|---|
| Bias strategy | `original`, `uniform`, `stratified` | `uniform` isolates stratification from the RNG change |
| Keep fraction (of groups) | 1.0, 0.75, 0.5, 0.3, 0.2, 0.1 | Pareto curve; fixed in advance, never tuned on TEST |
| Scorer | `ridge` (iterative backward elimination), `filter` (one-shot ANOVA F) | Embedded versus cheap ranking |
| Transform | `original` pooling, `joint` pooling | The 2x2: does pruning compound with joint PPV |
| Features | 9,996 (`--feature-count 10000`); 24,948 for the 25k run | As the published run; larger optional |

**Training-cost accounting** (`summarise.py`). Pruning needs the full model first, so it adds cost before it saves any.
`train cost = fit + full TRAIN transform + selection + classifier refit`
`inference cost per series = (TEST transform + predict) / n_test`
`break-even series = (extra train cost) / (inference saving per series)`, or never if inference is not cheaper. 0 means the pruned model's total training cost is already below the full model's (the smaller classifier fit saves more than the selection costs).

**Leakage rule.** Ranking and refitting use TRAIN only; the keep fraction is chosen from the fixed grid and reported for every value. The classifier's internal alpha search uses features and scaling fitted on the whole TRAIN split, as in the existing README's note on nested validation.

**Statistics.** Wilcoxon signed-rank over per-dataset mean differences (paired by dataset; no normality assumption); skipped with fewer than 5 non-zero differences. Dataset-bootstrap intervals in `report.md` resample datasets. Holm correction is applied only in the multiplicity check above; all other p-values are unadjusted.

## Conflicts with the current repo, and how each is resolved

| # | Conflict (where) | Why it matters | Resolution | Status |
|---|---|---|---|---|
| C1 | `benchmarks/results.py` expects exactly `('original','joint_ppv')` per seed and budget; `controller.build_variants` and `case_order` assume pairs; `worker.py` raises on any other implementation | A third or fourth variant would fail verification and break the report generator | Separate runner `experiments/group_pruning.py` with its own output; reuses only loaders and `make_classifier` | Resolved by design |
| C2 | `source_hashes()` lists ten files; `--resume` compares `source_sha256` | Editing any of them mid-run invalidates resume | No listed file is modified | Resolved |
| C3 | `minirocket.fit(X, ...)` has no labels; `_fit_biases` has a fixed Numba signature and cache | Stratification needs `y` | New `minirocket_stratified.py` takes an explicit example table; original untouched | Resolved |
| C4 | Original draws examples with Numba's RNG | Original vs stratified would mix two effects | `uniform` strategy on the identical code path as the control | Resolved; test shows the m = 1 loop equals the original given the same draws; control within noise |
| C5 | `_pool_joint(responses, biases, start, count, output)` reads biases and writes output at the same index | With pruning, output is compact and the bias layout is full | Pass slices with `start = 0`; generated kernels unchanged | Resolved; `generate_joint_ppv.py --check` passes; keep-all identical output |
| C6 | Biases depend on global feature index; padding parity depends on `(dilation_index + kernel_index) % 2` | Refitting or renumbering changes features | Fit once, mask in place; trap test included | Resolved |
| C7 | Published verification asserts identical accuracy between variants | Pruning changes accuracy by design | New contract: pruned columns equal `full[:, kept]`; flagged per row | Resolved; 0 mismatches in every run |
| C8 | Fixed-width assumptions in worker | Wrong for pruned output | Use `pruned_width` | Resolved |
| C9 | `benchmarks/memory.py` uses a macOS physical-footprint counter | Memory differs by platform | New runner does not measure memory; on Windows the existing benchmark reports peak working set via the shim | Open (reporting) |
| C10 | `.gitignore` ignores `/docs/`, `/test`, `/results/local/`; `.gitattributes` marks `results/**/*.csv` as `-text` | Docs or results could go untracked | README at repo root; results under `results/<name>/` | Resolved |
| C11 | Different protocol from the paper | Direct comparison of means invalid | Compare on the 109 common datasets against resample 0 | Resolved |
| C12 | `_default_classifier` duplicates `benchmarks.common.make_classifier` | Two copies could drift | Runner passes `make_classifier` into `prune_groups` | Resolved |
| C13 | Group budget is not a feature budget | "30% of groups" can keep more or fewer than 30% of features (see `filter` vs `ridge` rows) | Report both | Open (analysis) |
| C14 | POSIX-only `fcntl`, `resource`, `os.sysconf` in protected files and one test | Existing tests fail on Windows | `compat_windows/` shims via PYTHONPATH; protected files untouched | Resolved |

## Windows compatibility

**Why the existing tests failed on Windows.** The repo was written for macOS/Linux. Initially 56 tests gave 1 failure and 2 errors, from three POSIX-only things:
1. `import fcntl` (file locking) in `benchmarks/controller.py`, `benchmarks/run.py`, `tests/test_controller.py`: the module does not exist on Windows.
2. `os.sysconf('SC_PAGE_SIZE')` in `benchmarks/run.py::_hardware`: Windows raises `AttributeError`, and only `OSError`/`ValueError` are caught.
3. `import resource` in `benchmarks/memory.py`: worker processes crashed at start, so the controller's JIT prewarm failed.

**Fix: stand-in modules found through PYTHONPATH; no protected file edited.**

| File | Provides |
|---|---|
| `compat_windows/fcntl.py` | `flock`, `LOCK_EX/LOCK_NB/LOCK_UN` via `msvcrt.locking`; contention raises `BlockingIOError` as the code expects |
| `compat_windows/sitecustomize.py` | `os.sysconf` for `SC_PAGE_SIZE`/`SC_PHYS_PAGES` (auto-loaded at interpreter start, including subprocesses) |
| `compat_windows/resource.py` | `getrusage(RUSAGE_SELF).ru_maxrss` from the Windows peak working set |

Result: `python -B -m unittest discover -s tests` gives 64 tests OK, 1 skipped (`test_small_width_helpers_vectorize_on_audited_runtime`, specific to the audited ARM64 runtime). Caveats: on Windows, benchmark memory is the peak working set, not the macOS physical footprint of the published results (C9); locking uses byte 0 of the lock file, fine for the repo's non-blocking exclusive use; do not use the shims on Linux/macOS; `experiments/group_pruning.py` does not need them.

## What is and is not verified

**Verified on real Numba (Python 3.14.7, numba 0.63.1, Windows 11):**
- `python -B -m unittest discover -s tests`: 64 tests OK, 1 skipped, with `compat_windows` on PYTHONPATH. The 13 new tests pass.
- `python tools/generate_joint_ppv.py --check`: kernels current.
- Pruned features equal `full[:, kept]` bit for bit in all 15,744 rows of every run.
- Keep-all pruned joint transform gives output identical to the existing joint transform and is 0.7% (GunPoint) to 3% (Wafer) slower.
- Reproduction against the paper on 109 datasets.
- A `git diff` of the ten protected files against baseline `8aa4ae2` is empty.

**Not verified or not valid:**
- **Energy:** no valid measurement (next section).
- **SIMD audit test** is ARM64-specific and was skipped; SIMD behaviour of the sliced-view pooling calls was checked only indirectly through the keep-all timing.
- **Bitwise ties:** none were observed, so the tie-handling path never triggered.
- **Memory comparisons** with the published macOS results (C9).
- **Cross-machine timing:** one Windows laptop, `NUMBA_NUM_THREADS=1` but unrestricted BLAS threads in scikit-learn; use ratios, not absolute seconds.

## Energy: not measured

`experiments/energy.py` (loop the joint-PPV test transform for at least 30 s under CodeCarbon, subtract an idle period of equal length, also record process CPU time) was written and tested, but **no energy result is reported.** CodeCarbon 3.3.2 (CPU source `windows_emi`) did not respond to load on this machine: in a diagnostic of 20 s windows in alternating idle / busy-loop order, every window read about 995 J (about 46 W) whether idle or busy, and the idle-subtracted energy of the real transform came out negative (about -80 to -150 J per 30 s on GunPoint), which is impossible for a real measurement. Following the runbook ("if power measurement is not available, skip this phase and log why"), energy was skipped; the partial output was deleted and the diagnostic code was not kept. No energy or power claim is made. For a real figure, run `experiments/energy.py` on a machine with working RAPL counters (Linux, Intel) or an external meter; only the `measure()` helper needs swapping.

## Limitations

- One machine, one OS. The first 53 of 339 primary cases were timed in an earlier session than the other 286 (same machine and code, not a controlled comparison).
- Dataset-bootstrap intervals describe dataset mix, not seed noise; with 3 seeds, seed noise is only partly averaged out.
- The 30-dataset subgroups (10 datasets each) have little power; only `examples_per_group = 1` was tried.
- The 25k run has 10 datasets and 1 seed; the drop-0.3 run has 1 seed.
- Most p-values are unadjusted (only the multiplicity table is corrected).
- sklearn "Features ... are constant" warnings from the ANOVA scorer fill some logs; they are harmless (`filter_group_scores` sets those F values to 0).
- Timings come from a desktop with other processes possibly running; compare only within this machine.

## Chronological log

Branch `group-pruning`; nothing pushed. The folder was not a git repository: `git init`, baseline commit `8aa4ae2` of the existing files, new work from `1dabf25`. The empty embedded repo `minirocket/` was removed from the index with `git rm --cached` and the folder left untouched. Commit `0b1c245` named in the runbook does not exist here, so the reference CSVs were downloaded from the authors' GitHub repository. Data: `data/UCR-univariate.zip` (258 MB) from timeseriesclassification.com, extracted to `data/Univariate_ts/` (128 entries; not committed). No venv; the installed packages were used (Python 3.14.7, numba 0.63.1, numpy 2.3.5, scikit-learn 1.8.0, scipy 1.17.1, aeon 1.5.0, llvmlite 0.46.0, pandas 2.3.3; codecarbon 3.3.2 installed later for the energy attempt).

| Commit | What |
|---|---|
| `8aa4ae2` | Baseline: the repository as given |
| `1dabf25` | The six supplied code/test/experiment files, `sanity_keepall.py`, reference CSVs, pilot |
| `0730e36` | Windows shims; existing suite now runs (64 OK, 1 skipped) |
| `fe4cedf`, `9dfc8d9` | Reduced-scope Phase 6 (1 seed, drop 0.3, 113 datasets) documented and run |
| `6b2f8ad` | Phase 7: stratified-bias study and `analyse_bias.py` |
| `18eac86` | Phase 8: combined run |
| `7bba041` | Phase 9: energy script and why the measurement was skipped |
| `5890c97` | Phase 10: 25k run |
| `87c461b` | Phase 11: final README, results index, environment freeze |
| `2eacc28` | Full Phase 6 (113 datasets, 3 seeds, drop 0.2) completed by resuming; docs updated |
| `7921931` | `final_analysis.py`, filter-vs-ridge and multiplicity tests |
| next | Documentation consolidated into two files |

Phase by phase:
- **Phase 1/2.** Kernels current; 13/13 new tests pass under real Numba with no fixes needed (22 s). The existing suite initially failed on Windows (see above).
- **Phase 3.** Keep-all vs existing joint transform, best of 3: identical output; GunPoint 0.0252 s vs 0.0254 s (1.007); Wafer 0.9411 s vs 0.9691 s (1.030).
- **Phase 4.** Paper vs ours on 109 datasets: 85.67% vs 85.69%, p = 0.582.
- **Phase 5.** Pilot, 300 rows, 0 mismatches. `--bias-strategies` was left at its default, so all three strategies ran (the runbook command did not restrict it).
- **Phase 6.** A first run (all datasets, 3 seeds, drop 0.2) started 2026-11-06 23:33 and was stopped after 53 of 339 cases because large datasets are slow (ridge selection refits a 7,200 x ~10,000 matrix every round). At the user's request a faster substitute (1 seed, drop 0.3, new folder) was run on 2026-11-07 (about 1 h 20 min). Later the first run was resumed with the identical command plus `--resume` and completed: 339 cases, 7,458 rows, 0 mismatches, 0 errors.
- **Phase 7.** 900 cases, 1,800 rows, 0 mismatches.
- **Phase 8.** 300 cases, 3,600 rows, 0 mismatches.
- **Phase 9.** Skipped (see Energy).
- **Phase 10.** 100 rows, 0 mismatches.
- **Phase 11 and after.** Final assembly; later `final_analysis.py` showed that the earlier wording "`filter` is more accurate than `ridge` at every level" was only a same-sign pattern (significant only at 10%), and the combined run's nominal "10% keep" hint does not survive multiplicity correction; both statements were corrected here.
- The runbook (`RUNBOOK_CLAUDE_CODE.md`) was not copied into the repo (it is the user's own instruction file). It asked for a `RUN_LOG.md`; at the user's later request that log and the other change documents were merged into this file.

## Prior work and what is new here

Pruning ROCKET-type models is published: S-ROCKET ([arXiv:2203.03445](https://arxiv.org/abs/2203.03445)), POCKET ([arXiv:2309.08499](https://arxiv.org/abs/2309.08499), Knowledge-Based Systems 2024), and Detach-ROCKET ([arXiv:2309.14518](https://arxiv.org/abs/2309.14518), Data Mining and Knowledge Discovery 2024). From the text read earlier, POCKET reports training time but no inference time or energy, and Detach-ROCKET does not discuss MiniRocket's convolution cost. An energy-focused pruning study (Fischer et al., [arXiv:2604.07953](https://arxiv.org/abs/2604.07953)) covers Hydra and Quant; check its full text before claiming a gap. Cite these and position this work as: **group-level pruning on (dilation, kernel) groups, judged on measured transform time with joint PPV, with the cost of selection and a break-even point reported.** No prior work on class-stratified bias sampling was found in a few searches; that is not proof it is new. Do not claim pruning or stratification is new in general.

Please cite the original method as in the [main README](./README.md#minirocket).
