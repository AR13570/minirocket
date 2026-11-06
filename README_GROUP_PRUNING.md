[`ROCKET`](https://github.com/angus924/rocket) &middot; [**`MINIROCKET`**](https://github.com/angus924/minirocket) &middot; [`HYDRA`](https://github.com/angus924/hydra)

# MINIROCKET: group pruning and class-stratified biases

***An extension of the joint-PPV work: drop whole (dilation, kernel) groups, and make the thresholds fairer across classes.***

> <div align="justify">Joint PPV made counting thresholds cheap, so the cost of a MiniRocket transform now sits in the convolutions and in the classifier. This extension attacks both. <b>Group pruning</b> removes whole (dilation, kernel) groups: it skips their convolution and their thresholds, and it shrinks the classifier. Surviving groups keep every threshold, so joint PPV's per-group saving is kept. <b>Class-stratified biases</b> choose the example used to set each group's thresholds so that every class is represented, instead of one random example. The two changes are independent and can be combined.</div>

Status: **implemented and logic-checked, not yet run on real data.** See [What is and is not verified](#what-is-and-is-not-verified).
This document describes only the new work. The existing [README](./README.md) and [benchmark guide](./benchmarks/README.md) are unchanged.

## Contents

1. [Results so far](#results-so-far)
2. [What changes, and what does not](#what-changes-and-what-does-not)
3. [Concepts and the rules that keep the numbers exact](#concepts-and-the-rules-that-keep-the-numbers-exact)
4. [Examples](#examples)
5. [Step-by-step guide](#step-by-step-guide)
6. [Experiment grid](#experiment-grid)
7. [Conflicts with the current repo, and how each is resolved](#conflicts-with-the-current-repo-and-how-each-is-resolved)
8. [What is and is not verified](#what-is-and-is-not-verified)
9. [Report checklist](#report-checklist)
10. [Prior work and what is new here](#prior-work-and-what-is-new-here)

## Results so far

**Reproduction check (computed, not planned).** Original MiniRocket from the joint-PPV run (9,996 features, supplied split, mean of 3 fitting seeds) against the authors' published per-dataset numbers (their resample 0, which the paper says includes the default split), on the 109 datasets both cover. Produced by [`experiments/compare_to_paper.py`](./experiments/compare_to_paper.py).

| | Paper resample 0 | This repo (original method) |
|---|---:|---:|
| Mean accuracy, 109 datasets | 85.67% | 85.69% |

Mean paired difference +0.014 pp, median 0.000. Two datasets differ by more than 2 pp (BeetleFly +3.3, Computers +2.5) and none by more than 2 of the paper's own standard deviations across resamples. Wilcoxon signed-rank p = 0.58, so no systematic difference. The paper's 30-resample mean (87.24%) uses a different protocol and must not be compared with a single split.

**Pruning and stratified-bias results: none yet.** Fill this table from `report.md` after step 6.

| Variant | Groups kept | Features | Accuracy | Test transform vs full original | Break-even series |
|---|---:|---:|---:|---:|---:|
| full, original transform | 100% | 9,996 | | 1.00 | n/a |
| full, joint PPV | 100% | 9,996 | | | n/a |
| pruned 50%, joint PPV | | | | | |
| pruned 20%, joint PPV | | | | | |

## What changes, and what does not

**New files only. No existing file is modified**, so the published result package and the existing tests are untouched.

| File | Purpose |
|---|---|
| [`code/minirocket_pruned.py`](./code/minirocket_pruned.py) | Group layout and keep-mask; masked transform (joint pooling and original pooling); group scoring; iterative pruning |
| [`code/minirocket_stratified.py`](./code/minirocket_stratified.py) | Fit with an explicit example table: `uniform` (control) or `stratified`; optional pooling of m examples per group |
| [`tests/test_pruned.py`](./tests/test_pruned.py) | 13 tests, including a direct-convolution oracle that does not use the C-alpha / C-gamma trick |
| [`experiments/group_pruning.py`](./experiments/group_pruning.py) | The experiment grid; own CSV and metadata |
| [`experiments/summarise.py`](./experiments/summarise.py) | Pareto tables, dataset-bootstrap intervals, training-cost break-even |
| [`experiments/compare_to_paper.py`](./experiments/compare_to_paper.py) | Paper-vs-ours table for the reproduction section |

**Do not edit these files** while a benchmark run is in progress. `benchmarks/controller.py::source_hashes()` hashes them, and `--resume` refuses to continue if they change: `code/minirocket.py`, `code/minirocket_baseline.py`, `code/minirocket_joint_ppv_simd.py`, `code/_joint_ppv_simd_kernels.py`, `benchmarks/common.py`, `benchmarks/memory.py`, `benchmarks/controller.py`, `benchmarks/worker.py`, `benchmarks/inventory.py`, `tools/generate_joint_ppv.py`. The new runner only imports from `benchmarks/common.py`.

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

## Examples

Set up as in the main README (`export PYTHONPATH="$PWD/code${PYTHONPATH:+:$PYTHONPATH}"`), with `X_training`, `X_test` as finite 2D `np.float32` arrays and labels as 1D arrays.

**Group pruning with stratified biases**

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

`transform(..., implementation='original')` uses the original one-pass-per-threshold pooling with the same mask. Use it to measure the pruned-original and pruned-joint cells of the 2x2.

## Step-by-step guide

Each step ends with a check. Do not move on until it passes.

**Step 0. Branch and baseline.**
`git switch -c group-pruning`. Run `python tools/generate_joint_ppv.py --check` and the existing tests. *Check:* the generated kernels are current and the existing tests pass on your machine (they need aeon and llvmlite).

**Step 1. Run the new tests with real Numba.**
`python -B -m unittest tests.test_pruned -v`
*Check:* 13 tests pass. If the first call is slow, that is Numba compiling and caching. If a test fails on one-bit differences, see [V3](#what-is-and-is-not-verified).

**Step 2. Restore the authors' reference numbers and build the paper-vs-ours table.** The second commit of this repo removed them from the working tree; they are still in history.
```bash
mkdir -p results/reference
git show 0b1c245:results/results_ucr109_mean.csv        > results/reference/results_ucr109_mean.csv
git show 0b1c245:results/accuracy_ucr109_resamples.csv  > results/reference/accuracy_ucr109_resamples.csv
python -B -m experiments.compare_to_paper
```
*Check:* it prints 109 datasets compared and writes `results/reference/paper_vs_ours.csv`. Cite the authors' repository as the source of these files. This covers the "table comparing your numbers to the paper's" requirement.

**Step 3. Pilot run (minutes).** Use a few datasets that differ in size, class count and balance. Check class counts first; candidates: `GunPoint ECG200 Beef Computers Wafer`.
```bash
NUMBA_NUM_THREADS=1 python -B -m experiments.group_pruning \
  --data ./data/Univariate_ts/ --output-dir ./results/group_pruning_pilot/ \
  --datasets GunPoint ECG200 Beef Computers Wafer --seeds 0 1 \
  --keep-fractions 1.0 0.5 0.2 --scorers ridge filter
python -B -m experiments.summarise ./results/group_pruning_pilot/
```
*Check:* `report.md` says **0** pruned-feature mismatches. Any other number means a bug or a rare one-bit tie; stop and investigate. Look at `selection_seconds` before scaling up.

**Step 4. Full pruning run.** All datasets, 3 seeds, default fractions.
```bash
NUMBA_NUM_THREADS=1 python -B -m experiments.group_pruning \
  --data ./data/Univariate_ts/ --output-dir ./results/group_pruning_full/ \
  --bias-strategies original --scorers ridge filter
```
Use `--resume` after an interruption with unchanged settings. Run with nothing else heavy on the machine. Large datasets hold the full TRAIN feature matrix in memory (examples x 9,996 x 4 bytes); exclude them with `--datasets` if needed and say so in the report.

**Step 5. Read the pruning results.**
Open `summary.csv` and `report.md`. Plot accuracy against groups kept, accuracy against test transform time, and the break-even number of series. Answer: how much can be pruned before accuracy drops; whether the joint-PPV gain survives on the kept groups (compare the "same implementation" and "vs full original" ratio columns); whether the cheap `filter` ranking is good enough compared with `ridge`.

**Step 6. Stratified-bias study.** Separate from pruning at first, and with many seeds, because the effect is about stability.
```bash
NUMBA_NUM_THREADS=1 python -B -m experiments.group_pruning \
  --data ./data/Univariate_ts/ --output-dir ./results/group_pruning_bias/ \
  --bias-strategies original uniform stratified --keep-fractions 1.0 \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
```
Compare **stratified against uniform** (identical code and RNG policy). Original against uniform only shows the effect of the RNG change. Report per-dataset mean and standard deviation across seeds, min and max, a paired Wilcoxon test across datasets, and results split by imbalanced, small and noisy datasets. A null result on average is a valid finding; the MiniRocket paper found the whole-training-set variant essentially as accurate as the default. Optionally try `--examples-per-group 3` (fit cost about 3x).

**Step 7. Combined run.** Stratified biases and pruning together.
```bash
NUMBA_NUM_THREADS=1 python -B -m experiments.group_pruning \
  --data ./data/Univariate_ts/ --output-dir ./results/group_pruning_combined/ \
  --bias-strategies uniform stratified --seeds 0 1 2 3 4
```
Summarise as above. The question is whether stratified biases make the pruned model more stable across seeds, not only whether they change accuracy.

**Step 8. Energy (optional, report with care).**
CodeCarbon samples power at an interval, so a call shorter than a few seconds is below its resolution. Repeat the pruned and full test transforms in a loop until each measurement lasts at least about 30 seconds, and subtract an idle measurement taken the same way.
```python
from codecarbon import EmissionsTracker      # check the docs of your installed version
tracker = EmissionsTracker(measure_power_secs=1, save_to_file=False, log_level='error')
tracker.start()
for _ in range(repetitions):
    transform(X_test, parameters, keep)
tracker.stop()
energy_kwh = tracker.final_emissions_data.energy_consumed
```
State which method and machine you used. Report CPU time alongside energy, since estimates can be off. On Linux with an Intel CPU, RAPL counters are an alternative.

**Step 9. Larger feature counts (optional).** `--feature-count 25000` reuses the same code. Expect the classifier to become a larger share of the time.

**Step 10. Freeze and write up.** Commit the result folders under `results/<name>/` (not `results/local/`, which is git-ignored). Add the new README to the repo root. Do not put it under `docs/` (git-ignored). Fill the results table above and link the folders from the report.

## Experiment grid

| Factor | Values | Why |
|---|---|---|
| Bias strategy | `original`, `uniform`, `stratified` | `uniform` isolates stratification from the RNG change |
| Keep fraction (of groups) | 1.0, 0.75, 0.5, 0.3, 0.2, 0.1 | Pareto curve; fixed in advance, never tuned on TEST |
| Scorer | `ridge` (iterative backward elimination), `filter` (one-shot ANOVA F) | Embedded versus cheap ranking |
| Transform | `original` pooling, `joint` pooling | The 2x2: does pruning compound with joint PPV |
| Features | 9,996 (`--feature-count 10000`) | As the published run; larger optional |
| Seeds | 3 for pruning, 10 to 30 for stratified biases | Variance needs many seeds |

Size: datasets x seeds x strategies x configurations. For example 113 x 3 x 3 x 6 = 6,102 configurations (each timed with both transforms). The `ridge` selection loop dominates, so use the pilot first and larger drop fractions (`--drop-fraction 0.3`) if it is too slow.

Per row the CSV records the bias-fit time, full TRAIN transform time, selection time, classifier refit time, pruned transform times, accuracy, and an `equal_to_full_columns` check.

**Training-cost accounting** (used by `summarise.py`). Pruning needs the full model first, so it adds cost before it saves any.
`train cost = fit + full TRAIN transform + selection + classifier refit`
`inference cost per series = (TEST transform + predict) / n_test`
`break-even series = (extra train cost) / (inference saving per series)`, or never if inference is not cheaper. Selection reuses the cached full TRAIN features and slices columns; it does not re-transform TRAIN.

**Leakage rule.** Ranking and refitting use TRAIN only; the keep fraction is chosen from the fixed grid and reported for every value. The classifier's internal alpha search uses features and scaling fitted on the whole TRAIN split, as in the existing README's note on nested validation.

## Conflicts with the current repo, and how each is resolved

| # | Conflict (where) | Why it matters | Resolution | Status |
|---|---|---|---|---|
| C1 | `benchmarks/results.py` expects exactly `('original','joint_ppv')` per seed and budget; `controller.build_variants` and `case_order` assume pairs; `worker.py` raises on any other implementation | A third or fourth variant would fail verification and break the report generator and `tests/test_results.py`, `tests/test_controller.py` | Separate runner `experiments/group_pruning.py` with its own output; reuses only loaders and `make_classifier` | Resolved by design |
| C2 | `source_hashes()` lists ten files; `--resume` compares `source_sha256` | Editing any of them mid-run invalidates resume | No listed file is modified; rule above | Resolved |
| C3 | `minirocket.fit(X, ...)` has no labels; `_fit_biases` has a fixed Numba signature and cache, and the worker requires zero fit-cache misses | Stratification needs `y`; changing the signature breaks the "unchanged original" guarantee | New `minirocket_stratified.py` takes an explicit example table; original untouched | Resolved |
| C4 | Original draws examples with Numba's RNG, seeded by `minirocket_baseline` | Any difference between original and stratified would mix two effects | `uniform` strategy on the identical code path as the control; compare stratified against uniform | Resolved; test shows the m = 1 loop equals the original given the same draws |
| C5 | `_pool_joint(responses, biases, start, count, output)` reads biases and writes output at the same index | With pruning, the output is compact and the bias layout is full | Pass slices: `biases[full slice]` and `features[example, compact slice]` with `start = 0`; the generated kernels are unchanged, so `tools/generate_joint_ppv.py --check` still passes | Resolved; checked |
| C6 | Biases depend on global feature index; padding parity depends on `(dilation_index + kernel_index) % 2` | Refitting or renumbering after pruning changes features | Fit once, mask in place, never renumber; tests include a trap test | Resolved; trap shown |
| C7 | Published verification asserts identical accuracy between variants | Pruning changes accuracy by design | New exactness contract: pruned columns equal `full[:, kept]`; flagged per row | Resolved |
| C8 | Fixed-width assumptions: worker checks width `num_features // 84 * 84`; empty-input width is `len(parameters[-1])` | Wrong for pruned output | Use `pruned_width`; never pass pruned features to those code paths | Resolved |
| C9 | `benchmarks/memory.py` uses a macOS-only physical-footprint counter; the controller needs llvmlite | Memory numbers differ by platform; the new runner does not use them | Compare memory only on one machine; state the metric | Open (reporting) |
| C10 | `.gitignore` ignores `/docs/`, `/test`, `/results/local/`; `.gitattributes` marks `results/**/*.csv` as `-text` | Documentation or results can silently go untracked; line endings must not be normalised | README at repo root; committed results under `results/<name>/`; leave the attributes alone (Python's `csv` writes CRLF, which matches the existing files) | Resolved |
| C11 | Different protocol from the paper (supplied split vs 30 resamples; 113 vs 109 datasets; scaler plus ridge vs `normalize=True`) | Direct comparison of means is invalid | Compare on the 109 common datasets against resample 0; compare timing as ratios from your own run | Resolved; computed above |
| C12 | `_default_classifier` duplicates `benchmarks.common.make_classifier` | Two copies could drift | The experiment runner passes `make_classifier` into `prune_groups` | Resolved |
| C13 | Group budget is not a feature budget | A "30% of groups" model can keep more or fewer than 30% of features | Report both; consider a cost-weighted budget as an extension | Open (analysis) |

## What is and is not verified

**Verified in this sandbox**, using a small plain-Python stand-in for Numba (so the same logic ran without compilation):
- All 13 new tests pass.
- The masked transform matches `full[:, kept]` bit for bit for random masks, a removed dilation, a single group, odd-kernel masks and keep-all, with both pooling implementations.
- A direct-convolution oracle agrees with the original transform (max difference 3e-8), which also confirms the group layout and padding rule.
- The stratified fit's loop equals the original `_fit_biases` when given the same random draws, and its pooled quantiles match the oracle.
- On an imbalanced toy label set the stratified table gave 112 / 112 / 112 groups per class; uniform gave 190 / 47 / 99.
- The experiment runner, `--resume`, and `summarise.py` run end to end on a synthetic dataset, with 0 feature mismatches.
- `compare_to_paper.py` ran on your real files (result above). `tools/generate_joint_ppv.py --check` passes.

**Not verified:**
- **V1. Compilation under real Numba.** Possible first-compile issues: the module-level `_INDICES` array inside jitted functions (if Numba objects, inline the literal as the other modules do), boolean mask typing, and `np.quantile` dtype on assignment.
- **V2. SIMD behaviour.** The pooling helpers now receive sliced views. The hot loop is over the contiguous response array, so I expect no loss, but check by running the existing `test_small_width_helpers_vectorize_on_audited_runtime` and by timing keep-all against the existing joint transform.
- **V3. One-bit float differences.** Bitwise equality between different jitted functions held in the repo's own runs, so it should hold here. If a rare tie flips a threshold, the runner flags it rather than stopping.
- **V4. Any timing, accuracy or energy result.** Timings from the stand-in are meaningless; none are reported here.
- Two existing tests (`test_ts_reader...`, `test_normal_workflow...`) fail in the sandbox only because `aeon` and `llvmlite` are not installed.

## Report checklist

Mapped to the grading criteria in the project spec.

| Criterion | Evidence to include |
|---|---|
| Motivation (10) | MiniRocket's speed and use; why cost per inference and energy matter; the gap versus prior pruning work |
| Reproduction approach (10) | Steps 0 to 2; supplied split vs resamples; what was compared and why |
| Lessons learned (30) | `paper_vs_ours.csv`; the discrepancy analysis (within paper variance, no systematic difference); protocol and classifier differences; the original-code details found (e.g. bias sampling inside the jitted function) |
| Improvements and evaluation (30) | Steps 3 to 7: Pareto curves, 2x2 timing, stratified versus uniform with many seeds, break-even; null results reported honestly |
| Repository (20) | This README, tests, commands per result, environment versions, a results table against the paper |

Claims to **avoid**: any speed or energy claim that is not from your own measurement; any claim that pruning is new in general (see below); comparing 30-resample means with single-split numbers.

## Prior work and what is new here

Pruning ROCKET-type models is published: S-ROCKET ([arXiv:2203.03445](https://arxiv.org/abs/2203.03445)), POCKET ([arXiv:2309.08499](https://arxiv.org/abs/2309.08499), Knowledge-Based Systems 2024), and Detach-ROCKET ([arXiv:2309.14518](https://arxiv.org/abs/2309.14518), Data Mining and Knowledge Discovery 2024). From the text I read, POCKET reports training time but no inference time or energy, and Detach-ROCKET does not discuss MiniRocket's convolution cost. An energy-focused pruning study (Fischer et al., [arXiv:2604.07953](https://arxiv.org/abs/2604.07953)) covers Hydra and Quant; check its full text before claiming a gap. Cite these and position this work as: **group-level pruning on (dilation, kernel) groups, judged on measured transform time with joint PPV, with the cost of selection and a break-even point reported.** I found no prior work on class-stratified bias sampling in a few searches; that is not proof it is new.

Expect pruning to lose some accuracy on MiniRocket, which is less redundant than ROCKET, and a null average effect from stratified biases. Either outcome is reportable.

Please cite the original method as in the [main README](./README.md#minirocket).
