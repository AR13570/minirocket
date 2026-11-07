[`ROCKET`](https://github.com/angus924/rocket) &middot; [**`MINIROCKET`**](https://github.com/angus924/minirocket) &middot; [`HYDRA`](https://github.com/angus924/hydra)

# MINIROCKET: group pruning and class-stratified biases

***An extension of the joint-PPV work: drop whole (dilation, kernel) groups, and make the thresholds fairer across classes.***

> <div align="justify">Joint PPV made counting thresholds cheap, so the cost of a MiniRocket transform now sits in the convolutions and in the classifier. This extension attacks both. <b>Group pruning</b> removes whole (dilation, kernel) groups: it skips their convolution and their thresholds, and it shrinks the classifier. Surviving groups keep every threshold, so joint PPV's per-group saving is kept. <b>Class-stratified biases</b> choose the example used to set each group's thresholds so that every class is represented, instead of one random example. The two changes are independent and can be combined.</div>

Status: **implemented, tested under real Numba, and run on the UCR univariate archive at reduced scope** (see [Scope of what was run](#scope-of-what-was-run)). Energy was **not** measured (see [Energy](#energy-not-measured)).
This document describes only the new work. The existing [README](./README.md) and [benchmark guide](./benchmarks/README.md) are unchanged. A running list of every change, with commands and numbers, is in [`README_CHANGES.md`](./README_CHANGES.md); the phase-by-phase log is in [`RUN_LOG.md`](./RUN_LOG.md); each result folder is indexed in [`results/README_RESULTS.md`](./results/README_RESULTS.md).

## Contents

1. [Results](#results)
2. [Scope of what was run](#scope-of-what-was-run)
3. [What changes, and what does not](#what-changes-and-what-does-not)
4. [Concepts and the rules that keep the numbers exact](#concepts-and-the-rules-that-keep-the-numbers-exact)
5. [Examples](#examples)
6. [How to reproduce](#how-to-reproduce)
7. [Experiment grid](#experiment-grid)
8. [Conflicts with the current repo, and how each is resolved](#conflicts-with-the-current-repo-and-how-each-is-resolved)
9. [What is and is not verified](#what-is-and-is-not-verified)
10. [Energy: not measured](#energy-not-measured)
11. [Prior work and what is new here](#prior-work-and-what-is-new-here)

## Results

All numbers below are copied from files under `results/`. Pruning results are **single-seed** (Phase 6) and so are descriptive.

**Reproduction check.** Original MiniRocket from the joint-PPV run (9,996 features, supplied split, mean of 3 fitting seeds) against the authors' published per-dataset numbers (their resample 0, which the paper says includes the default split), on the 109 datasets both cover (`results/reference/paper_vs_ours.csv`, from [`experiments/compare_to_paper.py`](./experiments/compare_to_paper.py)).

| | Paper resample 0 | This repo (original method) |
|---|---:|---:|
| Mean accuracy, 109 datasets | 85.67% | 85.69% |

Mean paired difference +0.014 pp, median 0.000. Two datasets differ by more than 2 pp (BeetleFly +3.3, Computers +2.5) and none by more than 2 of the paper's own standard deviations. Wilcoxon p = 0.582: no systematic difference. The paper's 30-resample mean (87.24%) uses a different protocol and must not be compared with a single split. The authors' CSVs were fetched from github.com/angus924/minirocket (`results/`).

**Group pruning** (`results/group_pruning_full_drop30_1seed/`: 113 datasets, 1 seed, bias strategy original, drop fraction 0.3, 10,000 features). Mean over datasets, joint PPV. Time = TEST transform, single thread, one Windows machine. Pruned features equal `full[:, kept]` in all 2,486 rows.

| Variant | Groups kept | Features | Accuracy | Delta vs full | Test transform vs full original | Break-even series |
|---|---:|---:|---:|---:|---:|---:|
| full, original transform | 1,598 (100%) | 9,996 | 86.07% | 0 | 1.000 | n/a |
| full, joint PPV | 1,598 (100%) | 9,996 | 86.07% | 0 | 0.806 | n/a |
| ridge-pruned 50%, joint PPV | 799 | 4,508 | 85.70% | -0.38 pp | 0.359 | 530 |
| ridge-pruned 20%, joint PPV | 320 | 1,654 | 84.53% | -1.55 pp | 0.170 | 578 |
| filter-pruned 50%, joint PPV | 799 | 5,558 | 85.94% | -0.14 pp | 0.460 | 23 |
| filter-pruned 20%, joint PPV | 320 | 2,457 | 85.20% | -0.88 pp | 0.230 | 0 (cheaper to train than full) |

Findings: test-transform time falls roughly in proportion to groups kept and the joint-PPV saving carries over to the kept groups; moderate pruning (50% of groups) costs 0.1-0.4 pp, heavy pruning (10%) 1.6-2.4 pp. The cheap `filter` ranking was at least as accurate as `ridge` at every level in this run, but keeps more features; one seed makes that comparison weak. The full table (75% and 10% too, bootstrap intervals, train-cost ratios) is in that folder's `report.md`.

**Stratified vs uniform biases** (`results/group_pruning_bias_30ds_10seeds/`: 30 datasets, 10 seeds, full model). Mean accuracy difference stratified minus uniform: **-0.03 pp** (8 wins / 8 ties / 14 losses; Wilcoxon p = 0.35). Seed-to-seed standard deviation was not lower for stratified (0.73 vs 0.65 pp). Only the imbalanced subgroup is nominally significant, and in the wrong direction (-0.10 pp, p = 0.012, unadjusted across several subgroup tests on 9 datasets: exploratory). The original-vs-uniform control (RNG policy only) was also within noise (+0.06 pp, p = 0.29). **Null result on average.**

**Combined** (`results/group_pruning_combined_30ds_5seeds/`: 30 datasets, 5 seeds, ridge scorer, drop 0.3). Stratified minus uniform, all datasets: -0.02 pp (full), +0.06 (75%), 0.00 (50%), +0.17 (30%), +0.15 (20%), +0.49 (10%; p = 0.046, unadjusted, one of six slices). Seed-to-seed spread was lower for stratified at 30%, 20% and 10% of groups (e.g. 1.36 vs 1.59 pp at 10%) but those comparisons are not significant (p = 0.17-0.81). A hint worth a larger study, not a finding.

**25,000 features** (`results/group_pruning_25k/`: 10 datasets, 1 seed). Ridge pruning to 20% of groups kept accuracy at the full-model level (+0.13 pp) with about 0.18 of the transform time.

## Scope of what was run

The runbook's full-size plan was reduced because large datasets make the ridge selection loop slow on this machine (Crop alone took 18-24 min per seed with drop fraction 0.2). Every deviation:

| Phase | Planned | Run |
|---|---|---|
| 6 pruning | all datasets, 3 seeds, drop 0.2 | all 113 usable datasets, **1 seed, drop 0.3** (a first 3-seed run was stopped after 53 cases and is **not used**) |
| 7 stratified study | all datasets, 20 seeds | **30 datasets (fixed rule: 10 imbalanced, 10 small, 10 rest, evenly spaced alphabetically), 10 seeds** |
| 8 combined | all datasets, 5 seeds, default scorer | **the same 30 datasets, 5 seeds, ridge only, drop 0.3** |
| 9 energy | 5 datasets | **skipped**: no valid power measurement on this machine |
| 10 25k features | 10 datasets, reduced Phase 6 | 10 datasets, 1 seed, keep fractions 1.0/0.5/0.2, drop 0.3 |

15 of the 128 archive datasets are skipped by the loader (12 variable-length, 3 with missing values). Consequences: one seed gives no seed-to-seed noise estimate for pruning; dataset-bootstrap intervals describe dataset mix only; drop fraction 0.3 changes which groups survive compared with 0.2; the 30-dataset subset has little statistical power inside its subgroups.

## What changes, and what does not

**New files only. No file on the protected list was modified**, so the published result package and the existing tests are untouched.

| File | Purpose |
|---|---|
| [`code/minirocket_pruned.py`](./code/minirocket_pruned.py) | Group layout and keep-mask; masked transform (joint pooling and original pooling); group scoring; iterative pruning |
| [`code/minirocket_stratified.py`](./code/minirocket_stratified.py) | Fit with an explicit example table: `uniform` (control) or `stratified`; optional pooling of m examples per group |
| [`tests/test_pruned.py`](./tests/test_pruned.py) | 13 tests, including a direct-convolution oracle that does not use the C-alpha / C-gamma trick |
| [`experiments/group_pruning.py`](./experiments/group_pruning.py) | The experiment grid; own CSV and metadata |
| [`experiments/summarise.py`](./experiments/summarise.py) | Pareto tables, dataset-bootstrap intervals, training-cost break-even |
| [`experiments/compare_to_paper.py`](./experiments/compare_to_paper.py) | Paper-vs-ours table |
| [`experiments/sanity_keepall.py`](./experiments/sanity_keepall.py) | Keep-all pruned transform vs the existing joint transform: identical output, speed ratio |
| [`experiments/analyse_bias.py`](./experiments/analyse_bias.py) | Stratified-vs-uniform analysis with Wilcoxon tests and subgroups |
| [`experiments/energy.py`](./experiments/energy.py) | CodeCarbon energy script (written and tested; no valid results here, see [Energy](#energy-not-measured)) |
| [`compat_windows/`](./compat_windows) | Windows stand-ins for POSIX `fcntl`, `resource`, `os.sysconf` so the existing tests run (see `README_CHANGES.md` section 7) |

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

Set up as in the main README (`export PYTHONPATH="$PWD/code${PYTHONPATH:+:$PYTHONPATH}"`; on Windows also add `compat_windows`), with `X_training`, `X_test` as finite 2D `np.float32` arrays and labels as 1D arrays.

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

`transform(..., implementation='original')` uses the original one-pass-per-threshold pooling with the same mask.

## How to reproduce

Exact commands, in order, are in [`results/README_RESULTS.md`](./results/README_RESULTS.md). In short: install the packages (versions in `results/environment_freeze.txt`), download the UCR archive as in the benchmark guide, `export PYTHONPATH="$PWD/code:$PWD/compat_windows"` (Windows) or `"$PWD/code"` (Linux/macOS), run `python -B -m unittest discover -s tests`, then the commands per result folder. Use `NUMBA_NUM_THREADS=1`, a quiet machine, and the same machine for every timing you compare.

## Experiment grid

| Factor | Values | Why |
|---|---|---|
| Bias strategy | `original`, `uniform`, `stratified` | `uniform` isolates stratification from the RNG change |
| Keep fraction (of groups) | 1.0, 0.75, 0.5, 0.3, 0.2, 0.1 | Pareto curve; fixed in advance, never tuned on TEST |
| Scorer | `ridge` (iterative backward elimination), `filter` (one-shot ANOVA F) | Embedded versus cheap ranking |
| Transform | `original` pooling, `joint` pooling | The 2x2: does pruning compound with joint PPV |
| Features | 9,996 (`--feature-count 10000`); 24,948 for the 25k run | As the published run; larger optional |

**Training-cost accounting** (used by `summarise.py`). Pruning needs the full model first, so it adds cost before it saves any.
`train cost = fit + full TRAIN transform + selection + classifier refit`
`inference cost per series = (TEST transform + predict) / n_test`
`break-even series = (extra train cost) / (inference saving per series)`, or never if inference is not cheaper. A break-even of 0 means the pruned model's total training cost is already below the full model's (the smaller classifier fit saves more than the selection costs).

**Leakage rule.** Ranking and refitting use TRAIN only; the keep fraction is chosen from the fixed grid and reported for every value. The classifier's internal alpha search uses features and scaling fitted on the whole TRAIN split, as in the existing README's note on nested validation.

## Conflicts with the current repo, and how each is resolved

| # | Conflict (where) | Why it matters | Resolution | Status |
|---|---|---|---|---|
| C1 | `benchmarks/results.py` expects exactly `('original','joint_ppv')` per seed and budget; `controller.build_variants` and `case_order` assume pairs; `worker.py` raises on any other implementation | A third or fourth variant would fail verification and break the report generator | Separate runner `experiments/group_pruning.py` with its own output; reuses only loaders and `make_classifier` | Resolved by design |
| C2 | `source_hashes()` lists ten files; `--resume` compares `source_sha256` | Editing any of them mid-run invalidates resume | No listed file is modified | Resolved |
| C3 | `minirocket.fit(X, ...)` has no labels; `_fit_biases` has a fixed Numba signature and cache | Stratification needs `y` | New `minirocket_stratified.py` takes an explicit example table; original untouched | Resolved |
| C4 | Original draws examples with Numba's RNG | Original vs stratified would mix two effects | `uniform` strategy on the identical code path as the control | Resolved; test shows the m = 1 loop equals the original given the same draws; Phase 7 control within noise |
| C5 | `_pool_joint(responses, biases, start, count, output)` reads biases and writes output at the same index | With pruning, output is compact and the bias layout is full | Pass slices with `start = 0`; generated kernels unchanged | Resolved; `generate_joint_ppv.py --check` passes; keep-all identical output |
| C6 | Biases depend on global feature index; padding parity depends on `(dilation_index + kernel_index) % 2` | Refitting or renumbering changes features | Fit once, mask in place; trap test included | Resolved |
| C7 | Published verification asserts identical accuracy between variants | Pruning changes accuracy by design | New contract: pruned columns equal `full[:, kept]`; flagged per row | Resolved; 0 mismatches in every run |
| C8 | Fixed-width assumptions in worker | Wrong for pruned output | Use `pruned_width` | Resolved |
| C9 | `benchmarks/memory.py` uses a macOS physical-footprint counter | Memory differs by platform | New runner does not measure memory; on Windows the existing benchmark would report peak working set via the shim | Open (reporting) |
| C10 | `.gitignore` ignores `/docs/`, `/test`, `/results/local/`; `.gitattributes` marks `results/**/*.csv` as `-text` | Docs or results could go untracked | README at repo root; results under `results/<name>/` | Resolved |
| C11 | Different protocol from the paper | Direct comparison of means invalid | Compare on the 109 common datasets against resample 0 | Resolved; computed above |
| C12 | `_default_classifier` duplicates `benchmarks.common.make_classifier` | Two copies could drift | Runner passes `make_classifier` into `prune_groups` | Resolved |
| C13 | Group budget is not a feature budget | "30% of groups" can keep more or fewer than 30% of features (see filter vs ridge rows above) | Report both | Open (analysis) |
| C14 | POSIX-only `fcntl`, `resource`, `os.sysconf` in protected files and one test | Existing tests fail on Windows | `compat_windows/` shims via PYTHONPATH; protected files untouched | Resolved |

## What is and is not verified

**Verified on real Numba (Python 3.14.7, numba 0.63.1, Windows 11):**
- `python -B -m unittest discover -s tests`: 64 tests OK, 1 skipped (an ARM64-only compiler check), with `compat_windows` on PYTHONPATH. The 13 new tests pass.
- `python tools/generate_joint_ppv.py --check`: kernels current.
- Pruned features equal `full[:, kept]` bit for bit in every row of every run: pilot (300 rows), Phase 6 (2,486), Phase 7 (1,800), Phase 8 (3,600), Phase 10 (100).
- Keep-all pruned joint transform gives output identical to the existing joint transform and is 0.7% (GunPoint) to 3% (Wafer) slower (`experiments/sanity_keepall.py`).
- Reproduction against the paper on 109 datasets (above).

**Not verified or not valid:**
- **Energy:** no valid measurement (next section).
- **SIMD audit test** (`test_small_width_helpers_vectorize_on_audited_runtime`) is specific to ARM64 and was skipped; SIMD behaviour of the sliced-view pooling calls was checked only indirectly through the keep-all timing above.
- **Bitwise ties:** none were observed, so V3 never triggered.
- **Seed noise in pruning results** (single seed), and statistical power inside subgroups.
- **Memory comparisons** with the published macOS results (C9).
- **Cross-machine timing:** all timings come from one Windows desktop (AMD Ryzen 9 8940HX, 32 logical CPUs) with `NUMBA_NUM_THREADS=1` but unrestricted BLAS threads in scikit-learn; use ratios, not absolute seconds.

## Energy: not measured

CodeCarbon 3.3.2 (CPU source `windows_emi`) did not respond to load on this machine: 20 s windows read about 995 J (46 W) whether idle or running a busy loop, and the idle-subtracted energy of the real transform came out negative. No energy or power claim is made. Details in `README_CHANGES.md` section 12. For a real figure, run `experiments/energy.py` on a machine with working RAPL counters or an external meter.

## Prior work and what is new here

Pruning ROCKET-type models is published: S-ROCKET ([arXiv:2203.03445](https://arxiv.org/abs/2203.03445)), POCKET ([arXiv:2309.08499](https://arxiv.org/abs/2309.08499), Knowledge-Based Systems 2024), and Detach-ROCKET ([arXiv:2309.14518](https://arxiv.org/abs/2309.14518), Data Mining and Knowledge Discovery 2024). From the text read earlier, POCKET reports training time but no inference time or energy, and Detach-ROCKET does not discuss MiniRocket's convolution cost. An energy-focused pruning study (Fischer et al., [arXiv:2604.07953](https://arxiv.org/abs/2604.07953)) covers Hydra and Quant; check its full text before claiming a gap. Cite these and position this work as: **group-level pruning on (dilation, kernel) groups, judged on measured transform time with joint PPV, with the cost of selection and a break-even point reported.** No prior work on class-stratified bias sampling was found in a few searches; that is not proof it is new. Do not claim pruning or stratification is new in general.

Please cite the original method as in the [main README](./README.md#minirocket).
