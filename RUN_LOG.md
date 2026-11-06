# Run log: group pruning and stratified biases

Executed from `RUNBOOK_CLAUDE_CODE.md` in a Claude Code cloud session.

## Environment

- Branch: `claude/eager-hopper-3q5ufj` (the session's required branch; the runbook's `group-pruning` name was not used).
- Starting commit: `01c6e29ffe3c978e9b882b35d437b03bc9ed2cf9`
- Python 3.13.18 in `.venv` (not committed). Full list: `results/environment_freeze.txt`.
- Pinned to the published run's versions: numba 0.63.1, llvmlite 0.46.0, numpy 2.3.5, scipy 1.17.1, scikit-learn 1.9.1, aeon 1.6.0, threadpoolctl 3.7.0; plus pandas and codecarbon (latest).
- CPU: Intel(R) Xeon(R) Processor @ 2.10GHz, 4 logical CPUs, AVX2/AVX-512F; 15 GiB RAM. OS: Linux 6.18.44 x86_64 (cloud VM).
- NOTE: the published joint-PPV run was on Apple arm64 (macOS). Timings here are from this x86 VM only and are not comparable with that run's absolute times.
- No `/sys/class/powercap` (no RAPL) on this VM.

## Phase 0: Environment — PASS

Commands:
```
python3.13 -m venv .venv
.venv/bin/python -m pip install numba==0.63.1 llvmlite==0.46.0 numpy==2.3.5 scipy==1.17.1 scikit-learn==1.9.1 aeon==1.6.0 threadpoolctl==3.7.0 pandas codecarbon
.venv/bin/python -m pip freeze > results/environment_freeze.txt
curl --fail --location --output data/UCR-univariate.zip https://www.timeseriesclassification.com/aeon-toolkit/Archives/Univariate2018_ts.zip
unzip data/UCR-univariate.zip -d data
```
- First download attempt was refused by the session's network policy (HTTP 403 from the egress proxy). A personal cloud environment "Minirocket" allowing `www.timeseriesclassification.com` was created and the session moved onto it; the retry succeeded.
- `data/Univariate_ts/` holds **128** dataset folders, each with `*_TRAIN.ts` and `*_TEST.ts` (the full 2018 archive; the benchmark loader excludes the variable-length/missing-value ones, leaving 113 in the published run).
- Copied the supplied new files into place: `code/minirocket_pruned.py`, `code/minirocket_stratified.py`, `tests/test_pruned.py`, `experiments/{group_pruning,summarise,compare_to_paper}.py`, `README_GROUP_PRUNING.md`; added empty `experiments/__init__.py`.
- Check: `import numba, aeon, sklearn` works (numba 0.63.1).

## Phase 1: Baseline health — PASS

```
export PYTHONPATH="$PWD/code"; export NUMBA_NUM_THREADS=1
.venv/bin/python tools/generate_joint_ppv.py --check
.venv/bin/python -B -m unittest discover -s tests -v
```
- Kernel check: "Generated pooling kernels are current".
- Test suite: **64 tests, OK (skipped=1)** in 84 s. The skip is `test_small_width_helpers_vectorize_on_audited_runtime` ("specific to the audited ARM64/Numba runtime"), so README V2's suggested vectorisation check cannot run on this x86 machine; Phase 3 timing is used instead.
- (Discovery already included `tests/test_pruned.py`, since the new files were in place.)

## Phase 2: New unit tests under real Numba — PASS

```
.venv/bin/python -B -m unittest tests.test_pruned -v
```
- **13 tests, OK.** No code changes were needed: the module-level `_INDICES` array, the bool mask and `np.quantile` assignment all compiled under Numba 0.63.1 as written (README V1 resolved).

## Phase 3: Keep-all performance sanity — PASS

New file `experiments/sanity_keepall.py` (fit 9,996 features, seed 0; one warm-up then timed TEST transforms).
```
NUMBA_NUM_THREADS=1 .venv/bin/python -B -m experiments.sanity_keepall --data ./data/Univariate_ts/ \
    --datasets GunPoint Wafer --repetitions 3 --output results/sanity_keepall/sanity_keepall.csv
NUMBA_NUM_THREADS=1 .venv/bin/python -B -m experiments.sanity_keepall --data ./data/Univariate_ts/ --datasets Wafer --repetitions 7
```
| dataset | pooling | reference median s | pruned keep-all median s | ratio | identical |
|---|---|---:|---:|---:|---|
| GunPoint | joint | 0.0264 | 0.0265 | 1.005 | True |
| GunPoint | original | 0.0340 | 0.0337 | 0.991 | True |
| Wafer | joint | 1.3400 | 1.2880 | 0.961 | True |
| Wafer | original | 1.5751 | 1.7602 | 1.118 | True |
| Wafer (7 reps) | joint | 1.2498 | 1.3007 | 1.041 | True |
| Wafer (7 reps) | original | 1.6811 | 1.6117 | 0.959 | True |

- Outputs bit-identical in every case. Joint keep-all is within 10% of the existing joint transform (the required check). The one 1.118 reading (original pooling, Wafer) reversed to 0.959 on the 7-rep rerun, so run-to-run noise on this VM is roughly ±5–10%; no code change made. The sliced-view `_pool_joint` calls do not cost measurable time.
- CSV of the 3-rep run: `results/sanity_keepall/sanity_keepall.csv`.

## Phase 4: Reproduction table against the paper — PASS

```
mkdir -p results/reference
git show 0b1c245:results/results_ucr109_mean.csv       > results/reference/results_ucr109_mean.csv
git show 0b1c245:results/accuracy_ucr109_resamples.csv > results/reference/accuracy_ucr109_resamples.csv
.venv/bin/python -B -m experiments.compare_to_paper
```
Output (verbatim):
```
Compared 109 datasets (ours also has 4 extra: Fungi, HandOutlines, NonInvasiveFetalECGThorax1, NonInvasiveFetalECGThorax2)
Mean accuracy: paper resample 0 = 85.67% | ours = 85.69% | paper 30-resample mean = 87.24% (different protocol)
Difference (ours - paper resample 0), pp: mean +0.014, median +0.000, |diff| > 2 pp on 2 datasets, > 2 paper-stds on 0
Wilcoxon signed-rank on paired differences: p = 0.582
Largest differences: BeetleFly +3.3pp, Computers +2.5pp, DistalPhalanxTW +1.7pp, SemgHandSubjectCh2 -1.6pp, EOGVerticalSignal +1.5pp
```
- Matches the expected numbers. "Ours" is the committed `results/joint_ppv_2026-10-26/per_dataset.csv` (published arm64 run), not a rerun on this machine.

## Phase 5: Pilot run — PASS

**Deviation (timing hygiene):** the runbook sets only `NUMBA_NUM_THREADS=1`, but OpenBLAS defaulted to 4 threads here, so the classifier fits and ridge selection were multi-threaded in the first pilot. The repo's own `benchmarks/controller.py` pins `OPENBLAS_NUM_THREADS`/`OMP_NUM_THREADS` to the run's thread count, so the pilot was **rerun with all of them set to 1**, and every later timed run uses:
```
export PYTHONPATH="$PWD/code" NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
```
The first pilot (4 BLAS threads) is kept, uncommitted, in `results/local/group_pruning_pilot_blas4threads/`. Its accuracies and mismatch count (0) are identical to the rerun; only times differ. `-W ignore::UserWarning` was added to silence sklearn's "Features ... are constant" warnings from `f_classif`; it has no effect on results.

```
.venv/bin/python -B -W ignore::UserWarning -m experiments.group_pruning --data ./data/Univariate_ts/ \
  --output-dir ./results/group_pruning_pilot/ --datasets GunPoint ECG200 Beef Computers Wafer --seeds 0 1 \
  --keep-fractions 1.0 0.5 0.2 --scorers ridge filter          # wall time 5 min 20 s
.venv/bin/python -B -m experiments.summarise ./results/group_pruning_pilot/
```
- `report.md`: 300 rows, **0** pruned-feature mismatches (`equal_to_full_columns` true on every row).
- keep 1.0 rows: joint and original accuracy identical on every (dataset, seed, strategy).
- Independent check of the keep-all model: on GunPoint and Computers (seed 0) a separate `fit_univariate` + `transform_univariate` + `make_classifier` pipeline gives the same accuracy as the pilot's keep-all row (Computers 0.744).
- **Cross-platform finding:** for the original strategy, keep-all accuracy equals the published joint-PPV run (arm64) on GunPoint, ECG200 and Beef but differs on Computers (seed 0: 0.744 here vs 0.732 published; seed 1: 0.708 vs 0.684) and Wafer (±0.0002). The split hash and `fit_input_sha256` match the published run exactly, but the hash of the fitted `fit_univariate` parameters does not (GunPoint seed 0), so the unmodified original fit yields slightly different biases on this x86 machine (floating-point/fastmath differences). This is a property of the original code across platforms, not of the new code. Accuracy comparisons in this log are therefore within this machine only.
- Pilot headline (original strategy, mean over 5 datasets, 2 seeds): full 90.00%; ridge 50% groups 89.10% (−0.91 pp), ridge 20% 88.59% (−1.42 pp); filter 50% 89.33% (−0.68 pp), filter 20% 88.38% (−1.62 pp). Joint-PPV TEST transform at 20% groups (ridge) = 0.158 of full original.
- `selection_seconds` (ridge / filter, mean over runs, single-threaded): Beef, Computers, ECG200, GunPoint ≤ 0.4 s / ≤ 0.03 s; Wafer ~1.9 s / 0.2 s.

**Runtime extrapolation** (scratch script, not committed: scales the published run's per-dataset transform and classifier times by machine factors measured on the 5 pilot datasets: transform ×0.79–1.10, classifier ×4.03; ridge selection cost modelled as the sum of surviving widths over elimination rounds, which matched the pilot within ~2×):
- Phase 6 (113 datasets × 3 seeds × 11 configurations, drop 0.2): **≈ 25 h** (drop 0.3: ≈ 20 h). ElectricDevices (≈ 14 h) and Crop (≈ 7 h) dominate; all other datasets ≈ 4.4 h. Under 48 h, so `--drop-fraction 0.2` is kept as in the runbook. The classifier factor comes from small datasets, so the estimate for the two largest datasets is uncertain.
- Phase 7 (20 seeds × 3 strategies, keep 1.0): ≈ 27 h. Phase 8 (5 seeds × 2 strategies, ridge only): ≈ 66 h (≈ 11 h without Crop and ElectricDevices).

## Phase 6: Full pruning run — STARTED

Started 2026-11-06 22:03 UTC (pid 1375):
```
nohup .venv/bin/python -B -W ignore::UserWarning -m experiments.group_pruning \
  --data ./data/Univariate_ts/ --output-dir ./results/group_pruning_full/ \
  --bias-strategies original --scorers ridge filter > results/group_pruning_full.log 2>&1 &
```

### While Phase 6 runs: `experiments/analyse_bias.py` written and tested
- New file for Phases 7 and 8 (`--all-fractions` adds every pruned configuration for Phase 8). Uses joint rows only (accuracy is identical across implementations); subgroups from TRAIN labels read from `--data`.
- Smoke-tested on a scratch copy of the pilot CSV (2 seeds, so tests print `nan` below 10 non-zero differences).
- Sanity check prompted by that test: uniform showed zero across-seed spread on all 5 pilot datasets. A direct check on Computers (uniform seeds 0–4 → accuracy 0.728, 0.728, 0.704, 0.740, 0.720; stratified 0.724, 0.728, 0.708, 0.712, 0.672; biases differ by ~1.0 on average between seeds) confirmed it was a coincidence of 2 seeds, not a seeding bug.
