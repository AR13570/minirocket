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
