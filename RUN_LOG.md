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
