# RUN_LOG

Environment: Windows 11, Python 3.14.7, numba 0.63.1, numpy 2.3.5, scikit-learn 1.8.0, scipy 1.17.1, aeon 1.5.0, llvmlite 0.46.0, pandas 2.3.3. 32 logical CPUs. Not a git repository (no branch, no commit hashes). codecarbon not installed.

## Phase 0 - Environment
- Existing packages already installed; no venv created.
- NOT DONE: git branch `group-pruning` (folder is not a git repo), UCR data download (`data/` missing).

## Phase 1 - Baseline health
- `python tools/generate_joint_ppv.py --check`: PASS (kernels current).
- `python -B -m unittest discover -s tests`: 56 tests, 1 failure, 2 errors, 1 skipped, all from `import fcntl` (POSIX-only) in `benchmarks/run.py` (test_controller import, test_benchmark_common verbose test, test_results normal workflow). Windows-only; fixing needs a protected file, so left as is. New runner imports only `benchmarks.common`, which imports fine.

## Phase 2 - New tests under real Numba
- Created code/minirocket_pruned.py, code/minirocket_stratified.py, tests/test_pruned.py, experiments/{group_pruning,summarise,compare_to_paper}.py (verbatim from the supplied documents).
- `python -B -m unittest tests.test_pruned -v`: 13/13 PASS (22 s). No code changes were needed (V1 compile risks did not materialise).

## Phase 3-11
- Not started: need data (Phase 3 onward), reference CSVs from git history (Phase 4; commit 0b1c245 unavailable, zip of repo does not contain them).
