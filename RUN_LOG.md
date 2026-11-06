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

## Update (git, data, reference, Phase 3-5)
- `git init`; commits 8aa4ae2 (baseline), 1dabf25 (new files + reference CSVs); branch group-pruning.
- Data downloaded and extracted to data/Univariate_ts/ (not committed).
- Phase 4: `python -B -m experiments.compare_to_paper` -> 109 datasets, 85.67% vs 85.69%, Wilcoxon p=0.582.
- Phase 3: `python -B -m experiments.sanity_keepall` -> identical output, ratios 1.007 (GunPoint), 1.030 (Wafer).
- Phase 5: pilot run (command in README_CHANGES.md section 4); 300 rows, 0 mismatches; summarise OK. Selection time 0.1-2.3 s per case; full-run estimate far below 48 h.
- Stopped before Phase 6 as requested.

## Update (Windows test fix)
- Cause of failing existing tests: POSIX-only fcntl, os.sysconf, resource. Added compat_windows/{fcntl,sitecustomize,resource}.py (no protected file changed).
- `PYTHONPATH="$PWD/code:$PWD/compat_windows" python -B -m unittest discover -s tests` -> Ran 64 tests, OK (skipped=1, ARM64-only check).
