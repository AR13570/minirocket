# README_CHANGES - everything changed or added so far

Living document. Every change to this working copy is listed here with the reason. Numbers come only from files under `results/`. Companion files: `RUN_LOG.md` (phase-by-phase log of commands), `README_GROUP_PRUNING.md` and `RUNBOOK_CLAUDE_CODE.md` (the plan; not yet copied into the repo).

## 1. Existing files modified
**None.** The ten protected files named in the runbook (`code/minirocket.py`, `code/minirocket_baseline.py`, `code/minirocket_joint_ppv_simd.py`, `code/_joint_ppv_simd_kernels.py`, `benchmarks/{common,memory,controller,worker,inventory}.py`, `tools/generate_joint_ppv.py`) are untouched. `python tools/generate_joint_ppv.py --check` passes.

## 2. New files
| File | What it is |
|---|---|
| `code/minirocket_pruned.py` | Group layout, keep-mask, masked transform (joint and original pooling), group scoring, iterative pruning |
| `code/minirocket_stratified.py` | Bias fit from an explicit example table (`uniform` control or `stratified`) |
| `tests/test_pruned.py` | 13 tests, incl. direct-convolution oracle |
| `experiments/group_pruning.py` | Experiment grid runner (own CSV + metadata) |
| `experiments/summarise.py` | Pareto table, bootstrap intervals, break-even |
| `experiments/compare_to_paper.py` | Paper-vs-ours accuracy table |
| `experiments/sanity_keepall.py` | **Not in the supplied docs; written for runbook Phase 3.** Times existing joint transform vs pruned keep-all and checks identical output |
| `results/reference/results_ucr109_mean.csv`, `accuracy_ucr109_resamples.csv` | Authors' published numbers, from github.com/angus924/minirocket (`results/`) |
| `results/reference/paper_vs_ours.csv` | Output of `compare_to_paper.py` |
| `results/group_pruning_pilot/` (+ `results/group_pruning_pilot.log`) | Pilot run output |
| `RUN_LOG.md`, `README_CHANGES.md` | Logs |

The six code/test/experiment files were copied verbatim from the supplied documents (no logic edits were needed).

## 3. Environment and repo setup (deviations from the runbook)
- Folder was **not a git repo**. Ran `git init`, committed the existing files as `8aa4ae2 Baseline`, created branch `group-pruning`, and committed new work as `1dabf25`. `.gitignore` already excludes `data/`.
- An empty embedded git repo `minirocket/` was removed from the index (`git rm --cached`); the folder itself is untouched.
- Commit `0b1c245` does not exist here, so the reference CSVs were downloaded from the authors' GitHub repo instead of `git show`.
- Data: `data/UCR-univariate.zip` (258 MB) downloaded from timeseriesclassification.com and extracted to `data/Univariate_ts/` (128 entries; not committed).
- No venv; the already-installed packages were used: Python 3.14.7, numba 0.63.1, numpy 2.3.5, scikit-learn 1.8.0, scipy 1.17.1, aeon 1.5.0. `codecarbon` is not installed (matters only for the optional energy phase).
- Windows 11: the existing test suite initially had 1 failure and 2 errors (of 56). Fixed WITHOUT touching protected files, see section 7.

## 4. Results so far
**Phase 1/2.** Kernels current; 13/13 new tests pass under real Numba (no fixes needed).

**Phase 3 (keep-all vs existing joint transform, NUMBA_NUM_THREADS=1, best of 3).** Outputs identical. GunPoint 0.0252 s vs 0.0254 s (ratio 1.007); Wafer 0.9411 s vs 0.9691 s (ratio 1.030). Within the 10% limit.

**Phase 4 (paper vs ours, 109 datasets).** Paper resample 0 = 85.67%, ours = 85.69%; mean diff +0.014 pp, median 0.000; 2 datasets differ by more than 2 pp (BeetleFly +3.3, Computers +2.5), 0 beyond 2 paper stds; Wilcoxon p = 0.582. Matches the README's earlier claim.

**Phase 5 (pilot).** Command: `NUMBA_NUM_THREADS=1 python -B -m experiments.group_pruning --data ./data/Univariate_ts/ --output-dir ./results/group_pruning_pilot/ --datasets GunPoint ECG200 Beef Computers Wafer --seeds 0 1 --keep-fractions 1.0 0.5 0.2 --scorers ridge filter`.
`--bias-strategies` was left at its default, so all three strategies (original, uniform, stratified) ran; the runbook command did not restrict it. 300 rows, **0 pruned-feature mismatches**.
Strategy `original`, mean over the 5 datasets (from `report.md`): keep 50% of groups -> accuracy -0.68 pp (filter) / -0.91 pp (ridge), test-transform time ratio about 0.54 vs the unpruned model; keep 20% -> -1.62 pp (filter) / -1.42 pp (ridge), ratio about 0.22. Full-model joint PPV was 0.948 of full original on this small sample. Pilot size is tiny (5 datasets, 2 seeds), so treat these as a smoke test, not findings.
Selection cost was small: mean ridge selection 0.1-0.3 s per case on the small datasets and 2.3 s on Wafer, so the full run is nowhere near the runbook's 48-hour threshold and `--drop-fraction` stays at 0.2.
Keep-all accuracy equals the full model by construction (the keep-1.0 rows are the baseline).

## 5. Known caveats
- sklearn emits "Features ... are constant" warnings from the ANOVA scorer. They are harmless: `filter_group_scores` sets such F values to 0. They fill the pilot log.
- Timings come from a Windows desktop (32 logical CPUs) with other processes possibly running; compare only within this machine.

## 6. Not done yet (paused here at the user's request)
Phase 6 full run, Phase 7 stratified-bias study (+ `experiments/analyse_bias.py`), Phase 8 combined run, Phase 9 energy, Phase 10 25k features, Phase 11 final assembly (fill `README_GROUP_PRUNING.md` tables, `results/README_RESULTS.md`).

## 7. Windows compatibility shims (`compat_windows/`)
**Why existing tests failed on Windows.** The repo was written for macOS/Linux. Four POSIX-only things were used:
1. `import fcntl` (file locking) in `benchmarks/controller.py`, `benchmarks/run.py`, `tests/test_controller.py`. Module does not exist on Windows, so these modules failed to import.
2. `os.sysconf('SC_PAGE_SIZE')` in `benchmarks/run.py::_hardware` (only `OSError`/`ValueError` are caught, but Windows raises `AttributeError`).
3. `import resource` in `benchmarks/memory.py` (worker processes crashed at start, so the controller's JIT prewarm failed).
4. (Not a failure here) memory.py's macOS-only physical-footprint counter is skipped on non-macOS by design.

**Fix: stand-in modules found through PYTHONPATH; no protected file edited.**
| File | Provides |
|---|---|
| `compat_windows/fcntl.py` | `flock`, `LOCK_EX/LOCK_NB/LOCK_UN` via `msvcrt.locking`; contention raises `BlockingIOError` as the code expects |
| `compat_windows/sitecustomize.py` | `os.sysconf` for `SC_PAGE_SIZE`/`SC_PHYS_PAGES` (auto-loaded at interpreter start, including subprocesses) |
| `compat_windows/resource.py` | `getrusage(RUSAGE_SELF).ru_maxrss` from the Windows peak working set |

**Use (Windows only):** `export PYTHONPATH="$PWD/code:$PWD/compat_windows"` (Git Bash) before running tests or `benchmarks.run`.
**Result:** `python -B -m unittest discover -s tests` -> 64 tests, **OK (1 skipped)**. The skip is `test_small_width_helpers_vectorize_on_audited_runtime`, which is specific to the audited ARM64 runtime.
**Caveats:** (a) on Windows, benchmark memory = peak working set, not the macOS physical footprint used in the published results, so do not compare memory with them (C9). (b) Locking uses byte 0 of the lock file; fine for the repo's non-blocking exclusive use. (c) Do not use the shims on Linux/macOS; they are not needed. (d) `experiments/group_pruning.py` does not need them.
