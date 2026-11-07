# Files created or changed, and why

Baseline commit `8aa4ae2` (the repository as you gave it) to the final commit, branch `group-pruning`.
**No existing tracked file was modified** (`git diff --name-status 8aa4ae2 HEAD` shows no `M` entries).
The ten protected files are untouched. Everything below is new. "Add-on" means it builds on the existing code by
importing it without changing it; "from scratch" means it did not exist in any form in the documents you supplied.

Origin key:
- **Supplied**: written in the documents you pasted (runbook, code, tests, README), copied into the repo unchanged.
- **Scratch**: written by me during this task, not in your documents.
- **Adapted**: your supplied text, extended by me with results.

## Code

| File | Origin | Type | What it does and why |
|---|---|---|---|
| `code/minirocket_pruned.py` | Supplied | Add-on (imports the baseline) | Group-level pruning: keep mask per (dilation, kernel), exact pruned transform, ridge and filter scorers. The core of the task. |
| `code/minirocket_stratified.py` | Supplied | Add-on | Class-stratified bias sampling via an explicit example table, plus the uniform comparison. |
| `tests/test_pruned.py` | Supplied | New tests | 13 tests: the pruned transform equals `full[:, kept]`, masks, scorers, stratified sampling. |
| `experiments/group_pruning.py` | Supplied | New driver | Runs datasets x seeds x strategies x scorers x keep fractions; writes `measurements.csv`, `metadata.json`, `report.md`. |
| `experiments/summarise.py` | Supplied | New | Builds `summary.csv` and `report.md` with ratios, bootstrap intervals and break-even. |
| `experiments/compare_to_paper.py` | Supplied | New | Compares our baseline to the paper's accuracies. |
| `experiments/sanity_keepall.py` | Scratch | New | Times the existing joint transform against the pruned keep-all transform. Outputs are identical; ratios 1.007 (GunPoint), 1.030 (Wafer), so the pruning code adds negligible overhead. |
| `experiments/analyse_bias.py` | Scratch | New | Paired stratified vs uniform analysis: per-dataset mean and std, Wilcoxon, subgroups, per-keep-fraction mode. Needed to answer the bias question the runbook asked. |
| `experiments/final_analysis.py` | Scratch | New | Pruned-vs-full and filter-vs-ridge Wilcoxon tests, and Holm-corrected stratification tests, from the committed CSVs. Added because the earlier reports left these untested. |
| `experiments/energy.py` | Scratch | New (not usable here) | CodeCarbon CPU energy minus idle. Tested, but this machine's energy counter is constant (about 995 J per window), so no valid results exist. |
| `compat_windows/fcntl.py` | Scratch | Shim | `fcntl` via `msvcrt.locking`, so the existing benchmark tests import on Windows. |
| `compat_windows/resource.py` | Scratch | Shim | `resource.getrusage` via `GetProcessMemoryInfo`. |
| `compat_windows/sitecustomize.py` | Scratch | Shim | Provides `os.sysconf`. |

The shims exist because the existing test suite failed on Windows (`import fcntl`, `os.sysconf`, `import resource`), and
you asked me to fix that without editing protected files. Use them with `PYTHONPATH="$PWD/code:$PWD/compat_windows"`.
Result: 64 tests pass, 1 skipped (ARM64-only).

## Documentation

| File | Origin | Purpose |
|---|---|---|
| `FINAL_ANALYSIS.md` | Scratch | The finished analysis: findings, tests, recommendation, limitations. |
| `FILES_AND_CHANGES.md` | Scratch | This file. |
| `README_GROUP_PRUNING.md` | Adapted | Main README: method, results, scope of run, conflicts C1-C14, verified / not verified. |
| `README_CHANGES.md` | Scratch | Living change log, sections 1-15, written as you asked. |
| `RUN_LOG.md` | Scratch | Phase-by-phase commands and outcomes (runbook requirement). |

## Results (all new)

| Folder or file | What it is |
|---|---|
| `results/reference/` | Paper's accuracies and our baseline comparison (resample 0: 85.67% vs 85.69%). |
| `results/group_pruning_pilot/` | Pilot: 300 rows. |
| `results/group_pruning_full/` | **Primary**: 113 datasets, 3 seeds, drop 0.2, 7,458 rows. |
| `results/group_pruning_full_drop30_1seed/` | Robustness check: 1 seed, drop 0.3, 2,486 rows. |
| `results/group_pruning_bias_30ds_10seeds/` | Stratified-bias study: 1,800 rows, with `bias_analysis.md`. |
| `results/group_pruning_combined_30ds_5seeds/` | Stratified x pruning: 3,600 rows, with `combined_analysis.md`. |
| `results/group_pruning_25k/` | 25,000 features, 10 datasets, 1 seed, 100 rows. |
| `results/final_analysis_tables.md` | Output of `experiments/final_analysis.py`. |
| `results/dataset_stats.csv`, `results/bias_study_subset.csv` | Dataset properties and the 30-dataset subset with subgroup labels. |
| `results/README_RESULTS.md`, `results/environment_freeze.txt` | Folder index with commands and commits; `pip freeze` of the environment. |
| `results/*.log` | Console logs of each run. |

Every `results/group_pruning_*` folder was run with the stated command and has 0 pruned-feature mismatches.

## Why things differ from the runbook
- Phase 6 was run twice: a faster drop-0.3, 1-seed version first (you asked for fewer rounds and seeds), then the original 3-seed, drop-0.2 version, resumed to completion after being stopped at 53 cases.
- Phases 7, 8 and 10 used a 30-dataset subset, 10 datasets, or fewer seeds, for time. Each is labelled with its scope.
- Phase 9 (energy) was skipped because the measurement is invalid on this machine.

## Cleanup done
- Removed the two `__pycache__` folders (`code/`, `compat_windows/`). They are ignored by git and regenerate.
- `data/` and `results/local/` are not committed.

## Left in place on purpose
- `minirocket/` is an untracked nested clone (`https://github.com/AR13570/minirocket.git`) with an incomplete working tree. It is not part of this task and I removed it from the git index earlier. I did not delete it because it is yours and I can't tell whether anything in it is unpushed. Delete it if it is only a stale copy.
- `results/joint_ppv_2026-10-26/` and the other baseline results were there before and are unchanged.
- The `*.log` files are duplicates in spirit of the `report.md` files, but they record the exact run output, so I kept them.
