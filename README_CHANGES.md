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

## 8. Phase 6 full pruning run: scope reduced (deviation from the runbook)
**Planned (runbook):** all datasets, 3 seeds, `--drop-fraction 0.2` -> `results/group_pruning_full/`.
**What happened:** that run was started 2026-11-06 23:33 and **stopped manually** on 2026-11-07 after 53 of ~339 (dataset, seed) cases (datasets up to Crop; Crop seed 2 unfinished). Reason: large datasets are slow (Crop took ~24 min for seed 0 and ~18 min for seed 1, because ridge selection refits a 7,200 x ~10,000 matrix every round), so the whole run would take many hours.
`results/group_pruning_full/` is therefore **incomplete (no metadata.json, no report) and must not be used for any reported number.** It is kept only as a record; do not cite it.

**Replacement run (the one that counts), settings:**
| Setting | Value | Differs from runbook? |
|---|---|---|
| Output folder | `results/group_pruning_full_drop30_1seed/` | new folder |
| Datasets | all available univariate datasets (no subset) | no |
| Seeds | **1 (seed 0)** | yes (runbook: 3) |
| `--drop-fraction` | **0.3** | yes (runbook: 0.2, i.e. fewer ridge elimination rounds) |
| Bias strategy | `original` | no |
| Scorers | `ridge`, `filter` | no |
| Keep fractions | 1.0 0.75 0.5 0.3 0.2 0.1 (default) | no |
| Feature count | 10,000 (9,996 actual) | no |
| `NUMBA_NUM_THREADS` | 1 (BLAS threads in sklearn not limited) | no |
Log: `results/group_pruning_full_drop30_1seed.log`.

**Consequences to state in any report:**
- One seed per dataset, so accuracy and time differences are single-run values; the dataset-bootstrap intervals in `summarise.py` reflect dataset mix only, not seed-to-seed noise.
- With drop fraction 0.3 the ridge ranking removes groups in bigger steps, which can change which groups survive compared with 0.2. Results are not comparable with the pilot (0.2, 2 seeds) except qualitatively.
- Selection time is lower than with 0.2, so break-even numbers are optimistic relative to the 0.2 setting.

## 9. Phase 6 results (reduced-scope run, section 8 settings)
Folder `results/group_pruning_full_drop30_1seed/` (measurements.csv, summary.csv, report.md, metadata.json); command in `RUN_LOG.md`. Finished 2026-11-07, about 1h20 wall time.
**Scope:** 113 datasets run, 15 skipped automatically by the loader (12 variable-length series: AllGestureWiimoteX/Y/Z, GestureMidAirD1-3, GesturePebbleZ1/Z2, MelbournePedestrian, PLAID, PickupGestureWiimoteZ, ShakeGestureWiimoteZ; 3 with missing values: DodgerLoopDay/Game/Weekend). **1 seed (0), bias strategy `original`, drop fraction 0.3.** No dataset was excluded for memory or time.
**Exactness:** 2,486 rows, **0** rows where pruned features differ from `full[:, kept]`.
**Baseline:** unpruned accuracy 86.07% (mean over 113 datasets, single seed). Joint PPV on the full model takes 0.806 of the original transform time.

Mean over datasets (from `report.md`; delta = accuracy change vs unpruned; time ratio = test transform vs unpruned with the same implementation, joint PPV; "vs orig" = pruned joint vs unpruned original):
| Scorer | Groups kept | Features | Accuracy % | Delta pp | Time ratio (same impl) | Time ratio vs orig | Median break-even (series) |
|---|---:|---:|---:|---:|---:|---:|---:|
| filter | 75% | 7,952 | 86.10 | +0.03 | 0.811 | 0.654 | 237 |
| filter | 50% | 5,558 | 85.94 | -0.14 | 0.571 | 0.460 | 23 |
| filter | 30% | 3,536 | 85.51 | -0.57 | 0.379 | 0.306 | 0 |
| filter | 20% | 2,457 | 85.20 | -0.88 | 0.285 | 0.230 | 0 |
| filter | 10% | 1,271 | 84.51 | -1.57 | 0.174 | 0.140 | 0 |
| ridge | 75% | 7,060 | 85.92 | -0.15 | 0.657 | 0.530 | 656 |
| ridge | 50% | 4,508 | 85.70 | -0.38 | 0.445 | 0.359 | 530 |
| ridge | 30% | 2,575 | 85.14 | -0.93 | 0.285 | 0.230 | 594 |
| ridge | 20% | 1,654 | 84.53 | -1.55 | 0.210 | 0.170 | 578 |
| ridge | 10% | 776 | 83.72 | -2.35 | 0.138 | 0.112 | 642 |
Time-ratio 95% intervals (dataset bootstrap) are in `report.md`; all are narrow (e.g. ridge 50% joint 0.424-0.472).

**Reading it (single seed, so descriptive):**
- Test-transform time falls roughly in proportion to groups kept, and joint PPV's saving carries over to the kept groups (pruned joint vs unpruned original is lower than pruned joint vs unpruned joint at every level).
- Accuracy cost is small at moderate pruning: about -0.1 to -0.4 pp at 50% of groups, about -1.6 to -2.4 pp at 10%.
- The cheap `filter` scorer was at least as accurate as `ridge` at every keep level in this run (e.g. 50%: -0.14 vs -0.38 pp), while keeping more features (5,558 vs 4,508), so its time ratio is slightly worse. Do not conclude filter is better without more seeds; one seed and a different feature count make this a weak comparison.
- Break-even: `ridge` selection costs more than the inference saved for a few hundred series (median 472-656); `filter` repays quickly (23 series at 50%) or immediately (0 means total training cost is already below the unpruned model, because the classifier fit on fewer features saves more than the cheap ranking costs). Training-cost ratio values are in `report.md` (filter 30%: 0.53; ridge 30%: 3.28).
- Timings come from one Windows desktop; compare ratios, not absolute seconds.

## 10. Phase 7: stratified-bias study (reduced scope)
Folder `results/group_pruning_bias_30ds_10seeds/` (measurements.csv, metadata.json, `bias_analysis.md`). New file `experiments/analyse_bias.py` (not in the supplied documents; written per runbook Phase 7). Dataset statistics: `results/dataset_stats.csv`; chosen subset: `results/bias_study_subset.csv`.
**Settings (deviations from the runbook in bold):** strategies original, uniform, stratified; keep fraction 1.0 only; **10 seeds (0-9) instead of 20; 30 datasets instead of all 113**; default feature count 10,000; `--examples-per-group 1`. Command: `NUMBA_NUM_THREADS=1 python -B -m experiments.group_pruning --data ./data/Univariate_ts/ --output-dir ./results/group_pruning_bias_30ds_10seeds/ --datasets <30 names in bias_study_subset.csv> --bias-strategies original uniform stratified --keep-fractions 1.0 --seeds 0 1 2 3 4 5 6 7 8 9`.
**Subset rule (fixed before running, not chosen by hand):** all 113 usable datasets were classed as imbalanced (smallest TRAIN class share < 20%: 57 datasets), small (< 100 TRAIN series, not imbalanced: 26), or rest (30); within each class, 10 evenly spaced in alphabetical order. Note that the "imbalanced" rule also catches balanced datasets with more than 5 classes.
**Exactness:** 1,800 rows (900 cases x 2 implementations), 0 mismatches.

Result, stratified minus uniform (accuracy, pp; paired over datasets; from `bias_analysis.md`):
| Subgroup | Datasets | Mean diff | Wins/ties/losses | Wilcoxon p (n non-zero) | Seed-to-seed std, stratified vs uniform (pp) |
|---|---:|---:|---|---|---|
| all | 30 | -0.032 | 8/8/14 | 0.353 (22) | 0.727 vs 0.652 |
| imbalanced | 10 | -0.097 | 1/1/8 | 0.012 (9) | 0.619 vs 0.541 |
| small | 10 | -0.039 | 2/5/3 | 0.812 (5) | 0.826 vs 0.787 |
| rest | 10 | +0.041 | 5/2/3 | 0.461 (8) | 0.736 vs 0.627 |
Control, original minus uniform (RNG policy only): mean diff +0.061 pp, p = 0.290 over all 30.

**Honest reading:** the average effect of stratification is null (-0.03 pp, p = 0.35), and stratified did **not** have lower seed-to-seed variance (std lower on 9 datasets, higher on 14; mean std 0.73 vs 0.65 pp). The only nominally significant cell is the imbalanced subgroup, where stratified was slightly *worse* (-0.10 pp, p = 0.012). That p-value is unadjusted across several subgroup tests on 9 datasets and is exploratory; it should not be presented as an established effect. A null result is a valid finding (as the README anticipated). The original-vs-uniform difference is also within noise, so the RNG-policy change is not a confounder here. With 10 seeds and 30 datasets power is limited, and only `examples_per_group=1` was tried.

## 11. Phase 8: combined run (stratified/uniform biases x pruning), reduced scope
Folder `results/group_pruning_combined_30ds_5seeds/` (measurements.csv, summary.csv, report.md, `combined_analysis.md`, metadata.json). `analyse_bias.py --by-keep-fraction` produced `combined_analysis.md`.
**Settings (deviations in bold):** strategies uniform, stratified; **the same 30-dataset subset as Phase 7**; **5 seeds (0-4)**; **scorer ridge only**; **`--drop-fraction 0.3`** (as Phase 6); keep fractions 1.0 0.75 0.5 0.3 0.2 0.1. Command: `NUMBA_NUM_THREADS=1 python -B -m experiments.group_pruning --data ./data/Univariate_ts/ --output-dir ./results/group_pruning_combined_30ds_5seeds/ --datasets <30 names in bias_study_subset.csv> --bias-strategies uniform stratified --seeds 0 1 2 3 4 --drop-fraction 0.3 --scorers ridge`. Reason for the subset: the all-dataset version (113 datasets x 5 seeds x 2 strategies) would take roughly 10 hours.
**Exactness:** 3,600 rows, 0 mismatches.

Mean over the 30 datasets (ridge scorer, joint PPV, from `report.md`):
| Groups kept | Acc % stratified | Acc % uniform | Delta vs full, stratified | Delta vs full, uniform | Time ratio (same impl) |
|---:|---:|---:|---:|---:|---:|
| 100% | 81.56 | 81.58 | 0 | 0 | 1.000 |
| 75% | 81.60 | 81.54 | +0.03 | -0.04 | 0.69 |
| 50% | 81.45 | 81.45 | -0.12 | -0.14 | 0.475 |
| 30% | 81.13 | 80.96 | -0.43 | -0.62 | 0.31 |
| 20% | 80.60 | 80.45 | -0.97 | -1.14 | 0.22 |
| 10% | 79.76 | 79.27 | -1.81 | -2.32 | 0.14 |
Median break-even is about 370-490 series for both strategies at every keep level.

Stratified minus uniform, all 30 datasets (from `combined_analysis.md`): mean accuracy difference -0.02 pp (full), +0.06 (75%), 0.00 (50%), +0.17 (30%), +0.15 (20%), **+0.49 (10%)**; Wilcoxon p = 0.77, 0.31, 0.98, 0.44, 0.41, **0.046**. Seed-to-seed std (pp), stratified vs uniform: 0.64 vs 0.65 (full), 0.64/0.62 (75%), 0.91/0.84 (50%), 0.97/1.13 (30%), 1.06/1.23 (20%), 1.36/1.59 (10%).

**Honest reading:** stratified biases make no difference to the full model. At heavy pruning (30% of groups or fewer) the stratified models were on average 0.15-0.5 pp more accurate and had lower seed-to-seed spread, which is in the direction the project hoped for. But only the 10% slice reaches p < 0.05 (0.046, unadjusted, one of six slices tested, so it would not survive a multiple-comparison correction), the std comparisons are not significant (p = 0.17-0.81), and this is 5 seeds on 30 datasets with ridge only. Treat it as a hint worth a larger study, not a finding. Per-subgroup tables (imbalanced, small, rest) are in `combined_analysis.md`.
