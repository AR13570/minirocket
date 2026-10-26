# Run the MiniRocket benchmarks

These benchmarks compare **original MiniRocket** with **optimized joint PPV**.
Both produce the same features. We measure whether joint PPV computes them
faster, and record accuracy, RAM, fitting time and classifier time too.

Start with the small example below. It runs both implementations on one dataset.

## 1. Set up your terminal

Open a terminal in your cloned `minirocket` folder. Run all commands from this
repository root: the folder containing `code/`, `benchmarks/` and `.venv/`.

The examples use the existing `.venv` environment. It needs NumPy, Numba,
scikit-learn, SciPy, aeon and threadpoolctl. Keep the same environment for both
implementations; changing package versions can change performance.

Paste this setup once in your terminal:

```bash
export NUMBA_NUM_THREADS=1 NUMBA_NRT_STATS=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export NUMBA_CACHE_DIR="$PWD/results/local/numba-cache"

mkdir -p "$NUMBA_CACHE_DIR"
JOINT_RUN_DIR="$(mktemp -d results/local/joint-ppv-XXXXXX)"
echo "Save this results folder: $JOINT_RUN_DIR"
```

This requests one thread, enables a reusable compilation cache and creates a
new results folder so earlier runs are preserved. Use the same terminal for
these examples so `$JOINT_RUN_DIR` remains available.

The scripts use **local data only**. For the examples, the extracted archive
must be under `data/Univariate_ts/`, with files such as:

```text
data/Univariate_ts/GunPoint/GunPoint_TRAIN.ts
data/Univariate_ts/GunPoint/GunPoint_TEST.ts
```

Change `--data` if your archive is elsewhere. The loader also accepts TRAIN/TEST
TSV pairs, or a split NPZ containing `X_train`, `y_train`, `X_test` and `y_test`.
It does not download or unpack archives.

## 2. Try one dataset first

```bash
.venv/bin/python -B -m benchmarks.benchmark_joint_ppv \
  --data data/Univariate_ts \
  --datasets GunPoint \
  --budgets 2500 \
  --seeds 0 \
  --output "$JOINT_RUN_DIR/gunpoint.csv"
```

This runs **two cases**: original and joint PPV, each with 2,436 actual features
and one fitting seed. Each case fits once and repeats each TRAIN/TEST transform
three times. Compilation may make the first run take longer to start.

While it runs, you will see `Running GunPoint ...` messages. At the end, the
terminal shows a comparison table and the saved file paths. Add `--verbose` to
show individual seed results when using several seeds.

Summarize the completed run:

```bash
.venv/bin/python -B -m benchmarks.analyze_results \
  "$JOINT_RUN_DIR/gunpoint.csv" \
  --baseline original_2500 \
  --output-dir "$JOINT_RUN_DIR/gunpoint_summary"
```

This only reads the saved results; it does not run the models again. One dataset
is useful for checking that everything works, not for drawing broad performance
conclusions or estimating uncertainty across datasets.

## 3. Try a small subset

For a more useful trial, compare three datasets with three fitting seeds:

```bash
.venv/bin/python -B -m benchmarks.benchmark_joint_ppv \
  --data data/Univariate_ts \
  --datasets GunPoint Adiac CinCECGTorso \
  --budgets 5000 \
  --seeds 0 1 2 \
  --output "$JOINT_RUN_DIR/subset.csv"

.venv/bin/python -B -m benchmarks.analyze_results \
  "$JOINT_RUN_DIR/subset.csv" \
  --baseline original_5000 \
  --output-dir "$JOINT_RUN_DIR/subset_summary"
```

This runs **18 cases**: 3 datasets × 2 implementations × 3 seeds.

## 4. Run the full archive

First, record which local datasets can be used and fingerprint their contents:

```bash
.venv/bin/python -B -m benchmarks.inventory_joint_ppv \
  --data data/Univariate_ts \
  --output "$JOINT_RUN_DIR/archive.inventory.json"
```

The previous archive had 113 compatible datasets and 15 explicit skips. The
script reports the counts for your copy. It requires finite, equal-length,
univariate series of at least nine samples. It records unsupported datasets
instead of silently cleaning or padding them.

The following loop runs the full archive at each of the three feature budgets.
Each CSV contains only one budget, so its analysis compares the correct pair:

```bash
for budget in 2500 5000 10000; do
  .venv/bin/python -B -m benchmarks.benchmark_joint_ppv \
    --data data/Univariate_ts \
    --inventory "$JOINT_RUN_DIR/archive.inventory.json" \
    --budgets "$budget" \
    --seeds 0 1 2 \
    --output "$JOINT_RUN_DIR/archive_${budget}.csv" || break
done
```

With 113 compatible datasets, that is **678 cases per budget**, or **2,034 cases
in total**. These are three separate runs; compare original and joint PPV within
each run. Avoid other heavy work while measuring performance.

On macOS, you can prefix the benchmark command with `caffeinate -i` to prevent
idle sleep during that command. This does not prevent manual sleep or suspension.

After all three finish successfully, generate their summaries:

```bash
for budget in 2500 5000 10000; do
  .venv/bin/python -B -m benchmarks.analyze_results \
    "$JOINT_RUN_DIR/archive_${budget}.csv" \
    --baseline "original_${budget}" \
    --output-dir "$JOINT_RUN_DIR/summary_${budget}" || break
done
```

To inspect groups of datasets, run another analysis with a new output folder:

```bash
.venv/bin/python -B -m benchmarks.analyze_results \
  "$JOINT_RUN_DIR/archive_10000.csv" \
  --baseline original_10000 \
  --group-by classifier length_group train_size_group \
  --output-dir "$JOINT_RUN_DIR/groups_10000"
```

## Find and understand the results

For an example named `gunpoint.csv`, you get:

| File | What to use it for |
|---|---|
| `gunpoint.csv` | Every measured case, including timings, accuracy, RAM, feature hashes and array sizes. |
| `gunpoint.csv.json` | Configuration, package versions, source hashes, input fingerprints, progress and skipped datasets. |
| `gunpoint_summary/models.csv` | Start here: average accuracy and times for each implementation. |
| `gunpoint_summary/comparisons.csv` | Paired differences and time/memory ratios against the chosen original baseline. |
| `gunpoint_summary/per_dataset.csv` | See which individual datasets improved or regressed. |
| `gunpoint_summary/dataset_means.csv` | Results after averaging fitting seeds within each dataset. |
| `gunpoint_summary/report.json` | The complete analysis in machine-readable form. |

Times are **seconds for a complete split**, not seconds per example. A paired
time ratio of `0.80` means joint PPV took 80% of original's time on that dataset.
The summary's mean paired ratio averages these dataset ratios; it is different
from dividing the two overall mean elapsed times. Accuracy gain is in percentage
points. RAM and array sizes in CSV files are in **bytes**; divide by 1,048,576
for MiB.

The first console table's **Train (s)** includes extractor fitting, the TRAIN
transform and classifier fitting. **Test (s)** includes the TEST transform and
prediction. To examine the optimization itself, use `train_transform_seconds`
and `test_transform_seconds` in `models.csv`. `pipeline_seconds` includes both
sides plus prepared-input loading. Raw CSV `accuracy` is a fraction such as
`0.99`; summary `accuracy_percent` is a percentage such as `99.0`.

Seeds are averaged within each dataset before datasets receive equal weight.
Seeds repeat fitting on the same official split; they are not new train/test
splits. Both implementations should have identical accuracy and feature hashes.

## Change the settings

| Option | Meaning | Default |
|---|---|---|
| `--datasets GunPoint Adiac` | Select named datasets. Omit it for the full local archive. | All discovered datasets |
| `--budgets 2500` | Requested features for **each** implementation. | `2500 5000 10000` |
| `--seeds 0 1 2` | Fitting seeds. | `0 1 2` |
| `--transform-repetitions 3` | Timed transforms per TRAIN and TEST split. | `3` |
| `--threads 1` | Requested thread limit for both implementations. Keep the setup exports consistent with this. | `1` |
| `--batch-size 256` | Examples processed per transform batch. | `256` |
| `--verbose` | Print individual seed results. | Off |

Feature counts are rounded down to multiples of 84:
`2500 → 2436`, `5000 → 4956`, `10000 → 9996`. Model names use the requested
budget: `original_2500` and `joint_ppv_2500`, for example.

For the examples, keep one budget per CSV. Do not analyze a multi-budget CSV
against a single baseline, because that would also compare different widths.

## If something goes wrong

- **Output already exists:** repeat setup to create a fresh results folder. Do
  not reuse the completed study's filenames.
- **A run was interrupted:** wait until its controller and workers have exited,
  then repeat its exact command with `--resume` added. Keep the original results
  folder, environment, source files, settings, data and inventory unchanged.
  The controller refuses incompatible resumes and concurrent writers.
- **A new terminal cannot find `$JOINT_RUN_DIR`:** set it to the results folder
  printed during setup, for example
  `JOINT_RUN_DIR=results/local/joint-ppv-AbCdEf`, and repeat the environment exports.
  Do not create a new folder when resuming an interrupted run.
- **Dataset not found:** check `--data` and the exact dataset name. Files must
  already be extracted locally.
- **Feature equality or worker error:** stop and inspect the error. A timing
  result is not accepted as a valid comparison when paired outputs differ.

## What makes the comparison fair?

Both implementations use the same raw splits, fitting seed, feature count,
batching and StandardScaler/RidgeClassifierCV pipeline. They run in fresh
processes, with adjacent same-budget pairs in randomized order. The controller
checks exact features, learned parameters, predictions and accuracy.

Learning and discarded warmup use TRAIN only. Internal RidgeCV uses features
and scaling fitted to full outer TRAIN, so it is not fully nested validation.
Input time series receive no preprocessing.

Transform timings include validation, output allocation, checks and batch
copying. Compilation, warmup and hashing are excluded. RAM is a whole-process
lifetime peak, including imports, warmup and classifier work, not memory used
exclusively by the transform. macOS physical footprint is preferred; RSS is also
recorded. Actual Accelerate thread counts are not exposed on the audited build,
although a one-thread limit is requested before startup.

## Tests and the recorded study

To check the implementation without running the real archive, use the terminal
environment from setup and run:

```bash
.venv/bin/python -B -m unittest discover -s tests
.venv/bin/python -B tools/generate_joint_ppv.py --check
```

Tests include small synthetic worker runs. They do not run a full archive sweep.

The normal examples above need no separate run manifest. The specialized
`verify_joint_ppv`, `analyze_joint_ppv` and `report_joint_ppv` modules support the
recorded study's frozen, multi-budget workflow. They expect additional study
artifacts; the automatically saved `.csv.json` is not the separate run manifest.
In particular, `verify_joint_ppv` currently requires that study's exact settings,
and `analyze_joint_ppv` expects its verification output. Do not use them as
drop-in commands for an arbitrary small trial.

Existing local study results are in
[`joint_ppv_full_archive.report.md`](../results/local/joint_ppv_full_archive.report.md)
and [`joint_ppv_full_archive.overview.csv`](../results/local/joint_ppv_full_archive.overview.csv).
Those files are local artifacts and may not be present in another checkout.
Recorded studies retain their original source hashes. If sources have since
been cleaned up, use the corresponding archived source snapshot and its file
inventory for the original audit. Do not replace recorded hashes with current
ones.
