[`ROCKET`](https://github.com/angus924/rocket) &middot; [**`MINIROCKET`**](https://github.com/angus924/minirocket) &middot; [`HYDRA`](https://github.com/angus924/hydra)

# MINIROCKET

***MINIROCKET: A Very Fast (Almost) Deterministic Transform for Time Series Classification***

[KDD 2021](https://dl.acm.org/doi/abs/10.1145/3447548.3467231) / [arXiv:2012.08791](https://arxiv.org/abs/2012.08791) (preprint)

> <div align="justify">Until recently, the most accurate methods for time series classification were limited by high computational complexity.  ROCKET achieves state-of-the-art accuracy with a fraction of the computational expense of most existing methods by transforming input time series using random convolutional kernels, and using the transformed features to train a linear classifier.  We reformulate ROCKET into a new method, MINIROCKET, making it up to 75 times faster on larger datasets, and making it almost deterministic (and optionally, with additional computational expense, fully deterministic), while maintaining essentially the same accuracy.  Using this method, it is possible to train and test a classifier on all of 109 datasets from the UCR archive to state-of-the-art accuracy in less than 10 minutes.  MINIROCKET is significantly faster than any other method of comparable accuracy (including ROCKET), and significantly more accurate than any other method of even roughly-similar computational expense.  As such, we suggest that MINIROCKET should now be considered and used as the default variant of ROCKET.</div>

Please cite as:

```bibtex
@inproceedings{dempster_etal_2021,
  author    = {Dempster, Angus and Schmidt, Daniel F and Webb, Geoffrey I},
  title     = {{MiniRocket}: A Very Fast (Almost) Deterministic Transform for Time Series Classification},
  booktitle = {Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  publisher = {ACM},
  address   = {New York},
  year      = {2021},
  pages     = {248--257}
}
```

## Podcast

Hear more about MINIROCKET and time series classification on the [Data Skeptic](https://dataskeptic.com/blog/episodes/2021/minirocket) podcast!

## GPU Implementation \*NEW\*

A GPU implementation of MINIROCKET, developed by Malcolm McLean and Ignacio Oguiza, is available through [`tsai`](https://github.com/timeseriesAI/tsai).  See the [examples](https://github.com/timeseriesAI/tsai/blob/main/tutorial_nbs/10_Time_Series_Classification_and_Regression_with_MiniRocket.ipynb).  Many thanks to Malcolm and Ignacio for their work in developing the GPU implementation and making it part of `tsai`.

## `sktime`\* / Multivariate

MINIROCKET (including a basic multivariate implementation) is also available through [`sktime`](https://github.com/alan-turing-institute/sktime).  See the [examples](https://github.com/alan-turing-institute/sktime/blob/master/examples/minirocket.ipynb).

\* *for larger datasets (10,000+ training examples), the `sktime` methods should be integrated with SGD or similar as per [`softmax.py`](./code/softmax.py) (replace calls to `fit(...)` and `transform(...)` from `minirocket.py` with calls to the relevant `sktime` methods as appropriate)*

## Results

The current benchmark compares original MiniRocket with optimized joint PPV using
pooling helpers 1 through 8, at matching feature counts.

* [Full report: accuracy, transform time and RAM](./results/joint_ppv_2026-10-26/report.md)
* [Summary CSV](./results/joint_ppv_2026-10-26/summary.csv)
* [Per-dataset comparisons](./results/joint_ppv_2026-10-26/per_dataset.csv)
* [Raw measurements](./results/joint_ppv_2026-10-26/measurements.csv)
* [Verification](./results/joint_ppv_2026-10-26/verification.json)
* [How to run the benchmarks](./benchmarks/README.md)

The study covers 113 compatible UCR datasets, three fitting seeds on each supplied
TRAIN/TEST split, and 2,436 / 4,956 / 9,996 actual features. Features and predictions
match exactly between implementations. These measurements use a different
protocol from the paper's 109-dataset, 30-resample and large-dataset scalability
studies.

Across this fresh run, joint PPV reduced average transform time while preserving
accuracy exactly:

| Actual features | TRAIN time reduction | TEST time reduction | Full pipeline time reduction | Worker peak memory increase |
|---:|---:|---:|---:|---:|
| 2,436 | 28.7% | 23.9% | 14.6% | 1.1% |
| 4,956 | 32.2% | 35.1% | 18.7% | 1.3% |
| 9,996 | 48.7% | 46.9% | 31.8% | 1.8% |

These are ratios of arithmetic means after averaging fitting seeds within each
dataset and weighting datasets equally. Peak memory covers the whole worker
process, not transform-only allocation. Some datasets were slower; confidence
intervals and regressions are included in the full report. Results apply to the
measured machine and compiler environment and are not a universal speed guarantee
or an accuracy improvement.

### Reproduce these results

Follow the [benchmark guide](./benchmarks/README.md) to set up the environment,
[download the UCR archive](./benchmarks/README.md#setup-and-dataset-download),
and [run the full comparison](./benchmarks/README.md#run-the-full-benchmark)
on your own machine. The guide includes copyable commands for a quick single-dataset
check and the full archive at all three feature counts, plus instructions for
verifying the output. Use the [recorded settings and environment](./results/joint_ppv_2026-10-26/metadata.json)
when comparing results; timings will vary with hardware and software.

After setup and dataset download, only the input and a new output directory are
required:

```bash
.venv/bin/python -B -m benchmarks.run \
  --data ./data/Univariate_ts/ \
  --output-dir ./results/full-benchmark-run/
```

This uses all three feature counts and three fitting seeds, and generates the
analysis automatically. See the guide's [optional settings](./benchmarks/README.md#optional-change-experiment-settings)
to select datasets or change the experiment; the other options are not required.
For existing measurements, see [analysis without benchmarking](./benchmarks/README.md#run-analysis-without-benchmarking).

## Requirements\*

* Python, NumPy, pandas
* A Numba version compatible with your Python and NumPy versions
* scikit-learn or similar
* PyTorch or similar (for larger datasets)

\* *PyTorch is needed only for the softmax example. The archive benchmark also uses SciPy, aeon and threadpoolctl; see the [benchmark setup](./benchmarks/README.md#setup-and-dataset-download) and [recorded environment](./results/joint_ppv_2026-10-26/metadata.json).*

## Code

### [`minirocket_joint_ppv_simd.py`](./code/minirocket_joint_ppv_simd.py)

Optimized univariate transform. It keeps the original convolution arithmetic,
fitted parameters and feature order, and compares each response value against
several thresholds in one pass. Static helpers handle one through eight
thresholds, taking groups of eight followed by the remainder. Numba/LLVM compiles
the pooling loops into native code; SIMD vectorization depends on the target
CPU and compiler.

### [`_joint_ppv_simd_kernels.py`](./code/_joint_ppv_simd_kernels.py)

Generated static pooling helpers and dispatcher. Regenerate with
`python tools/generate_joint_ppv.py`; use `--check` to verify that the generated
file matches the generator. See [`generate_joint_ppv.py`](./tools/generate_joint_ppv.py).

### [`minirocket_baseline.py`](./code/minirocket_baseline.py)

Input validation and reproducible seeded fitting around the unchanged original
univariate implementation. Both benchmark methods use this fitting adapter;
the optimized module exposes it as `fit`.

### [`minirocket.py`](./code/minirocket.py)

Original univariate MiniRocket implementation and numerical reference.

### [`minirocket_dv.py`](./code/minirocket_dv.py) (MINIROCKET<sub>DV</sub>)
### [`softmax.py`](./code/softmax.py) (PyTorch / 10,000+ Training Examples)
### [`minirocket_multivariate.py`](./code/minirocket_multivariate.py) (equivalent to [sktime/MiniRocketMultivariate](https://github.com/alan-turing-institute/sktime/blob/master/sktime/transformations/panel/rocket/_minirocket_multivariate.py))
### [`minirocket_variable.py`](./code/minirocket_variable.py) (variable-length input; *experimental*)

## Important Notes

### Compilation

The functions in [`minirocket.py`](./code/minirocket.py) and [`minirocket_dv.py`](./code/minirocket_dv.py) are compiled by Numba on import, which may take some time.  Compiled functions are cached for reuse. The optimized transform uses lazy compilation, so its first call for an input signature may take longer. Cache invalidation or a different environment can require recompilation; warm up before timing steady-state execution.

### Input Data Type

Input data should be of type `np.float32`.  Alternatively, you can change the Numba signatures to accept, e.g., `np.float64`.

### Normalisation

Unlike ROCKET, MINIROCKET does **not** require the input time series to be normalised.  (However, whether or not it makes sense to normalise the input time series may depend on your particular application.)

## Examples

The modules live in `code/`. From the repository root, make them importable in
your current terminal before running a Python script or starting Python:

```bash
export PYTHONPATH="$PWD/code${PYTHONPATH:+:$PYTHONPATH}"
```

Use a Python environment with the dependencies above. For the univariate
examples, load `X_training` and `X_test` as finite two-dimensional `np.float32`
arrays shaped `(examples, time_points)`, with the same series length of at least
nine points. Labels are one-dimensional arrays. The snippets assume these data
variables have already been loaded.

The examples below use `StandardScaler` followed by `RidgeClassifierCV`, matching
the benchmark's classifier settings. Scaling is fitted on TRAIN features only;
TEST features are transformed using that fitted scaler. This replaces the old
`normalize=True` argument and is an explicit change from the historical examples.
The internal Ridge cross-validation uses features and scaling fitted on the full
outer TRAIN split; it is not fully nested validation.

**Optimized joint PPV (univariate)**

```python
import numpy as np
from minirocket_joint_ppv_simd import fit, transform
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

parameters = fit(X_training, num_features=10_000, random_state=0)
X_training_transform = transform(X_training, parameters)

classifier = make_pipeline(
    StandardScaler(),
    RidgeClassifierCV(alphas=X_training.shape[0] * np.logspace(-3, 3, 10)),
)
classifier.fit(X_training_transform, Y_training)

X_test_transform = transform(X_test, parameters)
predictions = classifier.predict(X_test_transform)
```

The requested 10,000 features round down to 9,996. This optimization supports
fixed-length univariate input. The original multivariate and variable-length
implementations below remain separate.

**MINIROCKET**

```python
from minirocket import fit, transform
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the data variables described above.

# note:
# * input time series do *not* need to be normalised
# * input data should be np.float32

parameters = fit(X_training)

X_training_transform = transform(X_training, parameters)

classifier = make_pipeline(
    StandardScaler(),
    RidgeClassifierCV(alphas=X_training_transform.shape[0] * np.logspace(-3, 3, 10)),
)
classifier.fit(X_training_transform, Y_training)

X_test_transform = transform(X_test, parameters)

predictions = classifier.predict(X_test_transform)
```

**MINIROCKET<sub>DV</sub>**

```python
from minirocket_dv import fit_transform
from minirocket import transform
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the data variables described above.

# note:
# * input time series do *not* need to be normalised
# * input data should be np.float32

parameters, X_training_transform = fit_transform(X_training)

classifier = make_pipeline(
    StandardScaler(),
    RidgeClassifierCV(alphas=X_training_transform.shape[0] * np.logspace(-3, 3, 10)),
)
classifier.fit(X_training_transform, Y_training)

X_test_transform = transform(X_test, parameters)

predictions = classifier.predict(X_test_transform)
```

**PyTorch / 10,000+ Training Examples**

```python
from softmax import train, predict

model_etc = train("InsectSound_TRAIN_shuffled.csv", num_classes = 10, training_size = 22952)
# note: 22,952 = 25,000 - 2,048 (validation)

predictions, accuracy = predict("InsectSound_TEST.csv", *model_etc)
```

**Variable-Length Input** (*Experimental*)

```python
from minirocket_variable import fit, transform, filter_by_length
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the data variables described above.

# note:
# * input time series do *not* need to be normalised
# * input data should be np.float32

# special instructions for variable-length input:
# * concatenate variable-length input time series into a single 1d numpy array
# * provide another 1d array with the lengths of each of the input time series
# * input data should be np.float32 (as above); lengths should be np.int32

# optionally, use a different reference length when setting dilation (default is
# the length of the longest time series), and use fit(...) with time series of
# at least this length, e.g.:
# >>> reference_length = X_training_lengths.mean()
# >>> X_training_1d_filtered, X_training_lengths_filtered = \
# >>> filter_by_length(X_training_1d, X_training_lengths, reference_length)
# >>> parameters = fit(X_training_1d_filtered, X_training_lengths_filtered, reference_length)

parameters = fit(X_training_1d, X_training_lengths)

X_training_transform = transform(X_training_1d, X_training_lengths, parameters)

classifier = make_pipeline(
    StandardScaler(),
    RidgeClassifierCV(alphas=X_training_transform.shape[0] * np.logspace(-3, 3, 10)),
)
classifier.fit(X_training_transform, Y_training)

X_test_transform = transform(X_test_1d, X_test_lengths, parameters)

predictions = classifier.predict(X_test_transform)
```

## Acknowledgements

We thank Professor Eamonn Keogh and all the people who have contributed to the UCR time series classification archive.  Figures in our paper showing mean ranks were produced using code from [Ismail Fawaz et al. (2019)](https://github.com/hfawaz/cd-diagram).

<div align="center">:rocket:<sub>:rocket:</sub><sub><sub>:rocket:</sub></sub></div>
