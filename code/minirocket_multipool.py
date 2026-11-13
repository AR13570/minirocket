"""
minirocket_multipool.py
=======================

Author : Shreyas Shinde (Student ID 26104692)
Feature: Multi-pooling + difference series for MiniRocket

WHAT THIS FILE DOES
-------------------
Original MiniRocket (and the faster "joint PPV" version, code/minirocket_joint_ppv_simd.py) turns every
convolution output C into ONE number per bias b:

    PPV  = proportion of positive values = (how many C[j] > b) / len(C)

PPV only says HOW OFTEN a pattern appears. This file adds three more numbers
per (kernel, dilation, bias), all computed in the SAME single pass over C:

    MPV  = mean of positive values
           -> average of (C[j] - b) over the positions where C[j] > b
           -> says HOW STRONGLY the pattern appears (0 if it never appears)

    MIPV = mean index of positive values, divided by len(C)
           -> average position j (as a fraction 0..1 of the series) where C[j] > b
           -> says WHERE in the series the pattern appears (-1 if it never appears)

    LSPV = longest stretch of positive values, divided by len(C)
           -> the longest run of consecutive positions with C[j] > b
           -> says HOW LONG the pattern lasts

The idea of these four pooling operators comes from the MultiRocket paper
(Tan, Dempster, Bergmeir & Webb, 2022). What is different here is that all
four are computed for ALL biases of a kernel in ONE pass over C, following the
same "read the data once" idea that the joint-PPV transform uses for PPV.

The second idea is the DIFFERENCE SERIES. For a series x we also build
    d[t] = x[t+1] - x[t]
which describes how fast the series is changing (its slope). We fit a second
MiniRocket on d and join both feature sets. This lets the classifier see
trend / rate-of-change patterns that are hard to see in the raw values.

HOW IT REUSES THE EXISTING CODE (WITHOUT CHANGING IT)
------------------------------------------------
* fit()  -> calls `fit_univariate` (code/minirocket_baseline.py), which
            checks the input, seeds the random generator and then calls the
            ORIGINAL MiniRocket fit. So dilations and biases are exactly the
            same as in the original and in the joint-PPV version.
* The convolution part of transform() is the original MiniRocket convolution
  (copied line for line from code/minirocket_joint_ppv_simd.py, which itself
  copied it from the original code/minirocket.py). It has to be copied because
  it lives inside fit_univariate and cannot be imported on its own.
* Only the POOLING step (what we do with C) is new.

FEATURE LAYOUT
--------------
For each representation ("raw" series, then "diff" series) the output columns
are four blocks of equal size:

    [ PPV block | MPV block | MIPV block | LSPV block ]

Inside each block the order is exactly the original MiniRocket feature order.
So the raw PPV block is IDENTICAL to the original / joint-PPV output. The tests
check this bit for bit.

UPDATE AFTER THE 10-DATASET EXPERIMENT (why the defaults are what they are)
--------------------------------------------------------------------------
The ablation on 10 UCR datasets showed:
  * MPV was the only new pooling feature that helped on its own.
  * MIPV, LSPV and the difference series lowered accuracy on these (small) datasets.
So:
  * use_difference now defaults to False (still available for experiments).
  * fit(..., pools=("PPV", "MPV")) computes ONLY PPV and MPV with a lighter,
    faster pooling function (_pool_two). pools=ALL_POOLS keeps all four for the
    ablation study.
  * The final model (minirocket_adaptive.py) uses PPV + MPV and decides per dataset,
    from training data only, whether MPV is worth adding.
"""

# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------
import numpy as np
from numba import njit, prange

# Input checking + seeded call of the original MiniRocket fit (code/minirocket_baseline.py).
from minirocket_baseline import fit_univariate, validate_input


def ridge_classifier(n_train, seed=0):
    """
    StandardScaler + RidgeClassifierCV, exactly as benchmarks.common.make_classifier(n_train, 'ridge', seed)
    (alphas = n_train * logspace(-3, 3, 10)). Defined here so code/ does not import benchmarks/;
    tests/test_adaptive.py checks the two stay identical. `seed` is unused (the classifier is
    deterministic) and kept for the same signature.
    """
    from sklearn.linear_model import RidgeClassifierCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), RidgeClassifierCV(alphas=n_train * np.logspace(-3, 3, 10)))


# Names of the four pooling operators, in the order their blocks appear.
POOLING_NAMES = ("PPV", "MPV", "MIPV", "LSPV")
NUM_POOLS = len(POOLING_NAMES)

# The two pooling sets this module can compute:
ALL_POOLS = POOLING_NAMES        # PPV, MPV, MIPV, LSPV -> used by the ablation study
PPV_MPV = ("PPV", "MPV")         # PPV, MPV only        -> used by the final adaptive model (faster)


# ---------------------------------------------------------------------------
# 2. The new pooling step: four features for many biases in ONE pass over C
# ---------------------------------------------------------------------------
@njit(fastmath=True, cache=True)
def _pool_four(C, biases, first_feature, num_biases, row, num_base_features,
               count, positive_sum, index_sum, current_run, longest_run):
    """
    Compute PPV, MPV, MIPV and LSPV of one convolution output C for
    `num_biases` consecutive biases, reading C only once.

    C                  : 1D array, convolution output of one kernel/dilation
    biases             : all fitted biases (we use biases[first_feature : first_feature + num_biases])
    first_feature      : index of the first of these features in the base feature order
    num_biases         : how many biases this kernel/dilation has
    row                : output row of ONE example for ONE representation (length 4 * num_base_features)
    num_base_features  : size of each pooling block (number of original MiniRocket features)
    count ... longest_run : small scratch arrays (one slot per bias), allocated once per
                            example by the caller so that we do not allocate memory here
    """
    n = C.shape[0]

    # Step 1: reset the running totals for every bias of this kernel/dilation.
    for k in range(num_biases):
        count[k] = 0            # how many values were above the bias
        positive_sum[k] = 0.0   # sum of (value - bias) for those values
        index_sum[k] = 0        # sum of their positions j
        current_run[k] = 0      # length of the run we are currently inside
        longest_run[k] = 0      # longest run seen so far

    # Step 2: ONE pass over the convolution output.
    for j in range(n):
        value = C[j]
        # Compare this single value against every bias at once.
        for k in range(num_biases):
            bias = biases[first_feature + k]
            if value > bias:
                # Same strict test as the original PPV ("value > bias").
                count[k] += 1
                positive_sum[k] += value - bias
                index_sum[k] += j
                current_run[k] += 1
                if current_run[k] > longest_run[k]:
                    longest_run[k] = current_run[k]
            else:
                # The run of positive values is broken.
                current_run[k] = 0

    # Step 3: turn the totals into the four features and store them in their blocks.
    ppv_block = 0 * num_base_features
    mpv_block = 1 * num_base_features
    mipv_block = 2 * num_base_features
    lspv_block = 3 * num_base_features
    for k in range(num_biases):
        f = first_feature + k   # position of this feature inside each block

        # PPV: written as count / n, exactly like the joint-PPV code, so the values are bit-identical.
        row[ppv_block + f] = count[k] / n

        if count[k] > 0:
            row[mpv_block + f] = positive_sum[k] / count[k]          # mean strength
            row[mipv_block + f] = (index_sum[k] / count[k]) / n      # mean position, 0..1
        else:
            row[mpv_block + f] = 0.0     # pattern never appeared: no strength
            row[mipv_block + f] = -1.0   # pattern never appeared: special "no position" value

        row[lspv_block + f] = longest_run[k] / n                     # longest run, 0..1


@njit(fastmath=True, cache=True)
def _pool_two(C, biases, first_feature, num_biases, row, num_base_features):
    """
    Fast pooling that computes ONLY PPV and MPV (used by the final model).
    Output layout of `row`: [PPV block | MPV block].

    Why it is written this way (speed):
      * One bias at a time, with two plain local variables (count, positive_sum)
        instead of small arrays. Local variables stay in CPU registers.
      * The inner loop has NO if-statement ("branchless"):
            count        += (value > bias)          -> adds 1 or 0
            positive_sum += max(value - bias, 0)    -> adds the excess or 0
        A loop without branches can be turned by the compiler into SIMD
        instructions that process several values of C at once.
      * C is short (one series) and stays in the CPU cache, so reading it once
        per bias is cheap.
    """
    n = C.shape[0]
    for k in range(num_biases):
        bias = biases[first_feature + k]
        count = 0
        positive_sum = 0.0
        for j in range(n):
            value = C[j]
            count += np.int64(value > bias)              # True counts as 1, False as 0
            positive_sum += max(value - bias, 0.0)       # only values above the bias add something

        f = first_feature + k
        row[f] = count / n                               # PPV: bit-identical to the original
        if count > 0:
            row[num_base_features + f] = positive_sum / count   # MPV: mean of positive values
        else:
            row[num_base_features + f] = 0.0


# ---------------------------------------------------------------------------
# 3. Transform of one representation (convolution = original, pooling = new)
# ---------------------------------------------------------------------------
@njit(fastmath=True, parallel=True, cache=True)
def _transform_multipool(X, dilations, num_features_per_dilation, biases, num_pools):
    """
    Same loops as original MiniRocket / the joint-PPV transform.
    The only difference is the pooling call: _pool_two (num_pools = 2: PPV, MPV)
    or _pool_four (num_pools = 4: PPV, MPV, MIPV, LSPV), and that the output has
    num_pools times as many columns.
    """
    num_examples, input_length = X.shape

    # The 84 fixed kernels: each row lists the 3 positions (out of 9) that have weight 2.
    # (Same table as in the original code; equivalent to itertools.combinations(range(9), 3).)
    indices = np.array((
        0,1,2,0,1,3,0,1,4,0,1,5,0,1,6,0,1,7,0,1,8,
        0,2,3,0,2,4,0,2,5,0,2,6,0,2,7,0,2,8,0,3,4,
        0,3,5,0,3,6,0,3,7,0,3,8,0,4,5,0,4,6,0,4,7,
        0,4,8,0,5,6,0,5,7,0,5,8,0,6,7,0,6,8,0,7,8,
        1,2,3,1,2,4,1,2,5,1,2,6,1,2,7,1,2,8,1,3,4,
        1,3,5,1,3,6,1,3,7,1,3,8,1,4,5,1,4,6,1,4,7,
        1,4,8,1,5,6,1,5,7,1,5,8,1,6,7,1,6,8,1,7,8,
        2,3,4,2,3,5,2,3,6,2,3,7,2,3,8,2,4,5,2,4,6,
        2,4,7,2,4,8,2,5,6,2,5,7,2,5,8,2,6,7,2,6,8,
        2,7,8,3,4,5,3,4,6,3,4,7,3,4,8,3,5,6,3,5,7,
        3,5,8,3,6,7,3,6,8,3,7,8,4,5,6,4,5,7,4,5,8,
        4,6,7,4,6,8,4,7,8,5,6,7,5,6,8,5,7,8,6,7,8
    ), dtype = np.int32).reshape(84, 3)

    num_kernels = len(indices)
    num_dilations = len(dilations)

    # Number of original MiniRocket features = size of each of our 4 blocks.
    num_base_features = num_kernels * np.sum(num_features_per_dilation)
    # Largest number of biases any kernel/dilation has (size of the scratch arrays).
    max_biases = np.max(num_features_per_dilation)

    features = np.zeros((num_examples, num_pools * num_base_features), dtype = np.float32)

    # prange = examples are processed in parallel on all CPU cores.
    for example_index in prange(num_examples):

        # Scratch arrays for _pool_four, created once per example and reused.
        count = np.zeros(max_biases, dtype = np.int64)
        positive_sum = np.zeros(max_biases, dtype = np.float64)
        index_sum = np.zeros(max_biases, dtype = np.int64)
        current_run = np.zeros(max_biases, dtype = np.int64)
        longest_run = np.zeros(max_biases, dtype = np.int64)

        _X = X[example_index]

        # ---- original MiniRocket convolution (unchanged) ----
        # Kernel weights are only -1 (alpha) and 2 (gamma), so the convolution is
        # built from shifted copies of -X and 3X; no multiplications needed.
        A = -_X          # A = alpha * X = -X
        G = _X + _X + _X # G = gamma * X = 3X

        feature_index_start = 0

        for dilation_index in range(num_dilations):

            _padding0 = dilation_index % 2

            dilation = dilations[dilation_index]
            padding = ((9 - 1) * dilation) // 2

            num_features_this_dilation = num_features_per_dilation[dilation_index]

            C_alpha = np.zeros(input_length, dtype = np.float32)
            C_alpha[:] = A

            C_gamma = np.zeros((9, input_length), dtype = np.float32)
            C_gamma[9 // 2] = G

            start = dilation
            end = input_length - padding

            for gamma_index in range(9 // 2):

                C_alpha[-end:] = C_alpha[-end:] + A[:end]
                C_gamma[gamma_index, -end:] = G[:end]

                end += dilation

            for gamma_index in range(9 // 2 + 1, 9):

                C_alpha[:-start] = C_alpha[:-start] + A[start:]
                C_gamma[gamma_index, :-start] = G[start:]

                start += dilation

            for kernel_index in range(num_kernels):

                feature_index_end = feature_index_start + num_features_this_dilation

                # Half of the kernel/dilation pairs use padding, half do not (original rule).
                _padding1 = (_padding0 + kernel_index) % 2

                index_0, index_1, index_2 = indices[kernel_index]

                # Convolution output of this kernel at this dilation.
                C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]

                if _padding1 != 0:
                    C = C[padding:-padding]   # "no padding" = keep only the valid part
                # ---- end of original convolution ----

                # NEW: pooling features for all biases of this kernel/dilation.
                if num_pools == 2:
                    _pool_two(C, biases, feature_index_start, num_features_this_dilation,
                              features[example_index], num_base_features)
                else:
                    _pool_four(C, biases, feature_index_start, num_features_this_dilation,
                               features[example_index], num_base_features,
                               count, positive_sum, index_sum, current_run, longest_run)

                feature_index_start = feature_index_end

    return features


# ---------------------------------------------------------------------------
# 4. Difference series
# ---------------------------------------------------------------------------
def difference(X):
    """
    First difference of every series: d[t] = x[t+1] - x[t].
    A series of length L becomes a series of length L - 1.
    """
    X = np.asarray(X, dtype=np.float32)
    return np.ascontiguousarray(np.diff(X, axis=1), dtype=np.float32)


# ---------------------------------------------------------------------------
# 5. Public functions: fit, transform and helpers
# ---------------------------------------------------------------------------
def fit(X, num_features=10_000, max_dilations_per_kernel=32, use_difference=False,
        pools=ALL_POOLS, random_state=0):
    """
    Fit multi-pooling MiniRocket.

    X                        : training series, shape (examples, length), length >= 9
                               (length >= 10 when use_difference=True)
    num_features             : number of ORIGINAL MiniRocket features per representation
                               (rounded down to a multiple of 84, like the original).
                               Each one gives len(pools) pooled features.
    max_dilations_per_kernel : same meaning as in the original MiniRocket (default 32)
    use_difference           : also fit on the difference series (default False, see the
                               update note at the top of this file)
    pools                    : ALL_POOLS = ("PPV", "MPV", "MIPV", "LSPV")  (default), or
                               PPV_MPV   = ("PPV", "MPV")  (faster, used by the final model)
    random_state             : seed, so the same seed always gives the same model

    Returns a dictionary with the fitted parameters of each representation.
    """
    # Only the two pooling sets that have a numba pooling function are allowed.
    pools = tuple(pools)
    if pools not in (ALL_POOLS, PPV_MPV):
        raise ValueError(f"pools must be {ALL_POOLS} or {PPV_MPV}, got {pools}")

    # Check the input once here (validate_input: 2D, numbers only, finite, length >= 9).
    X = validate_input(X, 2)

    parameters = {"raw": None, "diff": None, "pools": pools}

    # Representation 1: the raw series (same dilations/biases as original MiniRocket).
    parameters["raw"] = fit_univariate(X, num_features=num_features,
                                       max_dilations_per_kernel=max_dilations_per_kernel,
                                       random_state=random_state)

    # Representation 2: the difference series, with its OWN dilations and biases,
    # because it is one point shorter and has a different scale.
    if use_difference:
        if X.shape[1] < 10:
            raise ValueError("use_difference=True needs series of length >= 10 "
                             "(the difference series must still have >= 9 points)")
        parameters["diff"] = fit_univariate(difference(X), num_features=num_features,
                                            max_dilations_per_kernel=max_dilations_per_kernel,
                                            random_state=random_state)
    return parameters


def _pools_of(parameters):
    """Pooling set of a fitted model (models fitted before the 'pools' option used all four)."""
    return parameters.get("pools", ALL_POOLS)


def _transform_one(X, representation_parameters, pools=ALL_POOLS):
    """Run the multi-pooling transform for one representation (raw or diff)."""
    dilations, num_features_per_dilation, biases = representation_parameters
    return _transform_multipool(X, dilations, num_features_per_dilation, biases, len(pools))


def transform(X, parameters):
    """
    Turn series into features.
    Output columns with all four pools:
        [raw PPV | raw MPV | raw MIPV | raw LSPV | diff PPV | diff MPV | diff MIPV | diff LSPV]
    with pools = PPV_MPV:
        [raw PPV | raw MPV | diff PPV | diff MPV]
    (the diff part only if the model was fitted with use_difference=True)
    """
    X = validate_input(X, 2, allow_empty=True)
    pools = _pools_of(parameters)

    if X.shape[0] == 0:
        # No examples: return an empty table with the right number of columns.
        return np.empty((0, num_output_features(parameters)), dtype=np.float32)

    blocks = [_transform_one(X, parameters["raw"], pools)]
    if parameters["diff"] is not None:
        blocks.append(_transform_one(difference(X), parameters["diff"], pools))

    # Join the representations side by side.
    return np.hstack(blocks) if len(blocks) > 1 else blocks[0]


def fit_transform(X, **fit_options):
    """Convenience: fit on X and return (parameters, features of X)."""
    parameters = fit(X, **fit_options)
    return parameters, transform(X, parameters)


def num_base_features(representation_parameters):
    """Number of original MiniRocket features of one representation (= size of one block)."""
    return len(representation_parameters[2])   # one bias per original feature


def num_output_features(parameters):
    """Total number of output columns."""
    num_pools = len(_pools_of(parameters))
    total = num_pools * num_base_features(parameters["raw"])
    if parameters["diff"] is not None:
        total += num_pools * num_base_features(parameters["diff"])
    return total


def feature_blocks(parameters):
    """
    Where each block of features lives in the output table.
    Returns a dict like {"raw_PPV": slice(0, 9996), "raw_MPV": slice(9996, 19992), ...}.
    Used by the experiment script (ablation) and later by the explainability module.
    """
    blocks = {}
    position = 0
    for representation in ("raw", "diff"):
        if parameters[representation] is None:
            continue
        size = num_base_features(parameters[representation])
        for pool_name in _pools_of(parameters):
            blocks[f"{representation}_{pool_name}"] = slice(position, position + size)
            position += size
    return blocks


def select_blocks(features, parameters, block_names):
    """Keep only the chosen blocks, e.g. ["raw_PPV", "raw_MPV"] (used for ablation)."""
    blocks = feature_blocks(parameters)
    return np.hstack([features[:, blocks[name]] for name in block_names])
