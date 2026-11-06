# Group pruning results

Rows: 300. Pruned features differing from `full[:, kept]`: **0** (expected 0).

Ratios compare dataset-mean TEST transform time with the unpruned model. "vs full original" is the combined effect of pruning and joint PPV. Break-even is the median number of series after which the extra training cost (selection) is repaid by cheaper inference.

## Bias strategy: original

| scorer | keep | impl | groups | features | accuracy % | delta pp | ratio (same impl) [95% CI] | ratio (vs full original) | train cost ratio | break-even series |
|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|
| filter | 0.50 | joint | 756 | 5682 | 89.33 | -0.68 | 0.534 [0.515, 0.687] | 0.371 | 0.58 | 0 |
| filter | 0.50 | original | 756 | 5682 | 89.33 | -0.68 | 0.493 [0.472, 0.689] | 0.493 | 0.62 | 0 |
| filter | 0.20 | joint | 302 | 2440 | 88.38 | -1.62 | 0.222 [0.198, 0.426] | 0.154 | 0.50 | 0 |
| filter | 0.20 | original | 302 | 2440 | 88.38 | -1.62 | 0.205 [0.178, 0.368] | 0.205 | 0.55 | 0 |
| none | 1.00 | joint | 1512 | 9996 | 90.00 | +0.00 | 1.000 [1.000, 1.000] | 0.695 | 1.00 | inf |
| none | 1.00 | original | 1512 | 9996 | 90.00 | +0.00 | 1.000 [1.000, 1.000] | 1.000 | 1.00 | inf |
| ridge | 0.50 | joint | 756 | 5117 | 89.10 | -0.91 | 0.568 [0.487, 0.578] | 0.395 | 1.57 | 657 |
| ridge | 0.50 | original | 756 | 5117 | 89.10 | -0.91 | 0.485 [0.416, 0.605] | 0.485 | 1.51 | 410 |
| ridge | 0.20 | joint | 302 | 1828 | 88.59 | -1.42 | 0.227 [0.222, 0.270] | 0.158 | 2.30 | 695 |
| ridge | 0.20 | original | 302 | 1828 | 88.59 | -1.42 | 0.209 [0.192, 0.243] | 0.209 | 2.18 | 372 |

## Bias strategy: stratified

| scorer | keep | impl | groups | features | accuracy % | delta pp | ratio (same impl) [95% CI] | ratio (vs full original) | train cost ratio | break-even series |
|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|
| filter | 0.50 | joint | 756 | 5766 | 89.07 | -1.04 | 0.561 [0.550, 0.707] | 0.369 | 0.57 | 0 |
| filter | 0.50 | original | 756 | 5766 | 89.07 | -1.04 | 0.456 [0.418, 0.728] | 0.456 | 0.61 | 0 |
| filter | 0.20 | joint | 302 | 2491 | 89.20 | -0.91 | 0.212 [0.196, 0.354] | 0.140 | 0.57 | 0 |
| filter | 0.20 | original | 302 | 2491 | 89.20 | -0.91 | 0.190 [0.165, 0.356] | 0.190 | 0.61 | 0 |
| none | 1.00 | joint | 1512 | 9996 | 90.11 | +0.00 | 1.000 [1.000, 1.000] | 0.658 | 1.00 | inf |
| none | 1.00 | original | 1512 | 9996 | 90.11 | +0.00 | 1.000 [1.000, 1.000] | 1.000 | 1.00 | inf |
| ridge | 0.50 | joint | 756 | 5154 | 88.70 | -1.41 | 0.548 [0.515, 0.633] | 0.360 | 1.75 | 938 |
| ridge | 0.50 | original | 756 | 5154 | 88.70 | -1.41 | 0.471 [0.460, 0.615] | 0.471 | 1.69 | 728 |
| ridge | 0.20 | joint | 302 | 1882 | 88.48 | -1.62 | 0.212 [0.209, 0.248] | 0.139 | 2.64 | 560 |
| ridge | 0.20 | original | 302 | 1882 | 88.48 | -1.62 | 0.188 [0.186, 0.235] | 0.188 | 2.49 | 457 |

## Bias strategy: uniform

| scorer | keep | impl | groups | features | accuracy % | delta pp | ratio (same impl) [95% CI] | ratio (vs full original) | train cost ratio | break-even series |
|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|
| filter | 0.50 | joint | 756 | 5667 | 88.99 | -0.95 | 0.560 [0.542, 0.713] | 0.369 | 0.54 | 19 |
| filter | 0.50 | original | 756 | 5667 | 88.99 | -0.95 | 0.471 [0.437, 0.673] | 0.471 | 0.58 | 9 |
| filter | 0.20 | joint | 302 | 2476 | 89.37 | -0.57 | 0.248 [0.225, 0.446] | 0.163 | 0.57 | 0 |
| filter | 0.20 | original | 302 | 2476 | 89.37 | -0.57 | 0.204 [0.180, 0.339] | 0.204 | 0.61 | 0 |
| none | 1.00 | joint | 1512 | 9996 | 89.94 | +0.00 | 1.000 [1.000, 1.000] | 0.659 | 1.00 | inf |
| none | 1.00 | original | 1512 | 9996 | 89.94 | +0.00 | 1.000 [1.000, 1.000] | 1.000 | 1.00 | inf |
| ridge | 0.50 | joint | 756 | 5123 | 88.30 | -1.64 | 0.567 [0.562, 0.687] | 0.374 | 1.73 | 753 |
| ridge | 0.50 | original | 756 | 5123 | 88.30 | -1.64 | 0.504 [0.462, 0.626] | 0.504 | 1.67 | 367 |
| ridge | 0.20 | joint | 302 | 1840 | 87.93 | -2.01 | 0.244 [0.217, 0.297] | 0.161 | 2.51 | 685 |
| ridge | 0.20 | original | 302 | 1840 | 87.93 | -2.01 | 0.188 [0.183, 0.246] | 0.188 | 2.38 | 284 |

