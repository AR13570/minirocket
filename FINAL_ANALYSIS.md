# Final analysis: group pruning and stratified biases for MiniRocket

All numbers come from committed `measurements.csv` files (nothing estimated). Timing is one Windows 11 laptop
(Ryzen 9 8940HX), Python 3.14.7, numba 0.63.1, `NUMBA_NUM_THREADS=1`, single thread. Accuracy is the mean over datasets
(seeds averaged within a dataset first). Statistical tests are Wilcoxon signed-rank over datasets
(`experiments/final_analysis.py`, output in `results/final_analysis_tables.md`). Full detail: `README_GROUP_PRUNING.md`.

## 1. What was asked and what was run

| Question | Run | Scope |
|---|---|---|
| Does the new code reproduce the paper's baseline? | `results/reference/` | resample 0 of the UCR split |
| Does pruning change features it keeps? | every run | **0 mismatches in all 15,744 measured rows** (pilot 300, primary 7,458, drop-0.3 2,486, bias 1,800, combined 3,600, 25k 100; checked from the CSVs) |
| Accuracy / time / break-even of group pruning | `group_pruning_full/` | **113 datasets, 3 seeds, drop 0.2** (primary, the runbook's plan) |
| Same, faster setting | `group_pruning_full_drop30_1seed/` | 113 datasets, 1 seed, drop 0.3 (robustness check) |
| Do stratified biases help? | `group_pruning_bias_30ds_10seeds/`, `group_pruning_combined_30ds_5seeds/` | 30-dataset subset |
| Do conclusions hold at 25,000 features? | `group_pruning_25k/` | 10 datasets, 1 seed |
| Energy | not measured | CodeCarbon readings were constant on this machine, so invalid |

15 datasets are skipped by the loader (12 variable-length series, 3 with missing values), so averages cover 113.

## 2. Findings

### 2.1 Correctness: pruning is exact
Pruned features equal `full[:, kept_columns]` bit for bit in every run, and the 13 new tests pass (64 passing in the whole suite, 1 ARM64-only skip).
The baseline reproduces the paper: 85.67% vs 85.69% on resample 0.

### 2.2 Pruning trades a little accuracy for a lot of test time (primary run)
Time = test transform with joint PPV, relative to the unpruned model with the original transform. Joint PPV alone is 0.811.

| Scorer | Groups kept | Features | Accuracy | Delta vs full (86.08%) | Wilcoxon p | Time | Break-even |
|---|---:|---:|---:|---:|---:|---:|---:|
| filter | 75% | 7,946 | 86.07 | -0.01 pp | 0.91 | 0.657 | 233 series |
| filter | 50% | 5,555 | 86.03 | -0.05 pp | 0.50 | 0.463 | 25 |
| filter | 30% | 3,538 | 85.68 | -0.40 pp | 0.048 | 0.311 | 0 |
| filter | 20% | 2,455 | 85.29 | -0.79 pp | 0.002 | 0.232 | 0 |
| filter | 10% | 1,268 | 84.65 | -1.43 pp | <0.0001 | 0.142 | 0 |
| ridge | 75% | 7,133 | 85.97 | -0.11 pp | 0.23 | 0.545 | 1,347 |
| ridge | 50% | 4,527 | 85.81 | -0.27 pp | 0.006 | 0.362 | 1,019 |
| ridge | 30% | 2,582 | 85.29 | -0.79 pp | <0.0001 | 0.233 | 1,017 |
| ridge | 20% | 1,654 | 84.78 | -1.30 pp | <0.0001 | 0.172 | 993 |
| ridge | 10% | 784 | 83.85 | -2.23 pp | <0.0001 | 0.112 | 1,068 |

Reading:
- **Time falls roughly in proportion to the groups kept**, and the joint-PPV saving carries over to the kept groups.
- **Mild pruning is free in accuracy:** filter at 50% or 75% and ridge at 75% show no detectable loss. Ridge at 50% costs 0.27 pp, small but detectable (p=0.006).
- **Heavy pruning has a real cost:** 20% of groups loses 0.8 to 1.3 pp, and 10% loses 1.4 to 2.2 pp (all p<0.01).
- **Break-even depends on the scorer.** Ridge selection costs about 4 to 5 times the full model's training, so it repays only after roughly 1,000 test series. `filter` is nearly free: it repays in about 25 series at 50% and is cheaper to train than the full model at 30% or fewer.
- **Filter vs ridge accuracy** (filter minus ridge): +0.09, +0.23, +0.39, +0.51, +0.80 pp at 75, 50, 30, 20, 10% of groups. The sign is the same everywhere, but only the 10% level is nominally significant (p=0.038; 0.077 at 20%; the rest p>0.15), uncorrected for the 5 comparisons. Filter also keeps more features at the same group count, so it is not a like-for-like comparison. Treat "filter is at least as good as ridge here" as supported, "filter is better" as suggestive only.
- **Drop fraction 0.3, 1 seed** gives the same picture (ridge 50%: -0.38 pp at 0.359 of full-original time; ridge 20%: -1.55 pp at 0.170; break-even about 530-580 series).

### 2.3 At 25,000 features (10 datasets, 1 seed)
Full 88.90%. Ridge at 50%: +0.14 pp at 0.46 of the time (break-even about 400 series). Ridge at 20%: +0.13 pp at 0.18 (about 475). Filter at 50% and 20%: -0.31 and -0.33 pp at 0.60 and 0.30. With more features there is more redundancy to prune, and ridge loses nothing here. This is 10 datasets and one seed, so it is indicative only.

### 2.4 Stratified bias sampling: no demonstrated benefit
- **Bias study** (30 datasets, 10 seeds, full model): stratified minus uniform is -0.03 pp overall (Wilcoxon p=0.35, 8 wins, 8 ties, 14 losses). Seed-to-seed spread is not lower (0.73 vs 0.65 pp). The imbalanced subgroup is -0.10 pp (p=0.012, nominal, in the wrong direction).
- **Combined with pruning** (30 datasets, 5 seeds, 24 testable slices): the nominal highlights are ridge at 10% keep overall (+0.49 pp, p=0.046) and the small-dataset subgroup there (+1.35 pp, p=0.004). After Holm correction over all 24 slices, none survives (smallest adjusted p=0.094), and 2 raw p<0.05 hits out of 24 is about what chance gives (1.2 expected).
- The original-vs-uniform comparison (RNG policy only) is about +0.06 pp, i.e. the random-number policy by itself moves accuracy by about as much as any stratification effect.
- **Conclusion:** the data do not show that stratified biases help. The only candidate worth a dedicated follow-up is "small datasets under heavy ridge pruning", which would need new data, not more slicing of this data.

## 3. Recommendation
1. Use group pruning with the cheap `filter` scorer at 50% of groups: about 2.2 times faster test transform (0.463 of the original) for -0.05 pp (not significant), and it repays its training cost within about 25 series.
2. Use ridge selection only if you will classify more than roughly 1,000 series per trained model, or at very large feature counts where it did not lose accuracy (25k run, indicative).
3. Do not expect more than a 1 to 2 pp loss below 20% of groups.
4. Skip stratified biases: no benefit shown.

## 4. Limitations (what this does not show)
- One machine, one OS. Timings are relative, and the first 53 of 339 primary cases were timed in an earlier session than the other 286 (same machine and code, not a controlled comparison).
- Dataset-bootstrap intervals in `report.md` describe dataset mix, not seed noise. With 3 seeds, seed noise is only partly averaged out.
- The 30-dataset subgroups (10 datasets each) have little power.
- 25k run: 10 datasets, 1 seed. Drop-0.3 run: 1 seed.
- Energy is not measured. No energy claim is made.
- Memory metric differs by platform (conflict C9). The group budget is not the same as the feature budget (C13).
- Many p-values here are uncorrected. Only the Holm-corrected table in section 2.4 is corrected.
