# Experiment log

Generated 2026-10-08 20:20 by `python experiments.py` on CUDA. Data: 180 NSE stocks, 2017-09-25 to 2026-09-17 (frozen).

Every number below comes from **purged walk-forward validation folds**. The final test period is not used anywhere in this log.

* 5 time-ordered validation blocks, each trained on an expanding window that ends at least one label horizon (5 / 21 / 252 sessions) before the block, with every training label ending before the block starts.
* **Rank IC**: Spearman correlation between predicted probability and realised forward return, across stocks on each day, averaged over days. **AUC rel / abs**: ROC AUC against "beats the universe median" and against "rises more than 0.5%". **Top / bottom**: share of the day's top and bottom prediction quintiles that beat the median. Values are mean ± standard deviation across the 5 folds.
* A change is **kept** only if mean rank IC improves, rank IC improves in at least 4 of 5 folds, and mean AUC rel does not fall. Fixes (F) and the XGBoost switch (E0) are adopted regardless.
* F1 and F2 rows show the configuration *without* the fix (compare with B); F3 shows it *with* the fix. The "vs reference" column is always with-fix minus without-fix, or candidate minus the configuration it would replace.

## Weekly (5 sessions)

| Fold | Train | Validate | Gap (sessions) |
| --- | --- | --- | --- |
| fold1 | 2017-09-25 → 2020-10-06 | 2020-10-14 → 2021-09-09 | 5 |
| fold2 | 2017-09-25 → 2021-09-02 | 2021-09-13 → 2022-08-05 | 5 |
| fold3 | 2017-09-25 → 2022-07-29 | 2022-08-08 → 2023-07-07 | 5 |
| fold4 | 2017-09-25 → 2023-06-30 | 2023-07-10 → 2024-06-10 | 5 |
| fold5 | 2017-09-25 → 2024-06-03 | 2024-06-11 → 2025-05-08 | 5 |

| Step | Change | Rank IC | IC by fold | AUC rel | AUC abs | Top / bottom | vs reference | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B | Baseline: deployed model (LightGBM, 31 leaves, 400 trees, 26 features chosen by the old selection step) | 0.0368 ± 0.0122 | +0.039 / +0.028 / +0.031 / +0.057 / +0.029 | 0.506 ± 0.001 | 0.513 ± 0.028 | 52.6% / 47.7% |  | reference |
| F1 | NIFTY 50 index removed from the stock universe (row shows the old universe with the index; compare with B) | 0.0368 ± 0.0122 | +0.044 / +0.029 / +0.025 / +0.055 / +0.032 | 0.506 ± 0.002 | 0.513 ± 0.027 | 52.3% / 47.7% | vs B: -0.0000 IC, 2/5 folds, +0.0000 AUC | fix adopted |
| F2 | Purge on label end dates (row shows the old row-count gap; compare with B) | 0.0368 ± 0.0122 | +0.039 / +0.028 / +0.031 / +0.057 / +0.029 | 0.506 ± 0.001 | 0.513 ± 0.028 | 52.6% / 47.7% | vs B: +0.0000 IC, 0/5 folds, +0.0000 AUC | fix adopted |
| F3 | Full 69-feature candidate pool instead of feature sets chosen with the old validation data | 0.0317 ± 0.0128 | +0.034 / +0.028 / +0.022 / +0.053 / +0.022 | 0.504 ± 0.002 | 0.508 ± 0.024 | 52.0% / 47.5% | vs B: -0.0050 IC, 0/5 folds, -0.0018 AUC | fix adopted |
| E0 | XGBoost with settings equivalent to the previous model | 0.0335 ± 0.0120 | +0.034 / +0.029 / +0.030 / +0.053 / +0.021 | 0.505 ± 0.002 | 0.509 ± 0.027 | 52.2% / 47.8% | vs F3: +0.0017 IC, 3/5 folds, +0.0005 AUC | adopted (decision) |
| E1 | Relative target: beat the universe median over the horizon | 0.0289 ± 0.0105 | +0.019 / +0.025 / +0.020 / +0.039 / +0.042 | 0.513 ± 0.006 | 0.515 ± 0.014 | 51.6% / 47.7% | vs E0: -0.0045 IC, 1/5 folds, +0.0081 AUC | rejected |
| E2 (k=0.25) | Clear-move labels: drop training rows whose forward move is within k × volatility × √h of the label boundary (k=0.25) | 0.0332 ± 0.0137 | +0.030 / +0.029 / +0.028 / +0.057 / +0.022 | 0.504 ± 0.002 | 0.507 ± 0.019 | 52.3% / 48.0% | vs E0: -0.0003 IC, 2/5 folds, -0.0005 AUC | rejected |
| E2 (k=0.5) | Clear-move labels: drop training rows whose forward move is within k × volatility × √h of the label boundary (k=0.5) | 0.0291 ± 0.0154 | +0.031 / +0.022 / +0.021 / +0.055 / +0.016 | 0.503 ± 0.003 | 0.506 ± 0.022 | 51.9% / 48.3% | vs E0: -0.0043 IC, 1/5 folds, -0.0013 AUC | rejected |
| E3 | Cross-sectional ranks of stock-level features | 0.0284 ± 0.0223 | +0.023 / +0.032 / +0.016 / +0.064 / +0.006 | 0.504 ± 0.003 | 0.513 ± 0.022 | 51.7% / 48.1% | vs E0: -0.0051 IC, 2/5 folds, -0.0009 AUC | rejected |
| E4 | Weekly features: 5-day relative strength vs NIFTY and sector, volume surprise | 0.0349 ± 0.0128 | +0.031 / +0.042 / +0.029 / +0.053 / +0.019 | 0.505 ± 0.002 | 0.506 ± 0.025 | 52.3% / 47.6% | vs E0: +0.0014 IC, 1/5 folds, +0.0000 AUC | rejected |
| E5a (depth 4) | More regularisation: shallower trees, larger leaves, subsampling (depth 4) | 0.0319 ± 0.0135 | +0.029 / +0.039 / +0.018 / +0.051 / +0.022 | 0.504 ± 0.002 | 0.506 ± 0.020 | 52.0% / 48.2% | vs E0: -0.0016 IC, 2/5 folds, -0.0009 AUC | rejected |
| E5a (depth 3) | More regularisation: shallower trees, larger leaves, subsampling (depth 3) | 0.0303 ± 0.0120 | +0.031 / +0.035 / +0.017 / +0.047 / +0.021 | 0.504 ± 0.001 | 0.509 ± 0.025 | 52.0% / 48.2% | vs E0: -0.0032 IC, 2/5 folds, -0.0012 AUC | rejected |
| E5b (hold-out auc) | Early stopping on a purged hold-out at the end of each training window (hold-out auc) | 0.0025 ± 0.0080 | +0.000 / +0.001 / -0.008 / +0.012 / +0.009 | 0.500 ± 0.001 | 0.501 ± 0.040 | 51.2% / 47.6% | vs E0: -0.0310 IC, 0/5 folds, -0.0043 AUC | rejected |
| E5b (hold-out rank_ic) | Early stopping on a purged hold-out at the end of each training window (hold-out rank_ic) | 0.0241 ± 0.0283 | -0.023 / +0.027 / +0.033 / +0.053 / +0.031 | 0.505 ± 0.004 | 0.513 ± 0.024 | 51.5% / 48.0% | vs E0: -0.0094 IC, 3/5 folds, +0.0003 AUC | rejected |
| E5c | Prune features without stable importance across training sub-periods | 0.0006 ± 0.0087 | -0.011 / +0.000 / +0.000 / +0.014 / +0.000 | 0.500 ± 0.001 | 0.502 ± 0.028 | 48.6% / 49.1% | vs E0: -0.0329 IC, 0/5 folds, -0.0046 AUC | rejected |

**Final configuration: step E0.** xgboost, target absolute, 69 candidate features, clear-move k=None, cross-sectional=False, early stopping=None, pruning=False, params {'grow_policy': 'lossguide', 'max_leaves': 31, 'max_depth': 0, 'min_child_weight': 1000, 'learning_rate': 0.03, 'n_estimators': 400, 'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_lambda': 10.0}.

## Monthly (21 sessions)

| Fold | Train | Validate | Gap (sessions) |
| --- | --- | --- | --- |
| fold1 | 2017-09-25 → 2020-08-26 | 2020-09-25 → 2021-08-17 | 21 |
| fold2 | 2017-09-25 → 2021-07-16 | 2021-08-18 → 2022-07-08 | 21 |
| fold3 | 2017-09-25 → 2022-06-09 | 2022-07-11 → 2023-06-02 | 21 |
| fold4 | 2017-09-25 → 2023-05-04 | 2023-06-05 → 2024-04-30 | 21 |
| fold5 | 2017-09-25 → 2024-03-27 | 2024-05-02 → 2025-03-20 | 21 |

| Step | Change | Rank IC | IC by fold | AUC rel | AUC abs | Top / bottom | vs reference | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B | Baseline: deployed model (LightGBM, 31 leaves, 400 trees, baseline + daily features) | 0.0047 ± 0.0358 | -0.003 / +0.041 / -0.047 / +0.036 / -0.003 | 0.499 ± 0.008 | 0.518 ± 0.056 | 50.6% / 50.2% |  | reference |
| F1 | NIFTY 50 index removed from the stock universe (row shows the old universe with the index; compare with B) | 0.0056 ± 0.0409 | -0.010 / +0.043 / -0.051 / +0.048 / -0.002 | 0.499 ± 0.009 | 0.520 ± 0.057 | 50.8% / 50.2% | vs B: -0.0009 IC, 2/5 folds, +0.0002 AUC | fix adopted |
| F2 | Purge on label end dates (row shows the old row-count gap; compare with B) | 0.0047 ± 0.0358 | -0.003 / +0.041 / -0.047 / +0.036 / -0.003 | 0.499 ± 0.008 | 0.518 ± 0.056 | 50.6% / 50.2% | vs B: +0.0000 IC, 0/5 folds, +0.0000 AUC | fix adopted |
| F3 | Full 69-feature candidate pool instead of feature sets chosen with the old validation data | -0.0008 ± 0.0330 | +0.028 / +0.031 / -0.045 / +0.006 / -0.024 | 0.497 ± 0.006 | 0.489 ± 0.027 | 50.3% / 50.7% | vs B: -0.0056 IC, 2/5 folds, -0.0012 AUC | fix adopted |
| E0 | XGBoost with settings equivalent to the previous model | -0.0036 ± 0.0359 | +0.027 / +0.035 / -0.049 / -0.004 / -0.028 | 0.497 ± 0.007 | 0.490 ± 0.023 | 50.0% / 50.8% | vs F3: -0.0027 IC, 1/5 folds, -0.0006 AUC | adopted (decision) |
| E1 | Relative target: beat the universe median over the horizon | -0.0003 ± 0.0209 | +0.014 / +0.014 / -0.022 / +0.017 / -0.024 | 0.497 ± 0.008 | 0.505 ± 0.029 | 50.1% / 50.4% | vs E0: +0.0032 IC, 3/5 folds, +0.0003 AUC | rejected |
| E2 (k=0.25) | Clear-move labels: drop training rows whose forward move is within k × volatility × √h of the label boundary (k=0.25) | -0.0115 ± 0.0375 | +0.001 / +0.042 / -0.060 / -0.011 / -0.030 | 0.495 ± 0.008 | 0.495 ± 0.034 | 49.9% / 51.4% | vs E0: -0.0079 IC, 1/5 folds, -0.0021 AUC | rejected |
| E2 (k=0.5) | Clear-move labels: drop training rows whose forward move is within k × volatility × √h of the label boundary (k=0.5) | -0.0045 ± 0.0329 | +0.011 / +0.035 / -0.049 / +0.007 / -0.026 | 0.496 ± 0.006 | 0.493 ± 0.025 | 49.8% / 50.9% | vs E0: -0.0009 IC, 3/5 folds, -0.0012 AUC | rejected |
| E3 | Cross-sectional ranks of stock-level features | -0.0028 ± 0.0426 | +0.021 / +0.061 / -0.034 / -0.030 / -0.033 | 0.497 ± 0.009 | 0.483 ± 0.019 | 50.4% / 50.7% | vs E0: +0.0007 IC, 2/5 folds, +0.0001 AUC | rejected |
| E5a (depth 4) | More regularisation: shallower trees, larger leaves, subsampling (depth 4) | -0.0144 ± 0.0308 | -0.011 / +0.027 / -0.056 / -0.002 / -0.029 | 0.495 ± 0.006 | 0.490 ± 0.034 | 49.9% / 51.3% | vs E0: -0.0108 IC, 1/5 folds, -0.0016 AUC | rejected |
| E5a (depth 3) | More regularisation: shallower trees, larger leaves, subsampling (depth 3) | -0.0111 ± 0.0346 | -0.018 / +0.030 / -0.055 / +0.017 / -0.030 | 0.496 ± 0.006 | 0.489 ± 0.025 | 49.8% / 50.8% | vs E0: -0.0076 IC, 1/5 folds, -0.0007 AUC | rejected |
| E5b (hold-out auc) | Early stopping on a purged hold-out at the end of each training window (hold-out auc) | -0.0078 ± 0.0169 | +0.001 / +0.002 / -0.004 / +0.000 / -0.038 | 0.499 ± 0.002 | 0.504 ± 0.049 | 49.0% / 46.1% | vs E0: -0.0043 IC, 2/5 folds, +0.0023 AUC | rejected |
| E5b (hold-out rank_ic) | Early stopping on a purged hold-out at the end of each training window (hold-out rank_ic) | -0.0031 ± 0.0409 | -0.012 / +0.022 / -0.056 / +0.051 / -0.020 | 0.497 ± 0.008 | 0.480 ± 0.026 | 50.0% / 49.6% | vs E0: +0.0005 IC, 2/5 folds, -0.0002 AUC | rejected |
| E5c | Prune features without stable importance across training sub-periods | -0.0252 ± 0.0250 | -0.017 / +0.011 / -0.027 / -0.035 / -0.057 | 0.494 ± 0.005 | 0.484 ± 0.028 | 48.1% / 51.0% | vs E0: -0.0217 IC, 1/5 folds, -0.0025 AUC | rejected |

**Final configuration: step E0.** xgboost, target absolute, 69 candidate features, clear-move k=None, cross-sectional=False, early stopping=None, pruning=False, params {'grow_policy': 'lossguide', 'max_leaves': 31, 'max_depth': 0, 'min_child_weight': 1000, 'learning_rate': 0.03, 'n_estimators': 400, 'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_lambda': 10.0}.

## Yearly (252 sessions)

| Fold | Train | Validate | Gap (sessions) |
| --- | --- | --- | --- |
| fold1 | 2017-09-25 → 2019-01-04 | 2020-01-20 → 2020-09-24 | 252 |
| fold2 | 2017-09-25 → 2019-09-19 | 2020-09-25 → 2021-06-04 | 252 |
| fold3 | 2017-09-25 → 2020-06-03 | 2021-06-07 → 2022-02-09 | 252 |
| fold4 | 2017-09-25 → 2021-02-03 | 2022-02-10 → 2022-10-19 | 252 |
| fold5 | 2017-09-25 → 2021-10-13 | 2022-10-20 → 2023-06-27 | 252 |

| Step | Change | Rank IC | IC by fold | AUC rel | AUC abs | Top / bottom | vs reference | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B | Baseline: deployed model (HistGradientBoosting, depth 2, 100 iterations, 29 features) | -0.0552 ± 0.0261 | -0.085 / -0.070 / -0.015 / -0.050 / -0.056 | 0.476 ± 0.011 | 0.518 ± 0.059 | 43.3% / 50.4% |  | reference |
| F1 | NIFTY 50 index removed from the stock universe (row shows the old universe with the index; compare with B) | -0.0610 ± 0.0273 | -0.092 / -0.073 / -0.022 / -0.047 / -0.071 | 0.473 ± 0.012 | 0.517 ± 0.051 | 42.8% / 51.0% | vs B: +0.0059 IC, 4/5 folds, +0.0034 AUC | fix adopted |
| F2 | Purge on label end dates (row shows the old row-count gap; compare with B) | -0.0552 ± 0.0261 | -0.085 / -0.070 / -0.015 / -0.050 / -0.056 | 0.476 ± 0.011 | 0.518 ± 0.059 | 43.3% / 50.4% | vs B: +0.0000 IC, 0/5 folds, +0.0000 AUC | fix adopted |
| F3 | Full 69-feature candidate pool instead of feature sets chosen with the old validation data | -0.0627 ± 0.0528 | -0.057 / -0.063 / -0.008 / -0.037 / -0.149 | 0.475 ± 0.016 | 0.524 ± 0.059 | 42.7% / 50.6% | vs B: -0.0076 IC, 4/5 folds, -0.0011 AUC | fix adopted |
| E0 | XGBoost with settings equivalent to the previous model | -0.0658 ± 0.0565 | -0.067 / -0.051 / -0.009 / -0.042 / -0.160 | 0.473 ± 0.018 | 0.525 ± 0.057 | 42.4% / 50.9% | vs F3: -0.0031 IC, 1/5 folds, -0.0016 AUC | adopted (decision) |
| E1 | Relative target: beat the universe median over the horizon | -0.0662 ± 0.0764 | -0.086 / -0.059 / -0.033 / +0.027 / -0.181 | 0.469 ± 0.025 | 0.485 ± 0.053 | 43.8% / 50.8% | vs E0: -0.0003 IC, 1/5 folds, -0.0050 AUC | rejected |
| E2 (k=0.25) | Clear-move labels: drop training rows whose forward move is within k × volatility × √h of the label boundary (k=0.25) | -0.0717 ± 0.0563 | -0.069 / -0.061 / -0.018 / -0.044 / -0.166 | 0.472 ± 0.019 | 0.525 ± 0.066 | 42.4% / 51.0% | vs E0: -0.0059 IC, 0/5 folds, -0.0014 AUC | rejected |
| E2 (k=0.5) | Clear-move labels: drop training rows whose forward move is within k × volatility × √h of the label boundary (k=0.5) | -0.0530 ± 0.0261 | -0.069 / -0.061 / -0.012 / -0.045 / -0.078 | 0.482 ± 0.011 | 0.528 ± 0.073 | 43.2% / 50.8% | vs E0: +0.0128 IC, 1/5 folds, +0.0087 AUC | rejected |
| E3 | Cross-sectional ranks of stock-level features | -0.0892 ± 0.0961 | -0.115 / -0.083 / -0.013 / +0.003 / -0.237 | 0.459 ± 0.040 | 0.537 ± 0.092 | 40.9% / 52.1% | vs E0: -0.0234 IC, 1/5 folds, -0.0140 AUC | rejected |
| E5a | More regularisation: shallower trees, larger leaves, subsampling | -0.0790 ± 0.0546 | -0.117 / -0.063 / -0.009 / -0.057 / -0.149 | 0.472 ± 0.017 | 0.529 ± 0.084 | 42.5% / 51.0% | vs E0: -0.0132 IC, 2/5 folds, -0.0018 AUC | rejected |
| E5b (hold-out auc) | Early stopping on a purged hold-out at the end of each training window (hold-out auc) | -0.0412 ± 0.0218 | -0.067 / -0.050 / -0.015 / -0.051 / -0.023 | 0.482 ± 0.011 | 0.529 ± 0.055 | 43.9% / 51.1% | vs E0: +0.0246 IC, 2/5 folds, +0.0090 AUC | rejected |
| E5b (hold-out rank_ic) | Early stopping on a purged hold-out at the end of each training window (hold-out rank_ic) | -0.0583 ± 0.0800 | -0.067 / -0.070 / -0.005 / +0.030 / -0.180 | 0.473 ± 0.024 | 0.532 ± 0.060 | 42.8% / 51.5% | vs E0: +0.0075 IC, 2/5 folds, -0.0009 AUC | rejected |
| E5c | Prune features without stable importance across training sub-periods | -0.0340 ± 0.0492 | -0.019 / -0.119 / -0.032 / +0.000 / +0.000 | 0.483 ± 0.024 | 0.504 ± 0.058 | 43.4% / 49.7% | vs E0: +0.0319 IC, 3/5 folds, +0.0092 AUC | rejected |

**Final configuration: step E0.** xgboost, target absolute, 69 candidate features, clear-move k=None, cross-sectional=False, early stopping=None, pruning=False, params {'max_depth': 2, 'min_child_weight': 1250, 'learning_rate': 0.03, 'n_estimators': 100, 'colsample_bynode': 0.5, 'reg_lambda': 10.0}.
