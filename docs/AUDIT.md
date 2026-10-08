# Leakage and evaluation audit (October 2026)

This audit was done before any modelling change, on the code and artifacts at commit `7c0a1e3`. It covers where the
data comes from, how labels and splits are built, and every place future information could reach features,
training or the reported numbers. The fixes and what each one did to the metrics are at the end.

## Data

* **Source:** Yahoo Finance daily OHLCV through `yfinance` (`auto_adjust=True`, so prices are split- and
  dividend-adjusted with today's adjustment factors), 10 years per ticker, cached in `data/cache/`.
* **Universe:** 182 tickers hard-coded in `trainer.py`, of which 180 have usable history (`LTIM.NS` returns no
  data and `JBCHEPHARM.NS` a single row; both are now removed from the list).
* **Context series:** NIFTY 50, India VIX, USD/INR and three sector indices (Bank, IT, Pharma); other sectors use a
  leave-one-out equal-weight composite of their peers.
* **Date range:** prices from 2016-09-19; after the one-year indicator warm-up, feature rows run from 2017-09-25 to
  2026-09-17. The dataset is now frozen at that end date (`dataset_meta.json`).

## Labels (before)

`features/indicators.py:add_targets`, per stock, for h = 5, 21 and 252 rows of that stock's own history:

    y = 1 if Close[t+h] > 1.005 × Close[t] else 0        (NaN when Close[t+h] is not known yet)

This is an **absolute** direction label with a 0.5% hurdle. Base rates: weekly 46.8%, monthly 53.1%, yearly 65.0%.
About 20% of the label variance at every horizon is common to all stocks on a date (the whole market moving).

## Split (before)

`models/training.py:walk_forward_folds`: the last 15% of labelled dates is the test block; 4 expanding-window
validation folds sit between 45% of the history and the test block; every training block ends h sessions (counted
in rows of the date calendar) before the block it is scored on. There is no random or shuffled split anywhere.

## Findings

| # | Finding | Real problem? | Fix |
| --- | --- | --- | --- |
| 1 | **The deployed models were refit on the test period.** `trainer.py` refit the chosen configuration on *all* labelled rows, test block included, and saved that model. The test AUC shown in the app (0.488 / 0.526 / 0.547) belonged to a different model fitted on pre-test data. Scored on their own "test" period, the deployed models reach AUC **0.732 / 0.705 / 0.579**: in-sample numbers. | Yes: the deployed model could not be evaluated at all | The saved model is now exactly the model scored on the test period, trained only on purged pre-test rows |
| 2 | **Committed models and metrics predate the NIFTY-index fix.** Commit `adbb8f7` removed `^NSEI` from the stock list but nothing was retrained: the committed test sets still contain the index as a "stock" (328 of 60,173 weekly test rows). | Yes (stale artifacts) | Retrained on the 180-stock universe |
| 3 | **Feature selection used the validation folds.** Mutual information (on data spanning the folds) and permutation importance (on the validation blocks themselves) chose the weekly 26-feature set, which was then scored on the same folds; the model was then picked as the best of ~100 noisy fold scores. Weekly's selected set had the best CV AUC (0.528) and the worst test AUC (0.488). | Yes: inflates the CV estimate and biases model choice (not the test number) | Every horizon starts from the same unselected 69-feature pool; any pruning is nested inside each fold's training window |
| 4 | **The purge counted rows, not label end dates.** Labels shift h rows of the stock's own history, so a stock with missing sessions has a label ending 1–3 sessions after the gap assumes: 0.09% / 0.19% / 0.71% of weekly / monthly / yearly labels. | Yes, but tiny | Training rows must have a label window ending before the evaluated block starts, and a gap of at least h sessions |
| 5 | **The app overwrote the training cache.** `build_live_context` re-downloaded sector peers with `use_cache=False`, which also *wrote* them to `data/cache/`. Viewing a ticker silently changed training data (41 of 186 cached files had been refreshed to later dates). | Reproducibility, not leakage | The app no longer writes the cache; the dataset is cut at one `data_end` for every series |
| 6 | **Survivorship bias.** The universe is today's large and mid caps, including later IPOs and recent multi-baggers (RVNL, IRFC, MAZDOCK, BSE, KAYNES), with no delisted names. This inflates "UP" base rates (yearly 65%). | Yes, cannot be fully fixed without point-in-time index membership | Documented; cross-sectional metrics (rank IC, beat-the-median) are less exposed than absolute ones |
| 7 | **Scalers or indicators fitted on the full dataset.** None: tree models need no scaling, every indicator uses trailing windows (EWMs with `adjust=False`), and the point-in-time rebuild audit (`features/leakage.py`) finds no look-ahead in any of the 134 features (139 after this round's short-term features; still none). | No | — |
| 8 | **Same-day and external data.** Features at date D use D's close, which is legitimate for a forecast made at the close. USD/INR is lagged one session (its bar closes after NSE), intraday bars are aligned to the 15:30 close. | No | Fold and test metrics now also report rank IC against the return entered at the next session's open |
| 9 | **Absolute labels let the model memorise dates.** Market features are identical for every stock on a date, so with absolute labels the model can learn which dates were up-weeks (in-sample AUC 0.73 for the weekly model). | Overfitting, not leakage | Tested as the relative-target experiment |

Note: the test period of each horizon (weekly 2025-05-16 → 2026-09-10, monthly 2025-04-25 → 2026-08-19, yearly
2024-07-10 → 2025-09-16) had already been reported in the previous README for six feature setups, so it is not
completely unseen at the project level. No decision in this round used it.

**Correction after measuring:** finding 4's fix is a guarantee rather than a change in practice. With this dataset's
fold and test boundaries the old row-count rule happened not to let any label cross a boundary (0 rows at all 18
boundaries), so the fix changed no metric.

## What each fix did to the metrics

Fold numbers are mean rank IC over the 5 purged walk-forward folds (weekly / monthly / yearly), measured on the
previously deployed configuration; full tables in `reports/experiment_log.md` (rows F1–F3).

| Fix | Before → after | Effect |
| --- | --- | --- |
| 1. Deployed model = evaluated model, trained on purged pre-test rows only | Deployed models' AUC on their own test periods: 0.732 / 0.705 / 0.579 (in-sample) → 0.523 / 0.494 / 0.521 | The number shown in the app now belongs to the model the app runs |
| 2. NIFTY 50 index removed from the stock universe | Fold rank IC +0.0368 / +0.0056 / −0.0610 → +0.0368 / +0.0047 / −0.0552 | Negligible on weekly and monthly; the index no longer appears in test sets or backtests |
| 3. Unselected 69-feature pool instead of feature sets chosen on overlapping validation data | Fold rank IC +0.0368 / +0.0047 / −0.0552 → +0.0317 / −0.0008 / −0.0627 | Fold scores fall: part of the old cross-validation score came from selecting features on the same periods |
| 4. Purge on label end dates | No change (0 rows affected) | Guarantees that training labels never overlap an evaluated period |
| 5. App no longer writes the training cache; dataset frozen at 2026-09-17 | Rebuilt dataset matches the old one except for 0.07% of values (Yahoo revisions in 41 refreshed files) | Training is reproducible from the cache |

On the final test period, the previously deployed configuration re-scored with the purged split gets AUC (UP)
0.485 / 0.535 / 0.546 against the 0.488 / 0.526 / 0.547 it reported: the old test numbers were not inflated. What
the old pipeline got wrong was mostly *what* it measured (pooled absolute AUC, which rewards market timing) and *which*
model it deployed.

## AUCs above 0.60

The brief asked for any weekly or monthly AUC above about 0.60 to be treated as a leakage warning. Across every fold
and test evaluation in this work, the maxima are 0.568 (weekly), 0.602 (monthly) and 0.692 (yearly).

| Fold | Pooled AUC (UP) | Day-average score only | Within-day ranking (mean daily AUC) | Rank IC |
| --- | --- | --- | --- | --- |
| Monthly fold 3 (2022-07 → 2023-06), old configuration with the NIFTY index | 0.602 | 0.640 | 0.471 | −0.051 |
| Monthly fold 3, old configuration | 0.598 | 0.637 | 0.471 | −0.047 |
| Monthly fold 3, final XGBoost model | 0.515 | 0.533 | 0.466 | −0.049 |
| Yearly fold 1 (2020-01 → 2020-09, COVID crash and rebound), final model | 0.624 | 0.613 | 0.436 | −0.067 |

The high pooled AUCs come from the between-day part of the score: in that monthly fold the share of stocks rising
over the next month swung between 19% and 92%, and the old model's average daily score tracked it (correlation 0.53
over about 11 independent months, not statistically significant). Within each day the same models ranked stocks
*worse* than chance. Market features pass the point-in-time rebuild audit, so this is market timing that happened to
work in one short window, not leakage. It is also why the headline metrics are now the cross-sectional ones.
