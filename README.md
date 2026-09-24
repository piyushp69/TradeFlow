# TradeFlow

Multi-horizon trend forecasting for NSE stocks: a Streamlit dashboard on top of gradient-boosting models
trained on ~180 Indian stocks with technical, market-context and multi-timeframe features.

The models rank stocks against each other. A forecast is **relative**: UP means the stock ranks in the top
40% of the universe by P(UP), DOWN means the bottom 40%, NEUTRAL is in between. In a falling market even a
top-ranked stock can decline.

## Running

```bash
pip install -r requirements.txt
python trainer.py          # builds the dataset, trains, backtests and writes reports (~45 min)
streamlit run app.py       # dashboard on http://localhost:8501
```

`trainer.py` flags: `--rebuild-data` (rebuild the cached feature dataset), `--no-cache` (re-download prices),
`--horizons weekly monthly` (subset), `--no-intraday`, `--skip-1h-study`.

## Layout

| Path | Purpose |
| --- | --- |
| `data/fetcher.py` | Daily and intraday downloads, cleaning, on-disk cache, named loaders per source |
| `data/sectors.py` | Sector membership and which sectors have an official index |
| `features/technical.py` | Timeframe-agnostic indicators (RSI, MACD, ATR, EMA, momentum, regimes) |
| `features/indicators.py` | Original 29 baseline features and the labels (`add_targets`) |
| `features/market.py` | NIFTY, sector, India VIX, USD/INR and relative-strength features + alignment rules |
| `features/multitimeframe.py` | 5m / 15m / 1h / daily features aligned to each daily prediction row |
| `features/pipeline.py` | `build_feature_frame`, used by both training and the app |
| `features/leakage.py` | Point-in-time rebuild audit, target-leakage and alignment checks |
| `features/selection.py` | Correlation, mutual information, gain/split, permutation, SHAP, final selection |
| `models/training.py` | Splitting, walk-forward CV, model factories (HistGradientBoosting + LightGBM), metrics |
| `models/predictor.py` | Loads bundles, produces direction / percentile / SHAP explanation |
| `models/inference.py` | Live data assembly for one ticker (same feature builder as training) |
| `models/reports.py` | Reads `reports/` for the dashboard |
| `backtest/engine.py` | Cross-sectional long-only backtest with costs, slippage and a benchmark |
| `trainer.py` | Orchestration: dataset -> audit -> experiments -> selection -> final models -> backtest |
| `ui/`, `app.py` | Streamlit dashboard |

## Evaluation protocol

* The most recent 15% of dates is the **test block**. It is never used for feature selection, tuning or
  model choice, and is scored once per horizon.
* Model choice uses 4 **walk-forward folds** (expanding training window) inside the earlier data.
* Every training block ends `horizon_days` before the block it is evaluated on (**embargo**), so labels that
  look ahead cannot overlap the evaluation rows. The trainer asserts this.
* Metrics: ROC AUC, directional accuracy, precision/recall/F1 (same-day median split), Brier score, and the
  UP rate of the top vs bottom ranked quintile. MAE/RMSE/R² on price do not apply: the model predicts a
  direction, not a price.

## Leakage controls

A feature for date D may only use information available at the 15:30 IST close of D.

* `features/leakage.py` rebuilds every feature using only data that existed at several cutoff dates and
  compares it with the full-history build. Any difference fails the audit (`reports/leakage_audit.json`).
* USD/INR is lagged one session: its daily bar closes after the NSE close. NIFTY, sector indices, sector
  peers and India VIX trade in the same session and are joined on the exact date.
* Intraday features for date D come from the last bar of session D that ended by 15:30; a missing session
  is NaN, never a stale bar from an earlier day.
* Missing external data stays NaN instead of being forward-filled from stale values.
* Sector peer composites exclude the stock itself (leave-one-out).

## Data availability

Yahoo Finance serves ~60 days of 5m/15m bars and ~730 days of 1h bars, so intraday features cannot be
computed across the 10-year training history. The 5m/15m timeframes are therefore dashboard-only, and the
value of 1h features is measured in a separate study over the window where they exist
(`reports/intraday_1h_study_*.csv`). Only Bank Nifty, Nifty IT and Nifty Pharma have usable sector-index
history; other sectors use a leave-one-out equal-weight composite of their peers.
