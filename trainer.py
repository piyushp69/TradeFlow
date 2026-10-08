"""Training entry point.

    python trainer.py                      train and evaluate every horizon with the frozen configuration
    python trainer.py --rebuild-data       rebuild the cached feature dataset first (from the price cache)
    python trainer.py --horizons weekly    a subset of horizons (the others keep their metrics.json entries)

Steps:
    dataset (frozen at one end date) -> leakage audit -> for each horizon:
        purged walk-forward folds for the frozen configuration (models/config.py FINAL_SPECS)
        -> final model fit on every row before the final test period (purged: no label reaches into it)
        -> one evaluation on the final test period -> edge verdict -> backtest
        -> the previously deployed model through the same protocol, for the before/after comparison
    -> models/saved_models/model_<horizon>.pkl, models/saved_models/metrics.json and reports/*

The configuration was chosen by experiments.py on walk-forward folds only, before this script scored the final test
period. The saved model is exactly the model that was scored on the test period: it never saw test-period data.
"""
import argparse
import json
import logging
import os
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import roc_auc_score

from backtest.engine import BacktestConfig, run_backtest
from data.fetcher import (CACHE_DIR, INDIA_VIX_TICKER, NIFTY_TICKER, USDINR_TICKER, get_many_intraday,
                          get_many_stocks)
from data.sectors import SECTOR_INDEX_TICKERS, get_sector
from features.indicators import HORIZONS, add_targets
from features.leakage import (audit_point_in_time, check_intraday_alignment, check_suspicious_features,
                              check_target_leakage)
from features.market import MarketContext, SectorPanel
from features.pipeline import FEATURE_GROUPS, PRICE_LEVEL_FEATURES, build_feature_frame, requirements_for
from models.evaluation import (EDGE_MIN_POSITIVE_FOLDS, EDGE_MIN_TSTAT, FIRST_VAL_FRACTION, N_FOLDS, TEST_FRACTION,
                               daily_rank_ic, edge_verdict, purged_walk_forward)
from features.labels import label_column
from models.config import BASELINE_SPECS, FINAL_SPECS, matches
from models.panel import prepare_panel
from models.training import RANDOM_STATE, detect_device, to_cpu
from models.walkforward import kept_feature_share, score_split, walk_forward

# Tradable stocks only: NIFTY 50 is loaded separately as market context (load_context_sources).
# LTIM.NS and JBCHEPHARM.NS were removed in Oct 2026: Yahoo Finance returns no usable history for either.
STOCKS = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS', 'SBIN.NS',
    'BHARTIARTL.NS', 'ITC.NS', 'HINDUNILVR.NS', 'LT.NS', 'BAJFINANCE.NS', 'AXISBANK.NS',
    'ASIANPAINT.NS', 'MARUTI.NS', 'SUNPHARMA.NS', 'TITAN.NS', 'HCLTECH.NS', 'TATASTEEL.NS',
    'NTPC.NS', 'POWERGRID.NS', 'KOTAKBANK.NS', 'ADANIENT.NS', 'COALINDIA.NS', 'BAJAJFINSV.NS',
    'ULTRACEMCO.NS', 'JSWSTEEL.NS', 'GRASIM.NS', 'M&M.NS', 'INDUSINDBK.NS', 'HINDALCO.NS',
    'NESTLEIND.NS', 'ADANIPORTS.NS', 'ONGC.NS', 'TECHM.NS', 'WIPRO.NS', 'BRITANNIA.NS',
    'EICHERMOT.NS', 'BPCL.NS', 'DIVISLAB.NS', 'CIPLA.NS', 'APOLLOHOSP.NS', 'TATACONSUM.NS',
    'HEROMOTOCO.NS', 'DRREDDY.NS', 'BAJAJ-AUTO.NS', 'YESBANK.NS', 'IDEA.NS', 'UPL.NS',
    'BANDHANBNK.NS', 'DLF.NS', 'VBL.NS', 'PIDILITIND.NS', 'SIEMENS.NS', 'HAL.NS', 'BEL.NS',
    'ABB.NS', 'GAIL.NS', 'PNB.NS', 'BANKBARODA.NS', 'CANBK.NS', 'TRENT.NS', 'GODREJCP.NS',
    'DABUR.NS', 'CHOLAFIN.NS', 'TVSMOTOR.NS', 'POLYCAB.NS', 'HAVELLS.NS',
    'SRF.NS', 'COLPAL.NS', 'SHREECEM.NS', 'AMBUJACEM.NS', 'BHEL.NS', 'RECLTD.NS', 'PFC.NS',
    'IRFC.NS', 'RVNL.NS', 'MAZDOCK.NS', 'COCHINSHIP.NS', 'IRCTC.NS', 'CONCOR.NS',
    'TATACOMM.NS', 'LUPIN.NS', 'AUBANK.NS', 'IDFCFIRSTB.NS', 'FEDERALBNK.NS', 'RBLBANK.NS',
    'ABCAPITAL.NS', 'MFSL.NS', 'LICHSGFIN.NS', 'MPHASIS.NS', 'COFORGE.NS', 'PERSISTENT.NS',
    'ZENSARTECH.NS', 'KPITTECH.NS', 'TATAELXSI.NS', 'OIL.NS', 'HINDPETRO.NS', 'IOC.NS',
    'MRPL.NS', 'CHENNPETRO.NS', 'PETRONET.NS', 'TATAPOWER.NS', 'NHPC.NS', 'SJVN.NS',
    'TORNTPOWER.NS', 'CESC.NS', 'JINDALSTEL.NS', 'SAIL.NS', 'NMDC.NS', 'NATIONALUM.NS',
    'VEDL.NS', 'ASHOKLEY.NS', 'ESCORTS.NS', 'BALKRISIND.NS', 'MRF.NS', 'APOLLOTYRE.NS',
    'TATACHEM.NS', 'DEEPAKNTR.NS', 'AARTIIND.NS', 'PIIND.NS', 'COROMANDEL.NS', 'GNFC.NS',
    'JUBLFOOD.NS', 'PAGEIND.NS', 'METROBRAND.NS', 'BATAINDIA.NS', 'RELAXO.NS', 'ABFRL.NS',
    'NYKAA.NS', 'PAYTM.NS', 'DELHIVERY.NS', 'INDIAMART.NS', 'POLICYBZR.NS', 'CARBORUNIV.NS',
    'CUMMINSIND.NS', 'SKFINDIA.NS', 'TIMKEN.NS', 'VOLTAS.NS', 'BLUESTARCO.NS', 'DIXON.NS',
    'AMBER.NS', 'HINDZINC.NS', 'GLENMARK.NS', 'ALKEM.NS', 'ASTRAL.NS', 'OBEROIRLTY.NS',
    'PHOENIXLTD.NS', 'GODREJPROP.NS', 'BRIGADE.NS', 'PRESTIGE.NS', 'UNIONBANK.NS',
    'MAHABANK.NS', 'J&KBANK.NS', 'KARURVYSYA.NS', 'SOUTHBANK.NS', 'POONAWALLA.NS',
    'LTF.NS', 'M&MFIN.NS', 'MANAPPURAM.NS', 'ANGELONE.NS', 'MCX.NS', 'BSE.NS', 'CDSL.NS',
    'HUDCO.NS', 'NBCC.NS', 'KEC.NS', 'KPIL.NS', 'IRB.NS', 'RAYMOND.NS', 'KAYNES.NS',
    'SYNGENE.NS', 'LAURUSLABS.NS', 'ZYDUSLIFE.NS', 'NATCOPHARM.NS',
    'NAM-INDIA.NS', 'HDFCAMC.NS', 'RADICO.NS', 'TIINDIA.NS', 'KEI.NS', 'EXIDEIND.NS'
]

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT, "models", "saved_models")
METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics.json")
REPORTS_DIR = os.path.join(ROOT, "reports")
DATASET_CACHE = os.path.join(CACHE_DIR, "dataset.pkl")
DATASET_META = os.path.join(CACHE_DIR, "dataset_meta.json")
MIN_ROWS_PER_STOCK = 500
SHAP_SAMPLE = 20_000
SHAP_PLOT_ROWS = 4_000
LEAKAGE_ALERT_AUC = 0.60

TARGET_TEXT = {
    "absolute": "rises more than 0.5% over the horizon",
    "relative": "beats the median stock of the universe over the horizon",
}

logger = logging.getLogger("tradeflow.trainer")


def setup_logging(verbose=True):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    handlers = [logging.FileHandler(os.path.join(REPORTS_DIR, "training.log"), mode="w", encoding="utf-8")]
    if verbose:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=handlers, force=True)
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)


# Data

def resolve_data_end(prices, data_end=None):
    """The last date of the frozen dataset: the given date, else the last NIFTY 50 session in the cache."""
    if data_end is not None:
        return pd.Timestamp(data_end)
    return pd.Timestamp(prices[NIFTY_TICKER]["Date"].max())


def truncate(df, data_end, column="Date"):
    """Bars up to and including data_end. Cached files are refreshed at different times; this aligns them."""
    if df is None:
        return None
    return df[df[column] <= data_end].reset_index(drop=True)


def load_context_sources(use_cache=True, data_end=None):
    """Every external series needed for market features, plus daily prices for the whole universe.

    Every series is cut at the same `data_end`, so a dataset build is reproducible from the cache.
    Returns (sources, prices, data_end).
    """
    tickers = list(dict.fromkeys(STOCKS + [NIFTY_TICKER, INDIA_VIX_TICKER, USDINR_TICKER]
                                 + list(SECTOR_INDEX_TICKERS.values())))
    prices = get_many_stocks(tickers, use_cache=use_cache)
    data_end = resolve_data_end(prices, data_end)
    prices = {t: truncate(df, data_end) for t, df in prices.items()}
    sources = {
        "nifty": prices.get(NIFTY_TICKER),
        "vix": prices.get(INDIA_VIX_TICKER),
        "usdinr": prices.get(USDINR_TICKER),
        "prices": {t: df for t, df in prices.items() if t in STOCKS},
        "index_prices": {t: prices[t] for t in SECTOR_INDEX_TICKERS.values() if t in prices},
    }
    for name in ("nifty", "vix", "usdinr"):
        series = sources[name]
        logger.info("Context source %-8s rows=%s range=%s", name, len(series) if series is not None else 0,
                    f"{series['Date'].min():%Y-%m-%d}..{series['Date'].max():%Y-%m-%d}" if series is not None else "-")
    logger.info("Sector indices with history: %s | data frozen at %s", sorted(sources["index_prices"]),
                f"{data_end:%Y-%m-%d}")
    return sources, prices, data_end


def build_context(sources):
    return MarketContext(nifty=sources["nifty"], vix=sources["vix"], usdinr=sources["usdinr"],
                         sector_panel=SectorPanel(sources["prices"], sources["index_prices"]))


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def build_dataset(use_cache=True, use_intraday=True, rebuild=False, data_end=None):
    """Feature frame for every stock, concatenated, plus its metadata (date range, universe).

    Cached on disk between runs. The cache is rebuilt when it has no metadata (built by an older version) or
    was frozen at a different `data_end` than the one requested.
    """
    meta = _read_json(DATASET_META)
    wanted_end = None if data_end is None else f"{pd.Timestamp(data_end):%Y-%m-%d}"
    if not rebuild and os.path.exists(DATASET_CACHE) and meta and wanted_end in (None, meta.get("data_end")):
        data = pd.read_pickle(DATASET_CACHE)
        # A cache built from an older universe may hold tickers that have since been removed from STOCKS
        data = data[data["Stock"].isin(STOCKS)].reset_index(drop=True)
        logger.info("Loaded cached dataset: %d rows x %d columns, data frozen at %s", len(data), data.shape[1],
                    meta["data_end"])
        return data, meta

    sources, prices, data_end = load_context_sources(use_cache, data_end)
    context = build_context(sources)
    intraday = get_many_intraday(STOCKS, "1h") if use_intraday else {}
    intraday = {t: truncate(bars, data_end, "SessionDate") for t, bars in intraday.items()}

    frames, skipped = [], []
    start = time.time()
    for index, stock in enumerate(STOCKS, 1):
        df = prices.get(stock)
        if df is None or len(df) < MIN_ROWS_PER_STOCK:
            skipped.append(stock)
            continue
        bars = {"1h": intraday[stock]} if stock in intraday else None
        frame = build_feature_frame(add_targets(df), context, bars)
        if frame is not None:
            frame["sector"] = get_sector(stock) or "NONE"
            frames.append(frame)
        if index % 40 == 0:
            logger.info("   processed %d/%d stocks", index, len(STOCKS))

    if not frames:
        raise RuntimeError("no stock had enough price history to build features")

    data = pd.concat(frames, ignore_index=True)
    data = data.drop(columns=[c for c in data.columns if c.endswith("_bar_end")])
    float_cols = data.select_dtypes("float64").columns
    data[float_cols] = data[float_cols].astype("float32")

    logger.info("Dataset: %d rows x %d columns from %d stocks (skipped %d: %s) in %.0fs",
                len(data), data.shape[1], len(frames), len(skipped), skipped, time.time() - start)
    logger.info("Date range: %s .. %s", data["Date"].min().date(), data["Date"].max().date())
    for group, cols in FEATURE_GROUPS.items():
        logger.info("   group %-15s %3d features, coverage %.1f%%", group, len(cols),
                    data[cols].notna().mean().mean() * 100)
    data.to_pickle(DATASET_CACHE)
    meta = {"source": "Yahoo Finance daily OHLCV via yfinance (auto_adjust=True: split and dividend adjusted)",
            "price_history_start": f"{min(df['Date'].min() for df in sources['prices'].values() if df is not None):%Y-%m-%d}",
            "data_start": f"{data['Date'].min():%Y-%m-%d}", "data_end": f"{data_end:%Y-%m-%d}",
            "built_at": datetime.now().isoformat(timespec="seconds"), "stocks": int(data["Stock"].nunique()),
            "rows": int(len(data)), "skipped_stocks": skipped}
    with open(DATASET_META, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return data, meta


# Leakage audit

def run_leakage_audit(sources, data_end, sample_stocks=("TCS.NS", "MARUTI.NS", "HDFCBANK.NS")):
    """Rebuild every feature point-in-time on sample stocks and check intraday bar alignment."""
    logger.info("Leakage audit: point-in-time feature rebuild")
    audited = sum(FEATURE_GROUPS.values(), []) + PRICE_LEVEL_FEATURES
    mismatches, checked, alignment_problems = [], [], {}
    for stock in sample_stocks:
        raw = sources["prices"].get(stock)
        if raw is None:
            continue
        bars_path = os.path.join(CACHE_DIR, f"{stock}__1h.pkl")
        intraday = ({"1h": truncate(pd.read_pickle(bars_path), data_end, "SessionDate")}
                    if os.path.exists(bars_path) else None)
        result = audit_point_in_time(build_feature_frame, raw, sources, intraday, features=audited)
        checked.append(stock)
        if not result.empty:
            mismatches.append(result.assign(stock=stock))
        alignment = check_intraday_alignment(build_feature_frame(raw, build_context(sources), intraday))
        if alignment:
            alignment_problems[stock] = alignment
            logger.error("   intraday alignment problem for %s: %s", stock, alignment)
    failures = pd.concat(mismatches) if mismatches else pd.DataFrame()
    leaky = sorted(failures["feature"].unique()) if not failures.empty else []
    logger.info("   checked %d features on %s: %d look-ahead failures", len(audited), checked, len(leaky))
    return {"stocks_checked": checked, "features_checked": len(audited), "look_ahead_failures": leaky,
            "intraday_alignment_problems": alignment_problems}


def check_splits(panel, horizon, folds, test):
    """Re-check the purge on the actual rows before training (raises on any violation)."""
    days = HORIZONS[horizon]
    dates, label_end = panel["Date"], panel[f"label_end_rel_{horizon}"]
    test_start = dates.iloc[test.eval].min()
    for split in folds + [test]:
        if not (label_end.iloc[split.train].max() < dates.iloc[split.eval].min() and split.gap_sessions >= days):
            raise RuntimeError(f"{horizon} {split.name}: a training label overlaps its evaluation period")
    for split in folds:
        if not label_end.iloc[split.eval].max() < test_start:
            raise RuntimeError(f"{horizon} {split.name}: a validation label reaches into the test period")
    return {"test_start": f"{test_start:%Y-%m-%d}", "min_gap_sessions": min(s.gap_sessions for s in folds + [test]),
            "required_gap_sessions": days, "last_training_label_end": test.max_train_label_end, "ok": True}


def check_features(panel, train_idx, features, label_col):
    """Fail on target-like features, or on a single feature that predicts the label implausibly well."""
    train = panel.iloc[train_idx]
    label_cols = [c for c in panel.columns if c.startswith(("y_abs_", "y_rel_", "Target_"))]
    problems = check_target_leakage(train, list(features), label_cols)
    suspicious = check_suspicious_features(train[list(features)], train[label_col].astype(int))
    if problems or suspicious:
        raise RuntimeError(f"feature leakage check failed: {problems} {suspicious}")
    return {"target_like_features": [], "suspicious_single_feature_auc": {}, "checked": len(features)}


# Training

def reference_cross_section(fitted, panel):
    """Model output for every stock on the latest date: the app ranks a ticker against this cross-section."""
    latest = panel.groupby("Stock").tail(1)
    latest = latest[latest["Date"] >= panel["Date"].max() - pd.Timedelta(days=10)]
    return np.sort(fitted.predict(latest)), latest["Date"].max()


def feature_report(fitted, spec, panel, train_idx, horizon, fold_metrics):
    """Per-feature importance of the final model, measured on a sample of its own training rows."""
    rng = np.random.default_rng(RANDOM_STATE)
    sample = np.sort(rng.choice(train_idx, size=min(SHAP_SAMPLE, len(train_idx)), replace=False))
    rows = panel.iloc[sample]
    values = rows[fitted.columns].set_axis(fitted.features, axis=1)
    booster = fitted.model.get_booster()
    contrib = booster.predict(xgboost.DMatrix(rows[fitted.columns]), pred_contribs=True)[:, :-1]
    shap = pd.DataFrame(contrib, columns=fitted.features, index=rows.index)
    gain = booster.get_score(importance_type="total_gain")
    gain = pd.Series({f: gain.get(c, 0.0) for f, c in zip(fitted.features, fitted.columns)})
    train = panel.iloc[train_idx]
    single_ic = {f: daily_rank_ic(train["Date"], train[f], train[f"fwd_ret_{horizon}"]).mean()
                 for f in spec.features}
    kept = kept_feature_share(fold_metrics, list(spec.features))

    report = pd.DataFrame({"feature": list(spec.features)})
    report["used"] = report["feature"].isin(fitted.features)
    report["gain_share"] = report["feature"].map(gain / max(gain.sum(), 1e-12)).fillna(0.0)
    report["shap_importance"] = report["feature"].map(shap.abs().mean())
    report["shap_mean_signed"] = report["feature"].map(shap.mean())
    report["single_feature_rank_ic"] = report["feature"].map(single_ic)
    report["kept_in_folds"] = report["feature"].map(kept)
    report = report.sort_values(["used", "shap_importance"], ascending=[False, False]).reset_index(drop=True)

    plot_rows = shap.sample(min(SHAP_PLOT_ROWS, len(shap)), random_state=RANDOM_STATE).index
    shap_sample = {"shap": shap.loc[plot_rows], "values": values.loc[plot_rows]}
    return report, shap_sample, values.corr(method="spearman")


def run_backtests(panel, horizon, predictions):
    prices = panel[["Date", "Stock", "Open", "Close"]]
    config = BacktestConfig(horizon_days=HORIZONS[horizon])
    results = {name: run_backtest(preds, prices, config) for name, preds in predictions.items() if preds is not None}
    for name, result in results.items():
        logger.info("   [backtest %s] %d rebalances | return %s | benchmark %s | sharpe %s", name,
                    result["metrics"].get("periods", 0), result["metrics"].get("cumulative_return"),
                    result["benchmark"].get("cumulative_return"), result["metrics"].get("sharpe_ratio"))
    return results


def previous_model_on_test(bundle_path, panel, test, horizon):
    """AUC of the model deployed before this pipeline, on the current test period.

    That model was refit on every labelled row, test period included, so this AUC is in-sample. It is recorded to
    show what refitting on the test period did to the apparent score, never as a performance claim.
    """
    try:
        bundle = joblib.load(bundle_path)
    except Exception as exc:
        logger.info("   previous bundle not readable (%s)", exc)
        return None
    if not isinstance(bundle, dict) or bundle.get("model_type") not in ("lgbm", "hgb") or "model" not in bundle:
        return None
    rows = panel.iloc[test.eval]
    proba = bundle["model"].predict_proba(rows[bundle["features"]])[:, 1]
    return {"model": f"{bundle.get('experiment')} ({bundle.get('model_type')})",
            "auc_abs_on_test_period": round(float(roc_auc_score(rows[f"y_abs_{horizon}"].astype(int), proba)), 4),
            "note": "refit on every labelled row including this test period, so this AUC is in-sample"}


def originally_reported(previous, horizon):
    """Test metrics of the previously deployed model as they were reported before this pipeline."""
    if not previous:
        return None
    if previous.get("schema_version") == 2:
        baseline = previous.get("horizons", {}).get(horizon, {}).get("baseline") or {}
        return baseline.get("originally_reported")
    entry = previous.get(horizon)
    if not isinstance(entry, dict) or "test_metrics" not in entry:
        return None
    test = entry["test_metrics"]
    return {"model": f"{entry.get('experiment')} ({entry.get('model_type')})", "cv_auc": entry.get("cv_auc"),
            "test_auc": test.get("roc_auc"), "top_quintile_up_rate": test.get("up_rate_top_quintile"),
            "bottom_quintile_up_rate": test.get("up_rate_bottom_quintile"), "test_period": entry.get("test_period"),
            "test_rows": test.get("rows"),
            "caveats": "4 walk-forward folds without a label-end purge; the test rows included the NIFTY 50 index "
                       "as a stock; the weekly feature set was selected using the validation folds"}


def evaluate_baseline(panel, horizon, folds, test, device):
    """The previously deployed configuration through the same folds and test period (needs lightgbm)."""
    spec = BASELINE_SPECS[horizon]
    try:
        fold_metrics, fold_summary, _ = walk_forward(spec, panel, folds, horizon, device)
        _, test_metrics, _, predictions = score_split(spec, panel, test, horizon, device)
    except ImportError as exc:
        logger.warning("   baseline skipped: %s (pip install -r requirements-dev.txt)", exc)
        return None, None
    logger.info("   [baseline] test AUC rel %.4f abs %.4f | rank IC %+.4f", test_metrics["auc_rel"],
                test_metrics["auc_abs"], test_metrics["rank_ic"])
    return {"description": spec.description, "library": spec.library, "target": spec.target,
            "n_features": len(spec.features), "folds": fold_metrics, "fold_summary": fold_summary,
            "final_test": test_metrics}, predictions


def leakage_alerts(horizon, fold_metrics, test_metrics):
    """Weekly/monthly direction AUCs above 0.60 are implausible for daily price data: flag them for review."""
    if horizon == "yearly":
        return []
    alerts = []
    for name, metrics in [(m["fold"], m) for m in fold_metrics] + [("test", test_metrics)]:
        for key in ("auc_rel", "auc_abs"):
            if metrics[key] is not None and metrics[key] > LEAKAGE_ALERT_AUC:
                alerts.append(f"{name} {key}={metrics[key]:.3f}")
    for alert in alerts:
        logger.error("   [%s] AUC above %.2f (%s): investigate for leakage before trusting it", horizon,
                     LEAKAGE_ALERT_AUC, alert)
    return alerts


def train_horizon(panel, horizon, spec, step, device, previous, reports, with_baseline=True):
    days = HORIZONS[horizon]
    folds, test = purged_walk_forward(panel["Date"], panel[f"label_end_rel_{horizon}"], days, include_test=True)
    split_checks = check_splits(panel, horizon, folds, test)
    logger.info("=" * 100)
    logger.info("[%s] %d sessions ahead | target: %s | %d candidate features", horizon, days,
                TARGET_TEXT[spec.target], len(spec.features))
    for split in folds + [test]:
        logger.info("   %-5s train %s..%s (%d rows) | gap %d sessions | evaluate %s..%s (%d rows)", split.name,
                    *split.train_period, len(split.train), split.gap_sessions, *split.eval_period, len(split.eval))

    # 1. the frozen configuration on the walk-forward folds
    fold_metrics, fold_summary, fold_ic = walk_forward(spec, panel, folds, horizon, device)
    # 2. the final model: fit on every purged pre-test row, scored once on the test period
    feature_checks = check_features(panel, test.train, spec.features, label_column(spec.target, horizon))
    fitted, test_metrics, test_ic, predictions = score_split(spec, panel, test, horizon, device)
    edge = edge_verdict(fold_metrics, test_metrics, pd.concat([fold_ic, test_ic]), days)
    logger.info("   [test] AUC rel %.4f abs %.4f | rank IC %+.4f (t=%.2f) | top/bottom %.3f/%.3f | edge: %s",
                test_metrics["auc_rel"], test_metrics["auc_abs"], test_metrics["rank_ic"],
                test_metrics["rank_ic_tstat"], test_metrics["top_hit_rel"], test_metrics["bottom_hit_rel"],
                "reliable" if edge["reliable"] else "none (" + "; ".join(edge["reasons"]) + ")")
    alerts = leakage_alerts(horizon, fold_metrics, test_metrics)

    # 3. the previous model through the same protocol (and what the deployed copy scored after seeing the test)
    bundle_path = os.path.join(OUTPUT_DIR, f"model_{horizon}.pkl")
    previous_deployed = (previous_model_on_test(bundle_path, panel, test, horizon)
                         if os.path.exists(bundle_path) else None)
    baseline, baseline_predictions = (evaluate_baseline(panel, horizon, folds, test, device) if with_baseline
                                      else (None, None))
    if baseline is not None:
        carried = ((previous or {}).get("horizons", {}).get(horizon, {}).get("baseline") or {}
                   if (previous or {}).get("schema_version") == 2 else {})
        baseline["originally_reported"] = originally_reported(previous, horizon)
        # Only measurable while the old bundle is still on disk; later runs carry the first measurement over
        baseline["previous_deployed_model"] = previous_deployed or carried.get("previous_deployed_model")

    backtests = run_backtests(panel, horizon, {"model": predictions, "baseline": baseline_predictions})
    to_cpu(fitted)
    report, shap_sample, corr = feature_report(fitted, spec, panel, test.train, horizon, fold_metrics)
    probs, reference_date = reference_cross_section(fitted, panel)

    now = datetime.now().isoformat(timespec="seconds")
    bundle = {
        "model": fitted.model, "features": fitted.columns, "raw_features": fitted.features,
        "cross_sectional": spec.cross_sectional, "model_type": "xgboost", "library_version": xgboost.__version__,
        "experiment": step, "target": spec.target, "target_description": TARGET_TEXT[spec.target],
        "horizon_days": days,
        "feature_groups": [g for g, cols in FEATURE_GROUPS.items() if set(cols) & set(fitted.features)],
        "requirements": sorted(requirements_for(fitted.features)),
        "reference_probs": probs.tolist(), "reference_date": f"{reference_date:%Y-%m-%d}",
        "threshold": float(np.median(probs)), "train_period": list(test.train_period),
        "test_period": list(test.eval_period), "trained_at": now, "random_state": RANDOM_STATE,
    }
    entry = {
        "horizon_days": days, "target": spec.target, "target_description": TARGET_TEXT[spec.target],
        "model": {"library": "xgboost", "version": xgboost.__version__, "experiment_step": step,
                  "params": spec.params, "n_trees": fitted.info["n_trees"], "features": fitted.features,
                  "n_features": len(fitted.features), "candidate_features": len(spec.features),
                  "clear_move_k": spec.clear_move_k, "cross_sectional": spec.cross_sectional,
                  "early_stopping": spec.early_stopping, "prune": spec.prune,
                  "train_period": list(test.train_period), "train_rows": fitted.info["train_rows"],
                  "trained_at": now},
        "split": {"folds": [s.describe() for s in folds], "test": test.describe(), "checks": split_checks},
        "feature_checks": feature_checks,
        "folds": fold_metrics, "fold_summary": fold_summary,
        "final_test": test_metrics, "edge": edge, "leakage_alerts": alerts,
        "baseline": baseline,
        "backtest": {name: {"metrics": r["metrics"], "benchmark": r["benchmark"], "config": r["config"]}
                     for name, r in backtests.items()},
    }
    reports[f"backtest_{horizon}"] = entry["backtest"]
    reports[f"equity_curve_{horizon}"] = backtests["model"]["equity_curve"]
    if "baseline" in backtests:
        reports[f"equity_curve_baseline_{horizon}"] = backtests["baseline"]["equity_curve"]
    reports[f"feature_importance_{horizon}"] = report
    reports[f"shap_sample_{horizon}"] = shap_sample
    reports[f"correlation_{horizon}"] = corr
    return bundle, entry


# Output

def _clean(value):
    """JSON-safe copy: NaN/inf become null, numpy scalars become Python numbers."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def save_reports(reports):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    for key, value in reports.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(os.path.join(REPORTS_DIR, f"{key}.csv"), index=key.startswith("correlation"))
        elif key.startswith("shap_sample"):
            joblib.dump(value, os.path.join(REPORTS_DIR, f"{key}.pkl"), compress=3)
        else:
            with open(os.path.join(REPORTS_DIR, f"{key}.json"), "w", encoding="utf-8") as fh:
                json.dump(_clean(value), fh, indent=2)
    logger.info("Reports written to %s (%d files)", REPORTS_DIR, len(reports))


def train(use_cache=True, rebuild_data=False, horizons=None, use_intraday=True, data_end=None, cpu=False,
          with_baseline=True):
    setup_logging()
    np.random.seed(RANDOM_STATE)
    start = time.time()
    device = detect_device("cpu" if cpu else "cuda")
    horizons = [h for h in HORIZONS if not horizons or h in horizons]
    missing = [h for h in horizons if h not in FINAL_SPECS]
    if missing:
        raise SystemExit(f"No frozen configuration for {missing}: run experiments.py and freeze the result in "
                         "models/config.py FINAL_SPECS first.")
    logger.info("TradeFlow training run | xgboost=%s | device=%s | seed=%d", xgboost.__version__, device,
                RANDOM_STATE)

    data, meta = build_dataset(use_cache=use_cache, use_intraday=use_intraday, rebuild=rebuild_data,
                               data_end=data_end)
    sources, _, frozen_end = load_context_sources(use_cache=True, data_end=meta["data_end"])
    audit = run_leakage_audit(sources, frozen_end)
    used = {f for h in horizons for f in FINAL_SPECS[h].features}
    if used & set(audit["look_ahead_failures"]) or audit["intraday_alignment_problems"]:
        raise RuntimeError(f"leakage audit failed: {audit}")

    rank_features = sorted({f for h in horizons if FINAL_SPECS[h].cross_sectional for f in FINAL_SPECS[h].features})
    panel = prepare_panel(data, rank_features=rank_features)
    previous = _read_json(METRICS_PATH)
    steps = (_read_json(os.path.join(REPORTS_DIR, "experiments.json")) or {}).get("horizons", {})
    for horizon in horizons:
        recorded = steps.get(horizon, {}).get("final_spec")
        if recorded is not None and not matches(FINAL_SPECS[horizon], recorded):
            raise SystemExit(f"FINAL_SPECS['{horizon}'] in models/config.py differs from the configuration that "
                             "experiments.py chose; re-run the experiments or fix the config before training.")
    reports = {"leakage_audit": audit}

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    entries, bundles = {}, {}
    for horizon in horizons:
        step = steps.get(horizon, {}).get("final_step", "frozen")
        bundles[horizon], entries[horizon] = train_horizon(panel, horizon, FINAL_SPECS[horizon], step, device,
                                                           previous, reports, with_baseline)
    for horizon, bundle in bundles.items():
        joblib.dump(bundle, os.path.join(OUTPUT_DIR, f"model_{horizon}.pkl"), compress=3)
        logger.info("   [%s] saved model bundle (%s)", horizon, bundle["experiment"])

    kept = previous.get("horizons", {}) if previous and previous.get("schema_version") == 2 else {}
    metrics = {
        "schema_version": 2,
        "training_date": datetime.now().isoformat(timespec="seconds"),
        "data": {k: meta.get(k) for k in ("source", "price_history_start", "data_start", "data_end", "stocks",
                                          "rows")},
        "protocol": {
            "validation": f"purged walk-forward, {N_FOLDS} time-ordered folds, expanding training window",
            "purge": "training rows are dated at least one label horizon before the evaluated block, and their "
                     "label windows end before it starts",
            "final_test": f"most recent {TEST_FRACTION:.0%} of labelled sessions per horizon, scored once after "
                          "the configuration was frozen",
            "first_validation_fraction": FIRST_VAL_FRACTION,
            "edge_rule": f"rank IC > 0 on average and in >= {EDGE_MIN_POSITIVE_FOLDS} of {N_FOLDS} folds, > 0 on "
                         f"the final test, and Newey-West t-stat >= {EDGE_MIN_TSTAT:.0f} over all out-of-sample days",
        },
        "device": device, "random_state": RANDOM_STATE, "xgboost_version": xgboost.__version__,
        "leakage_audit": audit,
        "horizons": {**kept, **entries},
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(_clean(metrics), fh, indent=2)
    save_reports(reports)
    logger.info("Done in %.1f minutes", (time.time() - start) / 60)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TradeFlow models")
    parser.add_argument("--rebuild-data", action="store_true", help="rebuild the cached feature dataset")
    parser.add_argument("--no-cache", action="store_true", help="re-download price data")
    parser.add_argument("--no-intraday", action="store_true", help="skip 1h intraday features when rebuilding")
    parser.add_argument("--data-end", help="freeze the dataset at this date (default: last cached NIFTY session)")
    parser.add_argument("--horizons", nargs="*", choices=list(HORIZONS), help="train only these horizons")
    parser.add_argument("--cpu", action="store_true", help="train on the CPU even if a GPU is available")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="do not re-evaluate the previous models (otherwise needs lightgbm)")
    args = parser.parse_args()
    train(use_cache=not args.no_cache, rebuild_data=args.rebuild_data, horizons=args.horizons,
          use_intraday=not args.no_intraday, data_end=args.data_end, cpu=args.cpu,
          with_baseline=not args.skip_baseline)
