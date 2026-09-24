"""Training entry point.

Run `python trainer.py` to rebuild every artifact:

    dataset  -> leakage audit -> experiments (baseline vs enhanced feature groups)
             -> feature selection -> final model per horizon -> backtest -> reports

Model choice is made on walk-forward validation folds only. The most recent block of dates is held out as a
test set and is used exactly once per horizon, after everything else is decided.

Artifacts: models/saved_models/model_<horizon>.pkl, models/saved_models/metrics.json and reports/*.
"""
import argparse
import json
import logging
import os
import time
from datetime import datetime

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score

from backtest.engine import BacktestConfig, run_backtest
from data.fetcher import (CACHE_DIR, INDIA_VIX_TICKER, NIFTY_TICKER, USDINR_TICKER, get_many_intraday,
                          get_many_stocks)
from data.sectors import SECTOR_INDEX_TICKERS, get_sector
from features.indicators import HORIZONS, add_targets
from features.leakage import (audit_point_in_time, check_intraday_alignment, check_split_integrity,
                              check_suspicious_features, check_target_leakage)
from features.market import MarketContext, SectorPanel
from features.pipeline import (FEATURE_GROUPS, PRICE_LEVEL_FEATURES, build_feature_frame, feature_columns,
                               requirements_for)
from features.selection import run_feature_selection
from models.training import (RANDOM_STATE, evaluate, fit_model, predict_proba, run_experiment,
                             walk_forward_folds)

STOCKS = [
    '^NSEI', 'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS', 'SBIN.NS',
    'BHARTIARTL.NS', 'ITC.NS', 'HINDUNILVR.NS', 'LT.NS', 'BAJFINANCE.NS', 'AXISBANK.NS',
    'ASIANPAINT.NS', 'MARUTI.NS', 'SUNPHARMA.NS', 'TITAN.NS', 'HCLTECH.NS', 'TATASTEEL.NS',
    'NTPC.NS', 'POWERGRID.NS', 'KOTAKBANK.NS', 'ADANIENT.NS', 'COALINDIA.NS', 'BAJAJFINSV.NS',
    'ULTRACEMCO.NS', 'JSWSTEEL.NS', 'GRASIM.NS', 'M&M.NS', 'INDUSINDBK.NS', 'HINDALCO.NS',
    'NESTLEIND.NS', 'ADANIPORTS.NS', 'ONGC.NS', 'TECHM.NS', 'WIPRO.NS', 'BRITANNIA.NS',
    'EICHERMOT.NS', 'BPCL.NS', 'DIVISLAB.NS', 'CIPLA.NS', 'APOLLOHOSP.NS', 'TATACONSUM.NS',
    'HEROMOTOCO.NS', 'DRREDDY.NS', 'BAJAJ-AUTO.NS', 'YESBANK.NS', 'IDEA.NS', 'UPL.NS',
    'BANDHANBNK.NS', 'DLF.NS', 'VBL.NS', 'PIDILITIND.NS', 'SIEMENS.NS', 'HAL.NS', 'BEL.NS',
    'ABB.NS', 'GAIL.NS', 'PNB.NS', 'BANKBARODA.NS', 'CANBK.NS', 'TRENT.NS', 'GODREJCP.NS',
    'DABUR.NS', 'CHOLAFIN.NS', 'TVSMOTOR.NS', 'POLYCAB.NS', 'HAVELLS.NS', 'LTIM.NS',
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
    'SYNGENE.NS', 'LAURUSLABS.NS', 'JBCHEPHARM.NS', 'ZYDUSLIFE.NS', 'NATCOPHARM.NS',
    'NAM-INDIA.NS', 'HDFCAMC.NS', 'RADICO.NS', 'TIINDIA.NS', 'KEI.NS', 'EXIDEIND.NS'
]

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT, "models", "saved_models")
REPORTS_DIR = os.path.join(ROOT, "reports")
DATASET_CACHE = os.path.join(CACHE_DIR, "dataset.pkl")
MIN_ROWS_PER_STOCK = 500

logger = logging.getLogger("tradeflow.trainer")

# Feature-group combinations compared per horizon (experiment 1 is the existing pipeline)
EXPERIMENTS = [
    ("existing_hgb_baseline", "hgb", ["baseline"]),
    ("lgbm_baseline", "lgbm", ["baseline"]),
    ("lgbm_baseline_market", "lgbm", ["baseline", "market"]),
    ("lgbm_baseline_daily_mtf", "lgbm", ["baseline", "daily_mtf"]),
    ("lgbm_baseline_market_mtf", "lgbm", ["baseline", "market", "daily_mtf"]),
]


def setup_logging(verbose=True):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    handlers = [logging.FileHandler(os.path.join(REPORTS_DIR, "training.log"), mode="w", encoding="utf-8")]
    if verbose:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=handlers, force=True)
    logging.getLogger("lightgbm").setLevel(logging.WARNING)


# Data

def load_context_sources(use_cache=True):
    """Every external series needed for market features, plus daily prices for the whole universe."""
    tickers = list(dict.fromkeys(STOCKS + [NIFTY_TICKER, INDIA_VIX_TICKER, USDINR_TICKER]
                                 + list(SECTOR_INDEX_TICKERS.values())))
    prices = get_many_stocks(tickers, use_cache=use_cache)
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
    logger.info("Sector indices with history: %s", sorted(sources["index_prices"]))
    return sources, prices


def build_context(sources):
    return MarketContext(nifty=sources["nifty"], vix=sources["vix"], usdinr=sources["usdinr"],
                         sector_panel=SectorPanel(sources["prices"], sources["index_prices"]))


def build_dataset(use_cache=True, use_intraday=True, rebuild=False):
    """Feature frame for every stock, concatenated. Cached on disk between runs."""
    if not rebuild and os.path.exists(DATASET_CACHE):
        data = pd.read_pickle(DATASET_CACHE)
        logger.info("Loaded cached dataset: %d rows x %d columns", len(data), data.shape[1])
        return data

    sources, prices = load_context_sources(use_cache)
    context = build_context(sources)
    intraday = get_many_intraday([s for s in STOCKS if s != NIFTY_TICKER], "1h") if use_intraday else {}

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
        return None

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
    return data


# Leakage audit

def run_leakage_audit(sources, sample_stocks=("TCS.NS", "MARUTI.NS", "HDFCBANK.NS")):
    """Point-in-time rebuild check plus intraday alignment check. Returns (report, invalid-feature map)."""
    logger.info("Leakage audit: point-in-time feature rebuild")
    audited = sum(FEATURE_GROUPS.values(), []) + PRICE_LEVEL_FEATURES
    mismatches, checked, alignment_problems = [], [], {}
    for stock in sample_stocks:
        raw = sources["prices"].get(stock)
        if raw is None:
            continue
        bars_path = os.path.join(CACHE_DIR, f"{stock}__1h.pkl")
        intraday = {"1h": pd.read_pickle(bars_path)} if os.path.exists(bars_path) else None
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
    invalid = {f: "point-in-time audit: value changes when future data is removed" for f in leaky}
    invalid.update({f: "scale-dependent price level (not usable by a model pooled across stocks)"
                    for f in PRICE_LEVEL_FEATURES})
    return ({"stocks_checked": checked, "features_checked": len(audited), "look_ahead_failures": leaky,
             "intraday_alignment_problems": alignment_problems}, invalid)


# Training

class EnsembleModel:
    """Mean of component probabilities. Plain averaging keeps training and single-stock inference identical."""

    def __init__(self, components):
        self.components = components

    def predict_proba(self, X):
        p = np.mean([c["model"].predict_proba(X[c["features"]])[:, 1] for c in self.components], axis=0)
        return np.column_stack([1 - p, p])


def add_reference_cross_section(bundle, model, data, features, name):
    """Score each stock's most recent row: the app ranks a ticker against this cross-section."""
    latest = data.groupby("Stock").tail(1)
    latest = latest[latest["Date"] >= data["Date"].max() - pd.Timedelta(days=10)]
    probs = np.sort(predict_proba(model, latest[features]))
    bundle["reference_probs"] = probs.tolist()
    bundle["reference_date"] = f"{latest['Date'].max():%Y-%m-%d}"
    bundle["threshold"] = float(np.median(probs))
    logger.info("   [%s] reference cross-section: %d stocks, median P(UP)=%.3f", name, len(probs), bundle["threshold"])


def evaluate_ensemble(results):
    """Average the existing model with the best LightGBM model and score it on the validation folds."""
    existing = next((r for r in results if r["model_type"] == "hgb"), None)
    best_lgbm = max((r for r in results if r["model_type"] == "lgbm"), key=lambda r: r["cv_auc"], default=None)
    if existing is None or best_lgbm is None or existing["oof"] is None or best_lgbm["oof"] is None:
        return None
    merged = existing["oof"].merge(best_lgbm["oof"], on=["Date", "Stock", "y"], suffixes=("_a", "_b"))
    if merged.empty:
        return None
    merged["p"] = (merged["p_a"] + merged["p_b"]) / 2
    # Mean AUC per validation fold, exactly how single models are scored (a pooled AUC across folds would
    # also reward tracking the base rate as it drifts between folds, and would not be comparable)
    per_fold = [roc_auc_score(g["y"], g["p"]) for _, g in merged.groupby("fold_a") if g["y"].nunique() > 1]
    if not per_fold:
        return None
    return {"components": [existing["name"], best_lgbm["name"]],
            "cv_auc": round(float(np.mean(per_fold)), 4),
            "cv_fold_aucs": [round(float(a), 4) for a in per_fold],
            "parts": (existing, best_lgbm)}


def train_horizon(data, name, horizon_days, invalid_map, reports):
    target = f"Target_{name.capitalize()}"
    labelled = data.dropna(subset=[target]).reset_index(drop=True)
    folds, train_full, test = walk_forward_folds(labelled, horizon_days)

    logger.info("=" * 100)
    logger.info("[%s] horizon=%d days | labelled rows=%d", name, horizon_days, len(labelled))
    for i, (tr, va) in enumerate(folds, 1):
        logger.info("   fold %d train %s..%s (%d rows) -> validate %s..%s (%d rows)", i,
                    tr["Date"].min().date(), tr["Date"].max().date(), len(tr),
                    va["Date"].min().date(), va["Date"].max().date(), len(va))
    logger.info("   final train %s..%s (%d rows) | TEST %s..%s (%d rows)",
                train_full["Date"].min().date(), train_full["Date"].max().date(), len(train_full),
                test["Date"].min().date(), test["Date"].max().date(), len(test))

    ok, gap = check_split_integrity(train_full["Date"], test["Date"], horizon_days, labelled["Date"])
    logger.info("   embargo between training and test: %d sessions (need %d) -> %s", gap, horizon_days - 1,
                "OK" if ok else "FAILED")
    if not ok:
        raise RuntimeError(f"embargo violated for {name}: only {gap} sessions between train and test")

    all_candidates = feature_columns(["baseline", "market", "daily_mtf", "intraday_1h", "intraday_short"])
    horizon_invalid = dict(invalid_map)
    target_problems = check_target_leakage(train_full, all_candidates, [f"Target_{h.capitalize()}" for h in HORIZONS])
    if target_problems:
        logger.error("   target leakage detected: %s", target_problems)
        horizon_invalid.update({f: r for f, r in target_problems})
    suspicious = check_suspicious_features(train_full[all_candidates], train_full[target].astype(int))
    if suspicious:
        logger.warning("   implausibly strong single-feature AUC: %s", suspicious)
        horizon_invalid.update({f: f"suspicious single-feature AUC {v}" for f, v in suspicious.items()})

    results = []
    for exp_name, kind, groups in EXPERIMENTS:
        results.append(run_experiment(exp_name, kind, folds, train_full, test, feature_columns(groups), target))

    logger.info("   [feature selection] %d candidates (validation folds only, test never used)", len(all_candidates))
    selected, report, extras = run_feature_selection(folds, train_full, all_candidates, target, horizon_invalid)
    reports[f"feature_selection_{name}"] = report
    reports[f"correlation_{name}"] = extras["correlation"]
    reports[f"shap_sample_{name}"] = extras["shap_sample"]
    if selected:
        results.append(run_experiment("lgbm_selected", "lgbm", folds, train_full, test, selected, target))

    ensemble = evaluate_ensemble(results)
    if ensemble:
        logger.info("   [ensemble %s] CV AUC=%.4f (folds %s)", "+".join(ensemble["components"]),
                    ensemble["cv_auc"], ensemble["cv_fold_aucs"])

    best = max(results, key=lambda r: r["cv_auc"])
    use_ensemble = ensemble is not None and ensemble["cv_auc"] > best["cv_auc"]
    logger.info("   chosen on validation AUC: %s", "ensemble" if use_ensemble else best["name"])

    experiments_table = pd.DataFrame([{
        "experiment": r["name"], "model": r["model_type"], "config": r["config"], "iterations": r["n_iterations"],
        "n_features": r["n_features"], "cv_auc": r["cv_auc"], "cv_fold_aucs": str(r["cv_fold_aucs"]),
        **{f"val_{k}": v for k, v in (r["validation_metrics"] or {}).items()},
        **{f"test_{k}": v for k, v in r["test_metrics"].items()},
    } for r in results])

    if use_ensemble:
        # Score the test block with the components fit on pre-test data only (as for single models); the
        # final components below are refit on every labelled row, test block included
        test_proba = np.mean([part["test_predictions"]["p"].to_numpy() for part in ensemble["parts"]], axis=0)
        components = [{"model": fit_model(part["model_type"], part["config"], part["n_iterations"],
                                          labelled[part["features"]], labelled[target].astype(int)),
                       "features": part["features"], "model_type": part["model_type"], "name": part["name"]}
                      for part in ensemble["parts"]]
        final_model = EnsembleModel(components)
        chosen_features = sorted({f for c in components for f in c["features"]})
        test_metrics = evaluate(test[target].astype(int), test_proba, test["Date"])
        bundle = {"components": components, "features": chosen_features, "model_type": "ensemble",
                  "experiment": "ensemble(" + "+".join(ensemble["components"]) + ")", "config": None,
                  "n_iterations": None, "cv_auc": ensemble["cv_auc"],
                  "cv_fold_aucs": ensemble["cv_fold_aucs"],
                  "validation_metrics": None, "test_metrics": test_metrics}
        test_predictions = pd.DataFrame({"Date": test["Date"].to_numpy(), "Stock": test["Stock"].to_numpy(),
                                         "y": test[target].astype(int).to_numpy(), "p": test_proba})
        experiments_table.loc[len(experiments_table)] = {
            "experiment": bundle["experiment"], "model": "ensemble", "n_features": len(chosen_features),
            "cv_auc": ensemble["cv_auc"], **{f"test_{k}": v for k, v in test_metrics.items()}}
    else:
        final_model = fit_model(best["model_type"], best["config"], best["n_iterations"],
                                labelled[best["features"]], labelled[target].astype(int))
        chosen_features = best["features"]
        bundle = {"model": final_model, "features": chosen_features, "model_type": best["model_type"],
                  "experiment": best["name"], "config": best["config"], "n_iterations": best["n_iterations"],
                  "cv_auc": best["cv_auc"], "cv_fold_aucs": best["cv_fold_aucs"],
                  "validation_metrics": best["validation_metrics"], "test_metrics": best["test_metrics"]}
        test_predictions = best["test_predictions"]

    reports[f"experiments_{name}"] = experiments_table
    bundle.update({
        "horizon_days": horizon_days,
        "feature_groups": [g for g, cols in FEATURE_GROUPS.items() if set(cols) & set(chosen_features)],
        "requirements": sorted(requirements_for(chosen_features)),
        "selected_features": selected,
        "test_period": [f"{test['Date'].min():%Y-%m-%d}", f"{test['Date'].max():%Y-%m-%d}"],
        "validation_period": [f"{folds[0][1]['Date'].min():%Y-%m-%d}", f"{folds[-1][1]['Date'].max():%Y-%m-%d}"],
        "training_period": [f"{labelled['Date'].min():%Y-%m-%d}", f"{train_full['Date'].max():%Y-%m-%d}"],
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "lightgbm_version": lgb.__version__,
        "random_state": RANDOM_STATE,
    })
    add_reference_cross_section(bundle, final_model, data, chosen_features, name)

    reports[f"predictions_{name}"] = test_predictions
    baseline_predictions = next(r["test_predictions"] for r in results if r["name"] == EXPERIMENTS[0][0])
    return bundle, test_predictions, baseline_predictions


def run_backtests(name, horizon_days, predictions, baseline_predictions, prices):
    """Backtest the chosen model and the existing baseline over the same out-of-sample window."""
    # Only the predicted stocks: FX and index series have other calendars and are not tradable here
    stocks = set(predictions["Stock"])
    price_frame = pd.concat([df[["Date", "Stock", "Open", "Close"]] for t, df in prices.items() if t in stocks],
                            ignore_index=True)
    config = BacktestConfig(horizon_days=horizon_days)
    enhanced = run_backtest(predictions, price_frame, config)
    baseline = run_backtest(baseline_predictions, price_frame, config)
    periods = enhanced["metrics"].get("periods", 0)
    logger.info("   [backtest] %d rebalances | enhanced return=%s sharpe=%s | baseline return=%s | benchmark=%s",
                periods, enhanced["metrics"].get("cumulative_return"), enhanced["metrics"].get("sharpe_ratio"),
                baseline["metrics"].get("cumulative_return"), enhanced["benchmark"].get("cumulative_return"))
    if periods < 6:
        logger.warning("   [backtest] only %d non-overlapping periods for %s: indicative only", periods, name)
    return {"enhanced": enhanced, "baseline": baseline}


def run_intraday_study(data, name, horizon_days, reports):
    """Does adding 1h-timeframe features help, judged on the window where 1h bars exist?

    Yahoo Finance serves 1h bars for ~730 days only, so this is a separate study and never feeds the main
    model choice (most of the training history has no intraday data at all).
    """
    target = f"Target_{name.capitalize()}"
    subset = data.dropna(subset=[target]).dropna(subset=FEATURE_GROUPS["intraday_1h"], how="all").reset_index(drop=True)
    if subset.empty:
        return None
    folds, train_full, test = walk_forward_folds(subset, horizon_days, cv_folds=3, test_fraction=0.25,
                                                 cv_start_fraction=0.30)
    if min(len(f[1]) for f in folds) < 2000 or len(test) < 2000:
        logger.warning("   [1h study] %s: not enough labelled rows inside the 1h window (test=%d)", name, len(test))
        return None

    logger.info("   [1h study] %s: rows=%d (%s..%s), test=%d rows", name, len(subset),
                subset["Date"].min().date(), subset["Date"].max().date(), len(test))
    rows = []
    for label, groups in (("without_1h", ["baseline", "market", "daily_mtf"]),
                          ("with_1h", ["baseline", "market", "daily_mtf", "intraday_1h"])):
        features = feature_columns(groups)
        result = run_experiment(label, "lgbm", folds, train_full, test, features, target, keep_oof=False)
        rows.append({"variant": label, "n_features": len(features), "cv_auc": result["cv_auc"],
                     "cv_fold_aucs": str(result["cv_fold_aucs"]),
                     **{f"test_{k}": v for k, v in result["test_metrics"].items()}})
    table = pd.DataFrame(rows)
    reports[f"intraday_1h_study_{name}"] = table
    logger.info("   [1h study] %s: CV AUC without=%.4f with=%.4f | test AUC without=%.4f with=%.4f", name,
                table.cv_auc[0], table.cv_auc[1], table.test_roc_auc[0], table.test_roc_auc[1])
    return table


def save_reports(reports):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    for key, value in reports.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(os.path.join(REPORTS_DIR, f"{key}.csv"), index=key.startswith("correlation"))
        elif key.startswith("shap_sample"):
            joblib.dump(value, os.path.join(REPORTS_DIR, f"{key}.pkl"), compress=3)
        else:
            with open(os.path.join(REPORTS_DIR, f"{key}.json"), "w", encoding="utf-8") as fh:
                json.dump(value, fh, indent=2, default=str)
    logger.info("Reports written to %s (%d files)", REPORTS_DIR, len(reports))


def generate_base_model(use_cache=True, rebuild_data=False, horizons=None, use_intraday=True, run_1h_study=True):
    setup_logging()
    np.random.seed(RANDOM_STATE)
    start = time.time()
    logger.info("TradeFlow training run | sklearn=%s lightgbm=%s | seed=%d", sklearn.__version__, lgb.__version__,
                RANDOM_STATE)

    data = build_dataset(use_cache=use_cache, use_intraday=use_intraday, rebuild=rebuild_data)
    if data is None:
        logger.error("No valid data collected")
        return

    sources, prices = load_context_sources(use_cache)
    audit, invalid_map = run_leakage_audit(sources)
    reports = {"leakage_audit": audit}

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary = {}
    for name, days in HORIZONS.items():
        if horizons and name not in horizons:
            continue
        bundle, predictions, baseline_predictions = train_horizon(data, name, days, invalid_map, reports)
        backtests = run_backtests(name, days, predictions, baseline_predictions, prices)
        bundle["backtest"] = {k: {"metrics": v["metrics"], "benchmark": v["benchmark"], "config": v["config"]}
                              for k, v in backtests.items()}
        reports[f"backtest_{name}"] = bundle["backtest"]
        reports[f"equity_curve_{name}"] = backtests["enhanced"]["equity_curve"]
        reports[f"equity_curve_baseline_{name}"] = backtests["baseline"]["equity_curve"]
        reports[f"trades_{name}"] = backtests["enhanced"]["trades"]
        if run_1h_study:
            run_intraday_study(data, name, days, reports)

        joblib.dump(bundle, os.path.join(OUTPUT_DIR, f"model_{name}.pkl"), compress=3)
        summary[name] = {k: v for k, v in bundle.items()
                         if k not in ("model", "components", "reference_probs", "selected_features")}
        summary[name]["n_selected_features"] = len(bundle.get("selected_features") or [])
        logger.info("   [%s] saved model bundle (%s)", name, bundle["experiment"])

    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    save_reports(reports)
    logger.info("Done in %.1f minutes", (time.time() - start) / 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TradeFlow models")
    parser.add_argument("--rebuild-data", action="store_true", help="rebuild the cached feature dataset")
    parser.add_argument("--no-cache", action="store_true", help="re-download price data")
    parser.add_argument("--no-intraday", action="store_true", help="skip 1h intraday features")
    parser.add_argument("--skip-1h-study", action="store_true", help="skip the 1h ablation study")
    parser.add_argument("--horizons", nargs="*", choices=list(HORIZONS), help="train only these horizons")
    args = parser.parse_args()
    generate_base_model(use_cache=not args.no_cache, rebuild_data=args.rebuild_data, horizons=args.horizons,
                        use_intraday=not args.no_intraday, run_1h_study=not args.skip_1h_study)
