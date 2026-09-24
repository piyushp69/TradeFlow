"""One feature builder shared by training (trainer.py) and inference (models/predictor.py, app.py).

Groups:
    baseline        existing 29 features from features/indicators.py (unchanged, used by the existing model)
    market          NIFTY / sector / India VIX / USD-INR / relative-strength features
    daily_mtf       daily-timeframe indicators, trend and regime features
    intraday_1h     1h indicators + 1h-vs-daily momentum (history only from Oct 2023)
    intraday_short  5m / 15m indicators and cross-timeframe momentum (~60 days of history: dashboard only)
"""
import numpy as np
from features.indicators import FEATURES as BASELINE_FEATURES, calculate_indicators
from features.market import MARKET_CONTEXT_FEATURES, create_market_features
from features.multitimeframe import (create_multitimeframe_features, timeframe_model_columns,
                                     timeframe_price_levels, timeframe_columns)

FEATURE_GROUPS = {
    "baseline": list(BASELINE_FEATURES),
    "market": list(MARKET_CONTEXT_FEATURES),
    "daily_mtf": timeframe_model_columns("daily"),
    "intraday_1h": timeframe_model_columns("1h") + ["1h_vs_daily_momentum"],
    "intraday_short": timeframe_model_columns("5m") + timeframe_model_columns("15m")
                      + ["5m_vs_15m_momentum", "15m_vs_1h_momentum"],
}

# Raw price-level outputs (EMA, MACD, ATR): kept for display, never valid model inputs for a pooled model
PRICE_LEVEL_FEATURES = [c for tf in ("5m", "15m", "1h", "daily") for c in timeframe_price_levels(tf)]

DISPLAY_COLUMNS = [c for tf in ("5m", "15m", "1h", "daily") for c in timeframe_columns(tf)]

# Groups that need data the app must load before predicting
GROUP_REQUIREMENTS = {
    "market": {"market"},
    "intraday_1h": {"intraday_1h"},
    "intraday_short": {"intraday_short"},
}


def feature_columns(groups):
    cols = []
    for group in groups:
        cols += [c for c in FEATURE_GROUPS[group] if c not in cols]
    return cols


def requirements_for(features):
    """Which extra data sources a list of features needs."""
    needs = set()
    for group, reqs in GROUP_REQUIREMENTS.items():
        if set(features) & set(FEATURE_GROUPS[group]):
            needs |= reqs
    return needs


def build_feature_frame(raw_df, context=None, intraday=None):
    """Build every feature group for one stock.

    raw_df: cleaned daily OHLCV for one stock (may already contain Target_* columns; they pass through untouched).
    context: features.market.MarketContext (None -> market columns are NaN, baseline market columns too).
    intraday: optional {"5m": bars, "1h": bars} from data.fetcher intraday loaders.
    Rows match calculate_indicators(): the ~1-year warm-up needed by the baseline features is dropped.
    """
    nifty = context.nifty if context is not None else None
    base = calculate_indicators(raw_df, nifty)
    if base is None:
        return None

    extras = create_multitimeframe_features(raw_df, intraday)
    if context is not None:
        extras = create_market_features(extras, context)
    else:
        for col in MARKET_CONTEXT_FEATURES:
            extras[col] = np.nan

    new_cols = [c for c in extras.columns if c not in base.columns]
    merged = base.merge(extras[["Date"] + new_cols], on="Date", how="left")
    return merged
