"""Multi-timeframe features (5m, 15m, 1h, daily) aligned to daily prediction rows.

A daily row dated D is a prediction made at the NSE close (15:30 IST) of D. Intraday indicators are computed
on the continuous intraday bar series (trailing windows only) and the row takes the values of the **last bar
whose BarEnd <= D 15:30 and whose session is D**. A bar that ends after the close, or a day without intraday
data, yields NaN instead of reusing an older session's values.

Data availability (Yahoo Finance): 5m/15m ~60 days, 1h ~730 days. The 15m series is resampled from 5m bars
(anchored at 09:15) so both short timeframes come from the same source.
"""
import numpy as np
import pandas as pd
from data.fetcher import SESSION_CLOSE, resample_intraday
from features.technical import create_technical_features, price_level_columns

# prefix -> (ema_fast, ema_slow, momentum_bars). Momentum spans: 5m=1h, 15m=2h, 1h=~1 session, daily=1 week
TIMEFRAMES = {
    "5m": (9, 21, 12),
    "15m": (9, 21, 8),
    "1h": (9, 21, 7),
    "daily": (20, 50, 5),
}

REQUESTED_COLUMNS = ["RSI", "MACD", "MACD_signal", "ATR", "volatility", "return", "volume_change",
                     "EMA_{fast}", "EMA_{slow}", "price_vs_EMA"]
EXTRA_COLUMNS = ["MACD_norm", "MACD_signal_norm", "MACD_hist_norm", "ATR_pct", "EMA_fast_vs_slow",
                 "momentum", "momentum_z", "trend_direction", "rsi_regime"]
CROSS_FEATURES = {
    "5m_vs_15m_momentum": ("5m", "15m"),
    "15m_vs_1h_momentum": ("15m", "1h"),
    "1h_vs_daily_momentum": ("1h", "daily"),
}


def timeframe_columns(prefix):
    fast, slow, _ = TIMEFRAMES[prefix]
    cols = [c.format(fast=fast, slow=slow) for c in REQUESTED_COLUMNS] + EXTRA_COLUMNS
    return [f"{prefix}_{c}" for c in cols]


def timeframe_price_levels(prefix):
    fast, slow, _ = TIMEFRAMES[prefix]
    return price_level_columns(prefix, fast, slow)


def timeframe_model_columns(prefix):
    levels = set(timeframe_price_levels(prefix))
    return [c for c in timeframe_columns(prefix) if c not in levels]


def create_daily_timeframe_features(daily_df):
    fast, slow, mom = TIMEFRAMES["daily"]
    feats = create_technical_features(daily_df.reset_index(drop=True), "daily", fast, slow, momentum_bars=mom)
    feats.insert(0, "Date", daily_df["Date"].to_numpy())
    return feats


def create_intraday_features(bars, prefix):
    """Per-bar indicators for an intraday series. Keeps SessionDate/BarEnd for alignment."""
    fast, slow, mom = TIMEFRAMES[prefix]
    bars = bars.sort_values("BarStart").reset_index(drop=True)
    feats = create_technical_features(bars, prefix, fast, slow, momentum_bars=mom)
    feats.insert(0, "BarEnd", bars["BarEnd"].to_numpy())
    feats.insert(0, "SessionDate", bars["SessionDate"].to_numpy())
    return feats


def align_intraday_to_daily(daily_dates, bar_features):
    """For each daily date D, take the last bar of session D that ended by D 15:30."""
    cutoff = pd.to_datetime(daily_dates).to_numpy()
    complete = bar_features[bar_features["BarEnd"] <= bar_features["SessionDate"] + SESSION_CLOSE]
    last_bar = complete.sort_values("BarEnd").groupby("SessionDate").tail(1)
    left = pd.DataFrame({"Date": cutoff})
    merged = left.merge(last_bar, left_on="Date", right_on="SessionDate", how="left")
    return merged.drop(columns=["SessionDate"])


def create_multitimeframe_features(daily_df, intraday=None):
    """Daily frame + daily timeframe indicators + aligned intraday timeframes + cross-timeframe features.

    intraday: optional dict like {"5m": bars, "1h": bars}. "15m" is resampled from "5m" when not given.
    Timeframes without data produce NaN columns so the output schema is always the same.
    """
    intraday = dict(intraday or {})
    if "15m" not in intraday and intraday.get("5m") is not None:
        intraday["15m"] = resample_intraday(intraday["5m"], "15m")

    out = daily_df.reset_index(drop=True).copy()
    daily = create_daily_timeframe_features(out)
    for col in daily.columns.drop("Date"):
        out[col] = daily[col].to_numpy()

    for prefix in ("5m", "15m", "1h"):
        cols = timeframe_columns(prefix)
        bars = intraday.get(prefix)
        if bars is None or bars.empty:
            for col in cols:
                out[col] = np.nan
            out[f"{prefix}_bar_end"] = pd.NaT
            continue
        aligned = align_intraday_to_daily(out["Date"], create_intraday_features(bars, prefix))
        for col in cols:
            out[col] = aligned[col].to_numpy()
        out[f"{prefix}_bar_end"] = aligned["BarEnd"].to_numpy()

    for name, (fast_tf, slow_tf) in CROSS_FEATURES.items():
        out[name] = out[f"{fast_tf}_momentum_z"] - out[f"{slow_tf}_momentum_z"]

    return out.replace([np.inf, -np.inf], np.nan)
