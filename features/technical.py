"""Timeframe-agnostic technical indicators.

`create_technical_features` works on any OHLCV bar frame (daily or intraday). Every value at bar t
uses bars <= t only: rolling windows are trailing, EWMs use adjust=False, and nothing is shifted backwards.

Some outputs are raw price levels (EMA, MACD, ATR). They are useful for display, but they are
scale-dependent, so a model pooled across stocks with different prices cannot use them directly.
`price_level_columns` lists them so the feature-selection pipeline can reject them explicitly;
scale-free versions (`*_norm`, `*_pct`, ratios) are produced alongside.
"""
import numpy as np
import pandas as pd

EPS = 1e-10


def rsi(close, window=14):
    diff = close.diff()
    gain = diff.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = -diff.clip(upper=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    return 100 - (100 / (1 + gain / (loss + EPS)))


def ema(series, span):
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def macd(close, fast=12, slow=26, signal=9):
    line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    return line, line.ewm(span=signal, adjust=False).mean()


def atr(high, low, close, window=14):
    prev_close = close.shift(1)
    true_range = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return true_range.rolling(window).mean()


def price_level_columns(prefix, ema_fast, ema_slow):
    return [f"{prefix}_MACD", f"{prefix}_MACD_signal", f"{prefix}_ATR",
            f"{prefix}_EMA_{ema_fast}", f"{prefix}_EMA_{ema_slow}"]


def create_technical_features(bars, prefix, ema_fast, ema_slow, vol_window=20, momentum_bars=5):
    """Indicators for one timeframe. Returns a frame aligned to `bars` (same index)."""
    close, high, low = bars["Close"], bars["High"], bars["Low"]
    volume = bars["Volume"].astype(float)
    log_ret = np.log(close).diff()

    ema_f, ema_s = ema(close, ema_fast), ema(close, ema_slow)
    macd_line, macd_signal = macd(close)
    atr_raw = atr(high, low, close)
    vol = log_ret.rolling(vol_window).std()
    momentum = np.log(close / close.shift(momentum_bars))

    out = pd.DataFrame(index=bars.index)
    out[f"{prefix}_RSI"] = rsi(close)
    out[f"{prefix}_MACD"] = macd_line
    out[f"{prefix}_MACD_signal"] = macd_signal
    out[f"{prefix}_ATR"] = atr_raw
    out[f"{prefix}_volatility"] = vol
    out[f"{prefix}_return"] = log_ret
    # Volume relative to its trailing average (log). Raw bar-to-bar % change explodes on near-zero volume bars.
    out[f"{prefix}_volume_change"] = np.log((volume + 1) / (volume.rolling(vol_window).mean() + 1))
    out[f"{prefix}_EMA_{ema_fast}"] = ema_f
    out[f"{prefix}_EMA_{ema_slow}"] = ema_s
    out[f"{prefix}_price_vs_EMA"] = close / ema_f - 1

    # Scale-free versions usable by a model pooled across stocks
    out[f"{prefix}_MACD_norm"] = macd_line / close
    out[f"{prefix}_MACD_signal_norm"] = macd_signal / close
    out[f"{prefix}_MACD_hist_norm"] = (macd_line - macd_signal) / close
    out[f"{prefix}_ATR_pct"] = atr_raw / close
    out[f"{prefix}_EMA_fast_vs_slow"] = ema_f / ema_s - 1

    # Momentum and regimes
    out[f"{prefix}_momentum"] = momentum
    out[f"{prefix}_momentum_z"] = momentum / (vol * np.sqrt(momentum_bars) + EPS)
    up = (close > ema_f) & (ema_f > ema_s)
    down = (close < ema_f) & (ema_f < ema_s)
    trend = pd.Series(np.where(up, 1.0, np.where(down, -1.0, 0.0)), index=bars.index)
    out[f"{prefix}_trend_direction"] = trend.where(ema_s.notna())
    rsi_vals = out[f"{prefix}_RSI"]
    out[f"{prefix}_rsi_regime"] = pd.Series(np.where(rsi_vals > 60, 1.0, np.where(rsi_vals < 40, -1.0, 0.0)),
                                            index=bars.index).where(rsi_vals.notna())

    return out.replace([np.inf, -np.inf], np.nan)
