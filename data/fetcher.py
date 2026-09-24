"""Data loading.

Daily data: dates are tz-naive NSE session dates.
Intraday data: Yahoo returns bar-start timestamps in UTC; they are converted to Asia/Kolkata and every
bar gets an explicit `BarEnd` so later alignment can prove a bar was complete before it is used.

Yahoo Finance history limits (verified): 5m/15m bars ~60 days, 1h bars ~730 days, daily 10+ years.
"""
import logging
import os
import time
import yfinance as yf
import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

NIFTY_TICKER = "^NSEI"
INDIA_VIX_TICKER = "^INDIAVIX"
USDINR_TICKER = "INR=X"

IST = "Asia/Kolkata"
SESSION_CLOSE = pd.Timedelta(hours=15, minutes=30)
INTRADAY_PERIODS = {"5m": "60d", "15m": "60d", "1h": "730d"}
INTRADAY_BAR_LENGTH = {"5m": pd.Timedelta(minutes=5), "15m": pd.Timedelta(minutes=15), "1h": pd.Timedelta(hours=1)}


def _clean(df, ticker):
    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df = df.dropna(subset=["Close", "High", "Low"])
    df = df[df["Close"] > 0].drop_duplicates(subset="Date").sort_values("Date").reset_index(drop=True)
    df["Stock"] = ticker
    return df if not df.empty else None


def get_stock_data(ticker, period="10y"):
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        return _clean(df, ticker)
    except Exception:
        return None


def get_many_stocks(tickers, period="10y", use_cache=True):
    """Download several tickers in one batch, caching each to disk for repeat training runs."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    result, missing = {}, []

    for ticker in tickers:
        path = os.path.join(CACHE_DIR, f"{ticker}.pkl")
        if use_cache and os.path.exists(path):
            result[ticker] = pd.read_pickle(path)
        else:
            missing.append(ticker)

    if missing:
        raw = yf.download(missing, period=period, progress=False, auto_adjust=True,
                          group_by="ticker", threads=True)
        for ticker in missing:
            if isinstance(raw.columns, pd.MultiIndex) and ticker in raw.columns.get_level_values(0):
                df = _clean(raw[ticker].copy(), ticker)
            else:
                df = None
            if df is not None:
                df.to_pickle(os.path.join(CACHE_DIR, f"{ticker}.pkl"))
                result[ticker] = df

    logger.info("Daily data: %d/%d tickers loaded (%d downloaded)", len(result), len(tickers), len(missing))
    return result


# Named loaders used by the feature pipeline (training and the app share them)

def load_stock_data(ticker, period="10y"):
    return get_stock_data(ticker, period)


def load_nifty_data(period="10y"):
    return get_stock_data(NIFTY_TICKER, period)


def load_india_vix(period="10y"):
    return get_stock_data(INDIA_VIX_TICKER, period)


def load_usdinr_data(period="10y"):
    return get_stock_data(USDINR_TICKER, period)


def load_sector_data(sector, period="10y"):
    """Official sector index history, or None when the sector is represented by peers instead."""
    from data.sectors import SECTOR_INDEX_TICKERS
    ticker = SECTOR_INDEX_TICKERS.get(sector)
    return get_stock_data(ticker, period) if ticker else None


# Intraday

def _clean_intraday(df, ticker, interval):
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    ts_col = "Datetime" if "Datetime" in df.columns else df.columns[0]
    ts = pd.to_datetime(df[ts_col])
    ts = ts.dt.tz_localize("UTC") if ts.dt.tz is None else ts
    df["BarStart"] = ts.dt.tz_convert(IST).dt.tz_localize(None)
    df = df.drop(columns=[ts_col]).dropna(subset=["Close", "High", "Low"])
    df = df[df["Close"] > 0].drop_duplicates(subset="BarStart").sort_values("BarStart").reset_index(drop=True)

    df["SessionDate"] = df["BarStart"].dt.normalize()
    # A bar ends after its interval, but never after the 15:30 session close (the last 1h bar is 15:15-15:30)
    df["BarEnd"] = (df["BarStart"] + INTRADAY_BAR_LENGTH[interval]).clip(upper=df["SessionDate"] + SESSION_CLOSE)
    df["Stock"] = ticker
    return df if not df.empty else None


def drop_incomplete_bars(df, now=None):
    """Remove bars that have not finished yet (only relevant when running during market hours)."""
    if df is None or df.empty:
        return df
    now = now or pd.Timestamp.now(tz=IST).tz_localize(None)
    return df[df["BarEnd"] <= now].reset_index(drop=True)


def resample_intraday(df, interval):
    """Resample finer intraday bars (e.g. 5m) into coarser ones (e.g. 15m) inside each session.
    Bars are anchored to the 09:15 open and only emitted once every constituent bar exists."""
    if df is None or df.empty:
        return None
    rule = INTRADAY_BAR_LENGTH[interval]
    frames = []
    for session, day in df.groupby("SessionDate"):
        origin = session + pd.Timedelta(hours=9, minutes=15)
        agg = (day.set_index("BarStart")
                  .resample(rule, origin=origin, label="left", closed="left")
                  .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
                  .dropna(subset=["Close"]))
        agg = agg.reset_index()
        agg["SessionDate"] = session
        agg["BarEnd"] = (agg["BarStart"] + rule).clip(upper=session + SESSION_CLOSE)
        # Drop a coarse bar the source data does not cover to its end yet (session still in progress)
        frames.append(agg[agg["BarEnd"] <= day["BarEnd"].max()])
    out = pd.concat(frames, ignore_index=True)
    out["Stock"] = df["Stock"].iloc[0]
    return out


def load_intraday_data(ticker, interval="1h", period=None):
    period = period or INTRADAY_PERIODS[interval]
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        return _clean_intraday(df, ticker, interval)
    except Exception:
        return None


def get_many_intraday(tickers, interval="1h", use_cache=True, chunk_size=25):
    """Batch intraday download with per-ticker cache (`<ticker>__<interval>.pkl`)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    result, missing = {}, []
    for ticker in tickers:
        path = os.path.join(CACHE_DIR, f"{ticker}__{interval}.pkl")
        if use_cache and os.path.exists(path):
            result[ticker] = pd.read_pickle(path)
        else:
            missing.append(ticker)

    for i in range(0, len(missing), chunk_size):
        chunk = missing[i:i + chunk_size]
        raw = yf.download(chunk, period=INTRADAY_PERIODS[interval], interval=interval, progress=False,
                          auto_adjust=True, group_by="ticker", threads=True)
        for ticker in chunk:
            if isinstance(raw.columns, pd.MultiIndex) and ticker in raw.columns.get_level_values(0):
                df = _clean_intraday(raw[ticker].copy(), ticker, interval)
                if df is not None:
                    df.to_pickle(os.path.join(CACHE_DIR, f"{ticker}__{interval}.pkl"))
                    result[ticker] = df
        time.sleep(1)

    logger.info("Intraday %s data: %d/%d tickers loaded (%d downloaded)", interval, len(result), len(tickers), len(missing))
    return result
