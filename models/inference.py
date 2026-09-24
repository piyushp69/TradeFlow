"""Live data assembly for a single ticker.

Builds exactly the same feature frame as training (features.pipeline.build_feature_frame) from freshly
downloaded data, so no feature can drift between training and serving. The app wraps these calls in
Streamlit caches; nothing here caches by itself except the shared on-disk price cache.
"""
import logging

import pandas as pd

from data.fetcher import (drop_incomplete_bars, get_many_stocks, get_stock_data, load_india_vix,
                          load_intraday_data, load_nifty_data, load_usdinr_data)
from data.sectors import SECTOR_INDEX_TICKERS, get_sector, get_sector_peers
from features.market import MarketContext, SectorPanel
from features.pipeline import build_feature_frame

logger = logging.getLogger(__name__)

INTRADAY_INTERVALS = ("5m", "1h")


def build_live_context(ticker):
    """MarketContext for one ticker: NIFTY, VIX, USD/INR and either its sector index or its sector peers."""
    sector = get_sector(ticker)
    index_ticker = SECTOR_INDEX_TICKERS.get(sector)

    peers = get_sector_peers(ticker)
    prices, index_prices = {}, {}
    if index_ticker:
        index_df = get_stock_data(index_ticker)
        if index_df is not None:
            index_prices[index_ticker] = index_df
    elif peers:
        prices = get_many_stocks(peers, use_cache=False)

    context = MarketContext(nifty=load_nifty_data(), vix=load_india_vix(), usdinr=load_usdinr_data(),
                            sector_panel=SectorPanel(prices, index_prices))
    return context


def load_intraday(ticker, intervals=INTRADAY_INTERVALS):
    """Intraday bars for the dashboard/model, with any bar that has not finished yet removed."""
    bars = {}
    for interval in intervals:
        df = load_intraday_data(ticker, interval)
        if df is not None and not df.empty:
            df = drop_incomplete_bars(df)
            if not df.empty:
                bars[interval] = df
    return bars


def load_live_frame(ticker, with_intraday=True):
    """Returns (raw daily prices, full feature frame, metadata). Feature frame is None without enough history."""
    raw = get_stock_data(ticker)
    if raw is None or raw.empty:
        return None, None, {"error": "no price data"}

    context = build_live_context(ticker)
    intraday = load_intraday(ticker) if with_intraday else {}
    frame = build_feature_frame(raw, context, intraday)

    meta = {
        "sector": get_sector(ticker),
        "sector_source": context.sector_panel.source(ticker) if context.sector_panel else None,
        "intraday_intervals": sorted(intraday),
        "intraday_last_bar": {k: v["BarEnd"].max() for k, v in intraday.items()},
        "last_price_date": raw["Date"].max(),
    }
    if frame is not None:
        meta["feature_coverage"] = float(frame.tail(1).notna().mean(axis=1).iloc[0]) if len(frame) else 0.0
    return raw, frame, meta


def latest_row(frame):
    return frame.tail(1) if frame is not None and not frame.empty else None


def market_snapshot(frame):
    """Current market-context readings for the dashboard."""
    if frame is None or frame.empty:
        return {}
    row = frame.iloc[-1]
    keys = ["nifty_return_1", "nifty_return_5", "nifty_return_20", "nifty_volatility_20", "nifty_momentum",
            "nifty_sma_ratio", "nifty_sma200_ratio", "nifty_trend_direction", "sector_return_1", "sector_return_5",
            "sector_return_20", "sector_momentum", "sector_volatility", "sector_sma_ratio", "india_vix",
            "india_vix_change", "india_vix_return", "india_vix_volatility", "usdinr_return_1", "usdinr_return_5",
            "usdinr_return_20", "usdinr_volatility", "stock_vs_nifty_return", "stock_vs_sector_return",
            "stock_nifty_relative_strength", "stock_sector_relative_strength"]
    return {k: (None if pd.isna(row.get(k)) else float(row[k])) for k in keys if k in row}


def timeframe_snapshot(frame, prefixes=("5m", "15m", "1h", "daily")):
    """Per-timeframe readings (RSI, volatility, trend, momentum) for the multi-timeframe tab."""
    if frame is None or frame.empty:
        return {}
    row = frame.iloc[-1]
    out = {}
    for prefix in prefixes:
        values = {}
        for key in ("RSI", "volatility", "return", "momentum", "momentum_z", "trend_direction", "rsi_regime",
                    "price_vs_EMA", "ATR_pct", "MACD_hist_norm", "volume_change"):
            col = f"{prefix}_{key}"
            if col in row:
                values[key] = None if pd.isna(row[col]) else float(row[col])
        out[prefix] = values
    return out
