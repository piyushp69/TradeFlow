"""Market-context features: NIFTY 50, sector, India VIX, USD/INR, and stock-vs-market relatives.

Timestamp alignment (a stock row dated D represents information available at the NSE close, 15:30 IST, on D):

* NIFTY, sector indices, sector peers and India VIX trade in the same NSE session, so their close on D is
  known at the same time as the stock's close on D. They are joined on exact session date.
* USD/INR (Yahoo "INR=X") is a global FX series; its daily bar dated D closes after the NSE close.
  Using it on D would leak up to ~12 hours of future FX moves, so the value from the most recent FX date
  strictly before D is used (lag of at least one day).
* Joins are as-of with a tolerance. If a source has no data within the tolerance the features are NaN,
  never silently forward-filled from stale data.
"""
import logging
from dataclasses import dataclass
import numpy as np
import pandas as pd
from data.sectors import SECTOR_INDEX_TICKERS, SECTOR_MEMBERS, get_sector

logger = logging.getLogger(__name__)

EPS = 1e-10
SAME_SESSION_TOLERANCE = pd.Timedelta(days=0)
FX_TOLERANCE = pd.Timedelta(days=5)

NIFTY_FEATURES = ["nifty_return_1", "nifty_return_5", "nifty_return_20", "nifty_volatility_20",
                  "nifty_momentum", "nifty_sma_ratio", "nifty_sma200_ratio", "nifty_trend_direction"]
SECTOR_FEATURES = ["sector_return_1", "sector_return_5", "sector_return_20", "sector_momentum",
                   "sector_volatility", "sector_sma_ratio"]
VIX_FEATURES = ["india_vix", "india_vix_change", "india_vix_return", "india_vix_volatility"]
USDINR_FEATURES = ["usdinr_return_1", "usdinr_return_5", "usdinr_return_20", "usdinr_volatility"]
RELATIVE_FEATURES = ["stock_vs_nifty_return", "stock_vs_sector_return",
                     "stock_nifty_relative_strength", "stock_sector_relative_strength"]
MARKET_CONTEXT_FEATURES = NIFTY_FEATURES + SECTOR_FEATURES + VIX_FEATURES + USDINR_FEATURES + RELATIVE_FEATURES


def _index_features(dates, close, prefix, names):
    log_ret = np.log(close).diff()
    ret_20 = np.log(close / close.shift(20))
    vol_20 = log_ret.rolling(20).std()
    values = {
        "return_1": log_ret,
        "return_5": np.log(close / close.shift(5)),
        "return_20": ret_20,
        "volatility": vol_20,
        "momentum": ret_20 / (vol_20 * np.sqrt(20) + EPS),
        "sma_ratio": close / close.rolling(50).mean(),
    }
    out = pd.DataFrame({"Date": dates.to_numpy()})
    for key, col in names.items():
        out[col] = values[key].to_numpy()
    return out


def nifty_features(nifty_df):
    names = {"return_1": "nifty_return_1", "return_5": "nifty_return_5", "return_20": "nifty_return_20",
             "volatility": "nifty_volatility_20", "momentum": "nifty_momentum", "sma_ratio": "nifty_sma_ratio"}
    out = _index_features(nifty_df["Date"], nifty_df["Close"], "nifty", names)
    close = nifty_df["Close"].reset_index(drop=True)
    sma50, sma200 = close.rolling(50).mean(), close.rolling(200).mean()
    out["nifty_sma200_ratio"] = (close / sma200).to_numpy()
    trend = np.where((close > sma50) & (sma50 > sma200), 1.0, np.where((close < sma50) & (sma50 < sma200), -1.0, 0.0))
    out["nifty_trend_direction"] = pd.Series(trend).where(sma200.notna()).to_numpy()
    return out


def sector_features_from_close(dates, close):
    names = {"return_1": "sector_return_1", "return_5": "sector_return_5", "return_20": "sector_return_20",
             "momentum": "sector_momentum", "volatility": "sector_volatility", "sma_ratio": "sector_sma_ratio"}
    return _index_features(pd.Series(dates), pd.Series(close).reset_index(drop=True), "sector", names)


def vix_features(vix_df):
    vix = vix_df["Close"].reset_index(drop=True)
    log_ret = np.log(vix).diff()
    return pd.DataFrame({
        "Date": vix_df["Date"].to_numpy(),
        "india_vix": vix.to_numpy(),
        "india_vix_change": vix.diff().to_numpy(),
        "india_vix_return": log_ret.to_numpy(),
        "india_vix_volatility": log_ret.rolling(20).std().to_numpy(),
    })


def usdinr_features(fx_df):
    fx = fx_df["Close"].reset_index(drop=True)
    log_ret = np.log(fx).diff()
    return pd.DataFrame({
        "Date": fx_df["Date"].to_numpy(),
        "usdinr_return_1": log_ret.to_numpy(),
        "usdinr_return_5": np.log(fx / fx.shift(5)).to_numpy(),
        "usdinr_return_20": np.log(fx / fx.shift(20)).to_numpy(),
        "usdinr_volatility": log_ret.rolling(20).std().to_numpy(),
    })


def align_external(stock_dates, features, same_session, tolerance):
    """As-of join of source features onto stock dates.
    same_session=True: exact or earlier date allowed (NSE-session series).
    same_session=False: strictly earlier date only (series whose bar for D closes after the NSE close)."""
    left = pd.DataFrame({"Date": pd.to_datetime(stock_dates).to_numpy()}).reset_index()
    left = left.sort_values("Date")
    right = features.sort_values("Date").rename(columns={"Date": "SourceDate"})
    merged = pd.merge_asof(left, right, left_on="Date", right_on="SourceDate", direction="backward",
                           allow_exact_matches=same_session,
                           tolerance=tolerance if tolerance > pd.Timedelta(0) else None)
    if same_session and tolerance == pd.Timedelta(0):
        # Exact same-session match required: anything else is missing data, not a value to reuse
        mismatch = merged["SourceDate"].notna() & (merged["SourceDate"] != merged["Date"])
        merged.loc[mismatch, [c for c in right.columns if c != "SourceDate"]] = np.nan
    return merged.sort_values("index").drop(columns=["index"]).reset_index(drop=True)


class SectorPanel:
    """Pre-computed sector series. Official index where configured, otherwise a leave-one-out
    equal-weight composite of peers (the stock itself is excluded so it never explains itself)."""

    MIN_PEERS = 2

    def __init__(self, prices, index_prices=None):
        self.index_prices = index_prices or {}
        self.returns = {}
        for sector, members in SECTOR_MEMBERS.items():
            if sector in SECTOR_INDEX_TICKERS and SECTOR_INDEX_TICKERS[sector] in self.index_prices:
                continue
            series = {m: prices[m].set_index("Date")["Close"] for m in members if m in prices and prices[m] is not None}
            if series:
                self.returns[sector] = np.log(pd.DataFrame(series).sort_index()).diff()

    def source(self, ticker):
        sector = get_sector(ticker)
        if sector is None:
            return None
        if sector in SECTOR_INDEX_TICKERS and SECTOR_INDEX_TICKERS[sector] in self.index_prices:
            return f"index:{SECTOR_INDEX_TICKERS[sector]}"
        return f"peers:{sector}" if sector in self.returns else None

    def features_for(self, ticker):
        sector = get_sector(ticker)
        if sector is None:
            return None
        index_ticker = SECTOR_INDEX_TICKERS.get(sector)
        if index_ticker in self.index_prices:
            idx = self.index_prices[index_ticker]
            return sector_features_from_close(idx["Date"], idx["Close"])
        if sector not in self.returns:
            return None

        rets = self.returns[sector]
        peers = rets.drop(columns=[ticker], errors="ignore")
        count = peers.notna().sum(axis=1)
        loo = peers.mean(axis=1).where(count >= self.MIN_PEERS)
        # Synthetic level from cumulative peer log returns; NaN only before the composite first exists
        level = np.exp(loo.fillna(0).cumsum()).where(loo.notna().cummax())
        return sector_features_from_close(rets.index, level)


@dataclass
class MarketContext:
    """Everything needed to build market features for any stock (shared by training and the app)."""
    nifty: pd.DataFrame = None
    vix: pd.DataFrame = None
    usdinr: pd.DataFrame = None
    sector_panel: SectorPanel = None

    def __post_init__(self):
        self._nifty_feats = nifty_features(self.nifty) if self.nifty is not None else None
        self._vix_feats = vix_features(self.vix) if self.vix is not None else None
        self._fx_feats = usdinr_features(self.usdinr) if self.usdinr is not None else None


def create_market_features(df, context):
    """Add market-context columns to a daily stock frame (needs Date, Close). Missing sources -> NaN columns."""
    out = df.copy()
    dates = out["Date"]
    ticker = out["Stock"].iloc[0] if "Stock" in out.columns else None

    def attach(features, cols, same_session, tolerance):
        if features is None:
            for c in cols:
                out[c] = np.nan
            return
        merged = align_external(dates, features, same_session, tolerance)
        for c in cols:
            out[c] = merged[c].to_numpy()

    attach(context._nifty_feats, NIFTY_FEATURES, True, SAME_SESSION_TOLERANCE)
    attach(context._vix_feats, VIX_FEATURES, True, SAME_SESSION_TOLERANCE)
    attach(context._fx_feats, USDINR_FEATURES, False, FX_TOLERANCE)
    sector_feats = context.sector_panel.features_for(ticker) if context.sector_panel is not None and ticker else None
    attach(sector_feats, SECTOR_FEATURES, True, SAME_SESSION_TOLERANCE)

    stock_ret_1 = np.log(out["Close"]).diff()
    stock_ret_20 = np.log(out["Close"] / out["Close"].shift(20))
    out["stock_vs_nifty_return"] = stock_ret_1 - out["nifty_return_1"]
    out["stock_vs_sector_return"] = stock_ret_1 - out["sector_return_1"]
    out["stock_nifty_relative_strength"] = stock_ret_20 - out["nifty_return_20"]
    out["stock_sector_relative_strength"] = stock_ret_20 - out["sector_return_20"]
    return out.replace([np.inf, -np.inf], np.nan)
