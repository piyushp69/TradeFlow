import pandas as pd
import numpy as np

# Forecast horizons in trading days
HORIZONS = {"weekly": 5, "monthly": 21, "yearly": 252}

# Minimum forward return for a period to count as "UP"
UP_THRESHOLD = 0.005

# All features are scale-free (ratios / returns) so one model can be pooled across stocks
FEATURES = [
    # Momentum
    "Ret_1", "Ret_5", "Ret_21", "Ret_63", "Ret_126", "Ret_252", "Ret21_Z",
    # Trend
    "SMA20_Ratio", "SMA50_Ratio", "SMA200_Ratio", "SMA50_200_Ratio", "MACD_Ratio", "MACD_Hist_Ratio",
    # Oscillators / range position
    "RSI", "Stoch_K", "BB_PctB", "High252_Dist", "Low252_Dist",
    # Risk
    "Volatility", "Vol_60", "Vol_Ratio", "ATR_Ratio",
    # Volume
    "Volume_Ratio",
    # Market context (NIFTY 50)
    "Mkt_Ret_21", "Mkt_Ret_63", "Mkt_SMA200_Ratio", "Mkt_Vol_20", "Rel_Ret_21", "Rel_Ret_63",
]

# Compact short-term set: weekly moves are driven by short-term mean reversion, and the
# full set tends to overfit at that horizon. The trainer picks the set per horizon via walk-forward CV.
SHORT_TERM_FEATURES = ["Ret_1", "Ret_5", "RSI", "Volatility", "SMA20_Ratio", "SMA50_Ratio", "MACD_Ratio"]

FEATURE_SETS = {"full": FEATURES, "short_term": SHORT_TERM_FEATURES}

MARKET_FEATURES = ["Mkt_Ret_21", "Mkt_Ret_63", "Mkt_SMA200_Ratio", "Mkt_Vol_20", "Rel_Ret_21", "Rel_Ret_63"]

MARKET_TICKER = "^NSEI"
MIN_HISTORY = 260


def _rsi(close, window=14):
    diff = close.diff()
    gain = diff.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = -diff.clip(upper=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-10)))


def calculate_market_features(market_df):
    close = market_df["Close"]
    log_ret = np.log(close).diff()
    return pd.DataFrame({
        "Date": market_df["Date"],
        "Mkt_Ret_21": np.log(close / close.shift(21)),
        "Mkt_Ret_63": np.log(close / close.shift(63)),
        "Mkt_SMA200_Ratio": close / close.rolling(200).mean(),
        "Mkt_Vol_20": log_ret.rolling(20).std(),
    })


def calculate_indicators(df, market_df=None):
    if df is None or len(df) < MIN_HISTORY:
        return None

    df = df.copy()
    close, high, low = df["Close"], df["High"], df["Low"]
    log_ret = np.log(close).diff()

    # Momentum
    df["Daily_Return"] = close.pct_change()
    df["Ret_1"] = log_ret
    for n in (5, 21, 63, 126, 252):
        df[f"Ret_{n}"] = np.log(close / close.shift(n))

    # Risk
    df["Volatility"] = log_ret.rolling(20).std()
    df["Vol_60"] = log_ret.rolling(60).std()
    df["Vol_Ratio"] = df["Volatility"] / (df["Vol_60"] + 1e-10)
    prev_close = close.shift(1)
    true_range = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    df["ATR_Ratio"] = true_range.rolling(14).mean() / close
    df["Ret21_Z"] = df["Ret_21"] / (df["Volatility"] * np.sqrt(21) + 1e-10)

    # Trend Ratios
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    df["SMA20_Ratio"] = close / sma20
    df["SMA50_Ratio"] = close / sma50
    df["SMA200_Ratio"] = close / sma200
    df["SMA50_200_Ratio"] = sma50 / sma200

    # MACD
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    df["MACD_Ratio"] = macd / close
    df["MACD_Hist_Ratio"] = (macd - signal) / close

    # Oscillators
    df["RSI"] = _rsi(close)
    low14, high14 = low.rolling(14).min(), high.rolling(14).max()
    df["Stoch_K"] = 100 * (close - low14) / (high14 - low14 + 1e-10)
    std20 = close.rolling(20).std()
    df["BB_PctB"] = (close - (sma20 - 2 * std20)) / (4 * std20 + 1e-10)
    df["High252_Dist"] = close / high.rolling(252).max() - 1
    df["Low252_Dist"] = close / low.rolling(252).min() - 1

    # Volume
    volume = df["Volume"].astype(float)
    df["Volume_Ratio"] = np.log((volume.rolling(5).mean() + 1) / (volume.rolling(60).mean() + 1))

    # Market context
    if market_df is not None and not market_df.empty:
        mkt = calculate_market_features(market_df)
        df = df.merge(mkt, on="Date", how="left")
        mkt_cols = [c for c in mkt.columns if c != "Date"]
        df[mkt_cols] = df[mkt_cols].ffill()
    else:
        # The model handles missing values natively, so predictions still work without market data
        for col in ("Mkt_Ret_21", "Mkt_Ret_63", "Mkt_SMA200_Ratio", "Mkt_Vol_20"):
            df[col] = np.nan
    df["Rel_Ret_21"] = df["Ret_21"] - df["Mkt_Ret_21"]
    df["Rel_Ret_63"] = df["Ret_63"] - df["Mkt_Ret_63"]

    df = df.replace([np.inf, -np.inf], np.nan)
    stock_features = [f for f in FEATURES if f not in MARKET_FEATURES]
    return df.dropna(subset=stock_features).reset_index(drop=True)


def add_targets(df):
    """Label each row UP (1) / DOWN (0) per horizon. Rows whose future is unknown stay NaN."""
    for name, days in HORIZONS.items():
        future = df["Close"].shift(-days)
        target = (future > df["Close"] * (1 + UP_THRESHOLD)).astype(float)
        target[future.isna()] = np.nan
        df[f"Target_{name.capitalize()}"] = target
    return df
