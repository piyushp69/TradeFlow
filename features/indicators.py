import pandas as pd
import numpy as np

def calculate_indicators(df):
    if df is None or len(df) < 50:
        return None
    
    # Trend Ratios
    df["SMA20_Ratio"] = df["Close"] / df["Close"].rolling(window=20).mean()
    df["SMA50_Ratio"] = df["Close"] / df["Close"].rolling(window=50).mean()
    
    # Risk and Momentum
    df["Daily_Return"] = df["Close"].pct_change()
    df["Volatility"] = df["Daily_Return"].rolling(window=20).std()
    
    # RSI
    diff = df["Close"].diff()
    up = diff.clip(lower=0).rolling(window=14).mean()
    down = -diff.clip(upper=0).rolling(window=14).mean()
    df["RSI"] = 100 - (100 / (1 + (up / (down + 1e-10))))

    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Ratio'] = (exp1 - exp2) / df['Close']
    
    return df.dropna()