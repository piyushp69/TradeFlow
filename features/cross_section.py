"""Cross-sectional feature transforms.

`add_cross_sectional_ranks` replaces a stock-level feature with its percentile rank among all stocks on the same
day (0..1, NaN stays NaN). A rank for date D uses only values observed on D, so it cannot leak across time.

Market-level features (NIFTY, India VIX, USD/INR) take the same value for every stock on a day, so a rank would be
constant. They keep their raw values and give the model the market regime.
"""
from features.indicators import MARKET_FEATURES as BASELINE_MARKET_FEATURES
from features.market import NIFTY_FEATURES, USDINR_FEATURES, VIX_FEATURES

XS_SUFFIX = "__xs"

# Baseline Mkt_* columns are NIFTY-only; Rel_Ret_* compare the stock with NIFTY and vary across stocks
MARKET_LEVEL_FEATURES = frozenset(
    [f for f in BASELINE_MARKET_FEATURES if f.startswith("Mkt_")] + NIFTY_FEATURES + VIX_FEATURES + USDINR_FEATURES)


def is_stock_level(feature):
    return feature not in MARKET_LEVEL_FEATURES


def model_columns(features, cross_sectional):
    """Column names a model trains on: ranked versions of stock-level features when cross_sectional is set."""
    return [f + XS_SUFFIX if cross_sectional and is_stock_level(f) else f for f in features]


def add_cross_sectional_ranks(panel, features, date_col="Date"):
    """Add `<feature>__xs` percentile-rank columns for every stock-level feature."""
    ranked = {}
    by_day = panel.groupby(date_col)
    for feature in features:
        if is_stock_level(feature) and feature + XS_SUFFIX not in panel:
            ranked[feature + XS_SUFFIX] = by_day[feature].rank(pct=True).astype("float32")
    if not ranked:
        return panel
    import pandas as pd
    return pd.concat([panel, pd.DataFrame(ranked, index=panel.index)], axis=1)
