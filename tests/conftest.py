import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_panel(n_stocks=6, n_days=60, seed=0, start="2020-01-01"):
    """Random-walk OHLC panel: one row per stock per business day, rows deliberately shuffled."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    frames = []
    for i in range(n_stocks):
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n_days)))
        frames.append(pd.DataFrame({
            "Date": dates, "Stock": f"S{i}", "Close": close,
            "Open": close * np.exp(rng.normal(0, 0.005, n_days)),
            "Volatility": np.full(n_days, 0.02),
        }))
    panel = pd.concat(frames, ignore_index=True)
    return panel.sample(frac=1.0, random_state=seed).reset_index(drop=True)


@pytest.fixture
def panel():
    return make_panel()
