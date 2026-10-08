import numpy as np
import pandas as pd

from features.cross_section import XS_SUFFIX, add_cross_sectional_ranks, is_stock_level, model_columns


def _frame():
    return pd.DataFrame({
        "Date": pd.to_datetime(["2024-01-01"] * 4 + ["2024-01-02"] * 4),
        "Stock": list("ABCD") * 2,
        "Ret_5": [0.01, 0.03, -0.02, np.nan, 0.50, 0.40, 0.30, 0.20],
        "nifty_return_5": [0.01] * 4 + [0.02] * 4,
    })


def test_ranks_use_only_the_same_day_cross_section():
    out = add_cross_sectional_ranks(_frame(), ["Ret_5", "nifty_return_5"])
    day1 = out[out["Date"] == "2024-01-01"]["Ret_5" + XS_SUFFIX].tolist()
    day2 = out[out["Date"] == "2024-01-02"]["Ret_5" + XS_SUFFIX].tolist()
    np.testing.assert_allclose(day1[:3], [2 / 3, 1.0, 1 / 3], rtol=1e-6)
    assert np.isnan(day1[3])
    # Day 2's huge values do not change day 1's ranks, and vice versa
    np.testing.assert_allclose(day2, [1.0, 0.75, 0.5, 0.25], rtol=1e-6)


def test_market_level_features_are_not_ranked():
    out = add_cross_sectional_ranks(_frame(), ["Ret_5", "nifty_return_5"])
    assert "nifty_return_5" + XS_SUFFIX not in out
    assert not is_stock_level("nifty_return_5") and is_stock_level("Ret_5")
    assert model_columns(["Ret_5", "nifty_return_5"], True) == ["Ret_5" + XS_SUFFIX, "nifty_return_5"]
    assert model_columns(["Ret_5", "nifty_return_5"], False) == ["Ret_5", "nifty_return_5"]
