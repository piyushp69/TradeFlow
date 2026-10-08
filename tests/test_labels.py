import numpy as np
import pandas as pd
import pytest

from conftest import make_panel
from features.indicators import UP_THRESHOLD, add_targets
from features.labels import add_labels, clear_move_mask, label_column, target_move

HORIZONS = {"weekly": 5, "monthly": 21}


def _one(labelled, stock):
    return labelled[labelled["Stock"] == stock].sort_values("Date").reset_index(drop=True)


def test_forward_return_and_absolute_label_use_the_same_stock_h_rows_ahead(panel):
    out = add_labels(panel, HORIZONS)
    for stock in out["Stock"].unique():
        s = _one(out, stock)
        for name, h in HORIZONS.items():
            expected = s["Close"].shift(-h) / s["Close"] - 1
            np.testing.assert_allclose(s[f"fwd_ret_{name}"], expected, equal_nan=True)
            known = expected.notna()
            assert (s.loc[known, f"y_abs_{name}"] == (expected[known] > UP_THRESHOLD).astype(float)).all()
            assert s.loc[known, f"label_end_{name}"].equals(s["Date"].shift(-h)[known])


def test_rows_without_a_known_future_have_no_label(panel):
    out = add_labels(panel, HORIZONS)
    for stock in out["Stock"].unique():
        s = _one(out, stock)
        for name, h in HORIZONS.items():
            tail = s.tail(h)
            assert tail[f"y_abs_{name}"].isna().all() and tail[f"y_rel_{name}"].isna().all()
            assert tail[f"label_end_{name}"].isna().all() and tail[f"label_end_rel_{name}"].isna().all()
            assert s.iloc[:-h][f"y_abs_{name}"].notna().all()


def test_labels_never_mix_stocks_even_when_input_rows_are_shuffled():
    a = make_panel(n_stocks=1, seed=1).assign(Stock="A")
    b = make_panel(n_stocks=1, seed=2).assign(Stock="B")
    b["Close"] *= 50   # very different price level: any cross-stock shift would create huge returns
    out = add_labels(pd.concat([a, b]).sample(frac=1.0, random_state=3), HORIZONS)
    assert out["fwd_ret_monthly"].abs().max() < 1.0
    solo = add_labels(a, HORIZONS)
    np.testing.assert_allclose(_one(out, "A")["fwd_ret_weekly"], _one(solo, "A")["fwd_ret_weekly"], equal_nan=True)


def test_label_end_follows_the_stocks_own_rows_across_a_gap():
    panel = make_panel(n_stocks=1, n_days=30)
    s = panel.sort_values("Date").reset_index(drop=True)
    gapped = s.drop(index=[10, 11, 12]).reset_index(drop=True)   # three missing sessions (e.g. a trading halt)
    out = _one(add_labels(gapped, {"weekly": 5}), "S0")
    row = out[out["Date"] == s.loc[8, "Date"]].iloc[0]
    # five of this stock's rows ahead is calendar session 16, not 13
    assert row["label_end_weekly"] == s.loc[16, "Date"]


def test_relative_label_is_beat_the_same_day_median(panel):
    out = add_labels(panel, HORIZONS)
    for name in HORIZONS:
        known = out[out[f"y_rel_{name}"].notna()]
        median = known.groupby("Date")[f"fwd_ret_{name}"].transform("median")
        assert (known[f"y_rel_{name}"] == (known[f"fwd_ret_{name}"] > median).astype(float)).all()
        np.testing.assert_allclose(known[f"excess_ret_{name}"], known[f"fwd_ret_{name}"] - median)
        # Balanced by construction: with 6 stocks exactly 3 beat the median every day
        assert (known.groupby("Date")[f"y_rel_{name}"].sum() == 3).all()


def test_relative_label_end_covers_every_stock_used_by_the_median():
    panel = make_panel(n_stocks=4, n_days=40)
    halted = panel[~((panel["Stock"] == "S1") & panel["Date"].between("2020-01-10", "2020-01-16"))]
    out = add_labels(halted, {"weekly": 5})
    known = out[out["y_rel_weekly"].notna()]
    assert (known["label_end_rel_weekly"] >= known["label_end_weekly"]).all()
    per_day_max = known.groupby("Date")["label_end_weekly"].transform("max")
    assert (known["label_end_rel_weekly"] == per_day_max).all()
    assert (known["label_end_rel_weekly"] > known["label_end_weekly"]).any()


def test_open_entry_return_starts_at_the_next_sessions_open(panel):
    out = add_labels(panel, {"weekly": 5})
    s = _one(out, "S2")
    expected = s["Close"].shift(-5) / s["Open"].shift(-1) - 1
    np.testing.assert_allclose(s["fwd_ret_open_weekly"], expected, equal_nan=True)


def test_panel_label_matches_the_per_stock_add_targets(panel):
    out = add_labels(panel)
    for stock in out["Stock"].unique():
        s = _one(out, stock)
        legacy = add_targets(s[["Date", "Close"]].copy())
        for name in ("weekly", "monthly"):
            np.testing.assert_array_equal(legacy[f"Target_{name.capitalize()}"].to_numpy(),
                                          s[f"y_abs_{name}"].to_numpy())


def test_clear_move_mask_drops_only_small_moves_relative_to_volatility():
    move = np.array([0.001, -0.002, 0.05, -0.06, np.nan, 0.03])
    vol = np.array([0.02, 0.02, 0.02, 0.02, 0.02, np.nan])
    keep = clear_move_mask(move, vol, horizon_days=5, k=0.5)   # boundary 0.5 * 0.02 * sqrt(5) = 0.0224
    assert keep.tolist() == [False, False, True, True, True, True]


def test_target_helpers():
    out = add_labels(make_panel(), {"weekly": 5})
    assert label_column("relative", "weekly") == "y_rel_weekly"
    assert label_column("absolute", "weekly") == "y_abs_weekly"
    np.testing.assert_allclose(target_move(out, "absolute", "weekly"), out["fwd_ret_weekly"] - UP_THRESHOLD)
    with pytest.raises(ValueError):
        label_column("excess", "weekly")
