import numpy as np
import pandas as pd

from conftest import make_panel
from features.labels import add_labels
from models.evaluation import daily_rank_ic, edge_verdict, evaluate_block, newey_west_tstat


def _cross_section(n_days=30, n_stocks=50, seed=0):
    rng = np.random.default_rng(seed)
    dates = np.repeat(pd.bdate_range("2021-01-01", periods=n_days), n_stocks)
    market = np.repeat(rng.normal(0, 0.05, n_days), n_stocks)
    stock_part = rng.normal(0, 0.03, n_days * n_stocks)
    return dates, market, stock_part


def test_rank_ic_measures_the_cross_section_only():
    dates, market, stock_part = _cross_section()
    returns = market + stock_part
    assert np.allclose(daily_rank_ic(dates, stock_part, returns), 1.0)
    assert np.allclose(daily_rank_ic(dates, -stock_part, returns), -1.0)
    # A forecast of the market move alone is identical for every stock on a day: no cross-sectional information
    noise = np.random.default_rng(1).normal(0, 1e-9, len(market))
    assert abs(daily_rank_ic(dates, market + noise, returns).mean()) < 0.05


def test_a_constant_forecast_scores_zero_instead_of_being_skipped():
    dates, market, stock_part = _cross_section(n_days=5)
    ic = daily_rank_ic(dates, np.ones(len(dates)), market + stock_part)
    assert len(ic) == 5 and (ic == 0).all()


def test_rank_ic_skips_days_with_too_few_stocks():
    dates, market, stock_part = _cross_section(n_days=3, n_stocks=10)
    assert daily_rank_ic(dates, stock_part, market + stock_part, min_stocks=20).empty


def test_evaluate_block_on_a_perfect_relative_forecast():
    panel = add_labels(make_panel(n_stocks=30, n_days=80), {"weekly": 5}).dropna(subset=["y_rel_weekly"])
    metrics, ic = evaluate_block(panel, panel["excess_ret_weekly"].to_numpy(), "weekly")
    assert metrics["auc_rel"] == 1.0 and np.isclose(metrics["rank_ic"], 1.0)
    assert metrics["top_hit_rel"] == 1.0 and metrics["bottom_hit_rel"] == 0.0
    assert metrics["quintile_spread"] > 0 and len(ic) == metrics["days"]


def test_newey_west_widens_the_standard_error_of_autocorrelated_series():
    rng = np.random.default_rng(0)
    smooth = np.convolve(rng.normal(0.02, 0.1, 600), np.ones(20) / 20, mode="valid")
    assert newey_west_tstat(smooth, lag=20) < newey_west_tstat(smooth, lag=0)


def test_edge_verdict_needs_folds_test_and_significance():
    folds = [{"rank_ic": v} for v in (0.03, 0.02, 0.04, -0.01, 0.03)]
    strong = pd.Series(np.full(500, 0.03) + np.random.default_rng(0).normal(0, 0.01, 500))
    assert edge_verdict(folds, {"rank_ic": 0.02}, strong, 5)["reliable"]
    assert not edge_verdict(folds, {"rank_ic": -0.01}, strong, 5)["reliable"]
    weak_folds = [{"rank_ic": v} for v in (0.03, -0.02, 0.04, -0.01, 0.03)]
    verdict = edge_verdict(weak_folds, {"rank_ic": 0.02}, strong, 5)
    assert not verdict["reliable"] and "3 of 5" in verdict["reasons"][0]
    noisy = pd.Series(np.random.default_rng(1).normal(0.001, 0.2, 500))
    assert not edge_verdict(folds, {"rank_ic": 0.02}, noisy, 5)["reliable"]
