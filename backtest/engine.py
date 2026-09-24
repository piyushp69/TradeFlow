"""Lightweight cross-sectional backtester for the model's out-of-sample predictions.

Strategy: on each rebalance date, rank the universe by P(UP) and hold the top slice until the next
rebalance, equal weight (or inverse-volatility weighted). It is long-only and fully invested while it
holds positions, which matches how the app presents its forecasts.

Every assumption is configurable in `BacktestConfig`. Defaults:
    * signals are computed at the close, orders fill at the NEXT session's open (`execution="next_open"`),
      so nothing trades on a price that was not yet observable when the signal was produced;
    * costs are charged on traded value at each rebalance (brokerage/taxes `transaction_cost_bps`
      plus `slippage_bps`), including the final liquidation.

Known limitation: the universe is today's stock list, so results carry survivorship bias; a stock that was
delisted during the period is absent. The benchmark (equal-weight universe) shares that bias, which is why
the excess return over the benchmark is the more meaningful number.
"""
import logging
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


@dataclass
class BacktestConfig:
    horizon_days: int = 21
    rebalance_days: int = None          # defaults to horizon_days (non-overlapping holding periods)
    top_quantile: float = 0.20          # fraction of the universe held
    entry_percentile: float = None      # alternative rule: hold every stock ranked above this percentile
    exit_percentile: float = None       # position is closed when the rank falls below this (hysteresis)
    position_sizing: str = "equal"      # "equal" or "inverse_vol"
    transaction_cost_bps: float = 10.0
    slippage_bps: float = 5.0
    execution: str = "next_open"        # "next_open" or "close"
    initial_capital: float = 1_000_000
    risk_free_rate: float = 0.0         # annual, for the Sharpe ratio
    min_positions: int = 3

    def __post_init__(self):
        self.rebalance_days = self.rebalance_days or self.horizon_days

    @property
    def cost_rate(self):
        return (self.transaction_cost_bps + self.slippage_bps) / 10_000


def _price_panel(prices, field):
    """Wide Date x Stock frame of a price field."""
    frame = prices.pivot_table(index="Date", columns="Stock", values=field, aggfunc="last")
    return frame.sort_index()


def _execution_prices(config, opens, closes):
    if config.execution == "close":
        return closes
    # Orders placed on the signal date fill at each stock's next traded open. Shifting the whole panel one row
    # would pick up dates only other series trade on (e.g. exchange holidays) and leave the fill empty.
    return opens.apply(lambda col: col.dropna().shift(-1)).reindex(opens.index)


def _weights(config, chosen, vol):
    if config.position_sizing == "inverse_vol" and vol is not None:
        inv = 1 / vol.reindex(chosen).replace(0, np.nan)
        inv = inv.replace([np.inf, -np.inf], np.nan).dropna()
        if not inv.empty:
            return (inv / inv.sum()).reindex(chosen).fillna(0)
    return pd.Series(1 / len(chosen), index=chosen)


def _select(config, day_preds):
    ranks = day_preds["p"].rank(pct=True) * 100
    if config.entry_percentile is not None:
        chosen = day_preds.index[ranks >= config.entry_percentile]
    else:
        chosen = day_preds.index[ranks > (1 - config.top_quantile) * 100]
    return list(chosen)


def _metrics(returns, equity, trades, config, periods_per_year):
    returns = pd.Series(returns).astype(float)
    if returns.empty:
        return {}
    total_return = float(equity.iloc[-1] / config.initial_capital - 1)
    years = max(len(returns) / periods_per_year, 1e-9)
    ann_return = float((1 + total_return) ** (1 / years) - 1) if total_return > -1 else -1.0
    ann_vol = float(returns.std(ddof=1) * np.sqrt(periods_per_year)) if len(returns) > 1 else 0.0
    rf_period = config.risk_free_rate / periods_per_year
    sharpe = float((returns.mean() - rf_period) / returns.std(ddof=1) * np.sqrt(periods_per_year)) \
        if len(returns) > 1 and returns.std(ddof=1) > 0 else None
    drawdown = float((equity / equity.cummax() - 1).min())

    trade_returns = trades["net_return"] if not trades.empty else pd.Series(dtype=float)
    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]
    return {
        "cumulative_return": round(total_return, 4),
        "annualised_return": round(ann_return, 4),
        "annualised_volatility": round(ann_vol, 4),
        "sharpe_ratio": None if sharpe is None else round(sharpe, 3),
        "max_drawdown": round(drawdown, 4),
        "win_rate": round(float((trade_returns > 0).mean()), 4) if len(trade_returns) else None,
        "profit_factor": round(float(wins.sum() / abs(losses.sum())), 3) if losses.sum() else None,
        "number_of_trades": int(len(trade_returns)),
        "average_trade_return": round(float(trade_returns.mean()), 4) if len(trade_returns) else None,
        "periods": int(len(returns)),
        "total_cost_paid": round(float(trades["cost"].sum()) if not trades.empty else 0.0, 4),
    }


def run_backtest(predictions, prices, config=None):
    """Backtest cross-sectional predictions.

    predictions: DataFrame [Date, Stock, p] - out-of-sample probabilities only.
    prices:      long DataFrame [Date, Stock, Open, Close] covering the period plus one extra session.
    Returns dict with metrics, benchmark metrics, the equity curve and per-trade records.
    """
    config = config or BacktestConfig()
    preds = predictions.dropna(subset=["p"]).copy()
    preds["Date"] = pd.to_datetime(preds["Date"])
    prices = prices.copy()
    prices["Date"] = pd.to_datetime(prices["Date"])

    closes = _price_panel(prices, "Close")
    opens = _price_panel(prices, "Open")
    fills = _execution_prices(config, opens, closes)

    dates = np.sort(preds["Date"].unique())
    rebalance_dates = list(dates[::config.rebalance_days])
    if len(rebalance_dates) < 2:
        return {"metrics": {}, "benchmark": {}, "equity_curve": pd.DataFrame(), "trades": pd.DataFrame(),
                "config": asdict(config), "note": "not enough rebalance dates for a backtest"}

    vol_panel = None
    if config.position_sizing == "inverse_vol":
        vol_panel = np.log(closes).diff().rolling(20).std()

    equity, bench_equity = config.initial_capital, config.initial_capital
    equity_rows, trade_rows, period_returns, bench_returns = [], [], [], []
    held = {}

    for i, date in enumerate(rebalance_dates[:-1]):
        nxt = rebalance_dates[i + 1]
        day = preds[preds["Date"] == date].set_index("Stock")
        if day.empty:
            continue

        entry_prices = fills.loc[date].dropna() if date in fills.index else pd.Series(dtype=float)
        exit_prices = fills.loc[nxt].dropna() if nxt in fills.index else pd.Series(dtype=float)
        tradable = day.index.intersection(entry_prices.index).intersection(exit_prices.index)
        if len(tradable) < config.min_positions:
            continue
        day = day.loc[tradable]

        chosen = _select(config, day)
        if config.exit_percentile is not None:
            ranks = day["p"].rank(pct=True) * 100
            chosen = sorted(set(chosen) | {s for s in held if s in ranks.index and ranks[s] >= config.exit_percentile})
        if len(chosen) < config.min_positions:
            chosen = list(day["p"].nlargest(config.min_positions).index)

        vol = vol_panel.loc[date] if vol_panel is not None and date in vol_panel.index else None
        weights = _weights(config, chosen, vol)

        gross = (exit_prices[chosen] / entry_prices[chosen] - 1)
        # Turnover against the previous (drifted) portfolio, charged both when entering and leaving
        previous = pd.Series(held, dtype=float)
        turnover = (weights.subtract(previous, fill_value=0).abs().sum())
        cost = turnover * config.cost_rate
        period_return = float((weights * gross).sum() - cost)
        period_returns.append(period_return)
        equity *= (1 + period_return)
        equity_rows.append({"Date": nxt, "equity": equity, "positions": len(chosen), "turnover": turnover,
                            "cost": cost, "period_return": period_return})

        for stock in chosen:
            trade_rows.append({"entry_date": date, "exit_date": nxt, "stock": stock,
                               "weight": float(weights[stock]), "gross_return": float(gross[stock]),
                               "net_return": float(gross[stock] - 2 * config.cost_rate),
                               "cost": float(weights[stock] * 2 * config.cost_rate)})

        bench_gross = float((exit_prices[day.index] / entry_prices[day.index] - 1).mean())
        bench_returns.append(bench_gross)
        bench_equity *= (1 + bench_gross)
        equity_rows[-1]["benchmark_equity"] = bench_equity
        # Weights drift with each position's return before the next rebalance
        drifted = weights * (1 + gross)
        held = (drifted / drifted.sum()).to_dict() if drifted.sum() > 0 else {}

    if not equity_rows:
        return {"metrics": {}, "benchmark": {}, "equity_curve": pd.DataFrame(), "trades": pd.DataFrame(),
                "config": asdict(config), "note": "no rebalance date had enough tradable stocks"}

    # Final liquidation: selling the last portfolio costs the same rate as any other trade
    exit_cost = sum(held.values()) * config.cost_rate
    last = equity_rows[-1]
    period_returns[-1] = (1 + period_returns[-1]) * (1 - exit_cost) - 1
    equity *= (1 - exit_cost)
    last.update(equity=equity, turnover=last["turnover"] + sum(held.values()), cost=last["cost"] + exit_cost,
                period_return=period_returns[-1])

    equity_curve = pd.DataFrame(equity_rows)
    trades = pd.DataFrame(trade_rows)
    periods_per_year = TRADING_DAYS / config.rebalance_days
    metrics = _metrics(period_returns, equity_curve["equity"], trades, config, periods_per_year)
    bench_metrics = _metrics(bench_returns, equity_curve["benchmark_equity"], pd.DataFrame(), config, periods_per_year)
    if metrics and bench_metrics:
        metrics["excess_return_vs_benchmark"] = round(metrics["cumulative_return"] - bench_metrics["cumulative_return"], 4)

    return {"metrics": metrics, "benchmark": bench_metrics, "equity_curve": equity_curve, "trades": trades,
            "config": asdict(config)}


def run_single_stock_backtest(predictions, prices, stock, config=None):
    """Long/flat backtest for one stock: hold it while its cross-sectional rank stays above entry_percentile."""
    config = config or BacktestConfig(entry_percentile=60, exit_percentile=50, min_positions=1)
    ranks = predictions.copy()
    ranks["Date"] = pd.to_datetime(ranks["Date"])
    ranks["percentile"] = ranks.groupby("Date")["p"].rank(pct=True) * 100
    one = ranks[ranks["Stock"] == stock]
    if one.empty:
        return None

    prices = prices[prices["Stock"] == stock].copy()
    prices["Date"] = pd.to_datetime(prices["Date"])
    closes = _price_panel(prices, "Close")[stock]
    opens = _price_panel(prices, "Open")[stock]
    fills = opens.shift(-1) if config.execution == "next_open" else closes

    dates = np.sort(one["Date"].unique())[::config.rebalance_days]
    equity, rows, trades, returns = config.initial_capital, [], [], []
    holding = False
    for i, date in enumerate(dates[:-1]):
        nxt = dates[i + 1]
        if date not in fills.index or nxt not in fills.index or pd.isna(fills[date]) or pd.isna(fills[nxt]):
            continue
        pct = float(one.loc[one["Date"] == date, "percentile"].iloc[0])
        want = pct >= config.entry_percentile or (holding and pct >= (config.exit_percentile or 50))
        gross = float(fills[nxt] / fills[date] - 1) if want else 0.0
        cost = config.cost_rate * (0 if want == holding else 1)
        period = gross - cost
        equity *= (1 + period)
        returns.append(period)
        rows.append({"Date": nxt, "equity": equity, "in_position": want, "period_return": period, "cost": cost})
        if want:
            trades.append({"entry_date": date, "exit_date": nxt, "stock": stock, "weight": 1.0,
                           "gross_return": gross, "net_return": gross - 2 * config.cost_rate,
                           "cost": 2 * config.cost_rate})
        holding = want

    if not rows:
        return None
    curve = pd.DataFrame(rows)
    buy_hold = (fills.reindex(dates).dropna())
    bench_curve = config.initial_capital * (buy_hold / buy_hold.iloc[0])
    periods_per_year = TRADING_DAYS / config.rebalance_days
    return {
        "metrics": _metrics(returns, curve["equity"], pd.DataFrame(trades), config, periods_per_year),
        "benchmark": _metrics(buy_hold.pct_change().dropna().tolist(), bench_curve, pd.DataFrame(), config,
                              periods_per_year),
        "equity_curve": curve, "trades": pd.DataFrame(trades), "config": asdict(config),
    }
