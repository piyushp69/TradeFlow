"""Purged walk-forward validation, cross-sectional metrics and the edge rule.

Splits (`purged_walk_forward`)
------------------------------
For one horizon of h sessions:

* Final test block: the most recent `test_fraction` of labelled sessions. It is returned only when
  `include_test=True`, which only the trainer's final evaluation passes. Experiments never see it.
* Development rows: every row dated before the test block whose label ends before the test block starts, so no
  development decision depends on a test-period price.
* `n_folds` validation blocks: consecutive, equal-length blocks covering the later part of the development period.
* Training rows for an evaluation block starting at session S (expanding window):
    - dated at least h sessions before S (a gap of at least the label horizon), and
    - with a label window that ends before S (purge: no training label overlaps the evaluated period).
  Validation blocks never train on later data, so no embargo after a block is needed.

`purge="rows"` reproduces the previous pipeline (date gap only, no label-end check) for comparison.

Metrics (`evaluate_block`), all computed on one evaluation block
----------------------------------------------------------------
auc_rel / auc_abs   ROC AUC against "beats the universe median" and against "rises more than 0.5%"
rank_ic             Spearman correlation between predicted probability and realised forward return across stocks
                    on each day, averaged over days
rank_ic_open        the same against the return entered at the next session's open
top/bottom_hit_*    share of the day's top / bottom prediction quintile that beat the median (rel) or rose (abs)
quintile_spread     mean forward return of the top quintile minus the bottom quintile

Edge rule (`edge_verdict`), fixed before the final test was scored
------------------------------------------------------------------
A horizon has a reliable edge only if all three hold:
1. rank IC is positive on average over the walk-forward folds and in at least 4 of the 5 folds;
2. rank IC is positive on the final test block;
3. the daily rank IC over every out-of-sample day (validation folds and test) has a Newey-West t-statistic of at
   least 2, with the lag set to the horizon because consecutive days share most of their forward window.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

N_FOLDS = 5
TEST_FRACTION = 0.15
FIRST_VAL_FRACTION = 0.40
MIN_STOCKS_PER_DAY = 20

EDGE_MIN_POSITIVE_FOLDS = 4
EDGE_MIN_TSTAT = 2.0

FOLD_METRICS = ("auc_rel", "auc_abs", "rank_ic", "rank_ic_open", "ic_ir", "top_hit_rel", "bottom_hit_rel",
                "top_hit_abs", "bottom_hit_abs", "quintile_spread")


def _datetimes(values):
    return pd.to_datetime(pd.Series(values)).to_numpy(dtype="datetime64[ns]")


@dataclass
class Split:
    name: str
    train: np.ndarray          # positional row indices
    eval: np.ndarray
    train_period: tuple
    eval_period: tuple
    gap_sessions: int          # sessions strictly between the last training date and the first evaluation date
    max_train_label_end: str

    def describe(self):
        return {"name": self.name, "train_period": list(self.train_period), "eval_period": list(self.eval_period),
                "train_rows": int(len(self.train)), "eval_rows": int(len(self.eval)),
                "gap_sessions": self.gap_sessions, "max_train_label_end": self.max_train_label_end}


def _day(value):
    return f"{pd.Timestamp(value):%Y-%m-%d}"


def _train_mask(dates, label_end, labelled, sessions, eval_start, horizon_days, purge):
    start_pos = int(np.searchsorted(sessions, eval_start))
    last_pos = start_pos - horizon_days - 1
    if last_pos < 0:
        return np.zeros(len(dates), dtype=bool)
    mask = labelled & (dates <= sessions[last_pos])
    if purge == "label_end":
        mask &= label_end < eval_start
    elif purge != "rows":
        raise ValueError(f"unknown purge mode {purge!r}")
    return mask


def _make_split(name, train_mask, eval_mask, dates, label_end, sessions):
    train, evaluate = np.flatnonzero(train_mask), np.flatnonzero(eval_mask)
    if len(train) == 0 or len(evaluate) == 0:
        raise ValueError(f"{name}: empty training or evaluation set")
    last_train, first_eval = dates[train].max(), dates[evaluate].min()
    gap = int(np.searchsorted(sessions, first_eval) - np.searchsorted(sessions, last_train) - 1)
    return Split(name, train, evaluate, (_day(dates[train].min()), _day(last_train)),
                 (_day(first_eval), _day(dates[evaluate].max())), gap, _day(label_end[train].max()))


def purged_walk_forward(dates, label_end, horizon_days, n_folds=N_FOLDS, test_fraction=TEST_FRACTION,
                        first_val_fraction=FIRST_VAL_FRACTION, purge="label_end", include_test=False):
    """Return (folds, test_split). `test_split` is None unless include_test=True.

    dates: row dates. label_end: date of the last price each row's label uses (NaT for unlabelled rows,
    which are never used). Indices refer to positions in these arrays.
    """
    dates, label_end = _datetimes(dates), _datetimes(label_end)
    labelled = ~np.isnat(label_end)
    sessions = np.unique(dates[labelled])
    if len(sessions) < 10 * n_folds:
        raise ValueError("not enough labelled sessions for walk-forward validation")
    test_start = sessions[int(len(sessions) * (1 - test_fraction))]

    development = labelled & (dates < test_start) & (label_end < test_start)
    dev_sessions = np.unique(dates[development])
    blocks = np.array_split(dev_sessions[int(len(dev_sessions) * first_val_fraction):], n_folds)

    folds = []
    for k, block in enumerate(blocks, 1):
        eval_mask = development & (dates >= block[0]) & (dates <= block[-1])
        train_mask = _train_mask(dates, label_end, labelled, sessions, block[0], horizon_days, purge)
        folds.append(_make_split(f"fold{k}", train_mask, eval_mask, dates, label_end, sessions))

    test = None
    if include_test:
        test_mask = labelled & (dates >= test_start)
        train_mask = _train_mask(dates, label_end, labelled, sessions, test_start, horizon_days, purge)
        test = _make_split("test", train_mask, test_mask, dates, label_end, sessions)
    return folds, test


# Metrics

def daily_rank_ic(dates, proba, returns, min_stocks=MIN_STOCKS_PER_DAY):
    """Spearman correlation between predictions and realised returns across stocks, one value per day."""
    frame = pd.DataFrame({"d": _datetimes(dates), "p": np.asarray(proba, float), "r": np.asarray(returns, float)})
    frame = frame.dropna()
    frame = frame[frame.groupby("d")["p"].transform("size") >= min_stocks]
    if frame.empty:
        return pd.Series(dtype=float)
    by_day = frame.groupby("d")
    rp = by_day["p"].rank()
    rr = by_day["r"].rank()
    rp = rp - rp.groupby(frame["d"]).transform("mean")
    rr = rr - rr.groupby(frame["d"]).transform("mean")
    numerator = (rp * rr).groupby(frame["d"]).sum()
    p_spread = (rp ** 2).groupby(frame["d"]).sum()
    r_spread = (rr ** 2).groupby(frame["d"]).sum()
    ic = numerator / np.sqrt(p_spread * r_spread).replace(0, np.nan)
    # A day on which every stock gets the same score carries no ranking information: it counts as IC 0, because
    # skipping it would credit a model for the days it abstains
    ic[(p_spread == 0) & (r_spread > 0)] = 0.0
    return ic.dropna()


def _safe_auc(y, p):
    y = np.asarray(y)
    mask = ~np.isnan(y)
    if mask.sum() == 0 or len(np.unique(y[mask])) < 2:
        return np.nan
    return float(roc_auc_score(y[mask].astype(int), np.asarray(p)[mask]))


def evaluate_block(frame, proba, horizon):
    """Metrics for one evaluation block. `frame` holds the block's rows (Date and the horizon's label columns)."""
    proba = np.asarray(proba, dtype=float)
    dates = frame["Date"].to_numpy()
    fwd = frame[f"fwd_ret_{horizon}"].to_numpy(dtype=float)
    y_rel = frame[f"y_rel_{horizon}"].to_numpy(dtype=float)
    y_abs = frame[f"y_abs_{horizon}"].to_numpy(dtype=float)

    ic = daily_rank_ic(dates, proba, fwd)
    ic_open = daily_rank_ic(dates, proba, frame[f"fwd_ret_open_{horizon}"].to_numpy(dtype=float))

    pct = pd.Series(proba).groupby(pd.Series(dates)).rank(pct=True).to_numpy()
    top, bottom = pct > 0.8, pct <= 0.2

    def share(values, mask):
        values = values[mask]
        values = values[~np.isnan(values)]
        return float(values.mean()) if len(values) else np.nan

    metrics = {
        "auc_rel": _safe_auc(y_rel, proba),
        "auc_abs": _safe_auc(y_abs, proba),
        "rank_ic": float(ic.mean()) if len(ic) else np.nan,
        "rank_ic_open": float(ic_open.mean()) if len(ic_open) else np.nan,
        "ic_ir": float(ic.mean() / ic.std()) if len(ic) > 1 and ic.std() > 0 else np.nan,
        "top_hit_rel": share(y_rel, top),
        "bottom_hit_rel": share(y_rel, bottom),
        "top_hit_abs": share(y_abs, top),
        "bottom_hit_abs": share(y_abs, bottom),
        "quintile_spread": share(fwd, top) - share(fwd, bottom),
        "rows": int(len(frame)),
        "days": int(len(ic)),
        "period": [_day(dates.min()), _day(dates.max())],
    }
    return metrics, ic


def summarize_folds(fold_metrics, keys=FOLD_METRICS):
    """Mean and standard deviation across folds, plus how many folds had a positive rank IC."""
    summary = {}
    for key in keys:
        values = np.array([m[key] for m in fold_metrics], dtype=float)
        summary[key] = {"mean": float(np.nanmean(values)), "std": float(np.nanstd(values, ddof=1))
                        if np.sum(~np.isnan(values)) > 1 else np.nan}
    summary["positive_ic_folds"] = int(sum(m["rank_ic"] > 0 for m in fold_metrics))
    summary["n_folds"] = len(fold_metrics)
    return summary


def newey_west_tstat(values, lag):
    """t-statistic of the mean with a Newey-West (Bartlett) long-run variance."""
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return np.nan
    errors = x - x.mean()
    lag = int(min(max(lag, 0), n - 1))
    variance = errors @ errors / n
    for k in range(1, lag + 1):
        variance += 2 * (1 - k / (lag + 1)) * (errors[k:] @ errors[:-k]) / n
    if variance <= 0:
        return np.nan
    return float(x.mean() / np.sqrt(variance / n))


def edge_verdict(fold_metrics, test_metrics, oos_daily_ic, horizon_days):
    """Apply the edge rule from the module docstring. `oos_daily_ic`: daily IC over every out-of-sample day."""
    fold_ics = np.array([m["rank_ic"] for m in fold_metrics], dtype=float)
    positive_folds = int((fold_ics > 0).sum())
    tstat = newey_west_tstat(pd.Series(oos_daily_ic).sort_index().to_numpy(), lag=horizon_days)
    checks = {
        "folds": bool(positive_folds >= EDGE_MIN_POSITIVE_FOLDS and np.nanmean(fold_ics) > 0),
        "test": bool(test_metrics["rank_ic"] > 0),
        "significance": bool(tstat >= EDGE_MIN_TSTAT),
    }
    reasons = []
    if not checks["folds"]:
        reasons.append(f"rank IC positive in only {positive_folds} of {len(fold_ics)} walk-forward folds")
    if not checks["test"]:
        reasons.append(f"rank IC on the final test period is {test_metrics['rank_ic']:+.4f}")
    if not checks["significance"]:
        reasons.append(f"out-of-sample rank IC t-stat {tstat:.2f} (needs {EDGE_MIN_TSTAT:.0f})")
    return {
        "reliable": all(checks.values()),
        "checks": checks,
        "positive_ic_folds": positive_folds,
        "oos_ic_mean": float(pd.Series(oos_daily_ic).mean()),
        "oos_ic_tstat": tstat,
        "oos_days": int(len(oos_daily_ic)),
        "reasons": reasons,
        "rule": (f"rank IC > 0 on average and in >= {EDGE_MIN_POSITIVE_FOLDS} of the walk-forward folds, > 0 on the "
                 f"final test, and Newey-West t-stat >= {EDGE_MIN_TSTAT:.0f} over all out-of-sample days"),
    }
