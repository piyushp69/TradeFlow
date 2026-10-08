"""Labels for every horizon, built on the stacked panel of all stocks.

The relative label compares a stock with the rest of the universe on the same date, so labels are built on the
panel (every stock) rather than one stock at a time.

For a row (stock s, date t) and a horizon of h sessions:

    fwd_ret_h        Close[s, t+h] / Close[s, t] - 1       h rows ahead in the stock's own history
    fwd_ret_open_h   Close[s, t+h] / Open[s, t+1] - 1      the same move entered at the next session's open
    label_end_h      Date[s, t+h]                          the last price the label depends on
    y_abs_h          1 if fwd_ret_h > UP_THRESHOLD         the original "UP" label (absolute direction)
    excess_ret_h     fwd_ret_h - median over stocks of fwd_ret_h on date t
    y_rel_h          1 if excess_ret_h > 0                 "beats the universe median over the horizon"
    label_end_rel_h  max over stocks of label_end_h on date t

The relative label of one stock depends on every stock's forward return that day, so its label window ends at the
latest of those label ends (`label_end_rel_h`). Splits purge on that date, which is never earlier than the stock's
own label end. Rows whose future is not known yet keep NaN labels and NaT label ends.
"""
import numpy as np
import pandas as pd

from features.indicators import HORIZONS, UP_THRESHOLD

TARGETS = ("absolute", "relative")


def add_labels(panel, horizons=HORIZONS, up_threshold=UP_THRESHOLD):
    """Add forward returns, label ends and both labels for each horizon. Needs Date, Stock, Open, Close."""
    out = panel.sort_values(["Stock", "Date"], kind="mergesort").reset_index(drop=True)
    by_stock = out.groupby("Stock", sort=False)
    open_next = by_stock["Open"].shift(-1).where(lambda s: s > 0)
    new = {}
    for name, days in horizons.items():
        close_fwd = by_stock["Close"].shift(-days)
        known = close_fwd.notna()
        fwd = close_fwd / out["Close"] - 1
        label_end = by_stock["Date"].shift(-days)
        median = fwd.groupby(out["Date"]).transform("median")
        new[f"fwd_ret_{name}"] = fwd
        new[f"fwd_ret_open_{name}"] = (close_fwd / open_next - 1).replace([np.inf, -np.inf], np.nan)
        new[f"label_end_{name}"] = label_end
        new[f"y_abs_{name}"] = (fwd > up_threshold).astype("float32").where(known)
        new[f"excess_ret_{name}"] = fwd - median
        new[f"y_rel_{name}"] = (fwd > median).astype("float32").where(known)
        new[f"label_end_rel_{name}"] = label_end.groupby(out["Date"]).transform("max").where(known)
    out = out.drop(columns=[c for c in new if c in out.columns])
    return pd.concat([out, pd.DataFrame(new, index=out.index)], axis=1)


def label_column(target, horizon):
    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}; expected one of {TARGETS}")
    return f"y_abs_{horizon}" if target == "absolute" else f"y_rel_{horizon}"


def target_move(panel, target, horizon, up_threshold=UP_THRESHOLD):
    """The forward move the label thresholds: return over the hurdle, or return over the universe median."""
    if target == "absolute":
        return panel[f"fwd_ret_{horizon}"] - up_threshold
    return panel[f"excess_ret_{horizon}"]


def clear_move_mask(move, volatility, horizon_days, k):
    """True for rows whose forward move is at least k recent standard deviations away from the label boundary.

    `volatility` is the stock's trailing daily volatility at the prediction date, scaled to the horizon by
    sqrt(h). Rows with an unknown move or volatility are kept: the mask only ever removes clearly ambiguous rows.
    Use it to filter training rows only; evaluation always uses every row.
    """
    scale = np.asarray(volatility, dtype=float) * np.sqrt(horizon_days)
    move = np.asarray(move, dtype=float)
    # Comparisons with NaN are False, so rows with a missing move or volatility are never flagged ambiguous
    return ~(np.abs(move) < k * scale)
