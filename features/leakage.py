"""Leakage checks.

1. Point-in-time recomputation (`audit_point_in_time`): for cutoff dates T, rebuild all features using only the
   information set available at T 15:30 IST (daily prices <= T, USD/INR strictly before T, intraday bars that
   ended by T 15:30, sector peers <= T) and compare the row at T with the full-history build. Any difference
   means a feature used data from after T.
2. Target leakage (`check_target_leakage`): no feature may be named like a target/future quantity or equal a target.
3. Suspicious predictiveness (`check_suspicious_features`): a single feature that predicts the forward target far
   better than any technical signal plausibly can (AUC far from 0.5) is flagged for inspection.
4. Intraday alignment (`check_intraday_alignment`): every intraday bar used for daily row D ended by D 15:30
   and belongs to session D.
5. Split integrity (`check_split_integrity`): training/selection rows end before the test block minus the embargo.
"""
import logging
import re
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from data.fetcher import SESSION_CLOSE
from features.market import MarketContext, SectorPanel

logger = logging.getLogger(__name__)

FORBIDDEN_NAME = re.compile(r"target|future|forward|fwd|next_|label", re.IGNORECASE)


def _truncate_daily(df, cutoff, strictly_before=False):
    if df is None:
        return None
    mask = df["Date"] < cutoff if strictly_before else df["Date"] <= cutoff
    return df[mask].reset_index(drop=True)


def _truncate_intraday(bars, cutoff):
    if bars is None:
        return None
    return bars[bars["BarEnd"] <= cutoff + SESSION_CLOSE].reset_index(drop=True)


def audit_point_in_time(build_fn, stock_raw, context_sources, intraday=None, cutoffs=None, features=None):
    """Compare features at each cutoff between the full build and a point-in-time build.

    build_fn(raw_df, context, intraday) -> feature frame (features.pipeline.build_feature_frame)
    context_sources: dict(nifty=df, vix=df, usdinr=df, prices={ticker: df}, index_prices={ticker: df})
    Returns a DataFrame with one row per (cutoff, feature) mismatch; empty means no look-ahead detected.
    """
    def make_context(cutoff=None):
        t = (lambda d, strict=False: _truncate_daily(d, cutoff, strict)) if cutoff is not None else (lambda d, strict=False: d)
        prices = {k: t(v) for k, v in context_sources["prices"].items()}
        index_prices = {k: t(v) for k, v in context_sources["index_prices"].items()}
        return MarketContext(nifty=t(context_sources["nifty"]), vix=t(context_sources["vix"]),
                             usdinr=t(context_sources["usdinr"], strict=True),
                             sector_panel=SectorPanel(prices, index_prices))

    full = build_fn(stock_raw, make_context(), intraday).set_index("Date")
    features = features or [c for c in full.columns if pd.api.types.is_numeric_dtype(full[c])]
    if cutoffs is None:
        dates = full.index
        cutoffs = [dates[int(len(dates) * q)] for q in (0.3, 0.6, 0.9)] + [dates[-2]]

    mismatches = []
    for cutoff in cutoffs:
        pit_intraday = {k: _truncate_intraday(v, cutoff) for k, v in (intraday or {}).items()}
        pit = build_fn(_truncate_daily(stock_raw, cutoff), make_context(cutoff), pit_intraday)
        if pit is None or cutoff not in set(pit["Date"]):
            continue
        row_pit = pit.set_index("Date").loc[cutoff]
        row_full = full.loc[cutoff]
        for feat in features:
            a, b = row_full.get(feat), row_pit.get(feat)
            if (pd.isna(a) and pd.isna(b)) or (not pd.isna(a) and not pd.isna(b) and np.isclose(a, b, rtol=1e-7, atol=1e-10)):
                continue
            mismatches.append({"cutoff": cutoff, "feature": feat, "full_history_value": a, "point_in_time_value": b})
    return pd.DataFrame(mismatches, columns=["cutoff", "feature", "full_history_value", "point_in_time_value"])


def check_target_leakage(frame, features, target_cols):
    problems = []
    for feat in features:
        if FORBIDDEN_NAME.search(feat):
            problems.append((feat, "name suggests a target/future quantity"))
        for target in target_cols:
            if target in frame and feat in frame:
                both = frame[[feat, target]].dropna()
                if len(both) and np.allclose(both[feat].to_numpy(), both[target].to_numpy()):
                    problems.append((feat, f"identical to {target}"))
    return problems


def check_suspicious_features(X, y, max_abs_auc_edge=0.15, sample=200_000, seed=42):
    """Single-feature AUC. Honest technical features sit within a few points of 0.5 for forward returns."""
    if len(X) > sample:
        idx = np.random.default_rng(seed).choice(len(X), sample, replace=False)
        X, y = X.iloc[idx], y.iloc[idx]
    flagged = {}
    for col in X.columns:
        mask = X[col].notna()
        if mask.sum() < 1000 or y[mask].nunique() < 2:
            continue
        auc = roc_auc_score(y[mask], X.loc[mask, col])
        if abs(auc - 0.5) > max_abs_auc_edge:
            flagged[col] = round(float(auc), 4)
    return flagged


def check_intraday_alignment(frame, prefixes=("5m", "15m", "1h")):
    problems = {}
    for prefix in prefixes:
        col = f"{prefix}_bar_end"
        if col not in frame:
            continue
        bar_end = pd.to_datetime(frame[col])
        used = bar_end.notna()
        late = (bar_end[used] > frame.loc[used, "Date"] + SESSION_CLOSE).sum()
        wrong_session = (bar_end[used].dt.normalize() != frame.loc[used, "Date"]).sum()
        if late or wrong_session:
            problems[prefix] = {"bars_after_close": int(late), "bars_from_other_session": int(wrong_session)}
    return problems


def check_split_integrity(train_dates, eval_dates, embargo_days, all_dates):
    """Training rows must end at least `embargo_days` trading days before evaluation starts."""
    all_dates = np.sort(pd.to_datetime(pd.Series(all_dates).unique()))
    eval_start = pd.to_datetime(eval_dates).min()
    last_train = pd.to_datetime(train_dates).max()
    gap = int(((all_dates > last_train) & (all_dates < eval_start)).sum())
    return gap >= embargo_days - 1, gap
