"""Model fitting for the purged walk-forward pipeline.

A `ModelSpec` holds everything that defines a model for one horizon: library and hyper-parameters, features,
target (absolute "UP" or relative "beats the universe median"), and the optional training-side treatments the
experiments test:

    clear_move_k     drop training rows whose forward move is within k recent standard deviations of the label
                     boundary (evaluation always uses every row)
    cross_sectional  use each stock-level feature's percentile rank across stocks on the same day
    early_stopping   choose the number of trees on an inner, purged hold-out at the end of the training window
                     ("auc": XGBoost early stopping on pooled AUC; "rank_ic": best mean daily rank IC)
    prune            keep only features with non-trivial importance in every sub-period of the training window

Everything inside `fit_spec` sees training rows only. The early-stopping hold-out and the pruning sub-periods are
carved out of the training window, so a fold's validation block is never used to make a training decision.

The production models use XGBoost (GPU when available, CPU otherwise). LightGBM and scikit-learn's
HistGradientBoosting are kept only to reproduce the previous models for the baseline; they are imported lazily.
"""
import functools
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from features.cross_section import model_columns
from features.labels import clear_move_mask, label_column, target_move
from models.evaluation import daily_rank_ic

logger = logging.getLogger(__name__)

RANDOM_STATE = 42

XGB_BASE = dict(objective="binary:logistic", tree_method="hist", eval_metric="auc", random_state=RANDOM_STATE,
                n_jobs=-1, verbosity=0)

EARLY_STOPPING_HOLDOUT = 0.20
EARLY_STOPPING_PATIENCE = 100
EARLY_STOPPING_MAX_TREES = 2000
EARLY_STOPPING_GRID = (25, 50, 100, 150, 200, 300, 400, 600, 800, 1000, 1500, 2000)
MIN_EARLY_STOPPING_ROWS = 20_000
PRUNE_CHUNKS = 3
PRUNE_MIN_SHARE = 0.5       # keep a feature if its gain share is >= half of an equal share in every sub-period


@dataclass(frozen=True)
class ModelSpec:
    name: str
    features: tuple
    params: dict = field(default_factory=dict)
    library: str = "xgboost"
    target: str = "absolute"
    clear_move_k: float = None
    cross_sectional: bool = False
    early_stopping: str = None        # None, "auc" or "rank_ic" (hold-out metric)
    prune: bool = False
    description: str = ""

    def with_changes(self, **changes):
        values = {**self.__dict__, **changes}
        values["features"] = tuple(values["features"])
        return ModelSpec(**values)

    def to_dict(self):
        out = dict(self.__dict__)
        out["features"] = list(self.features)
        return out


@dataclass
class FittedModel:
    model: object
    features: list           # feature names before any cross-sectional transform
    columns: list            # columns the model was trained on
    info: dict

    def predict(self, frame):
        return self.model.predict_proba(frame[self.columns])[:, 1]


@functools.lru_cache(maxsize=None)
def detect_device(prefer="cuda"):
    """'cuda' when XGBoost was built with CUDA and a GPU accepts a tiny training job, else 'cpu'."""
    if prefer == "cpu":
        return "cpu"
    try:
        import xgboost as xgb
        if not xgb.build_info().get("USE_CUDA"):
            return "cpu"
        X = np.random.default_rng(0).normal(size=(512, 4))
        model = xgb.XGBClassifier(n_estimators=2, device="cuda", tree_method="hist", verbosity=0)
        model.fit(X, (X[:, 0] > 0).astype(int))
        return "cuda"
    except Exception as exc:   # no GPU, no driver, or a CPU-only build
        logger.info("GPU training unavailable (%s); using CPU", exc)
        return "cpu"


def make_model(library, params, device="cpu"):
    if library == "xgboost":
        import xgboost as xgb
        return xgb.XGBClassifier(**{**XGB_BASE, **params, "device": device})
    if library == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(**params)
    if library == "sklearn_hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(**params)
    raise ValueError(f"unknown model library {library!r}")


def training_rows(spec, panel, idx, horizon, horizon_days):
    """Labelled training rows for the spec's target, minus ambiguous moves when clear_move_k is set."""
    y = panel[label_column(spec.target, horizon)].to_numpy()[idx]
    keep = ~np.isnan(y)
    if spec.clear_move_k is not None:
        move = target_move(panel, spec.target, horizon).to_numpy()[idx]
        vol = panel["Volatility"].to_numpy()[idx]
        keep &= clear_move_mask(move, vol, horizon_days, spec.clear_move_k)
    return np.asarray(idx)[keep]


def _xy(panel, idx, columns, target_col):
    return panel[columns].iloc[idx], panel[target_col].to_numpy()[idx].astype(int)


def stable_features(spec, panel, idx, horizon, device):
    """Features whose gain share is non-trivial in every one of PRUNE_CHUNKS consecutive sub-periods."""
    features = list(spec.features)
    columns = model_columns(features, spec.cross_sectional)
    target_col = label_column(spec.target, horizon)
    dates = panel["Date"].to_numpy()[idx]
    shares = []
    for chunk in np.array_split(np.unique(dates), PRUNE_CHUNKS):
        part = idx[(dates >= chunk[0]) & (dates <= chunk[-1])]
        X, y = _xy(panel, part, columns, target_col)
        model = make_model(spec.library, spec.params, device).fit(X, y)
        gain = model.get_booster().get_score(importance_type="total_gain")
        share = pd.Series({c: gain.get(c, 0.0) for c in columns})
        shares.append(share / max(share.sum(), 1e-12))
    shares = pd.concat(shares, axis=1)
    keep = shares.min(axis=1) >= PRUNE_MIN_SHARE / len(columns)
    kept = [f for f, c in zip(features, columns) if keep[c]]
    if not kept:   # nothing is stable: fall back to the strongest feature rather than an empty model
        kept = [features[int(np.argmax(shares.min(axis=1).to_numpy()))]]
    return kept, {"kept": len(kept), "dropped": [f for f in features if f not in kept]}


def early_stopped_trees(spec, panel, idx, columns, horizon, horizon_days, device):
    """Number of trees chosen on a purged hold-out made of the last sessions of the training window.

    spec.early_stopping picks the hold-out metric:
        "auc"      XGBoost's built-in early stopping on pooled AUC (stops after EARLY_STOPPING_PATIENCE rounds
                   without improvement)
        "rank_ic"  the mean daily rank IC, the metric the experiments are judged on, read at EARLY_STOPPING_GRID
    """
    dates = panel["Date"].to_numpy()[idx]
    label_end = panel[f"label_end_rel_{horizon}"].to_numpy()[idx]
    sessions = np.unique(dates)
    start_pos = int(len(sessions) * (1 - EARLY_STOPPING_HOLDOUT))
    start = sessions[start_pos]
    holdout = idx[dates >= start]
    last_fit_pos = start_pos - horizon_days - 1
    fit_rows = idx[(label_end < start) & (dates <= sessions[max(last_fit_pos, 0)])] if last_fit_pos >= 0 else idx[:0]
    default = int(spec.params.get("n_estimators", 100))
    if len(fit_rows) < MIN_EARLY_STOPPING_ROWS or len(holdout) < MIN_EARLY_STOPPING_ROWS // 4:
        return default, {"skipped": f"too little data for a purged hold-out ({len(fit_rows)} fit rows)",
                         "n_trees": default}
    target_col = label_column(spec.target, horizon)
    X_fit, y_fit = _xy(panel, fit_rows, columns, target_col)
    X_hold, y_hold = _xy(panel, holdout, columns, target_col)
    info = {"metric": spec.early_stopping, "holdout_start": f"{pd.Timestamp(start):%Y-%m-%d}",
            "fit_rows": int(len(fit_rows)), "holdout_rows": int(len(holdout))}
    if spec.early_stopping == "auc":
        params = {**spec.params, "n_estimators": EARLY_STOPPING_MAX_TREES,
                  "early_stopping_rounds": EARLY_STOPPING_PATIENCE}
        model = make_model(spec.library, params, device)
        model.fit(X_fit, y_fit, eval_set=[(X_hold, y_hold)], verbose=False)
        trees = int(model.best_iteration) + 1
        return trees, {**info, "n_trees": trees, "holdout_score": round(float(model.best_score), 4)}
    if spec.early_stopping != "rank_ic":
        raise ValueError(f"unknown early-stopping metric {spec.early_stopping!r}")
    model = make_model(spec.library, {**spec.params, "n_estimators": EARLY_STOPPING_MAX_TREES}, device)
    model.fit(X_fit, y_fit)
    hold_dates = panel["Date"].to_numpy()[holdout]
    hold_returns = panel[f"fwd_ret_{horizon}"].to_numpy()[holdout]
    scores = {}
    for trees in EARLY_STOPPING_GRID:
        proba = model.predict_proba(X_hold, iteration_range=(0, trees))[:, 1]
        ic = daily_rank_ic(hold_dates, proba, hold_returns)
        scores[trees] = float(ic.mean()) if len(ic) else -np.inf
    trees = max(scores, key=scores.get)
    return trees, {**info, "n_trees": trees, "holdout_score": round(scores[trees], 4),
                   "holdout_ic_by_trees": {k: round(v, 4) for k, v in scores.items()}}


def fit_spec(spec, panel, train_idx, horizon, horizon_days, device="cpu"):
    """Fit a spec on the given training rows. Returns a FittedModel."""
    idx = training_rows(spec, panel, np.asarray(train_idx), horizon, horizon_days)
    info = {"train_rows": int(len(idx)), "train_rows_before_filter": int(len(train_idx))}
    features = list(spec.features)
    if spec.prune:
        features, info["prune"] = stable_features(spec, panel, idx, horizon, device)
    columns = model_columns(features, spec.cross_sectional)
    params = dict(spec.params)
    if spec.early_stopping:
        params["n_estimators"], info["early_stopping"] = early_stopped_trees(
            spec, panel, idx, columns, horizon, horizon_days, device)
    X, y = _xy(panel, idx, columns, label_column(spec.target, horizon))
    model = make_model(spec.library, params, device).fit(X, y)
    info["n_trees"] = int(params.get("n_estimators", params.get("max_iter", 0)))
    return FittedModel(model=model, features=features, columns=columns, info=info)


def to_cpu(fitted):
    """Make an XGBoost model load and predict on machines without a GPU (Streamlit Cloud)."""
    model = fitted.model
    if hasattr(model, "get_booster"):
        model.set_params(device="cpu")
        model.get_booster().set_param({"device": "cpu"})
    return fitted
