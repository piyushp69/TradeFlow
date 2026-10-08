"""Model configurations.

BASELINE_SPECS reproduce the models that were deployed before the purged walk-forward work (LightGBM for weekly and
monthly, scikit-learn HistGradientBoosting for yearly), with their exact feature lists and hyper-parameters.

FINAL_SPECS are the configurations chosen by the experiment ladder (`python experiments.py`, see
reports/experiment_log.md) using walk-forward folds only. They were frozen before the final test period was scored
and must not be changed after looking at it.
"""
from features.pipeline import feature_columns
from models.training import RANDOM_STATE, ModelSpec

# Every horizon starts the experiments from the same, unselected candidate pool
CANDIDATE_POOL = tuple(feature_columns(["baseline", "market", "daily_mtf"]))
SHORT_TERM_POOL = tuple(feature_columns(["short_term"]))

# ---------------------------------------------------------------- previous models (baseline)

_LGBM = dict(learning_rate=0.03, colsample_bytree=0.7, subsample=0.8, subsample_freq=1, reg_lambda=10.0,
             random_state=RANDOM_STATE, verbose=-1, n_jobs=-1, deterministic=True, force_row_wise=True,
             num_leaves=31, min_child_samples=4000, n_estimators=400)
_HGB = dict(learning_rate=0.03, max_iter=100, l2_regularization=10.0, early_stopping=False,
            random_state=RANDOM_STATE, max_depth=2, min_samples_leaf=5000, max_features=0.5)

_WEEKLY_SELECTED = (
    "Ret_5", "Ret_63", "SMA50_Ratio", "SMA50_200_Ratio", "MACD_Hist_Ratio", "High252_Dist", "Low252_Dist",
    "Mkt_Ret_63", "Mkt_SMA200_Ratio", "Mkt_Vol_20", "nifty_return_1", "nifty_return_5", "nifty_return_20",
    "nifty_sma_ratio", "sector_return_1", "sector_return_5", "sector_momentum", "sector_sma_ratio", "india_vix",
    "india_vix_return", "usdinr_return_1", "usdinr_return_5", "usdinr_return_20", "stock_vs_sector_return",
    "stock_sector_relative_strength", "daily_price_vs_EMA")

BASELINE_SPECS = {
    "weekly": ModelSpec("baseline", _WEEKLY_SELECTED, _LGBM, library="lightgbm", target="absolute",
                        description="LightGBM, 31 leaves, 400 trees, 26 features chosen by the old selection step"),
    "monthly": ModelSpec("baseline", tuple(feature_columns(["baseline", "daily_mtf"])), _LGBM, library="lightgbm",
                         target="absolute", description="LightGBM, 31 leaves, 400 trees, baseline + daily features"),
    "yearly": ModelSpec("baseline", tuple(feature_columns(["baseline"])), _HGB, library="sklearn_hgb",
                        target="absolute", description="HistGradientBoosting, depth 2, 100 iterations, 29 features"),
}

# ---------------------------------------------------------------- experiment settings (fixed before running)

# XGBoost with the closest equivalent of each previous model (LightGBM min_child_samples=4000 is about a hessian
# sum of 1000 for a logistic loss near p=0.5; HistGB max_features=0.5 samples features per split)
XGB_EQUIVALENT = {
    "weekly": dict(grow_policy="lossguide", max_leaves=31, max_depth=0, min_child_weight=1000, learning_rate=0.03,
                   n_estimators=400, subsample=0.8, colsample_bytree=0.7, reg_lambda=10.0),
    "yearly": dict(max_depth=2, min_child_weight=1250, learning_rate=0.03, n_estimators=100, colsample_bynode=0.5,
                   reg_lambda=10.0),
}
XGB_EQUIVALENT["monthly"] = XGB_EQUIVALENT["weekly"]

CLEAR_MOVE_K = (0.25, 0.5)

# Early stopping on the training window's purged hold-out. "auc" was the first version; it stops after a
# handful of trees when the target is absolute, because pooled AUC rewards market timing. "rank_ic" uses
# the metric the ladder is judged on. Both are run and logged.
EARLY_STOPPING_METRICS = ("auc", "rank_ic")

# More regularisation: shallower trees, larger leaves, row and column subsampling
_REGULARISED = dict(min_child_weight=2000, learning_rate=0.03, n_estimators=400, subsample=0.7,
                    colsample_bytree=0.6, reg_lambda=10.0)
REGULARISED_VARIANTS = {
    "weekly": {"depth 4": {**_REGULARISED, "max_depth": 4}, "depth 3": {**_REGULARISED, "max_depth": 3}},
    "monthly": {"depth 4": {**_REGULARISED, "max_depth": 4}, "depth 3": {**_REGULARISED, "max_depth": 3}},
    # The yearly model is already depth 2; regularising it means larger leaves and subsampling, not deeper trees
    "yearly": {"depth 2 + subsampling": {**_REGULARISED, "max_depth": 2, "min_child_weight": 2500,
                                         "n_estimators": 100}},
}

# ---------------------------------------------------------------- final configuration (frozen)

# Frozen on 2026-10-08 from reports/experiment_log.md, before the final test period was scored. For every horizon
# the ladder ended at E0: none of the changes in E1-E5 improved rank IC in at least 4 of the 5 walk-forward folds.
FINAL_SPECS = {
    horizon: ModelSpec("xgboost", CANDIDATE_POOL, XGB_EQUIVALENT[horizon], library="xgboost", target="absolute",
                       description="XGBoost with settings equivalent to the previous model")
    for horizon in ("weekly", "monthly", "yearly")
}


def matches(spec, recorded):
    """True when a spec equals the configuration recorded by experiments.py (ignoring names and descriptions)."""
    keys = ("features", "params", "library", "target", "clear_move_k", "cross_sectional", "early_stopping", "prune")
    current = spec.to_dict()
    return all(current[k] == recorded.get(k) for k in keys)
