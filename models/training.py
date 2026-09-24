"""Model training, time-series splitting and evaluation.

Splitting (unchanged from the existing pipeline, now shared):
    * the most recent `TEST_FRACTION` of dates is the final test block and is never used for feature
      selection, hyper-parameter tuning or model choice;
    * model selection uses `CV_FOLDS` walk-forward folds inside the earlier data (expanding training window);
    * every training block ends `horizon_days` before the block it is evaluated on (embargo), so labels that
      look `horizon_days` ahead cannot overlap the evaluation rows.

Two model families are supported: the existing HistGradientBoostingClassifier ("hgb") and LightGBM ("lgbm",
used for gain/split importance and native TreeSHAP contributions).
"""
import logging
from itertools import product
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)

logger = logging.getLogger(__name__)

RANDOM_STATE = 42

TEST_FRACTION = 0.15
CV_FOLDS = 4
CV_START_FRACTION = 0.45

# Existing HistGradientBoosting search space (kept identical so the baseline stays comparable)
CONFIGS = {
    "leaf15": dict(max_leaf_nodes=15, min_samples_leaf=2000, max_features=0.7),
    "leaf7": dict(max_leaf_nodes=7, min_samples_leaf=5000, max_features=0.5),
    "depth2": dict(max_depth=2, min_samples_leaf=5000, max_features=0.5),
}
ITER_CANDIDATES = [25, 50, 100, 200, 300]

# LightGBM search space: shallow, heavily regularised trees (financial data has a very low signal-to-noise ratio)
LGBM_CONFIGS = {
    "leaves7": dict(num_leaves=7, min_child_samples=1000),
    "leaves15": dict(num_leaves=15, min_child_samples=2000),
    "leaves31": dict(num_leaves=31, min_child_samples=4000),
    "depth2": dict(num_leaves=4, max_depth=2, min_child_samples=4000),
}
LGBM_ITERS = [50, 100, 200, 400]
LGBM_FIXED = dict(learning_rate=0.03, colsample_bytree=0.7, subsample=0.8, subsample_freq=1,
                  reg_lambda=10.0, n_estimators=max(LGBM_ITERS), random_state=RANDOM_STATE,
                  verbose=-1, n_jobs=-1, deterministic=True, force_row_wise=True)


# Splitting

def split_before(df, dates, start_idx, horizon_days):
    """Rows usable for training when evaluating from dates[start_idx] (embargo of `horizon_days`)."""
    return df[df["Date"] < dates[max(start_idx - horizon_days, 0)]]


def walk_forward_folds(df, horizon_days, cv_folds=CV_FOLDS, test_fraction=TEST_FRACTION,
                       cv_start_fraction=CV_START_FRACTION):
    dates = np.sort(df["Date"].unique())
    n = len(dates)
    test_start = int(n * (1 - test_fraction))
    cv_start = int(n * cv_start_fraction)
    fold_size = (test_start - horizon_days - cv_start) // cv_folds

    folds = []
    for k in range(cv_folds):
        a = cv_start + k * fold_size
        b = a + fold_size
        val = df[(df["Date"] >= dates[a]) & (df["Date"] < dates[b])]
        folds.append((split_before(df, dates, a, horizon_days), val))

    train_full = split_before(df, dates, test_start, horizon_days)
    test = df[df["Date"] >= dates[test_start]]
    return folds, train_full, test


# Models

def make_model(config, max_iter):
    """Existing model family (HistGradientBoostingClassifier)."""
    return HistGradientBoostingClassifier(
        learning_rate=0.03,
        max_iter=max_iter,
        l2_regularization=10.0,
        early_stopping=False,
        random_state=RANDOM_STATE,
        **CONFIGS[config],
    )


def make_lgbm(config, n_estimators=None, **overrides):
    params = {**LGBM_FIXED, **LGBM_CONFIGS[config], **overrides}
    if n_estimators:
        params["n_estimators"] = n_estimators
    return lgb.LGBMClassifier(**params)


def fit_model(kind, config, n_iter, X, y):
    model = make_model(config, n_iter) if kind == "hgb" else make_lgbm(config, n_iter)
    return model.fit(X, y)


def predict_proba(model, X):
    return model.predict_proba(X)[:, 1]


def staged_scores(kind, model, X, y, iters):
    """AUC after each candidate number of boosting iterations (one fit, several truncations)."""
    if kind == "hgb":
        return {i: roc_auc_score(y, p[:, 1]) for i, p in enumerate(model.staged_predict_proba(X), 1) if i in iters}
    return {i: roc_auc_score(y, model.predict_proba(X, num_iteration=i)[:, 1]) for i in iters}


# Evaluation

def evaluate(y_true, proba, dates, up_signal_percentile=60):
    """Metrics for a probability forecast judged against the same-day cross-section.

    A stock counts as predicted UP when its P(UP) is above that day's median P(UP), which is how the app
    turns probabilities into a direction. `up_signal_precision` uses the app's stricter UP band instead.
    Regression metrics (MAE/RMSE/R2 on price) do not apply: the model outputs a probability of a direction,
    so the probabilistic equivalents (Brier score / RMSE of the probability) are reported.
    """
    y_true = np.asarray(y_true).astype(int)
    frame = pd.DataFrame({"p": np.asarray(proba), "date": np.asarray(dates)})
    median = frame.groupby("date")["p"].transform("median")
    pred = (frame["p"] >= median).to_numpy()
    pct = frame.groupby("date")["p"].rank(pct=True).to_numpy() * 100
    base_rate = float(y_true.mean())
    strong_up = pct >= up_signal_percentile
    brier = float(brier_score_loss(y_true, proba))

    return {
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "directional_accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "up_signal_precision": round(float(y_true[strong_up].mean()), 4) if strong_up.any() else None,
        "up_rate_top_quintile": round(float(y_true[pct >= 80].mean()), 4),
        "up_rate_bottom_quintile": round(float(y_true[pct <= 20].mean()), 4),
        "up_rate": round(base_rate, 4),
        "majority_baseline_accuracy": round(max(base_rate, 1 - base_rate), 4),
        "log_loss": round(float(log_loss(y_true, proba)), 4),
        "brier_score": round(brier, 4),
        "rmse_probability": round(float(np.sqrt(brier)), 4),
        "rows": int(len(y_true)),
    }


# Walk-forward model selection

def cross_validate(kind, folds, features, target, configs=None, iters=None, keep_oof=False):
    """Mean walk-forward AUC for every (config, n_iterations) combination.

    Returns (best_key, best_score, per_fold_scores, oof) where best_key = (config, n_iter).
    """
    configs = configs or (CONFIGS if kind == "hgb" else LGBM_CONFIGS)
    iters = iters or (ITER_CANDIDATES if kind == "hgb" else LGBM_ITERS)
    scores, oof_store = {}, {}

    for config in configs:
        fold_scores, fold_oof = [], []
        for train, val in folds:
            model = fit_model(kind, config, max(iters), train[features], train[target].astype(int))
            y_val = val[target].astype(int)
            fold_scores.append(staged_scores(kind, model, val[features], y_val, iters))
            if keep_oof:
                if kind == "hgb":
                    preds = {i: p[:, 1] for i, p in enumerate(model.staged_predict_proba(val[features]), 1) if i in iters}
                else:
                    preds = {i: model.predict_proba(val[features], num_iteration=i)[:, 1] for i in iters}
                fold_oof.append((val["Date"].to_numpy(), val["Stock"].to_numpy(), y_val.to_numpy(), preds,
                                 np.full(len(val), len(fold_oof))))

        for n_iter in iters:
            per_fold = [f[n_iter] for f in fold_scores]
            scores[(config, n_iter)] = (float(np.mean(per_fold)), per_fold)
            if keep_oof:
                oof_store[(config, n_iter)] = pd.DataFrame({
                    "Date": np.concatenate([f[0] for f in fold_oof]),
                    "Stock": np.concatenate([f[1] for f in fold_oof]),
                    "y": np.concatenate([f[2] for f in fold_oof]),
                    "p": np.concatenate([f[3][n_iter] for f in fold_oof]),
                    "fold": np.concatenate([f[4] for f in fold_oof]),
                })
        best_iter = max(iters, key=lambda i: scores[(config, i)][0])
        logger.info("      %-8s %-8s best iters=%-4d mean CV AUC=%.4f", kind, config, best_iter,
                    scores[(config, best_iter)][0])

    best_key = max(scores, key=lambda k: scores[k][0])
    mean_auc, per_fold = scores[best_key]
    return best_key, mean_auc, per_fold, (oof_store.get(best_key) if keep_oof else None)


def run_experiment(name, kind, folds, train_full, test, features, target, keep_oof=True):
    """Tune on walk-forward folds, then score once on the untouched test block."""
    logger.info("   [%s] %d features, model=%s", name, len(features), kind)
    (config, n_iter), cv_auc, fold_aucs, oof = cross_validate(kind, folds, features, target, keep_oof=keep_oof)

    model = fit_model(kind, config, n_iter, train_full[features], train_full[target].astype(int))
    test_proba = predict_proba(model, test[features])
    test_metrics = evaluate(test[target].astype(int), test_proba, test["Date"])
    validation_metrics = evaluate(oof["y"], oof["p"], oof["Date"]) if oof is not None else None
    logger.info("   [%s] CV AUC=%.4f (folds %s) | test AUC=%.4f", name, cv_auc,
                [round(a, 3) for a in fold_aucs], test_metrics["roc_auc"])

    return {
        "name": name,
        "model_type": kind,
        "config": config,
        "n_iterations": n_iter,
        "features": list(features),
        "n_features": len(features),
        "cv_auc": round(cv_auc, 4),
        "cv_fold_aucs": [round(a, 4) for a in fold_aucs],
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "model": model,
        "oof": oof,
        "test_predictions": pd.DataFrame({"Date": test["Date"].to_numpy(), "Stock": test["Stock"].to_numpy(),
                                          "y": test[target].astype(int).to_numpy(), "p": test_proba}),
    }
