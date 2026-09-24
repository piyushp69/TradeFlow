"""Systematic feature selection.

Pipeline:
    raw candidates
      -> drop invalid features (scale-dependent price levels, target-like names, leakage-audit failures)
      -> drop features with too little history to train on
      -> drop constant / near-constant features
      -> drop exact duplicates
      -> correlation analysis (feature-feature redundancy, feature-target association)
      -> mutual information
      -> model gain / split importance
      -> permutation importance (validation folds, out-of-sample)
      -> SHAP (native LightGBM TreeSHAP, validation folds)
      -> final set by agreement between methods, then redundancy resolution

No method removes a feature on its own: a feature survives when at least `MIN_VOTES` of the four ranking
methods place it in the top half, unless permutation importance shows it actively hurts out-of-sample AUC.
Everything runs on pre-test data only (the folds are the walk-forward validation folds), so the final test
block never influences which features are kept.
"""
import logging
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from models.training import make_lgbm

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
MIN_VOTES = 2
NEAR_CONSTANT_FREQUENCY = 0.99
DUPLICATE_CORRELATION = 0.999
REDUNDANT_CORRELATION = 0.90
MAX_MISSING_RATE = 0.50
CORRELATION_SAMPLE = 50_000
MI_SAMPLE = 60_000
PERMUTATION_SAMPLE = 40_000
SHAP_SAMPLE = 20_000
SHAP_PLOT_ROWS = 4_000     # rows kept on disk for the dashboard beeswarm (means use the full sample)
SELECTION_MODEL = dict(config="leaves15", n_estimators=200)


def _sample(frame, n, seed=RANDOM_STATE):
    if len(frame) <= n:
        return frame
    return frame.sample(n, random_state=seed)


def drop_invalid(candidates, invalid_map):
    """invalid_map: feature -> reason. Returns (kept, removed_reasons)."""
    removed = {f: invalid_map[f] for f in candidates if f in invalid_map}
    return [f for f in candidates if f not in removed], removed


def find_low_coverage(frame, features, max_missing=MAX_MISSING_RATE):
    missing = frame[features].isna().mean()
    return {f: f"only {(1 - missing[f]) * 100:.0f}% of training rows have a value" for f in features
            if missing[f] > max_missing}


def find_near_constant(frame, features, threshold=NEAR_CONSTANT_FREQUENCY):
    out = {}
    for f in features:
        col = frame[f].dropna()
        if col.empty or col.nunique() <= 1:
            out[f] = "constant"
        else:
            top = col.value_counts(normalize=True).iloc[0]
            if top > threshold:
                out[f] = f"near-constant ({top:.1%} identical values)"
    return out


def correlation_analysis(frame, features, target, sample=CORRELATION_SAMPLE):
    """Spearman feature-feature matrix plus feature-target correlation (rank based, NaN-tolerant)."""
    data = _sample(frame[features + [target]].dropna(subset=[target]), sample)
    corr = data[features].corr(method="spearman")
    target_corr = data[features].corrwith(data[target], method="spearman")
    return corr, target_corr


def find_duplicates(corr, features, threshold=DUPLICATE_CORRELATION):
    """Exact/near-exact duplicates: keep the first occurrence in `features` order."""
    removed = {}
    for i, a in enumerate(features):
        if a in removed:
            continue
        for b in features[i + 1:]:
            if b in removed or a not in corr or b not in corr:
                continue
            if abs(corr.loc[a, b]) >= threshold:
                removed[b] = f"duplicate of {a} (|rho|>={threshold})"
    return removed


def correlation_clusters(corr, features, threshold=REDUNDANT_CORRELATION):
    """Connected components of |rho| >= threshold."""
    parent = {f: f for f in features}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(features):
        for b in features[i + 1:]:
            if a in corr and b in corr and abs(corr.loc[a, b]) >= threshold:
                parent[find(a)] = find(b)

    clusters = {}
    for f in features:
        clusters.setdefault(find(f), []).append(f)
    return [c for c in clusters.values() if len(c) > 1]


def mutual_information(frame, features, target, sample=MI_SAMPLE, seed=RANDOM_STATE):
    """Mutual information between each feature and the binary target.

    mutual_info_classif cannot handle NaN, so missing values are filled with the median of this
    (training-only) sample. The fill is used for ranking only; models keep the real NaNs.
    """
    data = _sample(frame[features + [target]].dropna(subset=[target]), sample, seed)
    X = data[features].fillna(data[features].median())
    scores = mutual_info_classif(X, data[target].astype(int), random_state=seed, n_neighbors=3)
    return pd.Series(scores, index=features)


def model_importance(folds, features, target, model_spec=SELECTION_MODEL):
    """Average LightGBM gain and split importance over the walk-forward training folds."""
    gain, split = [], []
    for train, _ in folds:
        model = make_lgbm(model_spec["config"], model_spec["n_estimators"])
        model.fit(train[features], train[target].astype(int))
        booster = model.booster_
        gain.append(pd.Series(booster.feature_importance("gain"), index=features))
        split.append(pd.Series(booster.feature_importance("split"), index=features))
    gain = pd.concat(gain, axis=1).mean(axis=1)
    split = pd.concat(split, axis=1).mean(axis=1)
    return gain / max(gain.sum(), 1e-12), split / max(split.sum(), 1e-12)


def calculate_permutation_importance(folds, features, target, model_spec=SELECTION_MODEL,
                                     n_repeats=3, sample=PERMUTATION_SAMPLE, seed=RANDOM_STATE):
    """Permutation importance measured on each fold's validation block (never on training rows)."""
    per_fold = []
    for train, val in folds:
        model = make_lgbm(model_spec["config"], model_spec["n_estimators"])
        model.fit(train[features], train[target].astype(int))
        val_sample = _sample(val, sample, seed)
        result = permutation_importance(
            model, val_sample[features], val_sample[target].astype(int),
            scoring="roc_auc", n_repeats=n_repeats, random_state=seed, n_jobs=1,
        )
        per_fold.append(pd.Series(result.importances_mean, index=features))
    frame = pd.concat(per_fold, axis=1)
    return frame.mean(axis=1), (frame > 0).mean(axis=1)


def calculate_shap(folds, features, target, model_spec=SELECTION_MODEL, sample=SHAP_SAMPLE, seed=RANDOM_STATE):
    """Native LightGBM TreeSHAP contributions on validation rows.

    Returns (mean |SHAP| per feature, mean signed SHAP, sample of SHAP values with feature values for plots).
    """
    abs_means, signed_means, samples = [], [], []
    for train, val in folds:
        model = make_lgbm(model_spec["config"], model_spec["n_estimators"])
        model.fit(train[features], train[target].astype(int))
        val_sample = _sample(val, sample, seed)
        contrib = model.predict(val_sample[features], pred_contrib=True)[:, :-1]  # last column is the bias
        contrib = pd.DataFrame(contrib, columns=features, index=val_sample.index)
        abs_means.append(contrib.abs().mean())
        signed_means.append(contrib.mean())
        samples.append({"shap": contrib, "values": val_sample[features]})
    last = samples[-1]
    plot_rows = last["shap"].index if len(last["shap"]) <= SHAP_PLOT_ROWS else         last["shap"].sample(SHAP_PLOT_ROWS, random_state=seed).index
    return (pd.concat(abs_means, axis=1).mean(axis=1), pd.concat(signed_means, axis=1).mean(axis=1),
            {"shap": last["shap"].loc[plot_rows], "values": last["values"].loc[plot_rows]})


def shap_explanation(model, row, features, top_n=6):
    """Per-prediction explanation: the features pushing this prediction up and down (LightGBM only)."""
    if not hasattr(model, "booster_"):
        return None
    contrib = model.predict(row[features], pred_contrib=True)[0]
    series = pd.Series(contrib[:-1], index=features).sort_values(key=abs, ascending=False).head(top_n)
    return [{"feature": f, "shap_value": float(v), "value": float(row[f].iloc[0])} for f, v in series.items()]


def run_feature_selection(folds, train_full, candidates, target, invalid_map=None, min_votes=MIN_VOTES):
    """Run every method and combine them into a final feature set.

    Returns (selected_features, report DataFrame, extras dict with the correlation matrix and SHAP sample).
    """
    invalid_map = dict(invalid_map or {})
    removed = {}

    kept, invalid_removed = drop_invalid(candidates, invalid_map)
    removed.update(invalid_removed)

    low_coverage = find_low_coverage(train_full, kept)
    removed.update(low_coverage)
    kept = [f for f in kept if f not in low_coverage]

    near_constant = find_near_constant(train_full, kept)
    removed.update(near_constant)
    kept = [f for f in kept if f not in near_constant]

    logger.info("      correlation analysis on %d features", len(kept))
    corr, target_corr = correlation_analysis(train_full, kept, target)
    duplicates = find_duplicates(corr, kept)
    removed.update(duplicates)
    kept = [f for f in kept if f not in duplicates]

    logger.info("      mutual information")
    mi = mutual_information(train_full, kept, target)
    logger.info("      model gain/split importance")
    gain, split = model_importance(folds, kept, target)
    logger.info("      permutation importance (validation folds)")
    perm_mean, perm_positive = calculate_permutation_importance(folds, kept, target)
    logger.info("      SHAP (LightGBM TreeSHAP)")
    shap_abs, shap_signed, shap_sample = calculate_shap(folds, kept, target)

    ranks = pd.DataFrame({"mi": mi, "gain": gain, "perm": perm_mean, "shap": shap_abs}).rank(pct=True)
    votes = (ranks > 0.5).sum(axis=1)
    harmful = (perm_mean < 0) & (perm_positive <= 0.25)
    selected = set(ranks.index[(votes >= min_votes) & ~harmful])
    for f in kept:
        if f not in selected:
            removed.setdefault(f, "low agreement between selection methods"
                               if not harmful.get(f, False) else "permutation importance negative out-of-sample")

    # Redundancy: inside a correlated cluster keep the feature with the best average rank
    composite = ranks.mean(axis=1)
    for cluster in correlation_clusters(corr, sorted(selected)):
        best = max(cluster, key=lambda f: composite.get(f, 0))
        for f in cluster:
            if f != best:
                selected.discard(f)
                removed[f] = f"redundant with {best} (|rho|>={REDUNDANT_CORRELATION})"

    ordered = [f for f in candidates if f in selected]
    report = pd.DataFrame({
        "feature": candidates,
        "correlation_score": [target_corr.get(f, np.nan) for f in candidates],
        "mutual_information": [mi.get(f, np.nan) for f in candidates],
        "model_importance_gain": [gain.get(f, np.nan) for f in candidates],
        "model_importance_split": [split.get(f, np.nan) for f in candidates],
        "permutation_importance": [perm_mean.get(f, np.nan) for f in candidates],
        "permutation_positive_folds": [perm_positive.get(f, np.nan) for f in candidates],
        "shap_importance": [shap_abs.get(f, np.nan) for f in candidates],
        "shap_mean_signed": [shap_signed.get(f, np.nan) for f in candidates],
        "votes": [votes.get(f, np.nan) for f in candidates],
        "selected": [f in selected for f in candidates],
        "removal_reason": [removed.get(f, "") for f in candidates],
    }).sort_values(["selected", "shap_importance"], ascending=[False, False]).reset_index(drop=True)

    logger.info("      selected %d of %d candidate features", len(ordered), len(candidates))
    return ordered, report, {"correlation": corr, "shap_sample": shap_sample}


def single_feature_auc(frame, features, target, sample=MI_SAMPLE, seed=RANDOM_STATE):
    """Diagnostic used by the leakage audit: AUC of each feature used alone."""
    data = _sample(frame[features + [target]].dropna(subset=[target]), sample, seed)
    y = data[target].astype(int)
    out = {}
    for f in features:
        mask = data[f].notna()
        if mask.sum() > 1000 and y[mask].nunique() == 2:
            out[f] = float(roc_auc_score(y[mask], data.loc[mask, f]))
    return pd.Series(out)
