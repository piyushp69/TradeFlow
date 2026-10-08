"""Fit a spec on each split and score it: shared by experiments.py (folds only) and trainer.py."""
import logging
import time

import numpy as np
import pandas as pd

from features.indicators import HORIZONS
from models.evaluation import evaluate_block, newey_west_tstat, summarize_folds
from models.training import fit_spec

logger = logging.getLogger(__name__)


def score_split(spec, panel, split, horizon, device):
    """Fit on the split's training rows, predict its evaluation rows. Returns (fitted, metrics, daily IC, preds)."""
    days = HORIZONS[horizon]
    fitted = fit_spec(spec, panel, split.train, horizon, days, device)
    rows = panel.iloc[split.eval]
    proba = fitted.predict(rows)
    metrics, ic = evaluate_block(rows, proba, horizon)
    metrics.update(fold=split.name, train_period=list(split.train_period), gap_sessions=split.gap_sessions,
                   train_rows=fitted.info["train_rows"], n_trees=fitted.info["n_trees"],
                   n_features=len(fitted.features), rank_ic_tstat=newey_west_tstat(ic.to_numpy(), days),
                   fit_info={k: v for k, v in fitted.info.items() if k in ("prune", "early_stopping")})
    predictions = pd.DataFrame({"Date": rows["Date"].to_numpy(), "Stock": rows["Stock"].to_numpy(), "p": proba})
    return fitted, metrics, ic, predictions


def walk_forward(spec, panel, splits, horizon, device):
    """Score a spec on every walk-forward fold. Returns (per-fold metrics, summary, daily IC over all folds)."""
    start = time.time()
    fold_metrics, ics = [], []
    for split in splits:
        _, metrics, ic, _ = score_split(spec, panel, split, horizon, device)
        fold_metrics.append(metrics)
        ics.append(ic)
    summary = summarize_folds(fold_metrics)
    logger.info("   %-34s IC %s | AUC rel %.4f abs %.4f | %.0fs", spec.name,
                " ".join(f"{m['rank_ic']:+.4f}" for m in fold_metrics), summary["auc_rel"]["mean"],
                summary["auc_abs"]["mean"], time.time() - start)
    daily_ic = pd.concat(ics).sort_index() if ics else pd.Series(dtype=float)
    return fold_metrics, summary, daily_ic


def kept_feature_share(fold_metrics, features):
    """Share of folds in which pruning kept each feature (1.0 everywhere when pruning is off)."""
    kept = np.ones(len(features))
    pruned = [m["fit_info"].get("prune") for m in fold_metrics]
    if any(pruned):
        kept = np.mean([[f not in (p or {}).get("dropped", []) for f in features] for p in pruned], axis=0)
    return dict(zip(features, kept.tolist()))
