"""Inference.

`StockPredictor` loads the model bundles written by trainer.py together with their out-of-sample evaluation
(models/saved_models/metrics.json) and turns the latest feature row into a forecast. Features are built with
features.pipeline.build_feature_frame, the same function training uses.

A forecast is relative to the cross-section: the stock's model score is ranked against the scores of the whole
training universe on the latest training date (`reference_probs`), and the call comes from that rank.

A horizon only gets a call when its out-of-sample evaluation passed the edge rule recorded in metrics.json.
Otherwise the forecast is reported as "no reliable edge" and no direction is given.
"""
import json
import logging
import os

import joblib
import numpy as np
import pandas as pd

from features.indicators import HORIZONS

logger = logging.getLogger(__name__)

# Stocks ranked inside this percentile band get NEUTRAL: any edge is concentrated in the extremes
NEUTRAL_BAND = (40, 60)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
METRICS_FILE = "metrics.json"

# The edge rule validates ranking across stocks (rank IC), so a call says how a stock ranks against its peers,
# whichever target the model was trained on
CALL_LABELS = {"UP": "OUTPERFORM", "DOWN": "UNDERPERFORM", "NEUTRAL": "NEUTRAL"}


def load_metrics(model_dir=MODEL_DIR):
    """The evaluation file written by trainer.py, or None when it is missing or from an older pipeline."""
    try:
        with open(os.path.join(model_dir, METRICS_FILE), encoding="utf-8") as fh:
            metrics = json.load(fh)
    except (OSError, ValueError):
        return None
    return metrics if isinstance(metrics, dict) and metrics.get("schema_version") == 2 else None


def percentile_rank(reference, value):
    """Mid-rank percentile of `value` within `reference` (ties count half)."""
    reference = np.asarray(reference)
    ties = np.isclose(reference, value, atol=1e-6)
    return float((np.sum((reference < value) & ~ties) + 0.5 * ties.sum()) / reference.size * 100)


class StockPredictor:
    def __init__(self, model_dir=MODEL_DIR):
        self.models = {}
        self.metrics = load_metrics(model_dir) or {}
        for period in HORIZONS:
            model_path = os.path.join(model_dir, f"model_{period}.pkl")
            if not os.path.exists(model_path):
                continue
            try:
                bundle = joblib.load(model_path)
            except Exception as exc:
                logger.warning("Could not load %s: %s", model_path, exc)
                continue
            # Only accept bundles produced by trainer.py
            if isinstance(bundle, dict) and {"features", "threshold", "model"} <= bundle.keys():
                self.models[period] = bundle

    def evaluation(self, period):
        """Out-of-sample results for one horizon from metrics.json ({} when unavailable)."""
        return (self.metrics.get("horizons") or {}).get(period) or {}

    def has_edge(self, period):
        return bool((self.evaluation(period).get("edge") or {}).get("reliable"))

    @property
    def requirements(self):
        """Union of the data sources the loaded models need (e.g. {"market"})."""
        needs = set()
        for bundle in self.models.values():
            needs |= set(bundle.get("requirements") or [])
        return needs

    @property
    def features(self):
        return sorted({f for b in self.models.values() for f in b.get("raw_features", b["features"])})

    @staticmethod
    def _score(bundle, row):
        return float(bundle["model"].predict_proba(row[bundle["features"]])[0][1])

    def predict_all(self, current_data, explain=False):
        if not self.models or current_data is None or current_data.empty:
            return None

        results = {}
        for period, bundle in self.models.items():
            try:
                last_row = current_data.tail(1)
                if bundle.get("cross_sectional"):
                    logger.warning("%s model needs same-day universe ranks; not available for one ticker", period)
                    continue
                missing = [f for f in bundle["features"] if f not in last_row.columns]
                if missing:
                    logger.warning("%s model: %d features missing from the input frame", period, len(missing))
                    continue
                score = self._score(bundle, last_row)
                reference = np.asarray(bundle.get("reference_probs") or [])
                percentile = percentile_rank(reference, score) if reference.size else None
                if percentile is None:
                    call = "UP" if score >= bundle["threshold"] else "DOWN"
                elif percentile >= NEUTRAL_BAND[1]:
                    call = "UP"
                elif percentile <= NEUTRAL_BAND[0]:
                    call = "DOWN"
                else:
                    call = "NEUTRAL"
                target = bundle.get("target", "absolute")
                reliable = self.has_edge(period)
                results[period] = {
                    "reliable": reliable,
                    "direction": call if reliable else "NO_EDGE",
                    "label": CALL_LABELS[call] if reliable else "NO RELIABLE EDGE",
                    "probability": round(score * 100, 2),
                    "threshold": bundle["threshold"],
                    "percentile": None if percentile is None else round(percentile, 1),
                    "reference_date": bundle.get("reference_date"),
                    "target": target,
                    "target_description": bundle.get("target_description"),
                    "model_type": bundle.get("model_type"),
                    "experiment": bundle.get("experiment"),
                    "feature_groups": bundle.get("feature_groups"),
                    "n_features": len(bundle["features"]),
                    "train_period": bundle.get("train_period"),
                    "evaluation": self.evaluation(period),
                    "explanation": self.explain(period, current_data) if explain and reliable else None,
                }
            except Exception:
                logger.exception("Prediction failed for the %s horizon", period)
        return results or None

    def explain(self, period, current_data, top_n=6):
        """SHAP contributions for the latest row, from XGBoost's exact TreeSHAP."""
        bundle = self.models.get(period)
        if bundle is None or not hasattr(bundle["model"], "get_booster"):
            return None
        try:
            import xgboost
            row = current_data.tail(1)
            columns = bundle["features"]
            names = bundle.get("raw_features", columns)
            contrib = bundle["model"].get_booster().predict(xgboost.DMatrix(row[columns]), pred_contribs=True)[0]
            series = pd.Series(contrib[:-1], index=names).sort_values(key=abs, ascending=False).head(top_n)
            values = dict(zip(names, row[columns].iloc[0].tolist()))
            return [{"feature": f, "shap_value": float(v), "value": float(values[f])} for f, v in series.items()]
        except Exception:
            logger.exception("SHAP explanation failed")
            return None
