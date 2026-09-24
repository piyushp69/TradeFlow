"""Inference.

`StockPredictor` loads the model bundles written by trainer.py and turns the latest feature row into a
direction. Features are built with features.pipeline.build_feature_frame, the same function training uses,
so a feature can never mean one thing in training and another in the app.

A forecast is relative to the cross-section: the stock's P(UP) is compared with the distribution of P(UP)
across the whole training universe (`reference_probs`), and the direction comes from that rank.
"""
import logging
import os

import joblib
import numpy as np

from features.indicators import HORIZONS
from features.selection import shap_explanation

logger = logging.getLogger(__name__)

# Stocks ranked inside this percentile band get NEUTRAL: the models' edge is concentrated in the extremes
NEUTRAL_BAND = (40, 60)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")


class StockPredictor:
    def __init__(self, model_dir=MODEL_DIR):
        self.models = {}

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
            if isinstance(bundle, dict) and {"features", "threshold"} <= bundle.keys() \
                    and ("model" in bundle or "components" in bundle):
                self.models[period] = bundle

    @property
    def requirements(self):
        """Union of the data sources the loaded models need (e.g. {"market", "intraday_1h"})."""
        needs = set()
        for bundle in self.models.values():
            needs |= set(bundle.get("requirements") or [])
        return needs

    @property
    def features(self):
        return sorted({f for b in self.models.values() for f in b["features"]})

    @staticmethod
    def _predict(bundle, row):
        if "components" in bundle:
            probs = [c["model"].predict_proba(row[c["features"]])[0][1] for c in bundle["components"]]
            return float(np.mean(probs))
        return float(bundle["model"].predict_proba(row[bundle["features"]])[0][1])

    def predict_all(self, current_data, explain=False):
        if not self.models or current_data is None or current_data.empty:
            return None

        try:
            results = {}
            for period, bundle in self.models.items():
                last_row = current_data.tail(1)
                missing = [f for f in bundle["features"] if f not in last_row.columns]
                if missing:
                    logger.warning("%s model: %d features missing from the input frame", period, len(missing))
                    continue
                up_prob = self._predict(bundle, last_row)
                threshold = bundle["threshold"]
                reference = np.asarray(bundle.get("reference_probs") or [])

                if reference.size:
                    # Mid-rank percentile among the training universe; ties (common with shallow trees) count half
                    ties = np.isclose(reference, up_prob, atol=1e-4)
                    percentile = float((np.sum((reference < up_prob) & ~ties) + 0.5 * ties.sum()) / reference.size * 100)
                    if percentile >= NEUTRAL_BAND[1]:
                        direction = "UP"
                    elif percentile <= NEUTRAL_BAND[0]:
                        direction = "DOWN"
                    else:
                        direction = "NEUTRAL"
                else:
                    percentile = None
                    direction = "UP" if up_prob >= threshold else "DOWN"

                results[period] = {
                    "direction": direction,
                    "up_probability": round(up_prob * 100, 2),
                    "threshold": threshold,
                    "percentile": None if percentile is None else round(percentile, 1),
                    "reference_date": bundle.get("reference_date"),
                    "model_type": bundle.get("model_type"),
                    "experiment": bundle.get("experiment"),
                    "feature_groups": bundle.get("feature_groups"),
                    "n_features": len(bundle["features"]),
                    "cv_auc": bundle.get("cv_auc"),
                    "test_metrics": bundle.get("test_metrics", {}),
                    "validation_metrics": bundle.get("validation_metrics") or {},
                    "test_period": bundle.get("test_period"),
                    "backtest": bundle.get("backtest"),
                    "explanation": self.explain(period, current_data) if explain else None,
                }

            return results or None

        except Exception:
            logger.exception("Prediction failed")
            return None

    def explain(self, period, current_data, top_n=6):
        """SHAP contributions for the latest row (LightGBM models only; None otherwise)."""
        bundle = self.models.get(period)
        if bundle is None:
            return None
        model, features = bundle.get("model"), bundle["features"]
        if model is None and "components" in bundle:
            # An ensemble's feature list is the union of its components; the LightGBM part needs its own
            component = next((c for c in bundle["components"] if c["model_type"] == "lgbm"), None)
            model, features = (component["model"], component["features"]) if component else (None, None)
        if model is None or not hasattr(model, "booster_"):
            return None
        try:
            return shap_explanation(model, current_data.tail(1), features, top_n)
        except Exception:
            logger.exception("SHAP explanation failed")
            return None
