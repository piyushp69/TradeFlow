"""Inference.

`StockPredictor` loads the model bundles written by trainer.py together with their out-of-sample evaluation
(models/saved_models/metrics.json) and turns the latest feature row into a forecast. Features are built with
features.pipeline.build_feature_frame, the same function training uses.

A horizon only gets a call when its out-of-sample evaluation passed the edge rule recorded in metrics.json.
Otherwise the forecast is reported as "no reliable edge" and no direction is given.

A call is relative: the stock's model score is ranked against the scores of every universe stock on the same
session (models.inference.load_universe_snapshot), because that same-day ranking is what the evaluation measured.
The training-time cross-section (`reference_probs`) is not used: the scores move with market-wide features, so a
rank against an older date would mostly reflect how the market has changed since then. Without the same-session
universe there is no call.
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
# A same-session rank needs most of the 180-stock universe
MIN_SAME_DAY_STOCKS = 90

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

    @property
    def needs_universe(self):
        """True when at least one horizon makes calls, which need the same-session universe."""
        return any(self.has_edge(period) for period in self.models)

    @staticmethod
    def _scores(bundle, frame):
        return bundle["model"].predict_proba(frame[bundle["features"]])[:, 1]

    def _same_day_rank(self, bundle, score, date, universe):
        """Percentile of the score among the universe's scores on the same session, or None."""
        if universe is None or universe.empty or any(f not in universe.columns for f in bundle["features"]):
            return None, "same-session universe data could not be loaded"
        same_day = universe[universe["Date"] == date]
        if len(same_day) < MIN_SAME_DAY_STOCKS:
            return None, f"universe data is for {universe['Date'].max():%d %b %Y}, not this ticker's last session"
        return percentile_rank(self._scores(bundle, same_day), score), f"{len(same_day)} stocks on {date:%d %b %Y}"

    def predict_all(self, current_data, explain=False, universe=None):
        if not self.models or current_data is None or current_data.empty:
            return None

        results = {}
        for period, bundle in self.models.items():
            try:
                last_row = current_data.tail(1)
                if bundle.get("cross_sectional"):
                    logger.warning("%s model needs same-day universe ranks of its inputs; not supported", period)
                    continue
                missing = [f for f in bundle["features"] if f not in last_row.columns]
                if missing:
                    logger.warning("%s model: %d features missing from the input frame", period, len(missing))
                    continue
                score = float(self._scores(bundle, last_row)[0])
                reliable = self.has_edge(period)
                percentile, basis = (self._same_day_rank(bundle, score, last_row["Date"].iloc[0], universe)
                                     if reliable else (None, None))
                if percentile is None:
                    call = None
                elif percentile >= NEUTRAL_BAND[1]:
                    call = "UP"
                elif percentile <= NEUTRAL_BAND[0]:
                    call = "DOWN"
                else:
                    call = "NEUTRAL"
                target = bundle.get("target", "absolute")
                direction = "NO_EDGE" if not reliable else call or "NO_RANK"
                results[period] = {
                    "reliable": reliable,
                    "direction": direction,
                    "label": {"NO_EDGE": "NO RELIABLE EDGE", "NO_RANK": "RANK UNAVAILABLE"}.get(
                        direction, CALL_LABELS.get(direction)),
                    "probability": round(score * 100, 2),
                    "percentile": None if percentile is None else round(percentile, 1),
                    "rank_basis": basis,
                    "target": target,
                    "target_description": bundle.get("target_description"),
                    "model_type": bundle.get("model_type"),
                    "experiment": bundle.get("experiment"),
                    "feature_groups": bundle.get("feature_groups"),
                    "n_features": len(bundle["features"]),
                    "train_period": bundle.get("train_period"),
                    "evaluation": self.evaluation(period),
                    "explanation": self.explain(period, current_data, universe) if explain and call else None,
                }
            except Exception:
                logger.exception("Prediction failed for the %s horizon", period)
        return results or None

    def explain(self, period, current_data, universe=None, top_n=6):
        """Why the stock ranks where it does: its SHAP contributions minus the average stock's on the same session.

        Market-wide features (NIFTY, VIX, USD/INR) take the same value for every stock, so they move every score
        together and cancel out here; what is left is what sets this stock apart from its peers, which is what a
        relative call is about. XGBoost's exact TreeSHAP, in log-odds.
        """
        bundle = self.models.get(period)
        if bundle is None or not hasattr(bundle["model"], "get_booster") or universe is None:
            return None
        try:
            import xgboost
            row = current_data.tail(1)
            same_day = universe[universe["Date"] == row["Date"].iloc[0]]
            if same_day.empty:
                return None
            columns = bundle["features"]
            names = bundle.get("raw_features", columns)
            booster = bundle["model"].get_booster()
            own = booster.predict(xgboost.DMatrix(row[columns]), pred_contribs=True)[0, :-1]
            peers = booster.predict(xgboost.DMatrix(same_day[columns]), pred_contribs=True)[:, :-1].mean(axis=0)
            series = pd.Series(own - peers, index=names).sort_values(key=abs, ascending=False).head(top_n)
            values = dict(zip(names, row[columns].iloc[0].tolist()))
            return [{"feature": f, "shap_value": float(v), "value": float(values[f])} for f, v in series.items()]
        except Exception:
            logger.exception("SHAP explanation failed")
            return None
