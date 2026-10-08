import json

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from models.predictor import StockPredictor, percentile_rank

DAY = pd.Timestamp("2026-10-08")


def _write_model_dir(tmp_path, reliable):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"f1": rng.normal(size=500), "f2": rng.normal(size=500)})
    model = xgb.XGBClassifier(n_estimators=5, max_depth=2).fit(X, (X["f1"] > 0).astype(int))
    bundle = {"model": model, "features": ["f1", "f2"], "threshold": 0.5, "reference_probs": [0.9] * 10,
              "target": "absolute", "model_type": "xgboost"}
    joblib.dump(bundle, tmp_path / "model_weekly.pkl")
    metrics = {"schema_version": 2, "horizons": {"weekly": {
        "edge": {"reliable": reliable, "reasons": [] if reliable else ["rank IC positive in only 2 of 5 folds"]},
        "final_test": {"rank_ic": 0.01, "auc_rel": 0.51, "period": ["2025-05-16", "2026-09-10"]}}}}
    (tmp_path / "metrics.json").write_text(json.dumps(metrics))
    universe = pd.DataFrame({"Date": DAY, "f1": np.linspace(-2, 2, 120), "f2": 0.0})
    return pd.DataFrame({"Date": [DAY], "f1": [3.0], "f2": [0.0]}), universe


def test_a_horizon_without_a_reliable_edge_makes_no_call(tmp_path):
    row, universe = _write_model_dir(tmp_path, reliable=False)
    result = StockPredictor(str(tmp_path)).predict_all(row, explain=True, universe=universe)["weekly"]
    assert result["direction"] == "NO_EDGE" and result["label"] == "NO RELIABLE EDGE"
    assert result["percentile"] is None and result["explanation"] is None
    assert result["evaluation"]["edge"]["reasons"]


def test_a_reliable_horizon_ranks_against_the_same_session_universe(tmp_path):
    row, universe = _write_model_dir(tmp_path, reliable=True)
    predictor = StockPredictor(str(tmp_path))
    assert predictor.needs_universe
    result = predictor.predict_all(row, explain=True, universe=universe)["weekly"]
    assert result["direction"] == "UP" and result["label"] == "OUTPERFORM"
    assert result["percentile"] > 60 and "120 stocks" in result["rank_basis"]
    # The explanation is relative to the same-session peers: an input equal for every stock (like a market index
    # feature) contributes nothing to the rank
    contributions = {e["feature"]: e["shap_value"] for e in result["explanation"]}
    assert result["explanation"][0]["feature"] == "f1" and abs(contributions.get("f2", 0.0)) < 1e-6
    # The stale training-time reference (all 0.9) is ignored: against it this stock would rank at the bottom
    low = predictor.predict_all(row.assign(f1=-3.0), universe=universe)["weekly"]
    assert low["label"] == "UNDERPERFORM"


def test_no_call_without_same_session_universe_data(tmp_path):
    row, universe = _write_model_dir(tmp_path, reliable=True)
    predictor = StockPredictor(str(tmp_path))
    assert predictor.predict_all(row, universe=None)["weekly"]["direction"] == "NO_RANK"
    stale = universe.assign(Date=DAY - pd.Timedelta(days=21))
    result = predictor.predict_all(row, universe=stale)["weekly"]
    assert result["label"] == "RANK UNAVAILABLE" and "not this ticker's last session" in result["rank_basis"]


def test_missing_metrics_file_means_no_call(tmp_path):
    row, universe = _write_model_dir(tmp_path, reliable=True)
    (tmp_path / "metrics.json").unlink()
    predictor = StockPredictor(str(tmp_path))
    assert not predictor.needs_universe
    assert predictor.predict_all(row, universe=universe)["weekly"]["direction"] == "NO_EDGE"


def test_percentile_rank_counts_ties_half():
    assert percentile_rank([0.1, 0.2, 0.3, 0.4], 0.25) == 50.0
    assert percentile_rank([0.1, 0.2, 0.2, 0.4], 0.2) == 50.0
