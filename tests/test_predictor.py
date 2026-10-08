import json

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from models.predictor import StockPredictor, percentile_rank


def _write_model_dir(tmp_path, reliable):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"f1": rng.normal(size=500), "f2": rng.normal(size=500)})
    model = xgb.XGBClassifier(n_estimators=5, max_depth=2).fit(X, (X["f1"] > 0).astype(int))
    reference = np.sort(model.predict_proba(X)[:, 1]).tolist()
    bundle = {"model": model, "features": ["f1", "f2"], "threshold": float(np.median(reference)),
              "reference_probs": reference, "target": "relative", "model_type": "xgboost"}
    joblib.dump(bundle, tmp_path / "model_weekly.pkl")
    metrics = {"schema_version": 2, "horizons": {"weekly": {
        "edge": {"reliable": reliable, "reasons": [] if reliable else ["rank IC positive in only 2 of 5 folds"]},
        "final_test": {"rank_ic": 0.01, "auc_rel": 0.51, "period": ["2025-05-16", "2026-09-10"]}}}}
    (tmp_path / "metrics.json").write_text(json.dumps(metrics))
    return pd.DataFrame({"f1": [3.0], "f2": [0.0]})


def test_a_horizon_without_a_reliable_edge_makes_no_call(tmp_path):
    row = _write_model_dir(tmp_path, reliable=False)
    result = StockPredictor(str(tmp_path)).predict_all(row, explain=True)["weekly"]
    assert result["direction"] == "NO_EDGE" and result["label"] == "NO RELIABLE EDGE"
    assert result["explanation"] is None
    assert result["evaluation"]["edge"]["reasons"]


def test_a_horizon_with_a_reliable_edge_gets_a_relative_call_and_explanation(tmp_path):
    row = _write_model_dir(tmp_path, reliable=True)
    result = StockPredictor(str(tmp_path)).predict_all(row, explain=True)["weekly"]
    assert result["direction"] == "UP" and result["label"] == "OUTPERFORM"
    assert result["percentile"] > 60 and result["explanation"][0]["feature"] == "f1"


def test_missing_metrics_file_means_no_call(tmp_path):
    row = _write_model_dir(tmp_path, reliable=True)
    (tmp_path / "metrics.json").unlink()
    assert StockPredictor(str(tmp_path)).predict_all(row)["weekly"]["direction"] == "NO_EDGE"


def test_percentile_rank_counts_ties_half():
    assert percentile_rank([0.1, 0.2, 0.3, 0.4], 0.25) == 50.0
    assert percentile_rank([0.1, 0.2, 0.2, 0.4], 0.2) == 50.0
