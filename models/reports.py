"""Read the artifacts written by trainer.py (reports/) for the dashboard.

Every loader returns None when the file is missing, so the app still works before a training run.
"""
import json
import os

import joblib
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(ROOT, "reports")


def _path(name, extension):
    return os.path.join(REPORTS_DIR, f"{name}.{extension}")


def available():
    if not os.path.isdir(REPORTS_DIR):
        return []
    return sorted(os.listdir(REPORTS_DIR))


def load_table(name):
    path = _path(name, "csv")
    if not os.path.exists(path):
        return None
    index_col = 0 if name.startswith("correlation") else None
    return pd.read_csv(path, index_col=index_col)


def load_json(name):
    path = _path(name, "json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_pickle(name):
    path = _path(name, "pkl")
    return joblib.load(path) if os.path.exists(path) else None


def feature_selection(horizon):
    return load_table(f"feature_selection_{horizon}")


def experiments(horizon):
    return load_table(f"experiments_{horizon}")


def correlation(horizon):
    return load_table(f"correlation_{horizon}")


def shap_sample(horizon):
    return load_pickle(f"shap_sample_{horizon}")


def backtest(horizon):
    return load_json(f"backtest_{horizon}")


def equity_curve(horizon, baseline=False):
    return load_table(f"equity_curve_baseline_{horizon}" if baseline else f"equity_curve_{horizon}")


def predictions(horizon):
    table = load_table(f"predictions_{horizon}")
    if table is not None:
        table["Date"] = pd.to_datetime(table["Date"])
    return table


def intraday_study(horizon):
    return load_table(f"intraday_1h_study_{horizon}")


def leakage_audit():
    return load_json("leakage_audit")


def training_log_tail(lines=200):
    path = os.path.join(REPORTS_DIR, "training.log")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return "".join(fh.readlines()[-lines:])
