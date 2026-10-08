"""Turn the cached feature dataset into the modelling panel: labels for every horizon plus cross-sectional ranks."""
import logging

import numpy as np

from features.cross_section import add_cross_sectional_ranks
from features.indicators import HORIZONS
from features.labels import add_labels

logger = logging.getLogger(__name__)

LABEL_FLOAT_COLUMNS = ("fwd_ret", "fwd_ret_open", "excess_ret")


def prepare_panel(data, rank_features=()):
    """Labels for every horizon, checked against the per-stock labels written at dataset build time."""
    panel = add_labels(data)
    for name in HORIZONS:
        for prefix in LABEL_FLOAT_COLUMNS:
            panel[f"{prefix}_{name}"] = panel[f"{prefix}_{name}"].astype("float32")
        legacy = panel.get(f"Target_{name.capitalize()}")
        if legacy is not None:
            both = legacy.notna() & panel[f"y_abs_{name}"].notna()
            mismatch = int((legacy[both] != panel.loc[both, f"y_abs_{name}"]).sum())
            if mismatch > 0.001 * both.sum():
                raise RuntimeError(f"{name}: panel labels disagree with the dataset's labels on {mismatch} rows")
    if rank_features:
        panel = add_cross_sectional_ranks(panel, list(rank_features))
    logger.info("Panel: %d rows, %d stocks, %s..%s", len(panel), panel["Stock"].nunique(),
                f"{panel['Date'].min():%Y-%m-%d}", f"{panel['Date'].max():%Y-%m-%d}")
    return panel


def horizon_rows(panel, horizon):
    """Positional mask of rows that carry a label for this horizon."""
    return np.flatnonzero(panel[f"label_end_rel_{horizon}"].notna().to_numpy())
