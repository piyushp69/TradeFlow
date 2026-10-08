import numpy as np
import pandas as pd
import pytest

from conftest import make_panel
from features.labels import add_labels
from models.evaluation import purged_walk_forward

HORIZONS = {"weekly": 5, "monthly": 21, "yearly": 252}


@pytest.fixture(scope="module")
def labelled():
    # ~10 years of sessions so the yearly horizon still has room for five folds and a test block
    return add_labels(make_panel(n_stocks=5, n_days=2600, seed=7), HORIZONS)


def _sessions(frame):
    return np.sort(frame.loc[frame["label_end_rel_weekly"].notna(), "Date"].unique())


@pytest.mark.parametrize("name,h", HORIZONS.items())
def test_no_training_label_reaches_into_its_evaluation_block(labelled, name, h):
    end = labelled[f"label_end_rel_{name}"]
    folds, test = purged_walk_forward(labelled["Date"], end, h, include_test=True)
    for split in folds + [test]:
        first_eval = labelled["Date"].iloc[split.eval].min()
        assert end.iloc[split.train].max() < first_eval
        assert labelled["Date"].iloc[split.train].max() < first_eval


@pytest.mark.parametrize("name,h", HORIZONS.items())
def test_gap_between_train_and_evaluation_is_at_least_the_horizon(labelled, name, h):
    folds, test = purged_walk_forward(labelled["Date"], labelled[f"label_end_rel_{name}"], h, include_test=True)
    sessions = np.sort(labelled["Date"].unique())
    for split in folds + [test]:
        last_train = labelled["Date"].iloc[split.train].max()
        first_eval = labelled["Date"].iloc[split.eval].min()
        between = ((sessions > last_train) & (sessions < first_eval)).sum()
        assert between >= h
        assert split.gap_sessions >= h


@pytest.mark.parametrize("name,h", HORIZONS.items())
def test_folds_are_ordered_disjoint_and_end_before_the_test_block(labelled, name, h):
    end = labelled[f"label_end_rel_{name}"]
    folds, test = purged_walk_forward(labelled["Date"], end, h, include_test=True)
    assert len(folds) >= 5
    test_start = labelled["Date"].iloc[test.eval].min()
    previous_end = None
    for split in folds:
        dates = labelled["Date"].iloc[split.eval]
        if previous_end is not None:
            assert dates.min() > previous_end
        previous_end = dates.max()
        # Validation labels are fully realised before the test block starts: no test-period price is used
        assert end.iloc[split.eval].max() < test_start
        assert not set(split.eval) & set(test.eval) and not set(split.train) & set(test.eval)
    assert not set(test.train) & set(test.eval)


def test_expanding_window_trains_on_earlier_folds(labelled):
    folds, _ = purged_walk_forward(labelled["Date"], labelled["label_end_rel_weekly"], 5)
    for earlier, later in zip(folds, folds[1:]):
        assert set(earlier.train) < set(later.train)
        # Most of the earlier validation block is training data for the next fold (its tail is purged)
        assert len(set(earlier.eval) & set(later.train)) > 0.8 * len(earlier.eval)


def test_test_block_is_the_most_recent_labelled_sessions_and_hidden_by_default(labelled):
    end = labelled["label_end_rel_weekly"]
    folds, test = purged_walk_forward(labelled["Date"], end, 5)
    assert test is None
    _, test = purged_walk_forward(labelled["Date"], end, 5, include_test=True)
    sessions = np.sort(labelled.loc[end.notna(), "Date"].unique())
    test_dates = np.sort(labelled["Date"].iloc[test.eval].unique())
    assert test_dates[-1] == sessions[-1]
    assert len(test_dates) == len(sessions) - int(len(sessions) * 0.85)


def test_unlabelled_rows_are_never_used(labelled):
    end = labelled["label_end_rel_monthly"]
    folds, test = purged_walk_forward(labelled["Date"], end, 21, include_test=True)
    unlabelled = set(np.flatnonzero(end.isna().to_numpy()))
    assert unlabelled
    for split in folds + [test]:
        assert not unlabelled & set(split.train) and not unlabelled & set(split.eval)


def test_label_end_purge_drops_rows_whose_label_crosses_the_boundary_after_a_gap():
    """A halted stock's 5-row label spans more than 5 sessions; a date-only gap would leak it into training."""
    panel = add_labels(make_panel(n_stocks=5, n_days=400, seed=11), {"weekly": 5})
    folds, _ = purged_walk_forward(panel["Date"], panel["label_end_weekly"], 5)
    boundary = panel["Date"].iloc[folds[2].eval].min()
    sessions = np.sort(panel["Date"].unique())
    halt = sessions[np.searchsorted(sessions, boundary) - 3:np.searchsorted(sessions, boundary)]
    # S0 does not trade in the three sessions before fold 3 starts (other stocks do, so the calendar is unchanged)
    halted = panel[~((panel["Stock"] == "S0") & panel["Date"].isin(halt))][["Date", "Stock", "Open", "Close"]]
    out = add_labels(halted, {"weekly": 5})
    end = out["label_end_weekly"]

    folds_rows, _ = purged_walk_forward(out["Date"], end, 5, purge="rows")
    folds_purged, _ = purged_walk_forward(out["Date"], end, 5, purge="label_end")
    assert end.iloc[folds_rows[2].train].max() >= boundary, "the old date-only rule lets this label leak"
    for split in folds_purged:
        assert end.iloc[split.train].max() < out["Date"].iloc[split.eval].min()
    assert [s.eval_period for s in folds_rows] == [s.eval_period for s in folds_purged]


def test_rejects_unknown_purge_mode(labelled):
    with pytest.raises(ValueError):
        purged_walk_forward(labelled["Date"], labelled["label_end_rel_weekly"], 5, purge="none")
