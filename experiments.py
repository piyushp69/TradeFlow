"""Experiment ladder on purged walk-forward folds. The final test period is never used here.

    python experiments.py [--horizons weekly monthly yearly] [--cpu]

For each horizon:

    B    the previously deployed model, unchanged, run through the new harness (the baseline)
    F1   NIFTY 50 removed from the stock universe (the committed models still traded the index as a stock)
    F2   purge on each label's end date instead of a fixed row-count gap
    F3   start from the full candidate feature pool instead of feature sets chosen with the old validation data
    E0   XGBoost in place of LightGBM / HistGB with equivalent settings
    E1+  one change at a time (relative target, clear-move labels, cross-sectional ranks, weekly short-term
         features, regularisation, early stopping, pruning)

Fixes F1-F3 and E0 are adopted whatever their effect, which is measured and logged. Every E1+ change is compared
with the configuration it would replace, fold by fold, and kept only if all three hold:
    (a) mean rank IC over the 5 folds improves,
    (b) rank IC improves in at least 4 of the 5 folds,
    (c) mean AUC against the relative label does not fall.
When two variants of one change both pass, the one with the higher mean rank IC is kept.

Writes reports/experiment_log.md, reports/experiments_<horizon>.csv and reports/experiments.json.
"""
import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from features.indicators import HORIZONS
from models.config import (BASELINE_SPECS, CANDIDATE_POOL, CLEAR_MOVE_K, EARLY_STOPPING_METRICS,
                           REGULARISED_VARIANTS, SHORT_TERM_POOL, XGB_EQUIVALENT)
from models.evaluation import FOLD_METRICS, purged_walk_forward
from models.panel import prepare_panel
from models.training import detect_device
from models.walkforward import walk_forward
from trainer import CACHE_DIR, REPORTS_DIR, STOCKS, build_dataset

logger = logging.getLogger("tradeflow.experiments")

LEGACY_DATASET = os.path.join(CACHE_DIR, "dataset_legacy_2026-09-17.pkl")
MIN_BETTER_FOLDS = 4


@dataclass
class RunResult:
    spec: object
    folds: list
    daily_ic: pd.Series
    summary: dict = field(default_factory=dict)
    seconds: float = 0.0


def run_spec(spec, panel, splits, horizon, device):
    start = time.time()
    fold_metrics, summary, daily_ic = walk_forward(spec, panel, splits, horizon, device)
    return RunResult(spec, fold_metrics, daily_ic, summary, time.time() - start)


def compare(candidate, incumbent):
    deltas = np.array([c["rank_ic"] - i["rank_ic"] for c, i in zip(candidate.folds, incumbent.folds)])
    auc_delta = candidate.summary["auc_rel"]["mean"] - incumbent.summary["auc_rel"]["mean"]
    better = int((deltas > 0).sum())
    passed = bool(deltas.mean() > 0 and better >= MIN_BETTER_FOLDS and auc_delta >= 0)
    return {"ic_delta": float(deltas.mean()), "better_folds": better, "auc_rel_delta": float(auc_delta),
            "passed": passed}


class Ladder:
    def __init__(self, horizon, panel, device):
        self.horizon, self.panel, self.device = horizon, panel, device
        self.folds, _ = purged_walk_forward(panel["Date"], panel[f"label_end_rel_{horizon}"], HORIZONS[horizon])
        self.rows = []
        self.runs = {}

    def run(self, spec, panel=None, splits=None):
        return run_spec(spec, self.panel if panel is None else panel, splits or self.folds, self.horizon, self.device)

    def record(self, step, change, result, decision, reference=None, comparison=None, kind="experiment"):
        self.runs[step] = result
        self.rows.append({"step": step, "kind": kind, "change": change, "decision": decision,
                          "spec": result.spec.to_dict(), "folds": result.folds, "summary": result.summary,
                          "reference": reference, "comparison": comparison, "seconds": round(result.seconds, 1)})

    def try_change(self, step, change, incumbent, candidates):
        """Run each candidate variant against the incumbent; keep the best one that passes the rule."""
        passing = []
        for label, spec in candidates:
            result = self.run(spec)
            comparison = compare(result, incumbent)
            name = f"{step} ({label})" if len(candidates) > 1 else step
            text = f"{change} ({label})" if len(candidates) > 1 else change
            self.record(name, text, result, None, reference=incumbent_step(self, incumbent),
                        comparison=comparison)
            if comparison["passed"]:
                passing.append((result.summary["rank_ic"]["mean"], name, result))
        winner = max(passing, key=lambda x: x[0]) if passing else None
        for row in self.rows:
            if row["decision"] is None:
                row["decision"] = ("kept" if winner and row["step"] == winner[1] else
                                   "passed, not chosen" if row["comparison"]["passed"] else "rejected")
        return winner[2] if winner else incumbent


def incumbent_step(ladder, result):
    return next(step for step, r in ladder.runs.items() if r is result)


def run_ladder(horizon, panel, device, legacy_panel=None):
    logger.info("=" * 90)
    ladder = Ladder(horizon, panel, device)
    for split in ladder.folds:
        logger.info("[%s] %s train %s..%s (%d rows) | gap %d sessions | validate %s..%s (%d rows)", horizon,
                    split.name, *split.train_period, len(split.train), split.gap_sessions, *split.eval_period,
                    len(split.eval))

    base = BASELINE_SPECS[horizon]
    baseline = ladder.run(base)
    ladder.record("B", f"Baseline: deployed model ({base.description})", baseline, "reference", kind="baseline")

    # Leakage fixes: adopted regardless; the log shows the configuration without the fix next to the one with it
    if legacy_panel is not None:
        legacy_folds, _ = purged_walk_forward(legacy_panel["Date"], legacy_panel[f"label_end_rel_{horizon}"],
                                              HORIZONS[horizon])
        with_index = ladder.run(base.with_changes(name="baseline + NIFTY index as a stock"), legacy_panel,
                                legacy_folds)
        ladder.record("F1", "NIFTY 50 index removed from the stock universe (row shows the old universe "
                            "with the index; compare with B)", with_index, "fix adopted", reference="B",
                      comparison=compare(baseline, with_index), kind="fix")
    row_gap_folds, _ = purged_walk_forward(panel["Date"], panel[f"label_end_rel_{horizon}"], HORIZONS[horizon],
                                           purge="rows")
    row_gap = ladder.run(base.with_changes(name="baseline, row-count gap"), splits=row_gap_folds)
    ladder.record("F2", "Purge on label end dates (row shows the old row-count gap; compare with B)", row_gap,
                  "fix adopted", reference="B", comparison=compare(baseline, row_gap), kind="fix")

    pool_spec = base.with_changes(name="candidate pool", features=CANDIDATE_POOL,
                                  description=f"{base.library}, previous settings, full candidate pool")
    pool = ladder.run(pool_spec)
    ladder.record("F3", f"Full {len(CANDIDATE_POOL)}-feature candidate pool instead of feature sets chosen with "
                        "the old validation data", pool, "fix adopted", reference="B",
                  comparison=compare(pool, baseline), kind="fix")

    xgb_spec = pool_spec.with_changes(name="xgboost", library="xgboost", params=XGB_EQUIVALENT[horizon],
                                      description="XGBoost with settings equivalent to the previous model")
    current = ladder.run(xgb_spec)
    ladder.record("E0", "XGBoost with settings equivalent to the previous model", current, "adopted (decision)",
                  reference="F3", comparison=compare(current, pool), kind="library")

    spec = current.spec
    current = ladder.try_change("E1", "Relative target: beat the universe median over the horizon", current,
                                [("", spec.with_changes(name="relative target", target="relative"))])
    spec = current.spec
    current = ladder.try_change("E2", "Clear-move labels: drop training rows whose forward move is within "
                                      "k × volatility × √h of the label boundary",
                                current, [(f"k={k}", spec.with_changes(name=f"clear-move k={k}", clear_move_k=k))
                                          for k in CLEAR_MOVE_K])
    spec = current.spec
    current = ladder.try_change("E3", "Cross-sectional ranks of stock-level features", current,
                                [("", spec.with_changes(name="cross-sectional ranks", cross_sectional=True))])
    spec = current.spec
    if horizon == "weekly":
        current = ladder.try_change(
            "E4", "Weekly features: 5-day relative strength vs NIFTY and sector, volume surprise", current,
            [("", spec.with_changes(name="short-term features", features=spec.features + SHORT_TERM_POOL))])
        spec = current.spec
    current = ladder.try_change("E5a", "More regularisation: shallower trees, larger leaves, subsampling", current,
                                [(label, spec.with_changes(name=f"regularised {label}", params=params))
                                 for label, params in REGULARISED_VARIANTS[horizon].items()])
    spec = current.spec
    current = ladder.try_change("E5b", "Early stopping on a purged hold-out at the end of each training window",
                                current, [(f"hold-out {metric}", spec.with_changes(name=f"early stopping ({metric})",
                                                                                  early_stopping=metric))
                                          for metric in EARLY_STOPPING_METRICS])
    spec = current.spec
    current = ladder.try_change("E5c", "Prune features without stable importance across training sub-periods",
                                current, [("", spec.with_changes(name="pruned", prune=True))])
    final_step = incumbent_step(ladder, current)
    logger.info("[%s] final configuration: step %s (%s)", horizon, final_step, current.spec.name)
    return ladder, final_step


# Reporting

def _ms(summary, key, digits=4):
    s = summary[key]
    return f"{s['mean']:.{digits}f} ± {s['std']:.{digits}f}"


def _fold_ics(result_row):
    return " / ".join(f"{m['rank_ic']:+.3f}" for m in result_row["folds"])


def _comparison_text(row):
    c = row["comparison"]
    if not c:
        return ""
    return f"{c['ic_delta']:+.4f} IC, {c['better_folds']}/5 folds, {c['auc_rel_delta']:+.4f} AUC"


def markdown_log(payload):
    meta, device = payload["data"], payload["device"]
    lines = [
        "# Experiment log",
        "",
        f"Generated {payload['generated_at'][:16].replace('T', ' ')} by `python experiments.py` on "
        f"{device.upper()}. Data: {meta.get('stocks')} NSE stocks, {meta.get('data_start')} to "
        f"{meta.get('data_end')} (frozen).",
        "",
        "Every number below comes from **purged walk-forward validation folds**. The final test period is not "
        "used anywhere in this log.",
        "",
        "* 5 time-ordered validation blocks, each trained on an expanding window that ends at least one label "
        "horizon (5 / 21 / 252 sessions) before the block, with every training label ending before the block starts.",
        "* **Rank IC**: Spearman correlation between predicted probability and realised forward return, across "
        "stocks on each day, averaged over days. **AUC rel / abs**: ROC AUC against \"beats the universe median\" "
        "and against \"rises more than 0.5%\". **Top / bottom**: share of the day's top and bottom prediction "
        "quintiles that beat the median. Values are mean ± standard deviation across the 5 folds.",
        f"* A change is **kept** only if mean rank IC improves, rank IC improves in at least {MIN_BETTER_FOLDS} of "
        "5 folds, and mean AUC rel does not fall. Fixes (F) and the XGBoost switch (E0) are adopted regardless.",
        "* F1 and F2 rows show the configuration *without* the fix (compare with B); F3 shows it *with* the fix. "
        "The \"vs reference\" column is always with-fix minus without-fix, or candidate minus the configuration "
        "it would replace.",
        "",
    ]
    for horizon in HORIZONS:
        if horizon not in payload["horizons"]:
            continue
        entry = payload["horizons"][horizon]
        lines += [f"## {horizon.capitalize()} ({HORIZONS[horizon]} sessions)", ""]
        lines += ["| Fold | Train | Validate | Gap (sessions) |", "| --- | --- | --- | --- |"]
        for s in entry["folds"]:
            lines.append(f"| {s['name']} | {s['train_period'][0]} → {s['train_period'][1]} | "
                         f"{s['eval_period'][0]} → {s['eval_period'][1]} | {s['gap_sessions']} |")
        lines += ["", "| Step | Change | Rank IC | IC by fold | AUC rel | AUC abs | Top / bottom | vs reference "
                      "| Decision |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for row in entry["steps"]:
            s = row["summary"]
            ref = f"vs {row['reference']}: {_comparison_text(row)}" if row["reference"] else ""
            lines.append(
                f"| {row['step']} | {row['change']} | {_ms(s, 'rank_ic')} | {_fold_ics(row)} | {_ms(s, 'auc_rel', 3)} "
                f"| {_ms(s, 'auc_abs', 3)} | {s['top_hit_rel']['mean']:.1%} / {s['bottom_hit_rel']['mean']:.1%} "
                f"| {ref} | {row['decision']} |")
        final = entry["final_spec"]
        lines += ["", f"**Final configuration: step {entry['final_step']}.** {final['library']}, target "
                      f"{final['target']}, {len(final['features'])} candidate features, clear-move "
                      f"k={final['clear_move_k']}, cross-sectional={final['cross_sectional']}, early "
                      f"stopping={final['early_stopping']}, pruning={final['prune']}, params {final['params']}.", ""]
    return "\n".join(lines)


def save(results, device, meta):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    payload = {"generated_at": datetime.now().isoformat(timespec="seconds"), "device": device, "data": meta,
               "rule": {"min_better_folds": MIN_BETTER_FOLDS, "metric": "rank_ic", "guard": "auc_rel"},
               "horizons": {}}
    for horizon, (ladder, final_step) in results.items():
        payload["horizons"][horizon] = {
            "folds": [s.describe() for s in ladder.folds], "final_step": final_step,
            "final_spec": ladder.runs[final_step].spec.to_dict(), "steps": ladder.rows}
        table = []
        for row in ladder.rows:
            entry = {"step": row["step"], "kind": row["kind"], "change": row["change"], "decision": row["decision"]}
            for key in FOLD_METRICS:
                entry[f"{key}_mean"] = row["summary"][key]["mean"]
                entry[f"{key}_std"] = row["summary"][key]["std"]
            entry["ic_by_fold"] = _fold_ics(row)
            entry["positive_ic_folds"] = row["summary"]["positive_ic_folds"]
            entry["vs_reference"] = _comparison_text(row)
            table.append(entry)
        pd.DataFrame(table).to_csv(os.path.join(REPORTS_DIR, f"experiments_{horizon}.csv"), index=False)
    existing = {}
    path = os.path.join(REPORTS_DIR, "experiments.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            existing = json.load(fh).get("horizons", {})
    payload["horizons"] = {**existing, **payload["horizons"]}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return payload


def main():
    parser = argparse.ArgumentParser(description="Run the TradeFlow experiment ladder on walk-forward folds")
    parser.add_argument("--horizons", nargs="*", choices=list(HORIZONS), default=list(HORIZONS))
    parser.add_argument("--cpu", action="store_true", help="train XGBoost on the CPU even if a GPU is available")
    args = parser.parse_args()

    os.makedirs(REPORTS_DIR, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True,
                        handlers=[logging.FileHandler(os.path.join(REPORTS_DIR, "experiments.log"), mode="w",
                                                      encoding="utf-8"), logging.StreamHandler()])
    device = detect_device("cpu" if args.cpu else "cuda")
    data, meta = build_dataset()
    panel = prepare_panel(data, rank_features=CANDIDATE_POOL + SHORT_TERM_POOL)
    legacy = None
    if os.path.exists(LEGACY_DATASET):
        legacy_data = pd.read_pickle(LEGACY_DATASET)
        legacy = prepare_panel(legacy_data[legacy_data["Stock"].isin(STOCKS + ["^NSEI"])].reset_index(drop=True))
    logger.info("Device: %s | panel %d rows", device, len(panel))

    results = {}
    for horizon in args.horizons:
        results[horizon] = run_ladder(horizon, panel, device, legacy)
    payload = save(results, device, meta)
    with open(os.path.join(REPORTS_DIR, "experiment_log.md"), "w", encoding="utf-8") as fh:
        fh.write(markdown_log(payload))
    logger.info("Saved reports/experiment_log.md, reports/experiments.json and reports/experiments_<horizon>.csv")


if __name__ == "__main__":
    main()
