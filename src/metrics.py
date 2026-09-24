"""Competition metrics.

Day 1: paste the official metric definition into the project notes and check the
matching function here against it (run `python -m src.metrics` for the self-checks).
"""
from __future__ import annotations

import numpy as np


def smape(y_true, y_pred) -> float:
    """2025 metric: mean(|p - a| / ((|a| + |p|) / 2)) * 100. Lower is better. 0/0 counts as 0."""
    a = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred, dtype=np.float64)
    denom = (np.abs(a) + np.abs(p)) / 2.0
    diff = np.abs(p - a)
    ratio = np.divide(diff, denom, out=np.zeros_like(diff), where=denom != 0)
    return float(np.mean(ratio) * 100.0)


def mape_score_2023(y_true, y_pred, eps: float = np.finfo(np.float64).eps) -> float:
    """2023 metric: max(0, 100 * (1 - MAPE)). Higher is better. Zero targets use eps like sklearn."""
    a = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred, dtype=np.float64)
    mape = np.mean(np.abs(p - a) / np.maximum(np.abs(a), eps))
    return float(max(0.0, 100.0 * (1.0 - mape)))


def _norm_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x).strip()


def f1_extraction(gt, pred) -> float:
    """2024 entity-extraction F1 over string outputs.

    TP: pred non-empty and pred == gt
    FP: pred non-empty and (gt empty or pred != gt)
    FN: pred empty and gt non-empty
    TN: both empty
    """
    tp = fp = fn = 0
    for g, p in zip(gt, pred):
        g, p = _norm_str(g), _norm_str(p)
        if p:
            if g and p == g:
                tp += 1
            else:
                fp += 1
        elif g:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def macro_f1(y_true, y_pred) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, y_pred, average="macro"))


def rmse(y_true, y_pred) -> float:
    a = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def mae(y_true, y_pred) -> float:
    a = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(a - p)))


# name -> (fn, greater_is_better)
METRICS = {
    "smape": (smape, False),
    "mape_score_2023": (mape_score_2023, True),
    "f1_extraction": (f1_extraction, True),
    "accuracy": (accuracy, True),
    "macro_f1": (macro_f1, True),
    "rmse": (rmse, False),
    "mae": (mae, False),
}


def get_metric(name: str):
    if name not in METRICS:
        raise KeyError(f"unknown metric {name!r}; options: {sorted(METRICS)}")
    return METRICS[name]


if __name__ == "__main__":
    assert smape([100], [100]) == 0.0
    assert abs(smape([100], [50]) - 66.6667) < 1e-3
    assert smape([0], [0]) == 0.0
    assert smape([0], [5]) == 200.0
    assert mape_score_2023([10, 20], [10, 20]) == 100.0
    assert mape_score_2023([10], [30]) == 0.0
    assert abs(mape_score_2023([10], [12]) - 80.0) < 1e-9
    # 1 TP, 1 FP (wrong), 1 FP (gt empty), 1 FN, 1 TN -> P=1/3, R=1/2, F1=0.4
    gt = ["34 gram", "5 volt", "", "10 watt", ""]
    pr = ["34 gram", "6 volt", "1 gram", "", None]
    assert abs(f1_extraction(gt, pr) - 0.4) < 1e-9
    assert accuracy([1, 2, 3], [1, 2, 0]) == 2 / 3
    print("metrics self-checks passed")
