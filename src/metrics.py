"""Competition metric (macro F0.5 per Source 1 entity) and blocking diagnostics.

Per S1 entity with true match set T and predicted set P:
    T empty, P empty      -> 1.0
    T empty, P non-empty  -> 0.0
    T non-empty, P empty  -> 0.0
    otherwise             -> 1.25 * p * r / (0.25 * p + r), with p = |T&P|/|P|, r = |T&P|/|T|
The leaderboard score is the mean over all S1 entities.

Run `python -m src.metrics` for the self-checks.
"""
from __future__ import annotations

import numpy as np


def f05_scores(tp, n_pred, n_true) -> np.ndarray:
    """Vectorized per-entity F0.5 from counts."""
    tp = np.asarray(tp, dtype=np.float64)
    n_pred = np.asarray(n_pred, dtype=np.float64)
    n_true = np.asarray(n_true, dtype=np.float64)
    f = np.zeros(len(tp), dtype=np.float64)
    f[(n_pred == 0) & (n_true == 0)] = 1.0
    ok = tp > 0
    p = tp[ok] / n_pred[ok]
    r = tp[ok] / n_true[ok]
    f[ok] = 1.25 * p * r / (0.25 * p + r)
    return f


def f05_macro(gt: dict, pred: dict, s1_ids) -> float:
    """Reference implementation on {s1_id: iterable of ids} dicts (slow; for checks)."""
    tp, n_pred, n_true = [], [], []
    for s1 in s1_ids:
        t, p = set(gt.get(s1, ())), set(pred.get(s1, ()))
        tp.append(len(t & p))
        n_pred.append(len(p))
        n_true.append(len(t))
    return float(f05_scores(tp, n_pred, n_true).mean())


def f05_from_pairs(n_s1: int, true_s1_of_rec: np.ndarray, n_true: np.ndarray,
                   pred_s1: np.ndarray, pred_rec: np.ndarray) -> np.ndarray:
    """Per-S1 F0.5 for predicted (s1 index, record index) pairs.

    true_s1_of_rec[r] is the S1 index record r belongs to (-1 if none); n_true[s] is the
    number of true matches of S1 s (including ones blocking never retrieved).
    """
    pred_s1 = np.asarray(pred_s1, dtype=np.int64)
    pred_rec = np.asarray(pred_rec, dtype=np.int64)
    hit = true_s1_of_rec[pred_rec] == pred_s1
    tp = np.bincount(pred_s1[hit], minlength=n_s1)
    n_pred = np.bincount(pred_s1, minlength=n_s1)
    return f05_scores(tp, n_pred, n_true)


def blocking_report(n_s1: int, true_s1_of_rec: np.ndarray, n_true: np.ndarray,
                    cand_s1: np.ndarray, cand_rec: np.ndarray) -> dict:
    """Pair recall, oracle F0.5 ceiling (perfect matcher on these candidates), sizes."""
    cand_s1 = np.asarray(cand_s1, dtype=np.int64)
    cand_rec = np.asarray(cand_rec, dtype=np.int64)
    hit = true_s1_of_rec[cand_rec] == cand_s1
    n_gt_pairs = int(n_true.sum())
    per_s1 = np.bincount(cand_s1, minlength=n_s1)
    oracle = f05_from_pairs(n_s1, true_s1_of_rec, n_true, cand_s1[hit], cand_rec[hit])
    return {
        "pairs": int(len(cand_s1)),
        "pair_recall": float(hit.sum() / max(n_gt_pairs, 1)),
        "oracle_f05": float(oracle.mean()),
        "cands_per_s1_mean": float(per_s1.mean()),
        "cands_per_s1_p99": float(np.percentile(per_s1, 99)),
        "s1_without_cands": float((per_s1 == 0).mean()),
    }


if __name__ == "__main__":
    # example from the problem statement
    gt = {"S1-00001": ["S2-00047", "S3-00812"]}
    pred = {"S1-00001": ["S2-00047", "S2-00193", "S3-00812"]}
    assert abs(f05_macro(gt, pred, ["S1-00001"]) - 0.7143) < 1e-3
    # singletons: empty/empty = 1, empty truth + any prediction = 0; missed entity = 0
    assert f05_macro({}, {}, ["a"]) == 1.0
    assert f05_macro({}, {"a": ["S2-1"]}, ["a"]) == 0.0
    assert f05_macro({"a": ["S2-1"]}, {}, ["a"]) == 0.0
    assert f05_macro({"a": ["S2-1"]}, {"a": ["S2-9"]}, ["a"]) == 0.0
    assert f05_macro({"a": ["S2-1"]}, {"a": ["S2-1"]}, ["a"]) == 1.0
    # pair-based version agrees with the dict version
    true_s1 = np.array([0, 0, -1, 1])           # records 0,1 -> S1 0; record 3 -> S1 1; record 2 unmatched
    n_true = np.bincount(true_s1[true_s1 >= 0], minlength=3)  # S1 2 is a singleton
    f = f05_from_pairs(3, true_s1, n_true, np.array([0, 0, 1]), np.array([0, 2, 3]))
    ref = f05_macro({0: [0, 1], 1: [3]}, {0: [0, 2], 1: [3]}, [0, 1, 2])
    assert abs(f.mean() - ref) < 1e-12, (f, ref)
    print("metrics self-checks passed")
