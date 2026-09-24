"""Turn pair probabilities into per-S1 match lists.

Every S2/S3 record belongs to at most one S1 entity (true for all 7.6M training pairs), so each
record keeps only its highest-probability S1, and only when
    p_best >= threshold  and  p_best - p_second >= margin.
Threshold and margin are grid-searched on out-of-fold predictions to maximize the exact
macro F0.5 over all training S1 entities (singletons and blocking misses included).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.metrics import f05_from_pairs


def best_per_record(rec: np.ndarray, prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Positions of each record's best pair, and the runner-up probability (0 if none)."""
    order = np.lexsort((-prob, rec))
    r = rec[order]
    start = np.flatnonzero(np.r_[True, r[1:] != r[:-1]])
    best = order[start]
    nxt = np.minimum(start + 1, len(order) - 1)
    has_second = (start + 1 < len(order)) & (r[nxt] == r[start])
    second = np.where(has_second, prob[order[nxt]], 0.0)
    return best, second


def select(rec: np.ndarray, prob: np.ndarray, threshold: float, margin: float,
           best=None, second=None) -> np.ndarray:
    """Boolean mask over pairs that become matches."""
    if best is None:
        best, second = best_per_record(rec, prob)
    ok = (prob[best] >= threshold) & (prob[best] - second >= margin)
    keep = np.zeros(len(prob), dtype=bool)
    keep[best[ok]] = True
    return keep


def tune(s1: np.ndarray, rec: np.ndarray, prob: np.ndarray, true_s1: np.ndarray, n_true: np.ndarray,
         thresholds, margins, groups: dict | None = None) -> tuple[dict, pd.DataFrame]:
    """Grid search; returns the best {threshold, margin, score} and the full table.

    groups: optional {name: boolean mask over S1 rows} for per-group scores (e.g. country).
    """
    n_s1 = len(n_true)
    best, second = best_per_record(rec, prob)
    rows = []
    for t in thresholds:
        for m in margins:
            keep = select(rec, prob, t, m, best, second)
            f = f05_from_pairs(n_s1, true_s1, n_true, s1[keep], rec[keep])
            row = {"threshold": t, "margin": m, "f05": f.mean(), "matches": int(keep.sum())}
            for name, mask in (groups or {}).items():
                row[f"f05_{name}"] = f[mask].mean()
            rows.append(row)
    table = pd.DataFrame(rows).sort_values("f05", ascending=False).reset_index(drop=True)
    top = table.iloc[0]
    return {"threshold": float(top.threshold), "margin": float(top.margin), "f05": float(top.f05)}, table


def match_lists(s1_ids: np.ndarray, rec_ids: np.ndarray, s1: np.ndarray, rec: np.ndarray,
                mask: np.ndarray | None = None) -> dict:
    """{s1_id: [record ids]} for the selected pairs."""
    if mask is not None:
        s1, rec = s1[mask], rec[mask]
    df = pd.DataFrame({"s1": s1_ids[s1], "rec": rec_ids[rec]})
    return df.groupby("s1", sort=False)["rec"].agg(list).to_dict()
