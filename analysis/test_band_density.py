"""Best-per-record pairs per 1000 S1 in probability bands: out of fold (with precision) vs test.

    python analysis/test_band_density.py lgb_stage3la_0926_033632 0.6

Result on 26 Sep (look-alike model, margin >= 0.6): test has 1.4-2.6x the out-of-fold density in the
0.6-0.99 bands for both US and India while the >= 0.99 band is ~1.0x. A stricter US/India threshold (0.80,
submission 18) nevertheless lost 0.00029 on the leaderboard, so much of that extra mass is true matches.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.blocking import pruned_path  # noqa: E402
from src.data import build_labels  # noqa: E402
from src.normalize import records_path  # noqa: E402
from src.utils import load_config  # noqa: E402

cfg = load_config([str(ROOT / "configs" / "base.yaml")])
run, min_margin = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.6
bands = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 1.01]


def profile(split, p, with_truth):
    c = pd.read_parquet(pruned_path(cfg, split), columns=["s1", "rec"])
    s1, rec = c["s1"].to_numpy(np.int64), c["rec"].to_numpy(np.int64)
    s1r = pd.read_parquet(records_path(cfg, split, "s1"), columns=["entity_id", "country"])
    ct = s1r["country"].to_numpy()
    order = np.lexsort((-p, rec))
    r_s, p_s = rec[order], p[order]
    first = np.r_[True, r_s[1:] != r_s[:-1]]
    idx = order[first]
    starts = np.flatnonzero(first)
    nxt = starts + 1
    has_second = (nxt < len(r_s)) & ~np.r_[first[1:], True][starts]
    second = np.zeros(len(idx))
    second[has_second] = p_s[nxt[has_second]]
    margin = p[idx] - second
    y = None
    if with_truth:
        rid = pd.read_parquet(records_path(cfg, split, "s23"), columns=["entity_id"])["entity_id"].to_numpy()
        true_s1, _ = build_labels(cfg, s1r["entity_id"].to_numpy(), rid)
        y = true_s1[rec[idx]] == s1[idx]
    out = {}
    for country in ("US", "India"):
        n_s1 = (ct == country).sum()
        sel = (ct[s1[idx]] == country) & (margin >= min_margin)
        rows = []
        for lo, hi in zip(bands[:-1], bands[1:]):
            m = sel & (p[idx] >= lo) & (p[idx] < hi)
            rows.append((lo, hi, 1000 * m.sum() / n_s1, y[m].mean() if with_truth and m.any() else None))
        out[country] = rows
    return out


d = ROOT / "data" / "cache" / "runs" / run
oof = profile("train", np.load(d / "oof.npy"), True)
test = profile("test", np.load(d / "test_prob.npy"), False)
for country in ("US", "India"):
    print(f"\n[{country}] band | OOF per 1000 S1 (precision) | test per 1000 S1 | test/OOF")
    for (lo, hi, a, prec), (_, _, b, _) in zip(oof[country], test[country]):
        ptxt = f"({prec:.2f})" if prec is not None else ""
        print(f"  [{lo:.2f}, {hi:.2f}): {a:8.1f} {ptxt:<7} {b:8.1f}   {b / a if a else float('nan'):.2f}")
