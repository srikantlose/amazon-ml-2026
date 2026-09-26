"""Invert the recorded leaderboard deltas of the France threshold steps (same model lgb_stage3_0925_183300:
sub 07 t=0.75 -> sub 08 t=0.85 (+0.000438), sub 08 -> sub 10 t=0.95 (+0.000434)) into the share of false
positives among the France pairs dropped at each step, using the real per-S1 structure of those pairs.
Assumes the S1's other accepted pairs are correct and its true matches were all found."""
from math import comb

import numpy as np
import runpy

g = runpy.run_path("sub10_probs.py", run_name="not_main")   # re-derives s1, rec, best, pb, mg, s1_country
s1, best, pb, mg, s1_country = g["s1"], g["best"], g["pb"], g["mg"], g["s1_country"]
N = len(s1_country)


def f05(tp, pred, true):
    if pred == 0 and true == 0:
        return 1.0
    if tp == 0:
        return 0.0
    p, r = tp / pred, tp / true
    return 1.25 * p * r / (0.25 * p + r)


fr_ok = (s1_country[s1[best]] == "France") & (mg >= 0.5)
for (lo, hi, dlb) in ((0.75, 0.85, 0.000438), (0.85, 0.95, 0.000434)):
    drop = np.bincount(s1[best[fr_ok & (pb >= lo) & (pb < hi)]], minlength=N)
    keep = np.bincount(s1[best[fr_ok & (pb >= hi)]], minlength=N)
    ss = np.flatnonzero(drop)
    def gain(q):
        tot = 0.0
        for s in ss:
            m, k = int(drop[s]), int(keep[s])
            for j in range(m + 1):
                w = comb(m, j) * (1 - q) ** j * q ** (m - j)
                tot += w * (f05(k, k, k + j) - f05(k + j, k + m, k + j))
        return tot / N
    qs = np.linspace(0, 1, 101)
    gs = np.array([gain(q) for q in qs])
    q_hat = qs[np.argmin(np.abs(gs - dlb))]
    print(f"France pairs in [{lo},{hi}): {drop.sum():,} on {len(ss):,} S1; recorded delta {dlb:+.6f} -> "
          f"implied FP share {q_hat:.2f} (precision {1 - q_hat:.2f}); delta if all FP {gs[-1]:+.6f}, if all TP {gs[0]:+.6f}")
