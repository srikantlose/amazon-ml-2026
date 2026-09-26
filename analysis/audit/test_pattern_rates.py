"""Look-alike pattern density among confident candidates: train (features_la) vs test (recomputed with the
real pair_row and test-split statistics), per 100 S1 of the country."""
import glob
import math
import pickle

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lalib import CACHE, la

IDX = {k: i for i, k in enumerate(la.COLUMNS)}

# train
c = pq.read_table(fr"{CACHE}\train\cands_pruned.parquet", columns=["s1", "p1"])
s1, p1 = c.column("s1").to_numpy(), c.column("p1").to_numpy()
ctry1 = np.array(pq.read_table(fr"{CACHE}\train\records_s1.parquet", columns=["country"]).column(0).to_pylist(),
                 dtype=object)
parts = sorted(glob.glob(fr"{CACHE}\train\features_la\part_*.parquet"))
F = {k: np.concatenate([pq.read_table(p, columns=[k]).column(0).to_numpy() for p in parts])
     for k in ("la_swap_vocab", "la_swap_same_initial", "la_extra_vocab_len_min")}
pat = (F["la_swap_vocab"] == 1) & (F["la_swap_same_initial"] == 0) & (F["la_extra_vocab_len_min"] >= 4)
swp = F["la_swap_vocab"] == 1
pc_ = ctry1[s1]
for cc in ("US", "India"):
    n = (ctry1 == cc).sum()
    for thr in (0.5, 0.9):
        m = (pc_ == cc) & (p1 >= thr)
        print(f"train {cc:6} p1>={thr}: pairs/S1 {m.sum() / n:.3f}; pattern per 100 S1 {(m & pat).sum() / n * 100:.2f}; "
              f"any swap per 100 S1 {(m & swp).sum() / n * 100:.2f}")
print(f"train pairs per S1 (all): {len(s1) / len(ctry1):.3f}")
del c, s1, p1, F, pat, swp, pc_

# test
res = pickle.load(open("tokstats.pkl", "rb"))["test"]
la._init({cc: {t: (math.log((b + 1) / (a + 1)) - math.log(d["n23"] / d["n1"]), math.log1p(a), b)
               for t, (a, b) in d["tok"].items()} for cc, d in res.items()})
c = pq.read_table(fr"{CACHE}\test\cands_pruned.parquet", columns=["s1", "rec", "p1"])
s1, rec, p1 = (c.column(k).to_numpy() for k in ("s1", "rec", "p1"))
del c
t1 = pq.read_table(fr"{CACHE}\test\records_s1.parquet", columns=["country", "name_n"])
ctry = np.array(t1.column("country").to_pylist(), dtype=object)
n1 = t1.column("name_n").combine_chunks()
n23 = pq.read_table(fr"{CACHE}\test\records_s23.parquet", columns=["name_n"]).column(0).combine_chunks()
print(f"test pairs per S1 (all): {len(s1) / len(ctry):.3f}")
sel = np.flatnonzero(p1 >= 0.5)
pat = np.zeros(len(sel), bool)
swp = np.zeros(len(sel), bool)
for s in range(0, len(sel), 400_000):
    rows = sel[s:s + 400_000]
    an = n1.take(pa.array(s1[rows])).to_pylist()
    bn = n23.take(pa.array(rec[rows])).to_pylist()
    cs = ctry[s1[rows]]
    for k, (x, z, cc) in enumerate(zip(an, bn, cs)):
        r = la.pair_row(x, z, cc)
        swp[s + k] = r[IDX["la_swap_vocab"]] == 1
        pat[s + k] = swp[s + k] and r[IDX["la_swap_same_initial"]] == 0 and r[IDX["la_extra_vocab_len_min"]] >= 4
pc_ = ctry[s1[sel]]
psel = p1[sel]
for cc in ("France", "India", "US"):
    n = (ctry == cc).sum()
    for thr in (0.5, 0.9):
        m = (pc_ == cc) & (psel >= thr)
        print(f"test  {cc:6} p1>={thr}: pairs/S1 {m.sum() / n:.3f}; pattern per 100 S1 {(m & pat).sum() / n * 100:.2f}; "
              f"any swap per 100 S1 {(m & swp).sum() / n * 100:.2f}")
