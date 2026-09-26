"""Recompute src.lookalike.token_stats per split/country with streaming (low memory) and save raw counts."""
import pickle
import time
from collections import Counter, defaultdict

import pyarrow.parquet as pq

from lalib import CACHE, la

OUT = "tokstats.pkl"
res = {}
for split in ("train", "test"):
    t0 = time.time()
    df1, df23 = defaultdict(Counter), defaultdict(Counter)
    n1, n23 = Counter(), Counter()
    for which, cnt, n in (("s1", df1, n1), ("s23", df23, n23)):
        pf = pq.ParquetFile(fr"{CACHE}\{split}\records_{which}.parquet")
        for batch in pf.iter_batches(batch_size=500_000, columns=["name_n", "country"]):
            names = batch.column(0).to_pylist()
            ctry = batch.column(1).to_pylist()
            for name, c in zip(names, ctry):
                cnt[c].update(la.name_tokens(name))
                n[c] += 1
            del names, ctry
    res[split] = {}
    for c in n1:
        d1, d23 = df1[c], df23[c]
        keep = {t: (d1[t], d23[t]) for t in set(d1) | set(d23)
                if d1[t] >= la.VOCAB_MIN_DF1 or d23[t] >= la.FILLER_MIN_DF23}
        res[split][c] = {"n1": n1[c], "n23": n23[c], "tok": keep, "distinct1": len(d1), "distinct23": len(d23)}
        print(f"{split} {c}: n1={n1[c]:,} n23={n23[c]:,} base={n23[c] / n1[c]:.3f} "
              f"tokens with stats={len(keep):,} (distinct S1 {len(d1):,}, S2/S3 {len(d23):,})", flush=True)
    del df1, df23
    print(f"{split} done in {time.time() - t0:.0f}s", flush=True)
with open(OUT, "wb") as f:
    pickle.dump(res, f)
print("saved", OUT)
