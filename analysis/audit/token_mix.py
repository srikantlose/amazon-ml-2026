"""Per name: how many tokens are filler / vocab / rare (by the split's own statistics), S1 vs S2/S3."""
import math
import pickle
from collections import defaultdict

import pyarrow.parquet as pq

from lalib import CACHE, la

res = pickle.load(open("tokstats.pkl", "rb"))
LF = la.FILLER_LRATIO
for split in ("train", "test"):
    classes = {}
    for c, d in res[split].items():
        base = math.log(d["n23"] / d["n1"])
        m = {}
        for t, (a, b) in d["tok"].items():
            lr = math.log((b + 1) / (a + 1)) - base
            m[t] = "f" if (lr >= LF and b >= la.FILLER_MIN_DF23) else ("v" if a >= la.VOCAB_MIN_DF1 else "r")
        classes[c] = m
    for which in ("s1", "s23"):
        acc = defaultdict(lambda: [0, 0, 0, 0, 0])  # names, filler, vocab, rare, digit tokens
        pf = pq.ParquetFile(fr"{CACHE}\{split}\records_{which}.parquet")
        for batch in pf.iter_batches(batch_size=500_000, columns=["name_n", "country", "src"]):
            names, ctry, src = (batch.column(i).to_pylist() for i in range(3))
            for name, c, s in zip(names, ctry, src):
                m = classes[c]
                a = acc[(c, s)]
                a[0] += 1
                for t in la.name_tokens(name):
                    k = m.get(t, "r")
                    if t.isdigit():
                        a[4] += 1
                    a[1 if k == "f" else 2 if k == "v" else 3] += 1
        for (c, s), (n, f, v, r, dg) in sorted(acc.items()):
            print(f"{split:5} S{s} {c:6} per name: filler {f / n:.3f}  vocab {v / n:.3f}  rare {r / n:.3f}  "
                  f"(digit tokens {dg / n:.3f})  total {(f + v + r) / n:.3f}", flush=True)
