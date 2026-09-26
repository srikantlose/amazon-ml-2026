"""How distinctive is the first name_core token (features.first / house key), and the scale of name_core counts."""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from lalib import CACHE

ARTICLES = {"de", "du", "des", "la", "le", "les", "l", "d", "au", "aux", "et", "chez", "en"}
for split in ("train", "test"):
    s1 = pq.read_table(fr"{CACHE}\{split}\records_s1.parquet", columns=["country", "name_core"]).to_pandas()
    s1["first"] = s1["name_core"].str.split(n=1).str[0].fillna("")
    for c, g in s1.groupby("country"):
        n = len(g)
        fcount = g.groupby("first")["first"].transform("size").to_numpy()
        ncount = g.groupby("name_core")["name_core"].transform("size").to_numpy()
        top = g["first"].value_counts().head(8)
        print(f"{split:5} {c:6} S1={n:>9,} | first token shared by >=0.1% of S1: {(fcount >= 0.001 * n).mean():.3f}; "
              f"median S1 sharing first token {np.median(fcount):.0f} ({np.median(fcount) / n * 1e5:.1f} per 100k S1); "
              f"first token is an article {g['first'].isin(ARTICLES).mean():.3f}")
        print(f"      name_core shared (count>=2): {(ncount >= 2).mean():.3f}; mean count {ncount.mean():.2f}; "
              f"p90 {np.quantile(ncount, 0.9):.0f}; p99 {np.quantile(ncount, 0.99):.0f}; max {ncount.max()} | "
              f"top first tokens: {dict(top)}")
    del s1
