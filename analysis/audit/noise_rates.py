"""Label-free noise indicators per split / country / source (S1, S2, S3)."""
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from lalib import CACHE

cols = ["src", "country", "is_domain", "name_non_latin", "addr_empty", "name_n", "house", "postal"]
for split in ("train", "test"):
    for which in ("s1", "s23"):
        t = pq.read_table(fr"{CACHE}\{split}\records_{which}.parquet", columns=cols)
        src = t.column("src").to_numpy()
        ctry = np.array(t.column("country").to_pylist(), dtype=object)
        dom = t.column("is_domain").to_numpy(zero_copy_only=False)
        nl = t.column("name_non_latin").to_numpy(zero_copy_only=False)
        ae = t.column("addr_empty").to_numpy(zero_copy_only=False)
        ntok = pc.list_value_length(pc.utf8_split_whitespace(t.column("name_n"))).to_numpy(zero_copy_only=False)
        nlen = pc.utf8_length(t.column("name_n")).to_numpy(zero_copy_only=False)
        house = (pc.utf8_length(t.column("house")).to_numpy(zero_copy_only=False) > 0)
        del t
        for c in sorted(set(ctry)):
            for s in sorted(set(src)):
                m = (ctry == c) & (src == s)
                if not m.any():
                    continue
                print(f"{split:5} S{s} {c:6} n={m.sum():>9,}  addr_empty={ae[m].mean():.4f}  domain={dom[m].mean():.4f}  "
                      f"non_latin={nl[m].mean():.4f}  name_tokens={ntok[m].mean():.3f}  name_len={nlen[m].mean():.2f}  "
                      f"has_house={house[m].mean():.3f}", flush=True)
        del src, ctry, dom, nl, ae, ntok, nlen, house
