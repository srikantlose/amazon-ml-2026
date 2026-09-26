"""IDF + SVD are refit per split (the test sample includes France): at a fixed exact string similarity, do the
char-3-gram SVD cosines (cos_name / cos_addr, used by stage 1 and stage 2) mean the same in train and test?"""
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz, process

from lalib import CACHE


def gather(path, cols, idx):
    """Rows idx (any order) of the given columns, streaming row groups to keep memory low."""
    pf = pq.ParquetFile(path)
    order = np.argsort(idx)
    sidx = idx[order]
    out = {c: np.empty(len(idx), dtype=object) for c in cols}
    start = 0
    for g in range(pf.num_row_groups):
        n = pf.metadata.row_group(g).num_rows
        lo, hi = np.searchsorted(sidx, [start, start + n])
        if hi > lo:
            t = pf.read_row_group(g, columns=cols)
            for c in cols:
                out[c][order[lo:hi]] = t.column(c).take(pa.array(sidx[lo:hi] - start)).to_pylist()
            del t
        start += n
    return out


rng = np.random.default_rng(0)
res = {}
for split in ("train", "test"):
    c = pq.read_table(fr"{CACHE}\{split}\cands_pruned.parquet", columns=["s1", "rec", "cos_name", "cos_addr"])
    idx = np.sort(rng.choice(c.num_rows, 600_000, replace=False))
    s1 = c.column("s1").to_numpy()[idx]
    rec = c.column("rec").to_numpy()[idx]
    cn = c.column("cos_name").to_numpy()[idx]
    ca = c.column("cos_addr").to_numpy()[idx]
    del c
    A = gather(fr"{CACHE}\{split}\records_s1.parquet", ["country", "name_core", "addr_n"], s1)
    B = gather(fr"{CACHE}\{split}\records_s23.parquet", ["name_core", "addr_n"], rec)
    rn = process.cpdist(A["name_core"], B["name_core"], scorer=fuzz.ratio, workers=4)
    ra = process.cpdist(A["addr_n"], B["addr_n"], scorer=fuzz.token_set_ratio, workers=4)
    res[split] = (A["country"], rn, ra, cn, ca)
    print(split, "sampled", len(idx), flush=True)

bins = [0, 40, 60, 70, 80, 90, 95, 100, 101]
for label, ri, ci in (("name: fuzz.ratio(name_core) bin -> mean cos_name train/test", 1, 3),
                      ("addr: token_set_ratio(addr_n) bin -> mean cos_addr train/test", 2, 4)):
    print(label)
    for cc in ("US", "India", "France"):
        row = f"  {cc:6}"
        for lo, hi in zip(bins[:-1], bins[1:]):
            cells = []
            for split in ("train", "test"):
                ctry, rn, ra, cn, ca = res[split]
                r = (rn, ra)[ri - 1]
                v = (cn, ca)[ci - 3]
                m = (ctry == cc) & (r >= lo) & (r < hi)
                cells.append(f"{v[m].mean():.3f}" if m.sum() >= 200 else "  -  ")
            row += f" [{lo},{hi}) {cells[0]}/{cells[1]}"
        print(row)
