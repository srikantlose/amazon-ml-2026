"""Stage-3 is trained on single-model OOF p2 but fed a 3-model average on test.

On train pairs that no stage-2 fold model trained or early-stopped on (S1 hash u >= train_frac + valid_frac),
compare the OOF-style p2 (the pair's own fold model) with the test-style p2 (mean of the three models),
and the record-level margin that stage 3 relies on most.
"""
import json
import zlib

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq

from lalib import CACHE

RUN = fr"{CACHE}\runs\lgb_stage2_0925_175903"
meta = json.load(open(fr"{RUN}\meta.json"))
feats = meta["features"]
N = 1_000_000
pf = pq.ParquetFile(fr"{CACHE}\train\features\part_000.parquet")
X = pf.read_row_group(0, columns=feats).slice(0, N).to_pandas()[feats]
c = pq.read_table(fr"{CACHE}\train\cands_pruned.parquet", columns=["s1", "rec"]).slice(0, N)
s1, rec = c.column("s1").to_numpy(), c.column("rec").to_numpy()
y = np.load("train_y.npy")[:N]
ids = pq.read_table(fr"{CACHE}\train\records_s1.parquet", columns=["entity_id"]).column(0)
us1 = np.unique(s1)
h = np.array([zlib.crc32(ids[int(i)].as_py().encode()) for i in us1], dtype=np.int64)
fold_s1 = dict(zip(us1, h % 3))
u_s1 = dict(zip(us1, (h // 3) % 10_000 / 10_000.0))
fold = np.array([fold_s1[i] for i in s1])
u = np.array([u_s1[i] for i in s1])
mcfg = meta["params"]
clean_pair = u >= mcfg["train_frac"] + mcfg["valid_frac"]
print(f"rows {N:,}; clean pairs {clean_pair.mean():.3f}; train_frac {mcfg['train_frac']} valid {mcfg['valid_frac']}")

P = np.zeros((3, N), np.float32)
for f in range(3):
    b = lgb.Booster(model_file=fr"{RUN}\model_fold{f}.txt")
    P[f] = b.predict(X, num_threads=4)
    del b
del X
oof = P[fold, np.arange(N)]
avg = P.mean(0)

# records whose candidates are all clean, and complete inside the slice (drop the last record)
last = rec[-1]
rec_clean = np.ones(rec.max() + 1, bool)
np.logical_and.at(rec_clean, rec, clean_pair)
ok_rec = rec_clean[rec] & (rec != last)
print(f"pairs in all-clean records {ok_rec.sum():,}")


def per_record(p, m):
    r, pp, yy = rec[m], p[m], y[m]
    order = np.lexsort((-pp, r))
    r, pp, yy = r[order], pp[order], yy[order]
    start = np.flatnonzero(np.r_[True, r[1:] != r[:-1]])
    best = pp[start]
    nxt = np.minimum(start + 1, len(r) - 1)
    has2 = (start + 1 < len(r)) & (r[nxt] == r[start])
    second = np.where(has2, pp[nxt], 0.0)
    return best, best - second, yy[start]


bo, mo, yo = per_record(oof, ok_rec)
ba, ma, ya = per_record(avg, ok_rec)
d = np.abs(P - avg)[:, ok_rec]
print(f"pair level |single - mean of 3|: mean {d.mean():.4f}; share > 0.05: {(d > 0.05).mean():.4f}; "
      f"> 0.2: {(d > 0.2).mean():.4f}")
for name, (b, m, yy) in (("OOF-style (train)", (bo, mo, yo)), ("3-model mean (test)", (ba, ma, ya))):
    acc = (b >= 0.65) & (m >= 0.5)
    print(f"{name:20} records {len(b):,}: best p in [0.2,0.8] {((b > 0.2) & (b < 0.8)).mean():.4f}; "
          f"margin in [0.3,0.7] {((m > 0.3) & (m < 0.7)).mean():.4f}; p>=0.99 {(b >= 0.99).mean():.4f}; "
          f"accepted at t=.65,m=.5 {acc.mean():.4f} (precision {yy[acc].mean():.4f}, "
          f"recall of true-best {acc[yy].mean():.4f})")
    for lo, hi in ((0.5, 0.7), (0.7, 0.9), (0.9, 0.97), (0.97, 1.01)):
        s = (b >= lo) & (b < hi)
        print(f"     best p in [{lo:.2f},{hi:.2f}): share {s.mean():.4f}, positive rate {yy[s].mean():.4f}")
