"""Counterfactual: score OOF-valid train pairs with the stage-2 fold models after shifting scale-dependent
features the way the test set shifts them; report direction of the change in accepted pairs."""
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
X = pq.ParquetFile(fr"{CACHE}\train\features\part_000.parquet").read_row_group(0, columns=feats).slice(0, N) \
    .to_pandas()[feats]
c = pq.read_table(fr"{CACHE}\train\cands_pruned.parquet", columns=["s1", "rec"]).slice(0, N)
s1, rec = c.column("s1").to_numpy(), c.column("rec").to_numpy()
y = np.load("train_y.npy")[:N]
ctry = np.array(pq.read_table(fr"{CACHE}\train\records_s1.parquet", columns=["country"]).column(0).to_pylist(),
                dtype=object)[s1]
ids = pq.read_table(fr"{CACHE}\train\records_s1.parquet", columns=["entity_id"]).column(0)
us1 = np.unique(s1)
fmap = dict(zip(us1, [zlib.crc32(ids[int(i)].as_py().encode()) % 3 for i in us1]))
fold = np.array([fmap[i] for i in s1])
models = [lgb.Booster(model_file=fr"{RUN}\model_fold{f}.txt") for f in range(3)]
last = rec[-1]
valid = rec != last


def score(Xs):
    p = np.zeros(N, np.float32)
    for f, b in enumerate(models):
        m = fold == f
        p[m] = b.predict(Xs[m], num_threads=4)
    return p


def decide(p, t=0.6, mg=0.5):
    o = np.lexsort((-p, rec))
    r = rec[o]
    st = np.flatnonzero(np.r_[True, r[1:] != r[:-1]])
    bi = o[st]
    nx = np.minimum(st + 1, len(o) - 1)
    sec = np.where((st + 1 < len(o)) & (r[nx] == r[st]), p[o[nx]], 0.0)
    acc = np.zeros(N, bool)
    ok = (p[bi] >= t) & (p[bi] - sec >= mg) & valid[bi]
    acc[bi[ok]] = True
    return acc


base = score(X)
acc0 = decide(base)
print(f"sample pairs {N:,}; base accepted {acc0.sum():,}, precision {y[acc0].mean():.5f}, "
      f"TP {int((y & acc0).sum()):,}")
scen = {
    "s1_n_cands x1.22 (test cands/S1)": {"s1_n_cands": 1.22},
    "name counts x0.5 (test US S1 table)": {"s1_name_freq": 0.5, "rec_name_s1_freq": 0.5},
    "s1_sum_p1 x1.05 + n_sib +0.3": {"s1_sum_p1": 1.05, "n_sib": "+0.3"},
}
for name, ch in scen.items():
    Xs = X.copy()
    for col, v in ch.items():
        if isinstance(v, str):
            Xs[col] = Xs[col] + float(v)
        elif col == "s1_n_cands":
            Xs[col] = np.round(Xs[col] * v)
        else:
            Xs[col] = np.maximum(np.round(Xs[col] * v), np.where(Xs[col] > 0, 1, 0))
    p = score(Xs)
    acc = decide(p)
    for cc in ("US", "India"):
        m = ctry == cc
        print(f"{name:38} {cc:5}: mean dp {np.mean(p[m] - base[m]):+.5f}; accepted {int(acc[m].sum()) - int(acc0[m].sum()):+,} "
              f"(new FP {int((acc & ~acc0 & ~y & m).sum()):,}, lost TP {int((acc0 & ~acc & y & m).sum()):,}, "
              f"new TP {int((acc & ~acc0 & y & m).sum()):,}, removed FP {int((acc0 & ~acc & ~y & m).sum()):,})", flush=True)
