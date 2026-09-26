"""Expected leaderboard change from dropping the swap-pattern pairs of submission 10, per country,
for a range of assumed false-positive rates q of those pairs (train, p1 >= 0.9: US 0.971, India 0.990)."""
import math
import pickle
from math import comb

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from lalib import CACHE, ROOT, la

res = pickle.load(open("tokstats.pkl", "rb"))["test"]
stats = {c: {t: (math.log((b + 1) / (a + 1)) - math.log(d["n23"] / d["n1"]), math.log1p(a), b)
             for t, (a, b) in d["tok"].items()} for c, d in res.items()}
la._init(stats)
IDX = {k: i for i, k in enumerate(la.COLUMNS)}


def codes(arr):
    src = pc.cast(pc.utf8_slice_codeunits(arr, 1, 2), pa.int64()).to_numpy()
    num = pc.cast(pc.utf8_slice_codeunits(arr, 3), pa.int64()).to_numpy()
    return src * 10**10 + num


def f05(tp, pred, true):
    if pred == 0 and true == 0:
        return 1.0
    if tp == 0:
        return 0.0
    p, r = tp / pred, tp / true
    return 1.25 * p * r / (0.25 * p + r)


s1t = pq.read_table(fr"{CACHE}\test\records_s1.parquet", columns=["entity_id", "country", "name_n"])
s1_code = codes(s1t.column("entity_id").combine_chunks())
s1_country = np.array(s1t.column("country").to_pylist(), dtype=object)
s1_name = s1t.column("name_n").combine_chunks()
del s1t
r = pq.read_table(fr"{CACHE}\test\records_s23.parquet", columns=["entity_id", "name_n"])
rec_code = codes(r.column("entity_id").combine_chunks())
rec_name = r.column("name_n").combine_chunks()
del r
sub = pacsv.read_csv(fr"{ROOT}\submissions\10_lgb_stage3_0925_183300\matching_results.tsv",
                     parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
                     convert_options=pacsv.ConvertOptions(column_types={"source1_entity_id": pa.string(),
                                                                        "matched_entity_ids": pa.string()}))
lists = pc.split_pattern(sub.column("matched_entity_ids").combine_chunks(), ",")
flat = pc.list_flatten(lists)
parent = pc.list_parent_indices(lists).to_numpy()
keep = pc.not_equal(flat, "").to_numpy(zero_copy_only=False)
m_rec = codes(pc.filter(flat, pa.array(keep)))
m_s1 = codes(sub.column("source1_entity_id").combine_chunks())[parent[keep]]
del sub, lists, flat
o1, o2 = np.argsort(s1_code), np.argsort(rec_code)
si = o1[np.searchsorted(s1_code[o1], m_s1)]
ri = o2[np.searchsorted(rec_code[o2], m_rec)]
n_pred = np.bincount(si, minlength=len(s1_code))

N_TOTAL = len(s1_code)
for c in ("France", "India", "US"):
    m = s1_country[si] == c
    a_idx, b_idx = si[m], ri[m]
    pat = np.zeros(len(a_idx), bool)
    for s in range(0, len(a_idx), 300_000):
        an = s1_name.take(pa.array(a_idx[s:s + 300_000])).to_pylist()
        bn = rec_name.take(pa.array(b_idx[s:s + 300_000])).to_pylist()
        for k, (x, z) in enumerate(zip(an, bn)):
            row = la.pair_row(x, z, c)
            pat[s + k] = (row[IDX["la_swap_vocab"]] == 1 and row[IDX["la_swap_same_initial"]] == 0
                          and row[IDX["la_extra_vocab_len_min"]] >= 4)
    mcount = np.bincount(a_idx[pat], minlength=len(s1_code))
    s1s = np.flatnonzero(mcount)
    out = []
    for q in (0.5, 0.7, 0.9, 0.97):
        tot = 0.0
        for s in s1s:
            mm, k = int(mcount[s]), int(n_pred[s] - mcount[s])   # k other predicted pairs, assumed correct
            for j in range(mm + 1):                             # j of the mm pattern pairs are true matches
                w = comb(mm, j) * (1 - q) ** j * q ** (mm - j)
                true = k + j                                    # assumes the rest of the S1 was found
                before = f05(k + j, k + mm, true)
                after = f05(k, k, true)
                tot += w * (after - before)
        out.append(f"q={q}: {tot / N_TOTAL:+.5f}")
    print(f"{c:6}: pattern pairs {pat.sum():,} on {len(s1s):,} S1 (S1 with no other accepted pair: "
          f"{(n_pred[s1s] == mcount[s1s]).sum():,}) -> expected LB change " + ", ".join(out), flush=True)
    np.save(f"sub10_pattern_{c}.npy", np.stack([a_idx[pat], b_idx[pat]]))
