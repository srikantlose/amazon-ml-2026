"""Re-align lgb_stage3_0925_183300/test_prob.npy with submission 10's candidate pairs (pruned order is
country segment, then record, then S1) and look at France's probability distribution above 0.95."""
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from lalib import CACHE, ROOT


def codes(arr):
    src = pc.cast(pc.utf8_slice_codeunits(arr, 1, 2), pa.int64()).to_numpy()
    num = pc.cast(pc.utf8_slice_codeunits(arr, 3), pa.int64()).to_numpy()
    return src * 10**10 + num


def read_pairs(path, col):
    t = pacsv.read_csv(path, parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
                       convert_options=pacsv.ConvertOptions(column_types={"source1_entity_id": pa.string(),
                                                                          col: pa.string()}))
    lists = pc.split_pattern(t.column(col).combine_chunks(), ",")
    flat = pc.list_flatten(lists)
    parent = pc.list_parent_indices(lists).to_numpy()
    keep = pc.not_equal(flat, "").to_numpy(zero_copy_only=False)
    return codes(t.column("source1_entity_id").combine_chunks())[parent[keep]], codes(pc.filter(flat, pa.array(keep)))


s1t = pq.read_table(fr"{CACHE}\test\records_s1.parquet", columns=["entity_id", "country"])
s1_code = codes(s1t.column("entity_id").combine_chunks())
s1_country = np.array(s1t.column("country").to_pylist(), dtype=object)
rec_code = codes(pq.read_table(fr"{CACHE}\test\records_s23.parquet", columns=["entity_id"]).column(0).combine_chunks())
o1, o2 = np.argsort(s1_code), np.argsort(rec_code)
to_s1 = lambda q: o1[np.searchsorted(s1_code[o1], q)]
to_rec = lambda q: o2[np.searchsorted(rec_code[o2], q)]

SUB = fr"{ROOT}\submissions\10_lgb_stage3_0925_183300"
cs, cr = read_pairs(fr"{SUB}\candidate_pairs.tsv", "candidate_entity_ids")
s1, rec = to_s1(cs), to_rec(cr)
crank = {c: i for i, c in enumerate(sorted(set(s1_country)))}
cr_arr = np.array([crank[c] for c in s1_country], np.int8)
order = np.lexsort((s1, rec, cr_arr[s1]))
s1, rec = s1[order], rec[order]
prob = np.load(fr"{CACHE}\runs\lgb_stage3_0925_183300\test_prob.npy")
if __name__ == "__main__": print(f"candidate pairs {len(s1):,}; test_prob {len(prob):,}")
assert len(prob) == len(s1)

# best per record and margin, as in src.decide
o = np.lexsort((-prob, rec))
r = rec[o]
start = np.flatnonzero(np.r_[True, r[1:] != r[:-1]])
best = o[start]
nxt = np.minimum(start + 1, len(o) - 1)
second = np.where((start + 1 < len(o)) & (r[nxt] == r[start]), prob[o[nxt]], 0.0)
pb, mg = prob[best], prob[best] - second
ctry_b = s1_country[s1[best]]

# alignment check against the accepted pairs of submission 10
ms, mr = read_pairs(fr"{SUB}\matching_results.tsv", "matched_entity_ids")
key = lambda a, b: a.astype(np.int64) * (1 << 24) + b.astype(np.int64)
acc = np.sort(key(to_s1(ms), to_rec(mr)))
thr = np.where(ctry_b == "France", 0.95, 0.65)
pred = (pb >= thr) & (mg >= 0.5)
mine = np.sort(key(s1[best[pred]], rec[best[pred]]))
print(f"alignment: accepted in submission {len(acc):,}, re-derived {len(mine):,}, "
      f"identical {len(acc) == len(mine) and bool((acc == mine).all())}")

pk = np.concatenate([key(*np.load(f"sub10_pattern_{c}.npy")) for c in ("France", "India", "US")])
is_pat = np.isin(key(s1[best], rec[best]), pk)
bands = [(0.5, 0.65), (0.65, 0.75), (0.75, 0.85), (0.85, 0.95), (0.95, 0.97), (0.97, 0.98), (0.98, 0.99),
         (0.99, 0.995), (0.995, 1.01)]
for c in ("France", "India", "US"):
    n_s1 = (s1_country == c).sum()
    m = (ctry_b == c) & (mg >= 0.5)
    print(f"{c}: records whose best pair has margin>=0.5, by best p (per 100 S1; pattern pairs among them):")
    print("   " + "  ".join(f"[{lo},{hi}) {((pb >= lo) & (pb < hi) & m).sum() / n_s1 * 100:.2f}"
                          f" ({((pb >= lo) & (pb < hi) & m & is_pat).sum():,})" for lo, hi in bands))
