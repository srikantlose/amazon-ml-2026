"""Verify the swap-pattern positive rates on train labels, with integer-coded ids (low memory)."""
import glob

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from lalib import CACHE, ROOT


def codes(arr: pa.Array) -> np.ndarray:
    """'S2-123' -> 2 * 10**10 + 123."""
    src = pc.cast(pc.utf8_slice_codeunits(arr, 1, 2), pa.int64()).to_numpy()
    num = pc.cast(pc.utf8_slice_codeunits(arr, 3), pa.int64()).to_numpy()
    return src * 10**10 + num


s1t = pq.read_table(fr"{CACHE}\train\records_s1.parquet", columns=["entity_id", "country"])
s1_code = codes(s1t.column("entity_id").combine_chunks())
s1_country = np.array(s1t.column("country").to_pylist(), dtype=object)
del s1t
rec_code = codes(pq.read_table(fr"{CACHE}\train\records_s23.parquet", columns=["entity_id"]).column(0).combine_chunks())

gt = pacsv.read_csv(fr"{ROOT}\student_resource\dataset\train\train_ground_truth.tsv",
                    parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
                    convert_options=pacsv.ConvertOptions(column_types={"source1_entity_id": pa.string(),
                                                                       "matched_entity_ids": pa.string()}))
lists = pc.split_pattern(gt.column("matched_entity_ids").combine_chunks(), ",")
flat = pc.list_flatten(lists)
parent = pc.list_parent_indices(lists).to_numpy()
keep = pc.not_equal(flat, "").to_numpy(zero_copy_only=False)
m_rec = codes(pc.filter(flat, pa.array(keep)))
m_s1 = codes(gt.column("source1_entity_id").combine_chunks())[parent[keep]]
del gt, lists, flat


def index_of(sorted_codes, order, q):
    pos = np.searchsorted(sorted_codes, q)
    assert (sorted_codes[pos] == q).all()
    return order[pos]


o1 = np.argsort(s1_code)
o2 = np.argsort(rec_code)
si = index_of(s1_code[o1], o1, m_s1)
ri = index_of(rec_code[o2], o2, m_rec)
true_s1 = np.full(len(rec_code), -1, np.int64)
true_s1[ri] = si
n_true = np.bincount(si, minlength=len(s1_code))
print(f"ground truth pairs {len(ri):,}; S1 {len(s1_code):,}; singletons {(n_true == 0).mean():.4f}; "
      f"mean matches {n_true.mean():.3f}")
del s1_code, rec_code, o1, o2

c = pq.read_table(fr"{CACHE}\train\cands_pruned.parquet", columns=["s1", "rec", "p1"])
s1 = c.column("s1").to_numpy()
rec = c.column("rec").to_numpy()
p1 = c.column("p1").to_numpy()
del c
y = true_s1[rec] == s1
ctry = s1_country[s1]
print(f"pairs {len(y):,}, positives {y.sum():,} ({y.mean():.4f}); pruned recall {y.sum() / n_true.sum():.4f}")

cols = ["la_n_missing", "la_extra_vocab", "la_swap_vocab", "la_swap_same_initial", "la_extra_vocab_len_min",
        "la_acronym", "la_extra_filler", "la_swap_jw"]
parts = sorted(glob.glob(fr"{CACHE}\train\features_la\part_*.parquet"))
F = {k: np.concatenate([pq.read_table(p, columns=[k]).column(0).to_numpy() for p in parts]) for k in cols}
assert len(F["la_swap_vocab"]) == len(y)

swap = F["la_swap_vocab"] == 1
diff_init = F["la_swap_same_initial"] == 0
long4 = F["la_extra_vocab_len_min"] >= 4
pat = swap & diff_init & long4
hi = p1 >= 0.5
for cc in ("US", "India"):
    m = ctry == cc
    n_s1 = (s1_country == cc).sum()
    for name, sel in (("swap, diff initial, len>=4", pat), ("swap, same initial", swap & ~diff_init),
                      ("swap, diff initial, len<4", swap & diff_init & ~long4), ("any swap", swap),
                      ("acronym", F["la_acronym"] == 1)):
        a = m & hi & sel
        print(f"{cc:6} p1>=0.5 & {name:28}: pairs {a.sum():>8,} ({a.sum() / n_s1 * 100:.2f} per 100 S1)  "
              f"positive rate {y[a].mean() if a.any() else float('nan'):.4f}  "
              f"share of all p1>=0.5 positives {(y & a).sum() / (y & m & hi).sum():.4f}")
    # p1 >= 0.9: pairs the ranker is sure about
    a = m & (p1 >= 0.9) & pat
    print(f"{cc:6} p1>=0.9 & pattern: pairs {a.sum():,} positive rate {y[a].mean():.4f}")
np.save("train_y.npy", y)
np.save("train_pair_country_is_us.npy", ctry == "US")
