"""Swap patterns among the pairs accepted by submission 10, per country, with test-split statistics."""
import math
import pickle
from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from lalib import CACHE, ROOT, la

ARTICLES = {"de", "du", "des", "la", "le", "les", "au", "aux", "et", "en"}
res = pickle.load(open("tokstats.pkl", "rb"))["test"]
stats = {}
for c, d in res.items():
    base = math.log(d["n23"] / d["n1"])
    stats[c] = {t: (math.log((b + 1) / (a + 1)) - base, math.log1p(a), b) for t, (a, b) in d["tok"].items()}
la._init(stats)


def codes(arr):
    src = pc.cast(pc.utf8_slice_codeunits(arr, 1, 2), pa.int64()).to_numpy()
    num = pc.cast(pc.utf8_slice_codeunits(arr, 3), pa.int64()).to_numpy()
    return src * 10**10 + num


def detail(an, bn, c):
    """(pattern, any swap, vocab words) with the same rules as la.pair_row."""
    a, b = la.name_tokens(an), la.name_tokens(bn)
    if not a or not b or (len(a) >= 2 and len(b) == 1 and b[0] == "".join(t[0] for t in a)):
        return False, False, []
    ja, jb = "".join(a), "".join(b)
    missing = [t for t in a if not any(la._close(t, u) for u in b) and not (len(t) >= 4 and t in jb)]
    extra = [u for u in b if not any(la._close(t, u) for t in a) and not (len(u) >= 4 and u in ja)]
    st = stats[c]
    vocab = []
    for u in extra:
        s = st.get(u)
        if s is None:
            continue
        lr, ldf1, df23 = s
        if lr >= la.FILLER_LRATIO and df23 >= la.FILLER_MIN_DF23:
            continue
        if ldf1 >= math.log1p(la.VOCAB_MIN_DF1):
            vocab.append(u)
    swap = bool(missing) and bool(vocab)
    if not swap:
        return False, False, vocab
    same_init = any(t[0] == u[0] for t in missing for u in vocab)
    pattern = (not same_init) and min(map(len, vocab)) >= 4
    return pattern, True, vocab


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
assert (s1_code[si] == m_s1).all() and (rec_code[ri] == m_rec).all()
print(f"accepted pairs {len(si):,}")

for c in ("France", "India", "US"):
    m = s1_country[si] == c
    a_idx, b_idx = si[m], ri[m]
    n_s1c = (s1_country == c).sum()
    pat = np.zeros(len(a_idx), bool)
    swp = np.zeros(len(a_idx), bool)
    art_swap = 0
    vocab_words = Counter()
    for s in range(0, len(a_idx), 300_000):
        an = s1_name.take(pa.array(a_idx[s:s + 300_000])).to_pylist()
        bn = rec_name.take(pa.array(b_idx[s:s + 300_000])).to_pylist()
        for k, (x, z) in enumerate(zip(an, bn)):
            p, w, v = detail(x, z, c)
            pat[s + k], swp[s + k] = p, w
            if w:
                vocab_words.update(v)
                if all(t in ARTICLES for t in v):
                    art_swap += 1
    s1_with = len(np.unique(a_idx[pat]))
    print(f"{c:6}: accepted {m.sum():>9,} | pattern pairs {pat.sum():>7,} ({pat.sum() / n_s1c * 100:.2f} per 100 S1; "
          f"S1 with >=1: {s1_with / n_s1c * 100:.2f}%) | any swap {swp.sum():,} "
          f"(of which only-article vocab {art_swap:,})")
    print("        most common swapped-in vocab words:", vocab_words.most_common(15))
