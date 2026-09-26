"""France recall check: for France S1 with 0-1 accepted matches, list records nobody accepted that share
the S1's house number + street and at least one name word, and say whether they were candidates.

    python analysis/france_missed_matches.py submissions/16_hybrid_la_us_in_lacat075_fr/matching_results.tsv \
        lgb_stage3lacat_0926_051023

26 Sep result (4,000 sampled S1): 677 such records were candidates but scored low, 671 were not candidates,
66 lost to another S1. The low-scored ones are mostly planted descriptor swaps that are correctly rejected
("vaillante ecole" <- "vaillante comite", p=0.000); only a few true matches were missed ("union du detudes"
<- "union du detudes & fils", p=0.29). The "not a candidate" ones are mostly other streets (key collisions).
"""
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.blocking import pruned_path  # noqa: E402
from src.lookalike import _close, name_tokens  # noqa: E402
from src.normalize import records_path  # noqa: E402
from src.utils import load_config  # noqa: E402

cfg = load_config([str(ROOT / "configs" / "base.yaml")])
matches_file, run = sys.argv[1], sys.argv[2]
S1 = pd.read_parquet(records_path(cfg, "test", "s1"), columns=["entity_id", "country", "name_n"])
S23 = pd.read_parquet(records_path(cfg, "test", "s23"), columns=["entity_id", "country", "name_n"])
T = ROOT / "student_resource" / "dataset" / "test"
raw1 = pd.read_csv(T / "test_source1.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=3).set_index("entity_id")
raw23 = pd.concat([pd.read_csv(T / f"test_source{i}.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=3)
                   for i in (2, 3)]).set_index("entity_id")
STREET = set("rue r avenue av ave bd boulevard blvd allee all place pl chemin ch impasse imp square sq cours quai "
             "route rte cite residence res lotissement lot passage voie hameau de des du la le l d bis ter b st "
             "saint".split())
NUM = re.compile(r"^(?:#+|no\.?|n°|nº)?\s*0*(\d+)")


def fold(s):
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch)).lower()


def key(address):
    for comp in fold(address).split(","):
        comp = comp.strip()
        m = NUM.match(comp)
        if m:
            rest = [t for t in re.findall(r"[a-z]+", comp[m.end():]) if t not in STREET]
            if rest:
                return (m.group(1),) + tuple(t[:5] for t in rest[:2])
    return None


rec_ids, s1_ids = S23["entity_id"].to_numpy(), S1["entity_id"].to_numpy()
n1, n23 = S1["name_n"].to_numpy(), S23["name_n"].to_numpy()
s1_row = {e: i for i, e in enumerate(s1_ids)}
rec_row = {e: i for i, e in enumerate(rec_ids)}
sub = pd.read_csv(matches_file, sep="\t", dtype=str, keep_default_na=False)
acc, owner = defaultdict(set), {}
for s, ids in zip(sub.source1_entity_id, sub.matched_entity_ids):
    for e in (ids.split(",") if ids else []):
        acc[s1_row[s]].add(rec_row[e])
        owner[rec_row[e]] = s1_row[s]
by_key = defaultdict(list)
for r in np.flatnonzero(S23["country"].to_numpy() == "France"):
    if r not in owner:
        k = key(raw23.at[rec_ids[r], "business_address"])
        if k:
            by_key[k].append(r)
c = pd.read_parquet(pruned_path(cfg, "test"), columns=["s1", "rec"])
cs, cr = c.s1.to_numpy(), c.rec.to_numpy()
p = np.load(ROOT / "data" / "cache" / "runs" / run / "test_prob.npy")
fr_s1 = S1["country"].to_numpy() == "France"
m = fr_s1[cs]
cand = dict(zip(zip(cs[m].tolist(), cr[m].tolist()), p[m].tolist()))
best = {}
for a, b, q in zip(cs[m].tolist(), cr[m].tolist(), p[m].tolist()):
    if q > best.get(b, (None, -1.0))[1]:
        best[b] = (a, q)
low = [i for i in np.flatnonzero(fr_s1) if len(acc[i]) <= 1]
rng = np.random.default_rng(7)
stats, printed = defaultdict(int), 0
for i in rng.choice(low, size=min(4000, len(low)), replace=False):
    k = key(raw1.at[s1_ids[i], "business_address"])
    a = name_tokens(n1[i])
    for r in (by_key.get(k, []) if k else []):
        b = name_tokens(n23[r])
        if not a or not b or not any(_close(x, y) for x in a for y in b):
            continue
        if (i, r) in cand:
            kind = "candidate, rejected (low p)" if best[r][0] == i else "candidate, lost to another S1"
        else:
            kind = "not a candidate"
        stats[kind] += 1
        if printed < 40:
            printed += 1
            extra = f" p={cand[(i, r)]:.3f}" if (i, r) in cand else ""
            print(f"[{kind}] {raw1.at[s1_ids[i], 'business_name']} | {raw1.at[s1_ids[i], 'business_address']}\n"
                  f"      <- {raw23.at[rec_ids[r], 'business_name']} | {raw23.at[rec_ids[r], 'business_address']}{extra}")
print("\nunaccepted same-address records sharing a name word:", dict(stats))
