"""Print random S1 entities of one country with the records a submission matched to them."""
import csv, random, sys
from pathlib import Path
csv.field_size_limit(10**9)
ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "student_resource/dataset/test"
sub, country, n, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])

def rows(p):
    with open(p, encoding="utf-8", newline="") as f:
        r = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE); next(r); yield from r

s1 = {e: (nm, ad) for e, nm, ad, c in rows(T / "test_source1.tsv") if c == country}
lists = {s: ids.split(",") for s, ids in rows(sub) if s in s1 and ids}
random.seed(seed)
pick = random.sample(sorted(lists), n)
need = {i for s in pick for i in lists[s]}
rec = {}
for fn in ("test_source2.tsv", "test_source3.tsv"):
    for e, nm, ad, c in rows(T / fn):
        if e in need: rec[e] = (nm, ad)
for s in pick:
    print(f"\n=== {s} | {s1[s][0]} | {s1[s][1]}")
    for i in sorted(lists[s]):
        print(f"    {i:>14} | {rec[i][0]} | {rec[i][1]}")
