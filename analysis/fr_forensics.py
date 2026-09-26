"""Label-free look at France pairs: the S1s whose lists changed between the 0.85 and 0.95
France thresholds (sub 08 -> sub 10). Streams the TSVs; stays well under 1 GB."""
import csv, random, sys
from pathlib import Path

csv.field_size_limit(10**9)
ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "student_resource/dataset/test"
SUB = ROOT / "submissions"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0


def rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        r = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        next(r)
        yield from r


def lists(path, keep):
    out = {}
    for s1, ids in rows(path):
        if s1 in keep:
            out[s1] = [i for i in ids.split(",") if i] if ids else []
    return out


fr_s1 = {}
for eid, name, addr, c in rows(T / "test_source1.tsv"):
    if c == "France":
        fr_s1[eid] = (name, addr)

a = lists(SUB / "08_lgb_stage3_0925_183300/matching_results.tsv", fr_s1)
b = lists(SUB / "10_lgb_stage3_0925_183300/matching_results.tsv", fr_s1)
changed = [s for s in fr_s1 if set(a[s]) != set(b[s])]
print(f"France S1 {len(fr_s1):,}; changed 0.85->0.95: {len(changed):,}")
random.seed(SEED)
pick = random.sample(changed, min(N, len(changed)))
need = {i for s in pick for i in a[s] + b[s]}
recs = {}
for fn in ("test_source2.tsv", "test_source3.tsv"):
    for eid, name, addr, c in rows(T / fn):
        if eid in need:
            recs[eid] = (name, addr)

for s in pick:
    n, ad = fr_s1[s]
    print(f"\n=== {s} | {n} | {ad}")
    for i in sorted(set(a[s]) | set(b[s])):
        tag = "KEEP" if i in b[s] else "DROP"
        rn, ra = recs.get(i, ("?", "?"))
        print(f"  {tag} {i:>14} | {rn} | {ra}")
