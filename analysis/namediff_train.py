"""How often are same-address records with a swapped name word true matches in train?

Samples ~1/150 of train S1, collects every S2/S3 record sharing (house number, next word) with a
sampled S1 in the same country, labels it (true match of this S1 / record of another S1 /
distractor) and classifies the name difference after dropping legal-form tokens:
  SAME  no unmatched tokens            DEL  S1 tokens missing from the record only
  ADD   extra record tokens only        SUB  both (a word swapped)
Tokens pair up when equal, prefix-related, or Jaro-Winkler >= 0.85 (typos)."""
import csv, re, sys, unicodedata, zlib, random
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from rapidfuzz.distance import JaroWinkler

csv.field_size_limit(10**9)
ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "student_resource/dataset/train"
MOD = int(sys.argv[1]) if len(sys.argv) > 1 else 150

LEGAL = set("""llc l l c inc incorporated corp corporation co company ltd limited pvt private lp llp pa p a pc
pllc plc the and of dba sarl s a r l sas sasu sa sci eurl cie et fils sons son group groupe services service
center centre associates associes holdings international france usa india""".split())


def fold(s):
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


TOK = re.compile(r"[a-z0-9]+")
KEY = re.compile(r"^(?:#+|no\.?|n°)?\s*0*(\d+)[a-z]?\W+([a-z0-9]+)")


def addr_key(addr):
    for comp in fold(addr).split(","):
        m = KEY.match(comp.strip())
        if m:
            return (m.group(1), m.group(2)[:4])
    return None


def name_tokens(name):
    return [t for t in TOK.findall(fold(name)) if t not in LEGAL and len(t) > 1]


def close(a, b):
    return a == b or (min(len(a), len(b)) >= 3 and (a.startswith(b) or b.startswith(a))) or JaroWinkler.similarity(a, b) >= 0.85


def diff_type(s1_name, rec_name):
    a, b = name_tokens(s1_name), name_tokens(rec_name)
    joined_b = "".join(TOK.findall(fold(rec_name)))
    missing = [t for t in a if not any(close(t, u) for u in b) and t not in joined_b]
    extra = [u for u in b if not any(close(t, u) for t in a) and u not in "".join(a)]
    kind = "SUB" if missing and extra else "DEL" if missing else "ADD" if extra else "SAME"
    return kind, missing, extra


def rows(p):
    with open(p, encoding="utf-8", newline="") as f:
        r = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE); next(r); yield from r


def h(s):
    return zlib.crc32(s.encode())


s1 = {}
by_key = defaultdict(list)
for e, nm, ad, c in rows(D / "train_source1.tsv"):
    if h(e) % MOD == 0:
        s1[e] = (nm, ad, c)
        k = addr_key(ad)
        if k:
            by_key[(c,) + k].append(e)
truth, matched = {}, []
for e, ids in rows(D / "train_ground_truth.tsv"):
    lst = [i for i in ids.split(",") if i]
    matched.extend(h(i) for i in lst)
    if e in s1:
        truth[e] = set(lst)
matched = np.unique(np.array(matched, dtype=np.int64))
print(f"sampled S1 {len(s1):,}, address keys {len(by_key):,}")

stats = Counter()
examples = defaultdict(list)
for fn in ("train_source2.tsv", "train_source3.tsv"):
    for e, nm, ad, c in rows(D / fn):
        k = addr_key(ad)
        if not k or (c,) + k not in by_key:
            continue
        hv = h(e)
        in_any = matched[np.searchsorted(matched, hv) % len(matched)] == hv
        for s in by_key[(c,) + k]:
            lab = "TRUE" if e in truth.get(s, ()) else ("OTHER_S1" if in_any else "DISTRACTOR")
            kind, miss, extra = diff_type(s1[s][0], nm)
            stats[(c, lab, kind)] += 1
            if kind == "SUB" and len(examples[(lab, c)]) < 400:
                examples[(lab, c)].append(f"{s1[s][0]} | {s1[s][1]}  <->  {nm} | {ad}   [-{' '.join(miss)} +{' '.join(extra)}]")

for c in ("US", "India"):
    print(f"\n{c}: label -> SAME / DEL / ADD / SUB (share SUB)")
    for lab in ("TRUE", "OTHER_S1", "DISTRACTOR"):
        v = [stats[(c, lab, k)] for k in ("SAME", "DEL", "ADD", "SUB")]
        print(f"  {lab:<10} {v}  sub={v[3] / max(1, sum(v)):.3f}")
    sub = [stats[(c, lab, 'SUB')] for lab in ('TRUE', 'OTHER_S1', 'DISTRACTOR')]
    print(f"  SUB pairs that are true matches: {sub[0] / max(1, sum(sub)):.3f}")
random.seed(0)
for key, ex in sorted(examples.items()):
    print(f"\n--- SUB examples {key} ({len(ex)} collected)")
    for x in random.sample(ex, min(12, len(ex))):
        print("  ", x)
