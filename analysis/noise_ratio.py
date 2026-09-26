"""Token document frequency in S2/S3 names vs S1 names, per split and country (no labels)."""
import csv, re, sys, unicodedata
from collections import Counter
from pathlib import Path
csv.field_size_limit(10**9)
ROOT = Path(__file__).resolve().parents[1]
TOK = re.compile(r"[a-z0-9]+")
def fold(s):
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c)).lower()
def rows(p):
    with open(p, encoding="utf-8", newline="") as f:
        r = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE); next(r); yield from r
split = sys.argv[1]
d = ROOT / f"student_resource/dataset/{split}"
s1, s23, n1, n23 = {}, {}, Counter(), Counter()
for src in ("1", "2", "3"):
    for e, nm, ad, c in rows(d / f"{split}_source{src}.tsv"):
        cnt = s1 if src == "1" else s23
        cnt.setdefault(c, Counter()).update(set(TOK.findall(fold(nm))))
        (n1 if src == "1" else n23)[c] += 1
words = sys.argv[2].split(",")
for c in sorted(s1):
    base = n23[c] / n1[c]
    print(f"\n[{split} {c}] records per S1 {base:.2f}; ratio = (df in S2/S3 / df in S1) / records-per-S1")
    for w in words:
        a, b = s1[c][w], s23[c][w]
        if a + b >= 20:
            print(f"   {w:<14} S1 {a:>7}  S2/S3 {b:>8}  ratio {b / max(a, 1) / base:6.2f}")
    # tokens with the highest ratio among reasonably frequent ones
    top = sorted(((s23[c][w] / max(s1[c][w], 1) / base, w) for w in s23[c] if s23[c][w] >= 2000), reverse=True)[:25]
    print("   top ratios:", ", ".join(f"{w} {r:.1f}" for r, w in top))
