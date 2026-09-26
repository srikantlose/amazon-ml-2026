"""Real-word swaps: same-address record whose name shares a word with the S1 name but swaps
another word for a *common* word (document frequency >= DF_MIN among the country's S1 names).
Train: split by truth label. Test: split by whether submission 10 accepted the pair.
Rates are per sampled S1, so train and test are comparable."""
import csv, re, sys, unicodedata, zlib
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from rapidfuzz.distance import JaroWinkler

csv.field_size_limit(10**9)
ROOT = Path(__file__).resolve().parents[1]
MOD = int(sys.argv[1]) if len(sys.argv) > 1 else 100
DF_MIN = 30
LEGAL = set("""llc l l c inc incorporated corp corporation co company ltd limited pvt private lp llp pa p a pc
pllc plc the and of dba sarl s a r l sas sasu sa sci eurl cie et de des du la le les""".split())
TOK = re.compile(r"[a-z0-9]+")
KEY = re.compile(r"^(?:#+|no\.?|n°|nº)?\s*0*(\d+)[a-z]?\W+([a-z0-9]+)")


def fold(s):
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


STREET = set("""rue r avenue av ave bd boulevard blvd allee all place pl chemin ch impasse imp square sq cours quai
route rte cite residence res lotissement lot passage voie hameau de des du la le l d st street road rd drive dr lane ln
court ct way circle cir n s e w north south east west bis ter b""".split())
WORD = re.compile(r"[a-z0-9]+")


def addr_key(addr):
    for comp in fold(addr).split(","):
        m = KEY.match(comp.strip())
        if m:
            rest = [t for t in WORD.findall(comp.strip()[m.start(2):]) if t not in STREET]
            return (m.group(1), rest[0][:4] if rest else "")
    return None


def toks(name):
    return [t for t in TOK.findall(fold(name)) if t not in LEGAL and len(t) > 1]


def close(a, b):
    return a == b or (min(len(a), len(b)) >= 3 and (a.startswith(b) or b.startswith(a))) or JaroWinkler.similarity(a, b) >= 0.85


def kind(s1_name, rec_name, df):
    a, b = toks(s1_name), toks(rec_name)
    if not a or not b:
        return "OTHER"
    shared = [t for t in a if any(close(t, u) for u in b)]
    missing = [t for t in a if not any(close(t, u) for u in b)]
    extra = [u for u in b if not any(close(t, u) for t in a)]
    if not shared:
        return "NOSHARE"
    if not missing and not extra:
        return "SAME"
    if extra and all(df[u] >= DF_MIN for u in extra):
        WORDS[(kind.country, kind.label, "extra")].update(extra)
        WORDS[(kind.country, kind.label, "missing")].update(missing)
        return "SWAPWORD" if missing else "ADDWORD"
    return "SUBTYPO" if missing and extra else ("DEL" if missing else "ADDTYPO")


WORDS = defaultdict(Counter)


def rows(p):
    with open(p, encoding="utf-8", newline="") as f:
        r = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE); next(r); yield from r


def h(s):
    return zlib.crc32(s.encode())


def run(split, countries, labels_of):
    d = ROOT / f"student_resource/dataset/{split}"
    df = {c: Counter() for c in countries}
    s1, by_key = {}, defaultdict(list)
    for e, nm, ad, c in rows(d / f"{split}_source1.tsv"):
        if c not in countries:
            continue
        df[c].update(set(toks(nm)))
        if h(e) % MOD == 0:
            s1[e] = (nm, ad, c)
            k = addr_key(ad)
            if k:
                by_key[(c,) + k].append(e)
    label = labels_of(s1)
    stats, ex = Counter(), defaultdict(list)
    n_s1 = Counter(c for _, _, c in s1.values())
    for fn in (f"{split}_source2.tsv", f"{split}_source3.tsv"):
        for e, nm, ad, c in rows(d / fn):
            k = addr_key(ad)
            if not k or (c,) + k not in by_key:
                continue
            for s in by_key[(c,) + k]:
                lab = label(s, e)
                kind.country, kind.label = c, lab
                kd = kind(s1[s][0], nm, df[c])
                stats[(c, lab, kd)] += 1
                if kd in ("SWAPWORD", "ADDWORD") and len(ex[(c, lab, kd)]) < 8:
                    ex[(c, lab, kd)].append(f"{s1[s][0]} | {s1[s][1]}  <->  {nm} | {ad}")
    kinds = ("SAME", "DEL", "ADDTYPO", "SUBTYPO", "ADDWORD", "SWAPWORD", "NOSHARE", "OTHER")
    for c in countries:
        print(f"\n[{split} {c}] sampled S1 {n_s1[c]:,}; pairs per 1000 S1 by kind")
        print("  " + " ".join(f"{k:>9}" for k in ("label",) + kinds))
        for lab in sorted({l for (cc, l, _) in stats if cc == c}):
            print("  " + f"{lab:>9} " + " ".join(f"{1000 * stats[(c, lab, k)] / n_s1[c]:>9.1f}" for k in kinds))
    for key, cnt in sorted(WORDS.items()):
        tot = sum(cnt.values())
        print()
        print(f"WORDS {key} total {tot}: " + ", ".join(f"{w} {n / tot:.3f}" for w, n in cnt.most_common(25)))
    return
    for key, v in sorted(ex.items()):
        print(f"\n-- {key}")
        for x in v:
            print("   ", x)


if sys.argv[2] == "train":
    def labels_of(s1):
        d = ROOT / "student_resource/dataset/train"
        truth, matched = {}, []
        for e, ids in rows(d / "train_ground_truth.tsv"):
            lst = [i for i in ids.split(",") if i]
            matched.extend(h(i) for i in lst)
            if e in s1:
                truth[e] = set(lst)
        m = np.unique(np.array(matched, dtype=np.int64))

        def lab(s, e):
            if e in truth.get(s, ()):
                return "TRUE"
            hv = h(e)
            return "OTHER_S1" if m[np.searchsorted(m, hv) % len(m)] == hv else "DISTRACT"
        return lab
    run("train", ("US", "India"), labels_of)
else:
    def labels_of(s1):
        acc = {}
        for s, ids in rows(ROOT / "submissions/10_lgb_stage3_0925_183300/matching_results.tsv"):
            if s in s1:
                acc[s] = set(ids.split(",")) if ids else set()
        return lambda s, e: "ACCEPTED" if e in acc[s] else "REJECTED"
    run("test", ("US", "India", "France"), labels_of)
