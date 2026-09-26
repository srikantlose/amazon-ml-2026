import math
import pickle

import numpy as np

from lalib import la

res = pickle.load(open("tokstats.pkl", "rb"))
LF = la.FILLER_LRATIO


def table(split, c):
    d = res[split][c]
    base = d["n23"] / d["n1"]
    toks = d["tok"]
    r = {t: ((b + 1) / (a + 1)) / base for t, (a, b) in toks.items()}
    return d, base, toks, r


def cls(toks, r, t):
    a, b = toks[t]
    if math.log(r[t]) >= LF and b >= la.FILLER_MIN_DF23:
        return "filler"
    if math.log1p(a) >= math.log1p(la.VOCAB_MIN_DF1):
        return "vocab"
    return "rare"


print("== where ordinary vocabulary sits: r = (df23/df1) / (n23/n1), tokens with df1 >= 200")
for split in res:
    for c in res[split]:
        d, base, toks, r = table(split, c)
        v = np.array([r[t] for t, (a, b) in toks.items() if a >= 200])
        w = np.array([a for t, (a, b) in toks.items() if a >= 200], float)
        o = np.argsort(v)
        cw = np.cumsum(w[o]) / w.sum()
        wmed = v[o][np.searchsorted(cw, 0.5)]
        q = np.quantile(v, [0.1, 0.25, 0.5, 0.75, 0.9])
        n_f = sum(cls(toks, r, t) == "filler" for t in toks)
        print(f"{split:5} {c:6} base={base:.3f} n={len(v):5d} q10/25/50/75/90 = "
              + " ".join(f"{x:.3f}" for x in q) + f" | df1-weighted median {wmed:.3f} | filler tokens {n_f:,}"
              + f" | share r>=1.15 among df1>=200: {(v >= 1.15).mean():.3f}")

words = ["services", "center", "partners", "solutions", "group", "enterprises", "consulting", "associates",
         "com", "dr", "developpement", "groupe", "fils", "freres", "cie", "ets",
         "club", "ecole", "amis", "collectif", "federation", "sportive", "livre", "mauges",
         "de", "du", "des", "la", "le", "les", "et", "au", "sri", "shri", "kumar", "john", "restaurant", "boulangerie"]
print("\n== r for selected words (blank = no statistics)")
hdr = [(s, c) for s in res for c in res[s]]
print(f"{'word':14}" + "".join(f"{s[:2]}-{c[:6]:8}" for s, c in hdr))
for wd in words:
    row = f"{wd:14}"
    for s, c in hdr:
        d, base, toks, r = table(s, c)
        row += f"{r[wd]:5.2f}{cls(toks, r, wd)[0]:1}     " if wd in toks else " " * 12
    print(row)

print("\n== same token, train vs test (US, India): r_test / r_train for tokens with df1 >= 100 in both")
for c in ("US", "India"):
    _, btr, ttr, rtr = table("train", c)
    _, bte, tte, rte = table("test", c)
    common = [t for t in ttr if t in tte and ttr[t][0] >= 100 and tte[t][0] >= 100]
    ratio = np.array([rte[t] / rtr[t] for t in common])
    fil = np.array([cls(ttr, rtr, t) == "filler" for t in common])
    print(f"{c}: {len(common)} tokens; median r_test/r_train all {np.median(ratio):.3f}, "
          f"train-filler {np.median(ratio[fil]):.3f} (n={fil.sum()}), train-vocab {np.median(ratio[~fil]):.3f}")
    # class flips among all tokens present in both, weighted by test df23 (how often the test pairs see them)
    both = [t for t in ttr if t in tte]
    flips = {}
    for t in both:
        k = (cls(ttr, rtr, t), cls(tte, rte, t))
        flips[k] = flips.get(k, 0) + tte[t][1]
    tot = sum(flips.values())
    print("   class (train -> test), share of test S2/S3 token occurrences:",
          {f"{a}->{b}": round(v / tot, 4) for (a, b), v in sorted(flips.items(), key=lambda x: -x[1])})
    f2v = sorted([t for t in both if cls(ttr, rtr, t) == "filler" and cls(tte, rte, t) == "vocab"],
                 key=lambda t: -tte[t][1])[:25]
    v2f = sorted([t for t in both if cls(ttr, rtr, t) == "vocab" and cls(tte, rte, t) == "filler"],
                 key=lambda t: -tte[t][1])[:25]
    print("   filler in train, vocab in test (top by test df23):",
          [(t, round(rtr[t], 2), round(rte[t], 2)) for t in f2v])
    print("   vocab in train, filler in test (top by test df23):",
          [(t, round(rtr[t], 2), round(rte[t], 2)) for t in v2f])

print("\n== France: most frequent S1 name tokens and article share")
d, base, toks, r = table("test", "France")
top = sorted(toks, key=lambda t: -toks[t][0])[:40]
print([(t, toks[t][0], round(r[t], 2), cls(toks, r, t)[0]) for t in top])
arts = ["de", "du", "des", "la", "le", "les", "et", "au", "aux", "en", "sur", "l", "d"]
print("articles present in France stats:", {a: (toks[a][0], round(r[a], 2), cls(toks, r, a)) for a in arts if a in toks})
top_f = sorted([t for t in toks if cls(toks, r, t) == "filler"], key=lambda t: -toks[t][1])[:40]
print("France filler (top by df23):", [(t, round(r[t], 2)) for t in top_f])
for c in ("US", "India"):
    d, base, toks, r = table("train", c)
    top_f = sorted([t for t in toks if cls(toks, r, t) == "filler"], key=lambda t: -toks[t][1])[:30]
    print(f"train {c} filler (top by df23):", [(t, round(r[t], 2)) for t in top_f])
