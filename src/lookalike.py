"""Look-alike name features: which words differ between the S1 name and the record name, and
whether the added words are injected filler or ordinary name vocabulary.

Sources add a few filler words to S2/S3 names ("services", "center", "partners" in the US,
"com", "dr" in India, "developpement", "groupe", "fils" in France), so those words are far more
frequent in S2/S3 names than in S1 names. Ordinary vocabulary ("club", "ecole", "amis", first
names) is about as frequent in both. A record that keeps the S1's other words but swaps one for
ordinary vocabulary ("mauges amis sas" -> "mauges collectif sas", same address) is usually a
different business; a record that swaps in filler ("pediatric clinic" -> "pediatric services")
usually is not. The pair features without this split treat both alike.

The frequency ratio is computed per split and country label from names alone (no labels), the
same way IDF is, so a country label that never appears in training gets its own statistics.
Written as an extra feature folder aligned with the pruned candidate pairs:

    python -m src.lookalike --config configs/base.yaml --split train
"""
from __future__ import annotations

import argparse
import math
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler

from src.blocking import pruned_path
from src.features import features_dir, write_parts
from src.normalize import LEGAL_FORMS, NAME_FILLER, records_path
from src.utils import add_config_arg, get_logger, load_config, timer

log = get_logger("lookalike")

SUBDIR = "features_la"
SKIP = LEGAL_FORMS | NAME_FILLER
# Countries differ in size (train US 1.3M S1, France 0.26M), so word frequencies are used as shares
# of the country's S1 names, and the vocabulary cut-off scales with the number of names.
VOCAB_MIN_SHARE = 2.3e-6          # ~3 names in train US; never fewer than 2 names
FILLER_MIN_DF23 = 20              # filler must be common in S2/S3 names
FILLER_LRATIO = math.log(1.15)    # S2/S3-vs-S1 frequency ratio relative to the country's base rate;
                                  # ordinary vocabulary sits at about 0.85
LRATIO_CLIP = (-1.5, 1.1)         # beyond ~3x the word is plainly filler; test injects more filler than train
COLUMNS = ["la_n_missing", "la_n_extra", "la_extra_vocab", "la_extra_filler", "la_extra_rare",
           "la_swap_vocab", "la_extra_lratio_min", "la_extra_lshare_max", "la_missing_lshare_min", "la_acronym",
           # garbled abbreviations keep the initial and some letters ("care" -> "cea"); descriptor
           # swaps ("amis" -> "collectif", "solutions" -> "products") usually do not
           "la_swap_jw", "la_swap_same_initial", "la_extra_vocab_len_min"]

_STATS: dict = {}


def name_tokens(name: str) -> list[str]:
    seen, out = set(), []
    for t in name.split():
        if t not in SKIP and len(t) > 1 and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _lshare(df1: int, n1: int) -> float:
    """log of the share of the country's S1 names containing the word, per million."""
    return math.log((df1 + 0.5) / max(n1, 1) * 1e6)


def token_stats(s1: pd.DataFrame, s23: pd.DataFrame) -> dict:
    """{country: (tokens, n1, vocab_min)}, tokens = {token: (clipped log ratio, log share, df23, df1)}.

    Every word seen in an S1 name is kept (df1 = 0 for absent words) plus words common in S2/S3 names."""
    stats = {}
    for country in s1["country"].unique():
        n1 = s23_n = 0
        df1, df23 = Counter(), Counter()
        for name in s1.loc[s1["country"] == country, "name_n"]:
            df1.update(name_tokens(name))
            n1 += 1
        for name in s23.loc[s23["country"] == country, "name_n"]:
            df23.update(name_tokens(name))
            s23_n += 1
        base = math.log(max(s23_n, 1) / max(n1, 1))
        lo, hi = LRATIO_CLIP
        tokens = {t: (min(hi, max(lo, math.log((df23[t] + 1) / (df1[t] + 1)) - base)), _lshare(df1[t], n1),
                      df23[t], df1[t])
                  for t in set(df1) | set(df23) if df1[t] >= 1 or df23[t] >= FILLER_MIN_DF23}
        vocab_min = max(2, round(VOCAB_MIN_SHARE * n1))
        stats[country] = (tokens, n1, vocab_min)
        log.info("%s: %d S1 / %d S2+S3 names, %d tokens with statistics, vocabulary = in >= %d S1 names",
                 country, n1, s23_n, len(tokens), vocab_min)
    return stats


def _subsequence(short: str, long: str) -> bool:
    it = iter(long)
    return all(ch in it for ch in short)


def _close(a: str, b: str) -> bool:
    """Same word up to typos, truncation ("pediatr") or contraction ("gaming" -> "gg", "homes" -> "hs")."""
    if a == b:
        return True
    if min(len(a), len(b)) >= 3 and (a.startswith(b) or b.startswith(a)):
        return True
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    # short contractions of short words, or long enough to be a spelling of the word ("bar" is not "boulangerie")
    contraction = (len(short) <= 3 and len(long) <= 7) or len(short) >= 0.6 * len(long)
    if contraction and short[0] == long[0] and _subsequence(short, long):
        return True
    return JaroWinkler.similarity(a, b) >= 0.85


def pair_row(s1_name: str, rec_name: str, country: str) -> tuple:
    a, b = name_tokens(s1_name), name_tokens(rec_name)
    nan = float("nan")
    if not a or not b:
        return (nan,) * len(COLUMNS)
    if len(a) >= 2 and len(b) == 1 and b[0] == "".join(t[0] for t in a):
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, nan, nan, nan, 1.0, nan, nan, nan)
    ja, jb = "".join(a), "".join(b)
    missing = [t for t in a if not any(_close(t, u) for u in b) and not (len(t) >= 4 and t in jb)]
    extra = [u for u in b if not any(_close(t, u) for t in a) and not (len(u) >= 4 and u in ja)]
    st, n1, vocab_min = _STATS.get(country, ({}, 1, 2))
    absent = _lshare(0, n1)
    filler = rare = 0
    vocab, lratios, lshares = [], [], []
    for u in extra:
        s = st.get(u)
        if s is None:
            rare += 1
            lshares.append(absent)
            continue
        lr, lsh, df23, df1 = s
        lshares.append(lsh)
        if lr >= FILLER_LRATIO and df23 >= FILLER_MIN_DF23:
            filler += 1
            lratios.append(lr)
        elif df1 >= vocab_min:
            vocab.append(u)
            lratios.append(lr)
        else:
            rare += 1
    miss_lshare = [st[t][1] if t in st else absent for t in missing]
    swap = bool(missing) and bool(vocab)
    swap_jw = max(JaroWinkler.similarity(t, u) for t in missing for u in vocab) if swap else nan
    same_initial = float(any(t[0] == u[0] for t in missing for u in vocab)) if swap else nan
    return (float(len(missing)), float(len(extra)), float(len(vocab)), float(filler), float(rare),
            float(swap), min(lratios) if lratios else nan,
            max(lshares) if lshares else nan, min(miss_lshare) if miss_lshare else nan, 0.0,
            swap_jw, same_initial, float(min(map(len, vocab))) if vocab else nan)


def _init(stats: dict) -> None:
    global _STATS
    _STATS = stats


def _chunk(args) -> np.ndarray:
    s1_names, rec_names, countries = args
    return np.array([pair_row(a, b, c) for a, b, c in zip(s1_names, rec_names, countries)], dtype=np.float32)


def build(cfg: dict, split: str, workers: int = 8, chunk: int = 200_000) -> None:
    s1 = pd.read_parquet(records_path(cfg, split, "s1"), columns=["name_n", "country"])
    s23 = pd.read_parquet(records_path(cfg, split, "s23"), columns=["name_n", "country"])
    with timer(f"token statistics {split}", log):
        stats = token_stats(s1, s23)
    cands = pd.read_parquet(pruned_path(cfg, split), columns=["s1", "rec"])
    ia, ib = cands["s1"].to_numpy(), cands["rec"].to_numpy()
    del cands
    a_names, a_country = s1["name_n"].to_numpy(dtype=object), s1["country"].to_numpy(dtype=object)
    b_names = s23["name_n"].to_numpy(dtype=object)
    del s1, s23
    jobs = ((a_names[ia[s:s + chunk]], b_names[ib[s:s + chunk]], a_country[ia[s:s + chunk]])
            for s in range(0, len(ia), chunk))
    with timer(f"look-alike features {split} ({len(ia):,} pairs)", log):
        with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(stats,)) as ex:
            out = np.concatenate(list(ex.map(_chunk, jobs)))
    df = pd.DataFrame(out, columns=COLUMNS)
    dest = features_dir(cfg, split, SUBDIR)
    write_parts(df, dest, cfg["features"]["chunk_pairs"])
    log.info("%s: wrote %d look-alike columns for %s pairs -> %s", split, df.shape[1], f"{len(df):,}", dest)
    for c in ("la_swap_vocab", "la_extra_vocab", "la_extra_filler", "la_acronym"):
        log.info("  %s > 0: %.4f of pairs", c, float((df[c] > 0).mean()))


def main():
    ap = argparse.ArgumentParser()
    add_config_arg(ap)
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    build(load_config(args.config), args.split, workers=args.workers)


if __name__ == "__main__":
    main()
