"""Pair features for candidate (S1, record) pairs.

String similarities come from rapidfuzz's multi-threaded cpdist over aligned arrays, so the
full candidate table (100M+ pairs) is processed in chunks without Python-level loops.
Competition features describe how a pair compares with the other candidates of the same
record (every record belongs to at most one S1) and of the same S1.
No country feature is used: labels in test include a country absent from train.

    python -m src.features --config configs/base.yaml --split train
"""
from __future__ import annotations

import argparse
import shutil

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from src.blocking import VIEW_BITS, pruned_path
from src.normalize import LEGAL_FORMS, STATES, load_records
from src.utils import add_config_arg, cache_dir, get_logger, load_config, timer

log = get_logger("features")

STRING_FIELDS = ["name_n", "name_core", "name_ns", "addr_n", "addr_nums", "house", "postal", "legal", "first"]


def record_arrays(df: pd.DataFrame) -> dict:
    """Per-record arrays used for gathering; strings as object arrays for rapidfuzz."""
    out = {}
    df = df.copy()
    df["legal"] = df["name_n"].map(lambda s: " ".join(sorted(set(s.split()) & LEGAL_FORMS)))
    df["first"] = df["name_core"].str.split(n=1).str[0].fillna("")
    for c in STRING_FIELDS:
        out[c] = df[c].to_numpy(dtype=object)
    out["states"] = df["addr_n"].map(_state_set).to_numpy(dtype=object)
    out["name_len"] = df["name_n"].str.len().to_numpy(np.float32)
    out["addr_len"] = df["addr_n"].str.len().to_numpy(np.float32)
    out["name_ntok"] = (df["name_n"].str.count(" ") + 1).to_numpy(np.float32)
    out["addr_ntok"] = np.where(df["addr_n"] == "", 0, df["addr_n"].str.count(" ") + 1).astype(np.float32)
    for c in ("is_domain", "name_non_latin", "addr_empty"):
        out[c] = df[c].to_numpy(np.float32)
    if "src" in df:
        out["src"] = df["src"].to_numpy(np.float32)
    return out


def _group_stats(key: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per row: rank of score within its key group (0 = best), group max, runner-up, group size.

    Written to keep temporaries small (int32 / float32): it runs on 100M+ rows.
    """
    n = len(key)
    rank, gmax, gsecond, gsize = (np.empty(n, np.float32) for _ in range(4))
    if n == 0:
        return rank, gmax, gsecond, gsize
    score = np.asarray(score, dtype=np.float32)
    order = np.lexsort((-score, key))
    k = key[order]
    start_flag = np.empty(n, dtype=bool)
    start_flag[0] = True
    np.not_equal(k[1:], k[:-1], out=start_flag[1:])
    del k
    starts = np.flatnonzero(start_flag).astype(np.int32)
    gid = np.cumsum(start_flag, dtype=np.int32) - 1
    del start_flag
    sizes = np.diff(np.r_[starts, n]).astype(np.int32)
    s_sorted = score[order]
    best = s_sorted[starts]
    second = np.where(sizes > 1, s_sorted[np.minimum(starts + 1, n - 1)], np.nan).astype(np.float32)
    del s_sorted
    rank[order] = np.arange(n, dtype=np.int32) - starts[gid]
    gmax[order] = best[gid]
    gsecond[order] = second[gid]
    gsize[order] = sizes[gid]
    return rank, gmax, gsecond, gsize


def country_segments(s1: np.ndarray, s1_country: np.ndarray) -> list[slice]:
    """Contiguous row ranges with one country each (candidate tables are written country by
    country and no record/S1 group crosses countries, so group statistics can run per slice)."""
    c = s1_country[s1]
    cuts = np.flatnonzero(c[1:] != c[:-1]) + 1
    bounds = np.r_[0, cuts, len(c)]
    return [slice(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:])]


def group_features(cands: pd.DataFrame, s1_country: np.ndarray | None = None) -> pd.DataFrame:
    """How each pair ranks against the other candidates of its record and of its S1.

    Scores: char cosine (full), token cosine (tok) and the stage-1 probability (p1).
    Runs one country slice at a time when s1_country is given (groups never cross countries).
    """
    if s1_country is not None:
        segs = country_segments(cands["s1"].to_numpy(), s1_country)
        if len(segs) > 1:
            return pd.concat([group_features(cands.iloc[s]) for s in segs], ignore_index=True)
    s1 = cands["s1"].to_numpy(np.int64)
    rec = cands["rec"].to_numpy(np.int64)
    scores = {
        "full": 0.5 * (cands["cos_name"].to_numpy(np.float32) + cands["cos_addr"].to_numpy(np.float32)),
        "tok": 0.5 * (cands["tcos_name"].to_numpy(np.float32) + cands["tcos_addr"].to_numpy(np.float32)),
        "p1": cands["p1"].to_numpy(np.float32),
    }
    out = {"cos_full": scores["full"], "tcos_full": scores["tok"]}
    for key, tag in ((rec, "rec"), (s1, "s1")):
        for sname, score in scores.items():
            rank, gmax, gsecond, gsize = _group_stats(key, score)
            out[f"{tag}_rank_{sname}"] = rank
            out[f"{tag}_gap_{sname}"] = gmax - score
            # for the best candidate: margin over the runner-up; otherwise 0
            out[f"{tag}_margin_{sname}"] = np.where(rank == 0, score - np.nan_to_num(gsecond, nan=0.0), 0.0)
        out[f"{tag}_n_cands"] = gsize
    # expected number of matches of the S1 / record under the stage-1 model
    out["s1_sum_p1"] = np.bincount(s1, weights=scores["p1"])[s1].astype(np.float32)
    out["rec_sum_p1"] = np.bincount(rec, weights=scores["p1"])[rec].astype(np.float32)
    return pd.DataFrame(out)


def sibling_features(cands: pd.DataFrame, B: dict, min_p1: float = 0.8, max_sib: int = 4,
                     prob: np.ndarray | None = None, prefix: str = "sib") -> pd.DataFrame:
    """Similarity of each record to the other records confidently linked to the same S1.

    Records of one business in S2 and S3 often resemble each other more than either resembles
    the S1 record (e.g. both drop the same word, or both carry the address S1 spells differently).
    "Confident" = the record's best candidate with probability >= min_p1 (stage-1 p1 by default,
    or the given `prob`, e.g. out-of-fold stage-2 probabilities); up to max_sib per S1.
    Uses only model probabilities (out-of-fold on train), never labels.
    """
    s1 = cands["s1"].to_numpy(np.int64)
    rec = cands["rec"].to_numpy(np.int64)
    p1 = cands["p1"].to_numpy(np.float32) if prob is None else np.asarray(prob, np.float32)
    rank = _group_stats(rec, p1)[0]
    conf = np.flatnonzero((rank == 0) & (p1 >= min_p1))
    sib = pd.DataFrame({"s1": s1[conf], "sib": rec[conf], "p": p1[conf]})
    sib = sib.sort_values(["s1", "p"], ascending=[True, False])
    sib = sib[sib.groupby("s1").cumcount() < max_sib + 1]      # +1: the record itself may be one of them
    pairs = pd.DataFrame({"pair": np.arange(len(cands)), "s1": s1, "rec": rec})
    t = pairs.merge(sib[["s1", "sib"]], on="s1")
    t = t[t["rec"] != t["sib"]]
    pi, ri, si = t["pair"].to_numpy(), t["rec"].to_numpy(), t["sib"].to_numpy()
    out = {}
    for name, field, scorer in (("name", "name_n", fuzz.token_set_ratio),
                                ("ns", "name_ns", fuzz.ratio),
                                ("addr", "addr_n", fuzz.token_set_ratio)):
        v = _cp(B[field][ri], B[field][si], scorer).astype(np.float32)
        best = np.full(len(cands), -1.0, np.float32)
        np.maximum.at(best, pi, v)
        out[f"{prefix}_{name}"] = np.where(best < 0, np.nan, best).astype(np.float32)
    out[f"n_{prefix}"] = np.bincount(pi, minlength=len(cands)).astype(np.float32)
    return pd.DataFrame(out)


# "la" (Louisiana) is left out: French localities start with it ("la teste de buch", "la baule")
STATE_CODES = frozenset(STATES.values()) - {"la"}


def _state_set(addr: str) -> str:
    """State codes at either end of the address (where US/India addresses put them).

    Only the first/last alphabetic token is considered: codes such as "de"/"la" are ordinary
    words inside French addresses ("rue de la paix"), which would make the feature mean
    something else there.
    """
    toks = [t for t in addr.split() if t.isalpha()]
    if not toks:
        return ""
    return " ".join(sorted({t for t in (toks[0], toks[-1]) if t in STATE_CODES}))


def name_frequency(s1_df: pd.DataFrame, rec_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """How many S1 records (same country) share each S1's / each record's name_core.

    A unique name with an empty address is strong evidence; a name shared by many S1 records
    ("modern foods llp") needs the address to decide.
    """
    counts = s1_df.groupby(["country", "name_core"]).size()
    s1_freq = counts.reindex(pd.MultiIndex.from_arrays([s1_df["country"], s1_df["name_core"]])).to_numpy()
    rec_freq = counts.reindex(pd.MultiIndex.from_arrays([rec_df["country"], rec_df["name_core"]])).fillna(0).to_numpy()
    return s1_freq.astype(np.float32), rec_freq.astype(np.float32)


def drop_common_tokens(s1_df: pd.DataFrame, rec_df: pd.DataFrame, col: str, max_share: float = 0.01
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Copies of `col` without the tokens found in more than `max_share` of the country's records.

    Frequent tokens are mostly administrative areas and generic words (state codes, region and
    department names, big cities, "pvt", "sarl", "de"). Sources write them inconsistently (a
    department in one, the region in another), which adds noise to the similarity of the specific
    parts (street, number, brand). Tokens with digits are always kept. Frequencies are computed
    per country label on the split being processed, so an unseen label gets its own list.
    """
    both = pd.concat([s1_df[["country", col]], rec_df[["country", col]]], ignore_index=True)
    toks = both[col].str.split()
    lens = toks.str.len().to_numpy()
    # document frequency: each token counted once per record
    rec_tok = pd.DataFrame({"row": np.repeat(np.arange(len(both)), lens),
                            "tok": np.concatenate(toks.to_numpy()) if lens.sum() else []}).drop_duplicates()
    rec_tok["country"] = both["country"].to_numpy()[rec_tok["row"].to_numpy()]
    dfreq = rec_tok.groupby(["country", "tok"]).size()
    n_per_country = both["country"].value_counts()
    share = dfreq / n_per_country.reindex(dfreq.index.get_level_values(0)).to_numpy()
    common = {c: set(share[c][share[c] > max_share].index) for c in share.index.get_level_values(0).unique()}
    del rec_tok, dfreq, share

    def strip(df):
        out = []
        for country, text in zip(df["country"].to_numpy(), df[col].to_numpy()):
            cs = common.get(country, set())
            out.append(" ".join(t for t in text.split() if t not in cs or any(ch.isdigit() for ch in t)))
        return np.array(out, dtype=object)
    return strip(s1_df), strip(rec_df)


def name_ambiguity(cands: pd.DataFrame, A: dict, B: dict) -> pd.DataFrame:
    """Within each record's candidates: rank/gap of core-name similarity and how many candidates
    have a near-identical name (several same-name S1 records -> the address must decide)."""
    s1 = cands["s1"].to_numpy(np.int64)
    rec = cands["rec"].to_numpy(np.int64)
    sim = _cp(A["name_core"][s1], B["name_core"][rec], fuzz.ratio).astype(np.float32)
    rank, gmax, _, _ = _group_stats(rec, sim)
    near = np.bincount(rec, weights=(sim >= 90).astype(np.float64))[rec].astype(np.float32)
    return pd.DataFrame({"rec_rank_cname": rank, "rec_gap_cname": gmax - sim, "rec_n_same_name": near})


def _cp(a, b, scorer, dtype=np.uint8):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=dtype)


def _eq_or_nan(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    both = (a != "") & (b != "")
    return np.where(both, (a == b).astype(np.float32), np.nan).astype(np.float32)


def pair_features(A: dict, B: dict, ia: np.ndarray, ib: np.ndarray) -> pd.DataFrame:
    """A = S1 arrays, B = record arrays; ia/ib = row positions of each pair."""
    g = lambda d, c, i: d[c][i]
    f = {}
    an, bn = g(A, "name_n", ia), g(B, "name_n", ib)
    f["n_ratio"] = _cp(an, bn, fuzz.ratio)
    f["n_partial"] = _cp(an, bn, fuzz.partial_ratio)
    f["n_tsort"] = _cp(an, bn, fuzz.token_sort_ratio)
    f["n_tset"] = _cp(an, bn, fuzz.token_set_ratio)
    f["n_jw"] = _cp(an, bn, JaroWinkler.normalized_similarity, np.float32)
    ac, bc = g(A, "name_core", ia), g(B, "name_core", ib)
    f["c_ratio"] = _cp(ac, bc, fuzz.ratio)
    f["c_tset"] = _cp(ac, bc, fuzz.token_set_ratio)
    f["c_jw"] = _cp(ac, bc, JaroWinkler.normalized_similarity, np.float32)
    ans, bns = g(A, "name_ns", ia), g(B, "name_ns", ib)
    f["ns_ratio"] = _cp(ans, bns, fuzz.ratio)
    f["ns_partial"] = _cp(ans, bns, fuzz.partial_ratio)
    aa, ba = g(A, "addr_n", ia), g(B, "addr_n", ib)
    f["a_ratio"] = _cp(aa, ba, fuzz.ratio)
    f["a_partial"] = _cp(aa, ba, fuzz.partial_ratio)
    f["a_tsort"] = _cp(aa, ba, fuzz.token_sort_ratio)
    f["a_tset"] = _cp(aa, ba, fuzz.token_set_ratio)
    anum, bnum = g(A, "addr_nums", ia), g(B, "addr_nums", ib)
    both_nums = (anum != "") & (bnum != "")
    f["num_tset"] = np.where(both_nums, _cp(anum, bnum, fuzz.token_set_ratio), np.nan).astype(np.float32)
    f["num_tsort"] = np.where(both_nums, _cp(anum, bnum, fuzz.token_sort_ratio), np.nan).astype(np.float32)
    f["house_eq"] = _eq_or_nan(g(A, "house", ia), g(B, "house", ib))
    f["state_eq"] = _eq_or_nan(g(A, "states", ia), g(B, "states", ib))
    f["s1_name_freq"] = A["name_freq"][ia]
    f["rec_name_s1_freq"] = B["name_freq"][ib]
    f["postal_eq"] = _eq_or_nan(g(A, "postal", ia), g(B, "postal", ib))
    f["legal_eq"] = _eq_or_nan(g(A, "legal", ia), g(B, "legal", ib))
    af, bf = g(A, "first", ia), g(B, "first", ib)
    f["first_eq"] = _eq_or_nan(af, bf)
    # brand token kept while the descriptive words change ("prock allstate llc" -> "prock llc services")
    f["s1_first_in_rec"] = _cp(af, bc, fuzz.partial_ratio)
    f["rec_first_in_s1"] = _cp(bf, ac, fuzz.partial_ratio)
    # cross-field: trade-name records sometimes carry the location in the name
    f["x_name_addr"] = _cp(bn, aa, fuzz.partial_ratio)
    # the specific parts only: country-frequent tokens (admin areas, generic words) removed
    for tag, field in (("ar", "addr_rare"), ("nr", "name_rare")):
        if field in A and field in B:
            x, y = g(A, field, ia), g(B, field, ib)
            both = (x != "") & (y != "")
            f[f"{tag}_tset"] = np.where(both, _cp(x, y, fuzz.token_set_ratio), np.nan).astype(np.float32)
            f[f"{tag}_ratio"] = np.where(both, _cp(x, y, fuzz.ratio), np.nan).astype(np.float32)
    for c in ("name_len", "addr_len", "name_ntok", "addr_ntok"):
        f[f"d_{c}"] = np.abs(A[c][ia] - B[c][ib])
        f[f"r_{c}"] = B[c][ib]
    for c in ("is_domain", "name_non_latin", "addr_empty", "src"):
        f[f"r_{c}"] = B[c][ib]
    f["s1_name_ntok"] = A["name_ntok"][ia]
    return pd.DataFrame(f)


def features_dir(cfg: dict, split: str, subdir: str = "features"):
    return cache_dir(cfg, split) / subdir


def build(cfg: dict, split: str) -> None:
    cands = pd.read_parquet(pruned_path(cfg, split))
    s1 = load_records(cfg, split, "s1")
    s23 = load_records(cfg, split, "s23")
    with timer("record arrays", log):
        A, B = record_arrays(s1), record_arrays(s23)
        A["name_freq"], B["name_freq"] = name_frequency(s1, s23)
    if cfg["features"].get("rare_tokens", False):
        # off by default: +0.0003 OOF F0.5 in-distribution, but worse transfer to an unseen
        # country in leave-one-country-out runs (US->India 0.9488 -> 0.9454)
        with timer("country-frequent token removal", log):
            A["addr_rare"], B["addr_rare"] = drop_common_tokens(s1, s23, "addr_n")
            A["name_rare"], B["name_rare"] = drop_common_tokens(s1, s23, "name_core")
    with timer("group features", log):
        grp = group_features(cands, s1["country"].to_numpy())
    with timer("sibling + name ambiguity features", log):
        grp = pd.concat([grp, sibling_features(cands, B), name_ambiguity(cands, A, B)], axis=1)
    views = cands["views"].to_numpy()
    out = features_dir(cfg, split)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    chunk = cfg["features"]["chunk_pairs"]
    for i, s in enumerate(range(0, len(cands), chunk)):
        e = min(s + chunk, len(cands))
        with timer(f"features {split} part {i} ({s:,}-{e:,})", log):
            ia = cands["s1"].to_numpy()[s:e]
            ib = cands["rec"].to_numpy()[s:e]
            f = pair_features(A, B, ia, ib)
            for c in ("cos_name", "cos_addr", "tcos_name", "tcos_addr", "p1"):
                f[c] = cands[c].to_numpy()[s:e]
            for v, bit in VIEW_BITS.items():
                f[f"view_{v}"] = ((views[s:e] & bit) > 0).astype(np.uint8)
            f = pd.concat([f, grp.iloc[s:e].reset_index(drop=True)], axis=1)
            f.to_parquet(out / f"part_{i:03d}.parquet", index=False)
    log.info("wrote %d parts to %s", i + 1, out)


def write_parts(df: pd.DataFrame, out_dir, chunk: int) -> None:
    """Write a pair-aligned feature table as part_XXX.parquet files (same chunking as features)."""
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    for i, s in enumerate(range(0, len(df), chunk)):
        df.iloc[s:s + chunk].to_parquet(out_dir / f"part_{i:03d}.parquet", index=False)


def part_columns(cfg: dict, split: str, subdir: str = "features") -> list[str]:
    import pyarrow.parquet as pq

    first = sorted(features_dir(cfg, split, subdir).glob("part_*.parquet"))[0]
    return pq.ParquetFile(first).schema_arrow.names


def load_part(cfg: dict, split: str, i: int, columns_by_dir: dict) -> pd.DataFrame:
    """Part i of several aligned feature folders, side by side: {subdir: columns}."""
    frames = [pd.read_parquet(features_dir(cfg, split, d) / f"part_{i:03d}.parquet", columns=cols)
              for d, cols in columns_by_dir.items()]
    return pd.concat(frames, axis=1) if len(frames) > 1 else frames[0]


def load_features(cfg: dict, split: str, rows: np.ndarray | None = None, columns=None,
                  subdir: str = "features") -> pd.DataFrame:
    """Load feature parts (optionally only the given global row positions, sorted ascending)."""
    parts = sorted(features_dir(cfg, split, subdir).glob("part_*.parquet"))
    chunk = cfg["features"]["chunk_pairs"]
    frames = []
    for i, p in enumerate(parts):
        if rows is None:
            frames.append(pd.read_parquet(p, columns=columns))
            continue
        lo, hi = np.searchsorted(rows, [i * chunk, (i + 1) * chunk])
        if hi > lo:
            frames.append(pd.read_parquet(p, columns=columns).iloc[rows[lo:hi] - i * chunk])
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    add_config_arg(ap)
    ap.add_argument("--split", required=True, choices=["train", "test"])
    args = ap.parse_args()
    build(load_config(args.config), args.split)


if __name__ == "__main__":
    main()
