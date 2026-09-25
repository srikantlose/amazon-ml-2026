"""Record normalization for business names and addresses.

Every record gets the same treatment regardless of source or country label:

1. Non-Latin text (Devanagari, Telugu, Kannada, Tamil, Bengali, Gujarati, accented Latin)
   is mapped to Latin. Tokens/components seen in training pairs use a map learned from
   those pairs (e.g. "प्राइवेट" -> "private", "தமிழ்நாடு" -> "tamil nadu"); anything else
   falls back to anyascii romanization.
2. Lowercase, "&" -> "and", dots/apostrophes deleted (l.l.c. -> llc), other punctuation -> space.
3. Canonical forms: legal suffixes in names (limited -> ltd, private -> pvt, corporation -> corp,
   S.A.S -> sas ...), street/address terms (street -> st, road -> rd, avenue/av -> ave, near -> nr,
   rue/r -> rue ...), full state names -> short codes (texas -> tx, haryana -> hr), ordinals (3rd -> 3).
4. Derived fields: name_core (legal forms and filler words removed), name_ns (core without
   spaces, catches "cardiologyphysicians.com"), digit tokens, house number, postal-like codes.

The tables below are hand-written normalization rules; the learned map uses training data only.

    python -m src.normalize --config configs/base.yaml
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from anyascii import anyascii
from rapidfuzz import fuzz

from src.data import ground_truth_pairs, read_source
from src.utils import add_config_arg, cache_dir, get_logger, load_config, timer

log = get_logger("normalize")

# ------------------------------------------------------------------ rule tables

LEGAL = {
    "llc": "llc", "pllc": "pllc", "inc": "inc", "incorporated": "inc",
    "corp": "corp", "corporation": "corp", "co": "co", "company": "co",
    "ltd": "ltd", "limited": "ltd", "pvt": "pvt", "private": "pvt",
    "llp": "llp", "lp": "lp", "plc": "plc", "pc": "pc", "opc": "opc",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "sci": "sci",
    "eurl": "eurl", "snc": "snc", "selarl": "selarl", "cie": "cie", "compagnie": "cie",
}
LEGAL_FORMS = frozenset(LEGAL.values())
NAME_FILLER = frozenset({"the", "and", "of", "india"})

ADDR = {
    "street": "st", "str": "st", "saint": "st", "road": "rd", "avenue": "ave", "av": "ave",
    "drive": "dr", "court": "ct", "place": "pl", "boulevard": "blvd", "bd": "blvd", "boul": "blvd",
    "lane": "ln", "highway": "hwy", "parkway": "pkwy", "circle": "cir", "square": "sq",
    "terrace": "ter", "trail": "trl", "suite": "ste", "apartment": "apt", "floor": "fl", "flr": "fl",
    "building": "bldg", "batiment": "bldg", "bat": "bldg",
    "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    "near": "nr", "opposite": "opp", "sector": "sec", "district": "dist",
    "r": "rue", "all": "allee", "chemin": "ch", "impasse": "imp", "route": "rte",
    "faubourg": "fbg", "residence": "res",
}
ADDR_DROP = frozenset({"no", "number", "num"})

STATES = {
    # US: full name -> USPS code (S1/S2 use codes, S3 spells names out)
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "district of columbia": "dc",
    "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id", "illinois": "il",
    "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
    "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or",
    "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc", "south dakota": "sd",
    "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va",
    "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    # India: full name -> code used by the sources (S1 spells names out, S3 uses codes)
    "maharashtra": "mh", "delhi": "dl", "uttar pradesh": "up", "karnataka": "ka",
    "tamil nadu": "tn", "gujarat": "gj", "west bengal": "wb", "telangana": "tg",
    "haryana": "hr", "rajasthan": "rj", "kerala": "kl", "bihar": "br",
    "madhya pradesh": "mp", "andhra pradesh": "ap", "punjab": "pb", "orissa": "od", "odisha": "od",
    # France: S1 always writes the region, S2/S3 write the region or the department. The
    # departments that occur map to their region's code, like US/India state names above.
    "hauts de france": "hdf", "nord": "hdf", "pas de calais": "hdf",
    "nouvelle aquitaine": "naq", "gironde": "naq",
    "pays de la loire": "pdl", "loire atlantique": "pdl",
}

# ------------------------------------------------------------------ low-level cleaning

_DEL = str.maketrans({".": None, "'": None, "`": None})
_PUNCT = str.maketrans({chr(i): " " for i in range(128) if not (chr(i).isalnum() or chr(i) == " ")})
_ORDINAL = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b")
_STATE_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, STATES), key=len, reverse=True)) + r")\b")
_DOMAIN_RE = re.compile(r"\b([a-z0-9][a-z0-9-]*)\s*\.\s*(?:co\s*\.\s*in|com|net|org|in|co|fr|biz|info|us)\b")
_HAS_DIGIT = re.compile(r"\d")
# "N° 14", "Nº14": the degree/ordinal sign would romanize to "deg" ("ndeg14")
_NUMERO = re.compile(r"\b[nN]\s*[°º]\s*")
# number tokens: "no169" / "ndeg14" / "n14" -> "169" / "14" / "14"; "0010" -> "10"
_NUM_PREFIX = re.compile(r"^(?:ndeg|num|no|n)(?=\d)")
_LEADING_ZEROS = re.compile(r"^0+(?=\d)")
_HOUSE = re.compile(r"^(\d+)")

# learned maps, set per process (see _init_worker)
_TOKEN_MAP: dict = {}
_COMP_MAP: dict = {}


def tokens(text: str) -> list[str]:
    """Lowercase, delete dots/apostrophes, ASCII punctuation -> space, split.

    Non-ASCII characters are untouched, so the same function tokenizes native-script text
    (used when learning the map) and already-romanized text.
    """
    return text.lower().replace("&", " and ").translate(_DEL).translate(_PUNCT).split()


def _romanize_token(tok: str) -> str:
    if tok.isascii():
        return tok
    mapped = _TOKEN_MAP.get(tok.lower())
    return mapped if mapped is not None else anyascii(tok)


def to_latin(text: str, components: bool = False) -> str:
    """Map non-Latin text to Latin: learned component map, learned token map, anyascii."""
    if text.isascii():
        return text
    if components:
        parts = []
        for comp in text.split(","):
            key = " ".join(tokens(comp))
            mapped = _COMP_MAP.get(key)
            parts.append(mapped if mapped is not None else " ".join(_romanize_token(t) for t in comp.split()))
        return ", ".join(parts)
    return " ".join(_romanize_token(t) for t in text.split())


def norm_name(raw: str) -> tuple[str, str, str, bool, bool]:
    """-> (name_n, name_core, name_ns, is_domain, was_non_latin)."""
    non_latin = not raw.isascii()
    s = to_latin(raw).lower()
    is_domain = False
    if "." in s:
        s2 = _DOMAIN_RE.sub(r"\1", s)
        is_domain = s2 != s
        s = s2
    toks = [LEGAL.get(t, t) for t in tokens(s)]
    if len(toks) == 1 and len(toks[0]) >= 8 and toks[0].endswith("com"):
        toks = [toks[0][:-3]]           # "blueengineeringcom"
        is_domain = True
    core = [t for t in toks if t not in LEGAL_FORMS and t not in NAME_FILLER] or toks
    return " ".join(toks), " ".join(core), "".join(core), is_domain, non_latin


def _canon_number(tok: str) -> str:
    """Formatting noise on numbers: glued "no"/"n°" prefixes and leading zeros."""
    if not _HAS_DIGIT.search(tok):
        return tok
    return _LEADING_ZEROS.sub("", _NUM_PREFIX.sub("", tok))


def norm_addr(raw: str) -> tuple[str, str, str, str]:
    """-> (addr_n, addr_nums, house, postal).

    Number tokens are canonicalized (no leading zeros, no "no"/"n°" prefix). The house number
    is the digit run of the first token that starts with a digit ("286b" -> "286"); tokens
    that start with letters ("cs21103", "bp557") are box/lot codes, not house numbers.
    """
    s = " ".join(tokens(to_latin(_NUMERO.sub("no ", raw), components=True)))
    if not s:
        return "", "", "", ""
    s = _ORDINAL.sub(r"\1", s)
    s = _STATE_RE.sub(lambda m: STATES[m.group(1)], s)
    toks = [_canon_number(ADDR.get(t, t)) for t in s.split() if t not in ADDR_DROP]
    toks = [t for t in toks if t and t not in ADDR_DROP]
    nums = [t for t in toks if _HAS_DIGIT.search(t)]
    houses = [_HOUSE.match(t).group(1) for t in nums if t[0].isdigit()]
    house = houses[0] if houses else ""
    postal = " ".join(sorted({t for t in nums if t.isdigit() and len(t) in (5, 6)}))
    return " ".join(toks), " ".join(sorted(set(nums))), house, postal


# ------------------------------------------------------------------ learned map (train pairs only)

def build_learned_map(s1: pd.DataFrame, s23: pd.DataFrame, pairs: pd.DataFrame,
                      min_count: int, min_share: float) -> dict:
    """Learn non-Latin -> Latin mappings from matched training pairs.

    Names: when the record name and its S1 name have the same number of tokens, tokens are
    aligned by position. Addresses: whole comma-separated components are matched to the S1
    component they co-occur with most often (ties broken by similarity to the romanization).
    """
    non_latin = ~(s23["business_name"].map(str.isascii) & s23["business_address"].map(str.isascii))
    rec = s23.loc[non_latin, ["entity_id", "business_name", "business_address"]]
    rec = rec.merge(pairs, left_on="entity_id", right_on="rec_id")
    rec = rec.merge(s1[["entity_id", "business_name", "business_address"]]
                    .rename(columns={"entity_id": "s1_id", "business_name": "s1_name", "business_address": "s1_addr"}),
                    on="s1_id")
    tok_align: dict[str, Counter] = defaultdict(Counter)
    comp_cooc: dict[str, Counter] = defaultdict(Counter)
    comp_count: Counter = Counter()
    for name, s1_name, addr, s1_addr in zip(rec["business_name"], rec["s1_name"],
                                            rec["business_address"], rec["s1_addr"]):
        if not name.isascii():
            rt, st = tokens(name), tokens(s1_name)
            if len(rt) == len(st):
                for a, b in zip(rt, st):
                    if not a.isascii():
                        tok_align[a][b] += 1
        if not addr.isascii():
            s1_comps = {" ".join(tokens(c)) for c in s1_addr.split(",")} - {""}
            for comp in addr.split(","):
                if comp.isascii():
                    continue
                key = " ".join(tokens(comp))
                comp_count[key] += 1
                for b in s1_comps:
                    comp_cooc[key][b] += 1

    token_map = {}
    for a, cnt in tok_align.items():
        total = sum(cnt.values())
        b, n = cnt.most_common(1)[0]
        if total >= min_count and n / total >= min_share:
            token_map[a] = b

    comp_map = {}
    for a, cnt in comp_cooc.items():
        total = comp_count[a]
        if total < min_count:
            continue
        best = max(cnt.values()) / total
        if best < 0.6:
            continue
        roman = anyascii(a).lower()
        close = [b for b, n in cnt.items() if n / total >= 0.8 * best]
        comp_map[a] = max(close, key=lambda b: (fuzz.ratio(roman, b.replace(" ", "")), cnt[b]))
    return {"token_map": token_map, "comp_map": comp_map}


# ------------------------------------------------------------------ batch processing

def _init_worker(maps: dict) -> None:
    global _TOKEN_MAP, _COMP_MAP
    _TOKEN_MAP = maps.get("token_map", {})
    _COMP_MAP = maps.get("comp_map", {})


def _normalize_chunk(args):
    names, addrs = args
    n = list(zip(*map(norm_name, names))) if names else [[]] * 5
    a = list(zip(*map(norm_addr, addrs))) if addrs else [[]] * 4
    return n, a


NAME_FIELDS = ["name_n", "name_core", "name_ns", "is_domain", "name_non_latin"]
ADDR_FIELDS = ["addr_n", "addr_nums", "house", "postal"]


def normalize_frame(df: pd.DataFrame, maps: dict, workers: int, chunk: int) -> pd.DataFrame:
    names, addrs = df["business_name"].tolist(), df["business_address"].tolist()
    jobs = [(names[i:i + chunk], addrs[i:i + chunk]) for i in range(0, len(df), chunk)]
    cols = {k: [] for k in NAME_FIELDS + ADDR_FIELDS}
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(maps,)) as ex:
        for n, a in ex.map(_normalize_chunk, jobs):
            for k, v in zip(NAME_FIELDS, n):
                cols[k].extend(v)
            for k, v in zip(ADDR_FIELDS, a):
                cols[k].extend(v)
    out = pd.DataFrame({"entity_id": df["entity_id"].to_numpy(), "country": df["country"].to_numpy(), **cols})
    out["is_domain"] = out["is_domain"].astype(bool)
    out["name_non_latin"] = out["name_non_latin"].astype(bool)
    out["addr_empty"] = out["addr_n"] == ""
    return out


def learned_map_path(cfg: dict):
    return cache_dir(cfg) / "learned_map.json"


def load_or_build_map(cfg: dict, force: bool = False) -> dict:
    path = learned_map_path(cfg)
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))
    ncfg = cfg["normalize"]
    with timer("learned map", log):
        s1 = read_source(cfg, "train", 1)
        s23 = pd.concat([read_source(cfg, "train", 2), read_source(cfg, "train", 3)], ignore_index=True)
        maps = build_learned_map(s1, s23, ground_truth_pairs(cfg), ncfg["learned_map_min_count"],
                                 ncfg["learned_map_min_share"])
    path.write_text(json.dumps(maps, ensure_ascii=False, indent=0), encoding="utf-8")
    log.info("learned map: %d tokens, %d address components -> %s",
             len(maps["token_map"]), len(maps["comp_map"]), path)
    return maps


def records_path(cfg: dict, split: str, which: str):
    return cache_dir(cfg, split) / f"records_{which}.parquet"


def normalize_split(cfg: dict, split: str, maps: dict, force: bool = False) -> None:
    ncfg = cfg["normalize"]
    for which, sources in (("s1", [1]), ("s23", [2, 3])):
        out = records_path(cfg, split, which)
        if out.exists() and not force:
            log.info("cached: %s", out)
            continue
        with timer(f"normalize {split}/{which}", log):
            parts = []
            for n in sources:
                raw = read_source(cfg, split, n)
                norm = normalize_frame(raw, maps, ncfg["workers"], ncfg["chunk_size"])
                norm.insert(1, "src", np.int8(n))
                parts.append(norm)
            df = pd.concat(parts, ignore_index=True)
            df.to_parquet(out, index=False)
            log.info("wrote %s %s", out, df.shape)


def load_records(cfg: dict, split: str, which: str) -> pd.DataFrame:
    return pd.read_parquet(records_path(cfg, split, which))


def main():
    ap = argparse.ArgumentParser()
    add_config_arg(ap)
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    maps = load_or_build_map(cfg, force=args.force)
    for split in args.splits:
        normalize_split(cfg, split, maps, force=args.force)


if __name__ == "__main__":
    main()
