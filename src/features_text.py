"""Handcrafted text features + TF-IDF -> TruncatedSVD.

    python -m src.features_text --config configs/base.yaml

Writes {cache_dir}/hand_{split}.parquet and {cache_dir}/svd_{split}.npy, and skips work
when they already exist (use --force to rebuild).
"""
from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils import add_config_arg, cache_file, get_logger, load_config, read_split, timer

log = get_logger("features_text")

_NUM = r"(\d+(?:[.,]\d+)?)"

# (unit alternation, kind, factor to canonical unit). Canonical: grams, millilitres, count.
# Order matters: "fl oz" must be tried before "oz".
UNIT_TABLE = [
    (r"fl\.?\s*oz|fluid\s*ounces?", "volume", 29.5735),
    (r"millilit(?:er|re)s?|ml", "volume", 1.0),
    (r"centilit(?:er|re)s?|cl", "volume", 10.0),
    (r"lit(?:er|re)s?|ltrs?|l", "volume", 1000.0),
    (r"gallons?|gal", "volume", 3785.41),
    (r"kilograms?|kgs?", "weight", 1000.0),
    (r"milligrams?|mg", "weight", 0.001),
    (r"grams?|gms?|gr|g", "weight", 1.0),
    (r"ounces?|oz", "weight", 28.3495),
    (r"pounds?|lbs?|lb", "weight", 453.592),
    (r"counts?|ct|pcs|pieces?|tablets?|capsules?|sheets?|bags?|pods?", "count", 1.0),
]
KINDS = ["weight", "volume", "count"]

_UNIT_REGEXES = [
    (re.compile(_NUM + r"\s*-?\s*(?:" + alt + r")\b", re.I), kind, factor)
    for alt, kind, factor in UNIT_TABLE
]
_UNIT_EXACT = [(re.compile(r"^(?:" + alt + r")\.?$", re.I), kind, factor) for alt, kind, factor in UNIT_TABLE]

_PACK_RES = [
    re.compile(r"pack\s*of\s*(\d+)", re.I),
    re.compile(r"(\d+)\s*-?\s*(?:pack|pk)\b", re.I),
    re.compile(r"set\s*of\s*(\d+)", re.I),
]
_IPQ_RE = re.compile(r"(?:item\s*pack\s*quantity|\bipq)\s*[:=]?\s*" + _NUM, re.I)
# 2025 catalog_content ends with "Value: 12.0\nUnit: Ounce"
_VALUE_RE = re.compile(r"\bvalue\s*:\s*" + _NUM, re.I)
_UNITFIELD_RE = re.compile(r"\bunit\s*:\s*([A-Za-z][A-Za-z .]*?)\s*(?:$|\n)", re.I | re.M)
_ITEM_NAME_RE = re.compile(r"item\s*name\s*:\s*", re.I)


def _to_float(s: str | None) -> float:
    if s is None:
        return np.nan
    try:
        return float(s.replace(",", "."))
    except ValueError:
        return np.nan


def normalize_unit(unit: str) -> tuple[str | None, float]:
    """Map a free-text unit ("Fl Oz", "grams", "Count") to (kind, factor-to-canonical)."""
    u = (unit or "").strip().lower()
    for rx, kind, factor in _UNIT_EXACT:
        if rx.match(u):
            return kind, factor
    return None, np.nan


def extract_quantities(text: str) -> dict:
    """First number+unit mention per kind, converted to canonical units."""
    out = {f"qty_{k}": np.nan for k in KINDS}
    out["n_unit_mentions"] = 0
    for rx, kind, factor in _UNIT_REGEXES:
        for m in rx.finditer(text):
            out["n_unit_mentions"] += 1
            key = f"qty_{kind}"
            if np.isnan(out[key]):
                out[key] = _to_float(m.group(1)) * factor

    pack = np.nan
    for rx in _PACK_RES:
        m = rx.search(text)
        if m:
            pack = _to_float(m.group(1))
            break
    out["pack_of"] = pack

    m = _IPQ_RE.search(text)
    out["ipq"] = _to_float(m.group(1)) if m else np.nan

    m = _VALUE_RE.search(text)
    value = _to_float(m.group(1)) if m else np.nan
    m = _UNITFIELD_RE.search(text)
    kind, factor = normalize_unit(m.group(1)) if m else (None, np.nan)
    out["value_field"] = value
    out["value_field_canonical"] = value * factor if kind else np.nan
    out["value_field_kind"] = KINDS.index(kind) if kind else -1

    mult = np.nanmax([out["pack_of"], out["ipq"], 1.0])
    for k in ("weight", "volume"):
        out[f"total_{k}"] = out[f"qty_{k}"] * mult
    return out


def first_token(text: str) -> str:
    """Brand-like first token of the title (after an optional 'Item Name:' prefix)."""
    t = _ITEM_NAME_RE.sub("", text, count=1).strip()
    tok = re.split(r"[\s,|\-–:()]+", t, maxsplit=1)[0] if t else ""
    return tok.lower()


def build_handcrafted(texts: pd.Series) -> pd.DataFrame:
    texts = texts.fillna("").astype(str)
    feats = pd.DataFrame({
        "n_chars": texts.str.len(),
        "n_words": texts.str.split().str.len(),
        "n_digits": texts.str.count(r"\d"),
        "n_upper": texts.str.count(r"[A-Z]"),
        "n_lines": texts.str.count("\n") + 1,
        "n_bullets": texts.str.count(r"(?i)bullet\s*point"),
        "has_description": texts.str.contains(r"(?i)description", regex=True).astype(int),
        "has_ipq": texts.str.contains(r"(?i)item\s*pack\s*quantity|\bipq", regex=True).astype(int),
    })
    feats["upper_ratio"] = feats["n_upper"] / feats["n_chars"].clip(lower=1)
    feats["digit_ratio"] = feats["n_digits"] / feats["n_chars"].clip(lower=1)
    qty = pd.DataFrame([extract_quantities(t) for t in texts], index=texts.index)
    return pd.concat([feats, qty], axis=1)


def encode_brand(train_text: pd.Series, test_text: pd.Series, min_count: int = 3):
    """Frequency-ranked integer code for the first token; rare tokens -> -1. Uses no targets."""
    tr = train_text.fillna("").astype(str).map(first_token)
    te = test_text.fillna("").astype(str).map(first_token)
    counts = tr.value_counts()
    vocab = {tok: i for i, tok in enumerate(counts[counts >= min_count].index)}
    enc = lambda s: s.map(vocab).fillna(-1).astype(int)
    freq = lambda s: s.map(counts).fillna(0).astype(int)
    return (enc(tr), freq(tr)), (enc(te), freq(te))


def build_tfidf_svd(train_text: pd.Series, test_text: pd.Series, cfg: dict):
    tcfg = cfg["text_features"]
    tr = train_text.fillna("").astype(str)
    te = test_text.fillna("").astype(str)
    fit_text = pd.concat([tr, te]) if tcfg.get("fit_on_test") else tr

    word = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=tcfg["word_max_features"],
                           sublinear_tf=True, dtype=np.float32)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                           max_features=tcfg["char_max_features"], sublinear_tf=True, dtype=np.float32)
    word.fit(fit_text)
    char.fit(fit_text)
    X_fit = sparse.hstack([word.transform(fit_text), char.transform(fit_text)]).tocsr()
    svd = TruncatedSVD(n_components=tcfg["svd_components"], random_state=cfg["seed"])
    svd.fit(X_fit)
    transform = lambda s: svd.transform(sparse.hstack([word.transform(s), char.transform(s)]).tocsr())
    log.info("tfidf vocab: word=%d char=%d; svd explained var=%.3f",
             len(word.vocabulary_), len(char.vocabulary_), svd.explained_variance_ratio_.sum())
    return transform(tr).astype(np.float32), transform(te).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    add_config_arg(ap)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    text_col = cfg["columns"]["text"]

    train, test = read_split(cfg, "train"), read_split(cfg, "test")
    cache_file(cfg, "x", "x").parent.mkdir(parents=True, exist_ok=True)

    hand_paths = [cache_file(cfg, "hand", s, "parquet") for s in ("train", "test")]
    if args.force or not all(p.exists() for p in hand_paths):
        with timer("handcrafted", log):
            (b_tr, f_tr), (b_te, f_te) = encode_brand(train[text_col], test[text_col])
            for df, brand, freq, path in ((train, b_tr, f_tr, hand_paths[0]), (test, b_te, f_te, hand_paths[1])):
                h = build_handcrafted(df[text_col])
                h["brand_code"] = brand.values
                h["brand_freq"] = freq.values
                h.astype(np.float32).to_parquet(path, index=False)
                log.info("wrote %s %s", path, h.shape)
    else:
        log.info("handcrafted cache exists, skipping")

    svd_paths = [cache_file(cfg, "svd", s) for s in ("train", "test")]
    if args.force or not all(p.exists() for p in svd_paths):
        with timer("tfidf+svd", log):
            s_tr, s_te = build_tfidf_svd(train[text_col], test[text_col], cfg)
            np.save(svd_paths[0], s_tr)
            np.save(svd_paths[1], s_te)
            log.info("wrote %s %s", svd_paths[0], s_tr.shape)
    else:
        log.info("svd cache exists, skipping")


if __name__ == "__main__":
    main()
