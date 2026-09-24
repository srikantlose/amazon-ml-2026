"""Candidate generation (blocking).

Every S2/S3 record belongs to at most one S1 entity, and matches never cross country labels,
so blocking works record-centric inside each country label (a generic groupby, so unseen
labels such as France are handled the same way):

1. Each text field (name_core, addr_n) becomes a character 3-gram TF-IDF vector, reduced to
   `svd_dim` dimensions with TruncatedSVD and L2-normalized, so inner products approximate
   TF-IDF cosine similarity. IDF and SVD are fit on a sample of the split being processed.
2. For every S2/S3 record, exact top-k S1 records by
      full : 0.5 * cos(name) + 0.5 * cos(address)
      name : cos(name)
      addr : cos(address)          (skipped for records without an address)
   computed on the GPU in fp16 (brute force, chunked).
3. Plus exact name_core matches (skipping names shared by many S1 records).

The union of these pairs is the candidate set the matcher scores (candidate_pairs.tsv).
Cosines for every candidate pair are stored as features.

    python -m src.blocking --config configs/base.yaml --split train [--sample-frac 0.1]
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize as l2_normalize

from src.metrics import blocking_report
from src.normalize import load_records, records_path
from src.data import build_labels
from src.utils import add_config_arg, cache_dir, get_logger, load_config, seed_everything, timer

log = get_logger("blocking")

ALPHABET = " abcdefghijklmnopqrstuvwxyz0123456789"
GRAMS = [a + b + c for a in ALPHABET for b in ALPHABET for c in ALPHABET]
FIELDS = {"name": "name_core", "addr": "addr_n"}
VIEW_BITS = {"full": 1, "name": 2, "addr": 4, "key": 8, "tfull": 16, "tname": 32, "taddr": 64}

_VEC = None
_IDF = None
_COMP = None
_TOK_INDEX = None


def _vectorizer() -> CountVectorizer:
    return CountVectorizer(analyzer="char_wb", ngram_range=(3, 3), vocabulary=GRAMS, dtype=np.float32)


def _init_worker(idf, comp):
    global _VEC, _IDF, _COMP
    _VEC, _IDF, _COMP = _vectorizer(), idf, comp


def _tfidf(counts: sparse.csr_matrix, idf: np.ndarray) -> sparse.csr_matrix:
    x = counts.tocsr(copy=True)
    x.data = 1.0 + np.log(x.data)         # sublinear tf
    x = x @ sparse.diags(idf)
    return l2_normalize(x, copy=False)


def _counts_chunk(texts):
    return _VEC.transform(texts)


def _embed_chunk(texts):
    z = _tfidf(_VEC.transform(texts), _IDF) @ _COMP.T
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    np.divide(z, norms, out=z, where=norms > 0)
    return z.astype(np.float16)


def _chunks(texts: list, size: int):
    return [texts[i:i + size] for i in range(0, len(texts), size)]


def embed_field(texts: list[str], cfg: dict, workers: int) -> np.ndarray:
    """Char 3-gram TF-IDF -> SVD embedding (fp16, unit norm; zero rows for empty text)."""
    bcfg = cfg["blocking"]
    rng = np.random.default_rng(cfg["seed"])
    n_fit = min(bcfg["svd_fit_rows"], len(texts))
    sample = [texts[i] for i in rng.choice(len(texts), n_fit, replace=False)]
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(None, None)) as ex:
        counts = sparse.vstack(list(ex.map(_counts_chunk, _chunks(sample, 50_000)))).tocsr()
    df = np.bincount(counts.indices, minlength=len(GRAMS))
    idf = (np.log((1 + n_fit) / (1 + df)) + 1.0).astype(np.float32)
    svd = TruncatedSVD(n_components=bcfg["svd_dim"], algorithm="randomized", n_iter=5,
                       random_state=cfg["seed"])
    svd.fit(_tfidf(counts, idf))
    log.info("svd(%d) explained variance on sample: %.3f", bcfg["svd_dim"], svd.explained_variance_ratio_.sum())
    comp = svd.components_.astype(np.float32)
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(idf, comp)) as ex:
        return np.vstack(list(ex.map(_embed_chunk, _chunks(texts, 100_000))))


def vectors(cfg: dict, split: str, s1: pd.DataFrame, s23: pd.DataFrame, force: bool = False) -> dict:
    """{(which, field): fp16 array}, cached as .npy per split."""
    out = {}
    d = cache_dir(cfg, split)
    for field, col in FIELDS.items():
        paths = {w: d / f"vec_{field}_{w}.npy" for w in ("s1", "s23")}
        if all(p.exists() for p in paths.values()) and not force:
            for w, p in paths.items():
                out[(w, field)] = np.load(p, mmap_mode="r")
            continue
        with timer(f"embed {split}/{field}", log):
            texts = s1[col].tolist() + s23[col].tolist()
            z = embed_field(texts, cfg, cfg["blocking"]["workers"])
            del texts
        np.save(paths["s1"], z[:len(s1)])
        np.save(paths["s23"], z[len(s1):])
        del z
        for w, p in paths.items():
            out[(w, field)] = np.load(p, mmap_mode="r")
    return out


@torch.no_grad()
def gpu_topk(qn: np.ndarray, qa: np.ndarray, q_has_name: np.ndarray, q_has_addr: np.ndarray,
             In: torch.Tensor, Ia: torch.Tensor, ks: dict, batch: int) -> dict:
    """Top-k index rows for each query row, per view. Returns {view: (q, idx, score)} arrays.

    Queries with an empty field are skipped for that field's single-field view.
    """
    dev = In.device
    n, N = len(qn), In.shape[0]
    res = {v: ([], [], []) for v in ("full", "name", "addr") if ks.get(v, 0) > 0}
    if ks.get("name", 0) == 0 and ks.get("addr", 0) == 0 and ks.get("full", 0) > 0:
        # fused path: [n, a] . [n', a'] = cos_name + cos_addr in one matmul and one top-k
        k = min(ks["full"], N)
        I = torch.cat([In, Ia], dim=1)
        for s in range(0, n, 2 * batch):
            e = min(s + 2 * batch, n)
            q = torch.from_numpy(np.hstack([qn[s:e], qa[s:e]])).to(dev, non_blocking=True)
            val, idx = (q @ I.T).topk(k, dim=1)
            res["full"][0].append(np.repeat(np.arange(s, e), k))
            res["full"][1].append(idx.cpu().numpy().ravel())
            res["full"][2].append(0.5 * val.float().cpu().numpy().ravel())
        del I
        return {v: tuple(np.concatenate(a) for a in parts) for v, parts in res.items()}
    for s in range(0, n, batch):
        e = min(s + batch, n)
        q_rows = np.arange(s, e)
        tn = torch.from_numpy(np.ascontiguousarray(qn[s:e])).to(dev, non_blocking=True)
        ta = torch.from_numpy(np.ascontiguousarray(qa[s:e])).to(dev, non_blocking=True)
        sn = tn @ In.T
        sa = ta @ Ia.T
        for view, sim, keep in (("name", sn, q_has_name[s:e]), ("addr", sa, q_has_addr[s:e])):
            k = min(ks.get(view, 0), N)
            if k <= 0:
                continue
            val, idx = sim.topk(k, dim=1)
            res[view][0].append(np.repeat(q_rows[keep], k))
            res[view][1].append(idx.cpu().numpy()[keep].ravel())
            res[view][2].append(val.float().cpu().numpy()[keep].ravel())
        k = min(ks.get("full", 0), N)
        if k > 0:
            sn.add_(sa).mul_(0.5)
            val, idx = sn.topk(k, dim=1)
            res["full"][0].append(np.repeat(q_rows, k))
            res["full"][1].append(idx.cpu().numpy().ravel())
            res["full"][2].append(val.float().cpu().numpy().ravel())
        del sn, sa
    return {v: tuple(np.concatenate(a) if a else np.empty(0) for a in parts) for v, parts in res.items()}


@torch.no_grad()
def pair_cosines(s1_local: np.ndarray, q_local: np.ndarray, In: torch.Tensor, Ia: torch.Tensor,
                 qn: np.ndarray, qa: np.ndarray, chunk: int = 2_000_000) -> tuple[np.ndarray, np.ndarray]:
    dev = In.device
    cn, ca = np.empty(len(s1_local), np.float32), np.empty(len(s1_local), np.float32)
    for s in range(0, len(s1_local), chunk):
        e = min(s + chunk, len(s1_local))
        si = torch.from_numpy(s1_local[s:e].astype(np.int64)).to(dev)
        qi = q_local[s:e]
        vn = torch.from_numpy(np.asarray(qn[qi])).to(dev)
        va = torch.from_numpy(np.asarray(qa[qi])).to(dev)
        cn[s:e] = (In[si].float() * vn.float()).sum(1).cpu().numpy()
        ca[s:e] = (Ia[si].float() * va.float()).sum(1).cpu().numpy()
    return cn, ca


# ------------------------------------------------------------------ word-token inverted index

def token_tfidf(texts: pd.Series, n_index: int, max_df_index: int):
    """Binary word-token TF-IDF (IDF over all given rows, rows L2-normalized).

    The first n_index rows are the index side (S1); columns whose document frequency among
    them exceeds max_df_index are dropped from the retrieval index so posting lists stay short:
    very common tokens ("services", "road", city names) cost a lot and rarely identify a business.
    Returns (query rows, transposed retrieval index, full index rows for exact pair cosines).
    """
    lists = texts.str.split()
    lens = lists.str.len().to_numpy()
    flat = lists.explode().dropna()
    codes, uniq = pd.factorize(flat.to_numpy())
    rows = np.repeat(np.arange(len(texts)), lens)
    X = sparse.csr_matrix((np.ones(len(codes), np.float32), (rows, codes)), shape=(len(texts), len(uniq)))
    X.sum_duplicates()
    X.data[:] = 1.0
    df = np.bincount(X.indices, minlength=X.shape[1])
    idf = np.log((1 + X.shape[0]) / (1 + df)).astype(np.float32) + 1.0
    X.data = idf[X.indices]
    X = l2_normalize(X, copy=False)
    full_idx, qry = X[:n_index].tocsr(), X[n_index:].tocsr()
    df_index = np.bincount(full_idx.indices, minlength=X.shape[1])
    keep_col = (df_index <= max_df_index).astype(np.float32)
    idx = (full_idx @ sparse.diags(keep_col)).tocsr()
    idx.eliminate_zeros()
    return qry, idx.T.tocsr(), full_idx


def rowwise_dot(A: sparse.csr_matrix, ia: np.ndarray, B: sparse.csr_matrix, ib: np.ndarray,
                chunk: int = 2_000_000) -> np.ndarray:
    """Exact <A[ia[i]], B[ib[i]]> for aligned index arrays."""
    out = np.empty(len(ia), np.float32)
    for s in range(0, len(ia), chunk):
        e = min(s + chunk, len(ia))
        out[s:e] = np.asarray(A[ia[s:e]].multiply(B[ib[s:e]]).sum(axis=1)).ravel()
    return out


def _init_tok_worker(index_t):
    global _TOK_INDEX
    _TOK_INDEX = index_t


def _topk_rows(P: sparse.csr_matrix, k: int, min_score: float = 0.0):
    """Top-k columns per row of a CSR matrix -> (row, col, value) arrays."""
    rows = np.repeat(np.arange(P.shape[0], dtype=np.int64), np.diff(P.indptr))
    data, cols = P.data, P.indices
    if min_score > 0:
        m = data >= min_score
        rows, data, cols = rows[m], data[m], cols[m]
    # sort by (row asc, value desc) with one float64 key; values are cosines in [0, 2]
    order = np.argsort(rows * 4.0 - data, kind="stable")
    r = rows[order]
    start = np.flatnonzero(np.r_[True, r[1:] != r[:-1]])
    pos = np.arange(len(r)) - np.repeat(start, np.diff(np.r_[start, len(r)]))
    keep = order[pos < k]
    return rows[keep], cols[keep], data[keep]


def _tok_chunk(args):
    qn, qa, k_full, k_name, k_addr, offset, min_score = args
    tn, ta = _TOK_INDEX
    pn = (qn @ tn).tocsr()
    pa = (qa @ ta).tocsr()
    out = {}
    for view, P, k in (("tname", pn, k_name), ("taddr", pa, k_addr), ("tfull", (pn + pa).tocsr(), k_full)):
        if k > 0:
            r, c, v = _topk_rows(P, k, min_score)
            out[view] = (r + offset, c, v)
    return out


def token_matrices(s1c: pd.DataFrame, s23c: pd.DataFrame, max_df: int) -> dict:
    tn = token_tfidf(pd.concat([s1c["name_core"], s23c["name_core"]], ignore_index=True), len(s1c), max_df)
    ta = token_tfidf(pd.concat([s1c["addr_n"], s23c["addr_n"]], ignore_index=True), len(s1c), max_df)
    return {"name": tn, "addr": ta}


def token_topk(tok: dict, n_queries: int, ks: dict, workers: int, chunk: int = 10_000,
               min_score: float = 0.0) -> dict:
    """Exact word-token TF-IDF top-k S1 (local rows) per record (local rows), per token view."""
    tn_q, tn_idx, _ = tok["name"]
    ta_q, ta_idx, _ = tok["addr"]
    jobs = [(tn_q[s:s + chunk], ta_q[s:s + chunk], ks.get("tfull", 0), ks.get("tname", 0), ks.get("taddr", 0), s,
             min_score) for s in range(0, n_queries, chunk)]
    res = {v: ([], [], []) for v in ("tfull", "tname", "taddr") if ks.get(v, 0) > 0}
    with ProcessPoolExecutor(workers, initializer=_init_tok_worker, initargs=((tn_idx, ta_idx),)) as ex:
        for out in ex.map(_tok_chunk, jobs):
            for v, (r, c, s) in out.items():
                res[v][0].append(r)
                res[v][1].append(c)
                res[v][2].append(s)
    return {v: tuple(np.concatenate(a) for a in parts) for v, parts in res.items()}


def name_key_pairs(s1c: pd.DataFrame, s23c: pd.DataFrame, max_group: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact name_core matches; names shared by more than max_group S1 records are skipped."""
    sizes = s1c.groupby("name_core")["s1"].transform("size")
    left = s1c.loc[sizes <= max_group, ["s1", "name_core"]]
    m = s23c[["rec", "name_core"]].merge(left, on="name_core")
    return m["s1"].to_numpy(np.int64), m["rec"].to_numpy(np.int64)


def generate(cfg: dict, split: str, sample_frac: float = 1.0, force: bool = False) -> pd.DataFrame:
    bcfg = cfg["blocking"]
    # only the columns blocking needs: the full record tables cost several GB at this scale
    s1 = pd.read_parquet(records_path(cfg, split, "s1"), columns=["country", "name_core", "addr_n"])
    s23 = pd.read_parquet(records_path(cfg, split, "s23"), columns=["country", "name_core", "addr_n", "addr_empty"])
    vec = vectors(cfg, split, s1, s23, force=force)
    s1["s1"] = np.arange(len(s1))
    s23["rec"] = np.arange(len(s23))
    rng = np.random.default_rng(cfg["seed"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ks = bcfg["views"]

    parts = []
    for country in sorted(set(s1["country"]) | set(s23["country"])):
        s1c = s1[s1["country"] == country]
        s23c = s23[s23["country"] == country]
        if sample_frac < 1.0:
            s23c = s23c[rng.random(len(s23c)) < sample_frac]
        if len(s1c) == 0 or len(s23c) == 0:
            continue
        with timer(f"block {split}/{country}: {len(s23c):,} records x {len(s1c):,} S1", log):
            s1_idx = s1c["s1"].to_numpy()
            q_idx = s23c["rec"].to_numpy()
            In = torch.from_numpy(np.asarray(vec[("s1", "name")][s1_idx])).to(dev)
            Ia = torch.from_numpy(np.asarray(vec[("s1", "addr")][s1_idx])).to(dev)
            qn = np.asarray(vec[("s23", "name")][q_idx])
            qa = np.asarray(vec[("s23", "addr")][q_idx])
            got = gpu_topk(qn, qa, (s23c["name_core"] != "").to_numpy(), ~s23c["addr_empty"].to_numpy(),
                           In, Ia, ks, bcfg["query_batch"])
            tok = token_matrices(s1c, s23c, bcfg["token_max_df"])
            got.update(token_topk(tok, len(s23c), ks, bcfg.get("token_workers", bcfg["workers"]),
                                  bcfg.get("token_chunk", 10_000), bcfg.get("token_min_score", 0.0)))

            # union of views, with a bitmask saying which views produced each pair
            s1_local = [got[v][1] for v in got]
            q_local = [got[v][0] for v in got]
            bits = [np.full(len(got[v][0]), VIEW_BITS[v], np.uint8) for v in got]
            ks1, krec = name_key_pairs(s1c, s23c, bcfg["name_key_max_group"])
            if len(ks1):
                pos_s1 = pd.Index(s1_idx).get_indexer(ks1)
                pos_q = pd.Index(q_idx).get_indexer(krec)
                s1_local.append(pos_s1)
                q_local.append(pos_q)
                bits.append(np.full(len(ks1), VIEW_BITS["key"], np.uint8))
            a = np.concatenate(s1_local).astype(np.int64)
            b = np.concatenate(q_local).astype(np.int64)
            m = np.concatenate(bits)
            del got, s1_local, q_local, bits
            key = b * len(s1_idx) + a
            del a, b
            order = np.argsort(key, kind="stable")
            key, m = key[order], m[order]
            first = np.r_[True, key[1:] != key[:-1]]
            starts = np.flatnonzero(first)
            views = np.bitwise_or.reduceat(m, starts)
            ukey = key[starts]
            q_u, s1_u = ukey // len(s1_idx), ukey % len(s1_idx)
            cn, ca = pair_cosines(s1_u, q_u, In, Ia, qn, qa)
            tcn = rowwise_dot(tok["name"][0], q_u, tok["name"][2], s1_u)
            tca = rowwise_dot(tok["addr"][0], q_u, tok["addr"][2], s1_u)
            parts.append(pd.DataFrame({
                "s1": s1_idx[s1_u].astype(np.int32), "rec": q_idx[q_u].astype(np.int32),
                "views": views, "cos_name": cn, "cos_addr": ca, "tcos_name": tcn, "tcos_addr": tca,
            }))
            del tok
            del In, Ia
            if dev == "cuda":
                torch.cuda.empty_cache()
    cands = pd.concat(parts, ignore_index=True)
    log.info("%s: %s candidate pairs", split, f"{len(cands):,}")
    return cands


def candidates_path(cfg: dict, split: str):
    """Union of all retrieval views (input to stage-1 pruning)."""
    return cache_dir(cfg, split) / "cands.parquet"


def pruned_path(cfg: dict, split: str):
    """Stage-1 pruned candidates: the pairs the matcher scores (= candidate_pairs.tsv)."""
    return cache_dir(cfg, split) / "cands_pruned.parquet"


def report(cfg: dict, split: str, cands: pd.DataFrame, sample_frac: float = 1.0) -> None:
    """Recall by view and for the union (train only)."""
    s1 = load_records(cfg, split, "s1")
    s23 = load_records(cfg, split, "s23")
    true_s1, n_true = build_labels(cfg, s1["entity_id"].to_numpy(), s23["entity_id"].to_numpy())
    if sample_frac < 1.0:
        # queries were subsampled: recall over the ground-truth pairs whose record was queried
        queried = np.zeros(len(s23), bool)
        queried[cands["rec"].to_numpy()] = True
        hit = true_s1[cands["rec"].to_numpy()] == cands["s1"].to_numpy()
        n_gt = int(((true_s1 >= 0) & queried).sum())
        log.info("sampled recall (lower bound on queried records): union %.4f over %d GT pairs; pairs/S1 ~ %.1f",
                 hit.sum() / max(n_gt, 1), n_gt, len(cands) / sample_frac / len(s1))
        for view, bit in VIEW_BITS.items():
            sel = (cands["views"].to_numpy() & bit) > 0
            log.info("  view %-4s recall %.4f  pairs %s", view, (hit & sel).sum() / max(n_gt, 1), f"{sel.sum():,}")
        return
    rep = blocking_report(len(s1), true_s1, n_true, cands["s1"].to_numpy(), cands["rec"].to_numpy())
    log.info("union: %s", {k: round(v, 4) for k, v in rep.items()})
    for view, bit in VIEW_BITS.items():
        sel = (cands["views"].to_numpy() & bit) > 0
        r = blocking_report(len(s1), true_s1, n_true, cands["s1"].to_numpy()[sel], cands["rec"].to_numpy()[sel])
        log.info("  view %-4s: recall %.4f  oracle %.4f  pairs %s", view, r["pair_recall"], r["oracle_f05"],
                 f"{r['pairs']:,}")


def main():
    ap = argparse.ArgumentParser()
    add_config_arg(ap)
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--sample-frac", type=float, default=None)
    ap.add_argument("--force", action="store_true", help="recompute vectors too")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    frac = args.sample_frac if args.sample_frac is not None else cfg["blocking"]["sample_frac"]
    cands = generate(cfg, args.split, sample_frac=frac, force=args.force)
    if frac >= 1.0:
        cands.to_parquet(candidates_path(cfg, args.split), index=False)
        log.info("wrote %s", candidates_path(cfg, args.split))
    if args.split == "train":
        report(cfg, args.split, cands, frac)


if __name__ == "__main__":
    main()
