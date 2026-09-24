"""Stage-1 pruning of the blocking union.

Blocking keeps recall high by unioning several retrieval views, which leaves ~20 S1 candidates
per S2/S3 record. A light LightGBM ranker over the retrieval scores alone (char/token cosines,
which views fired, ranks within the record and within the S1) keeps the top `top_n` S1 per
record with probability >= `min_prob`. The pruned set is what the matcher scores, i.e. the
content of candidate_pairs.tsv. Train predictions are out-of-fold (same S1-grouped folds as
the matcher); test uses the average of the fold models.

The union has 200M+ pairs, so group statistics are kept compact (uint8 ranks, float16 gaps)
and feature matrices are assembled chunk by chunk.

    python -m src.prune --config configs/base.yaml --split train
    python -m src.prune --config configs/base.yaml --split test
"""
from __future__ import annotations

import argparse
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.blocking import VIEW_BITS, candidates_path, pruned_path
from src.data import build_labels
from src.features import _group_stats, country_segments
from src.metrics import blocking_report
from src.normalize import load_records
from src.train_matcher import s1_hash
from src.utils import add_config_arg, cache_dir, get_logger, load_config, seed_everything, timer

log = get_logger("prune")

PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_child_samples=200,
              feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, n_estimators=400, verbose=-1)
# (group, score, with_gap)
GROUP_SPECS = [("rec", "cos_full", True), ("rec", "tcos_full", True), ("rec", "cos_name", False),
               ("rec", "tcos_name", False), ("rec", "cos_addr", False), ("rec", "tcos_addr", False),
               ("s1", "cos_full", True), ("s1", "tcos_full", True)]
CHUNK = 20_000_000


def _score(cands: pd.DataFrame, name: str, rows=None) -> np.ndarray:
    get = (lambda c: cands[c].to_numpy(np.float32)) if rows is None else (lambda c: cands[c].to_numpy(np.float32)[rows])
    if name == "cos_full":
        return 0.5 * (get("cos_name") + get("cos_addr"))
    if name == "tcos_full":
        return 0.5 * (get("tcos_name") + get("tcos_addr"))
    return get(name)


def group_stats(cands: pd.DataFrame, s1_country: np.ndarray) -> dict:
    """Compact per-pair group statistics over the whole candidate table, one country at a time."""
    n = len(cands)
    s1_all = cands["s1"].to_numpy()
    out = {}
    for tag, sc, with_gap in GROUP_SPECS:
        out.setdefault(f"{tag}_rank_{sc}", np.empty(n, np.uint8))
        if with_gap:
            out.setdefault(f"{tag}_gap_{sc}", np.empty(n, np.float16))
        out.setdefault(f"{tag}_n", np.empty(n, np.uint16))
    for seg in country_segments(s1_all, s1_country):
        part = cands.iloc[seg]
        keys = {"rec": part["rec"].to_numpy(np.int32), "s1": part["s1"].to_numpy(np.int32)}
        for tag, sc, with_gap in GROUP_SPECS:
            score = _score(part, sc)
            rank, gmax, _, gsize = _group_stats(keys[tag], score)
            out[f"{tag}_rank_{sc}"][seg] = np.minimum(rank, 255)
            if with_gap:
                out[f"{tag}_gap_{sc}"][seg] = gmax - score
            out[f"{tag}_n"][seg] = np.minimum(gsize, 65535)
            del rank, gmax, gsize, score
    return out


def assemble(cands: pd.DataFrame, stats: dict, s23: pd.DataFrame, rows: np.ndarray) -> pd.DataFrame:
    f = {}
    for c in ("cos_name", "cos_addr", "tcos_name", "tcos_addr", "cos_full", "tcos_full"):
        f[c] = _score(cands, c, rows)
    views = cands["views"].to_numpy()[rows]
    for v, bit in VIEW_BITS.items():
        f[f"view_{v}"] = ((views & bit) > 0).astype(np.float32)
    f["n_views"] = sum(f[f"view_{v}"] for v in VIEW_BITS)
    for k, arr in stats.items():
        f[k] = arr[rows].astype(np.float32)
    rec = cands["rec"].to_numpy(np.int64)[rows]
    f["r_addr_empty"] = s23["addr_empty"].to_numpy(np.float32)[rec]
    f["r_non_latin"] = s23["name_non_latin"].to_numpy(np.float32)[rec]
    return pd.DataFrame(f)


def predict_chunked(boosters, cands, stats, s23, rows_by_model=None) -> np.ndarray:
    """rows_by_model: optional per-pair model index (OOF); otherwise average all models."""
    prob = np.zeros(len(cands), np.float32)
    for s in range(0, len(cands), CHUNK):
        rows = np.arange(s, min(s + CHUNK, len(cands)))
        X = assemble(cands, stats, s23, rows)
        if rows_by_model is None:
            prob[rows] = np.mean([b.predict(X) for b in boosters], axis=0)
        else:
            which = rows_by_model[rows]
            for i, b in enumerate(boosters):
                m = which == i
                if m.any():
                    prob[rows[m]] = b.predict(X[m])
    return prob


def keep_mask(rec: np.ndarray, prob: np.ndarray, top_n: int, min_prob: float, rank=None) -> np.ndarray:
    if rank is None:
        rank, _, _, _ = _group_stats(rec, prob)
    return (rank < top_n) & (prob >= min_prob)


def train(cfg: dict) -> None:
    pcfg = cfg["prune"]
    mdir = cache_dir(cfg, "prune")
    cands = pd.read_parquet(candidates_path(cfg, "train"))
    s1_rec = load_records(cfg, "train", "s1")
    s23 = load_records(cfg, "train", "s23")[["entity_id", "addr_empty", "name_non_latin"]]
    true_s1, n_true = build_labels(cfg, s1_rec["entity_id"].to_numpy(), s23["entity_id"].to_numpy())
    s1 = cands["s1"].to_numpy(np.int64)
    rec = cands["rec"].to_numpy(np.int64)
    y = (true_s1[rec] == s1).astype(np.uint8)
    with timer(f"group stats over {len(cands):,} pairs", log):
        stats = group_stats(cands, s1_rec["country"].to_numpy())
    h = s1_hash(s1_rec["entity_id"].to_numpy())
    n_folds = cfg["model"]["n_folds"]
    fold = (h % n_folds)[s1].astype(np.int8)
    u = ((h // n_folds) % 10_000 / 10_000.0).astype(np.float32)[s1]
    boosters = []
    for f in range(n_folds):
        tr = np.flatnonzero((fold != f) & (u < pcfg["train_frac"]))
        with timer(f"stage-1 fold {f} on {len(tr):,} pairs", log):
            m = lgb.LGBMClassifier(**PARAMS, random_state=cfg["seed"] + f, n_jobs=-1)
            m.fit(assemble(cands, stats, s23, tr), y[tr])
            m.booster_.save_model(str(mdir / f"stage1_fold{f}.txt"))
            boosters.append(m.booster_)
    with timer("stage-1 OOF prediction", log):
        oof = predict_chunked(boosters, cands, stats, s23, rows_by_model=fold)
    del stats
    full = blocking_report(len(s1_rec), true_s1, n_true, s1, rec)
    log.info("before pruning: recall %.4f oracle %.4f pairs %s", full["pair_recall"], full["oracle_f05"],
             f"{full['pairs']:,}")
    rank = _group_stats(rec, oof)[0]
    for top_n in (3, 4, 5, 6, 8):
        for min_prob in (0.001, 0.01):
            k = keep_mask(rec, oof, top_n, min_prob, rank)
            r = blocking_report(len(s1_rec), true_s1, n_true, s1[k], rec[k])
            log.info("top_n=%d min_prob=%.3f: recall %.4f oracle %.4f pairs %s (%.2f/record)", top_n, min_prob,
                     r["pair_recall"], r["oracle_f05"], f"{r['pairs']:,}", r["pairs"] / len(s23))
    k = keep_mask(rec, oof, pcfg["top_n"], pcfg["min_prob"], rank)
    del rank
    out = cands[k].reset_index(drop=True)
    out["p1"] = oof[k]
    out.to_parquet(pruned_path(cfg, "train"), index=False)
    (mdir / "meta.json").write_text(json.dumps(pcfg, indent=2), encoding="utf-8")
    log.info("kept %s of %s pairs (top_n=%d, min_prob=%.3f) -> %s", f"{k.sum():,}", f"{len(k):,}",
             pcfg["top_n"], pcfg["min_prob"], pruned_path(cfg, "train"))


def apply_test(cfg: dict) -> None:
    pcfg = cfg["prune"]
    mdir = cache_dir(cfg, "prune")
    cands = pd.read_parquet(candidates_path(cfg, "test"))
    s23 = load_records(cfg, "test", "s23")[["entity_id", "addr_empty", "name_non_latin"]]
    s1_country = load_records(cfg, "test", "s1")["country"].to_numpy()
    with timer(f"group stats over {len(cands):,} pairs", log):
        stats = group_stats(cands, s1_country)
    boosters = [lgb.Booster(model_file=str(p)) for p in sorted(mdir.glob("stage1_fold*.txt"))]
    with timer("stage-1 test prediction", log):
        prob = predict_chunked(boosters, cands, stats, s23)
    del stats
    k = keep_mask(cands["rec"].to_numpy(np.int64), prob, pcfg["top_n"], pcfg["min_prob"])
    out = cands[k].reset_index(drop=True)
    out["p1"] = prob[k]
    out.to_parquet(pruned_path(cfg, "test"), index=False)
    log.info("test: kept %s of %s pairs (%.2f per record)", f"{k.sum():,}", f"{len(k):,}", k.sum() / len(s23))


def main():
    ap = argparse.ArgumentParser()
    add_config_arg(ap)
    ap.add_argument("--split", required=True, choices=["train", "test"])
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    train(cfg) if args.split == "train" else apply_test(cfg)


if __name__ == "__main__":
    main()
