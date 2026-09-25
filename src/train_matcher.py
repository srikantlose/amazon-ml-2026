"""LightGBM pair classifier with out-of-fold predictions for every training candidate pair.

Folds are assigned per S1 entity (crc32 of the id), so all candidates of an S1 share a fold.
Each fold model is fit on a hash-sampled subset of the other folds' S1 entities (all of their
candidate pairs, so negatives keep their natural mix) and early-stopped on a disjoint slice.
Predicting every held-out fold gives OOF probabilities for all pairs, which the decision step
needs because records choose between competing S1 entities across folds.

    python -m src.train_matcher --config configs/base.yaml [--name baseline] [--notes "..."]
"""
from __future__ import annotations

import argparse
import json
import zlib

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.blocking import pruned_path
from src.data import build_labels
from src.decide import tune
from src.features import features_dir, load_features, load_part, part_columns
from src.normalize import records_path
from src.utils import (add_config_arg, cache_dir, get_logger, load_config, log_experiment, make_run_id,
                       seed_everything, timer)

log = get_logger("train_matcher")


def s1_hash(ids: np.ndarray) -> np.ndarray:
    return np.fromiter((zlib.crc32(s.encode()) for s in ids), dtype=np.int64, count=len(ids))


def run_dir(cfg: dict, run_id: str):
    d = cache_dir(cfg, "runs") / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def main():
    ap = argparse.ArgumentParser()
    add_config_arg(ap)
    ap.add_argument("--name", default="")
    ap.add_argument("--notes", default="")
    ap.add_argument("--drop", nargs="*", default=[], help="feature columns to leave out")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="additional pair-aligned feature folders, e.g. features_s3_<run> from src.refine")
    args = ap.parse_args()
    cfg = load_config(args.config)
    mcfg = cfg["model"]
    seed_everything(cfg["seed"])
    run_id = make_run_id("lgb" + (f"_{args.name}" if args.name else ""))
    out = run_dir(cfg, run_id)

    cands = pd.read_parquet(pruned_path(cfg, "train"), columns=["s1", "rec"])
    s1 = cands["s1"].to_numpy(np.int64)
    rec = cands["rec"].to_numpy(np.int64)
    s1_rec = pd.read_parquet(records_path(cfg, "train", "s1"), columns=["entity_id", "country"])
    rec_ids = pd.read_parquet(records_path(cfg, "train", "s23"), columns=["entity_id"])["entity_id"].to_numpy()
    s1_ids = s1_rec["entity_id"].to_numpy()
    true_s1, n_true = build_labels(cfg, s1_ids, rec_ids)
    del rec_ids
    y = (true_s1[rec] == s1).astype(np.uint8)
    log.info("%s pairs, %.4f positive; blocking recall %.4f", f"{len(y):,}", y.mean(), y.sum() / n_true.sum())

    h = s1_hash(s1_ids)
    n_folds = mcfg["n_folds"]
    fold_s1 = h % n_folds
    u_s1 = (h // n_folds) % 10_000 / 10_000.0
    fold, u = fold_s1[s1], u_s1[s1]
    need = np.flatnonzero(u < mcfg["train_frac"] + mcfg["valid_frac"])
    drop = set(args.drop)
    cols_by_dir = {d: [c for c in part_columns(cfg, "train", d) if c not in drop]
                   for d in ["features", *args.extra]}
    cols = [c for cs in cols_by_dir.values() for c in cs]
    with timer(f"load {len(need):,} sampled rows", log):
        X_need = pd.concat([load_features(cfg, "train", rows=need, columns=cs, subdir=d)
                            for d, cs in cols_by_dir.items()], axis=1)

    models = []
    for f in range(n_folds):
        sel_tr = (fold[need] != f) & (u[need] < mcfg["train_frac"])
        sel_va = (fold[need] != f) & (u[need] >= mcfg["train_frac"])
        Xtr, ytr = X_need[sel_tr], y[need][sel_tr]
        Xva, yva = X_need[sel_va], y[need][sel_va]
        with timer(f"fold {f}: fit on {len(ytr):,} pairs", log):
            model = lgb.LGBMClassifier(**mcfg["lightgbm"], random_state=cfg["seed"] + f, n_jobs=-1)
            model.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="binary_logloss",
                      callbacks=[lgb.early_stopping(mcfg["early_stopping_rounds"], verbose=False),
                                 lgb.log_evaluation(200)])
        log.info("fold %d best_iter=%s", f, model.best_iteration_)
        model.booster_.save_model(str(out / f"model_fold{f}.txt"), num_iteration=model.best_iteration_)
        models.append(model.booster_)
    del X_need

    oof = np.zeros(len(y), dtype=np.float32)
    chunk = cfg["features"]["chunk_pairs"]
    with timer("OOF prediction", log):
        for i, _ in enumerate(sorted(features_dir(cfg, "train").glob("part_*.parquet"))):
            X = load_part(cfg, "train", i, cols_by_dir)
            lo = i * chunk
            fp = fold[lo:lo + len(X)]
            for f, booster in enumerate(models):
                m = fp == f
                if m.any():
                    oof[lo:lo + len(X)][m] = booster.predict(X[m], num_iteration=booster.best_iteration)
    np.save(out / "oof.npy", oof)

    countries = s1_rec["country"].to_numpy()
    groups = {c: countries == c for c in sorted(set(countries))}
    best, table = tune(s1, rec, oof, true_s1, n_true, cfg["decision"]["thresholds"], cfg["decision"]["margins"],
                       groups=groups)
    log.info("decision grid (top 8):\n%s", table.head(8).to_string(index=False))
    log.info("OOF macro F0.5 = %.5f at threshold=%.2f margin=%.2f", best["f05"], best["threshold"], best["margin"])

    imp = pd.Series(np.mean([m.feature_importance("gain") for m in models], axis=0), index=cols)
    imp.sort_values(ascending=False).to_csv(out / "importance.csv")
    log.info("top features:\n%s", imp.sort_values(ascending=False).head(15).round(0).to_string())
    meta = {"run_id": run_id, "features": cols, "feature_dirs": cols_by_dir, "decision": best, "n_folds": n_folds,
            "params": mcfg, "best_iterations": [m.best_iteration for m in models],
            "per_country": {k: float(v) for k, v in table.iloc[0].items() if k.startswith("f05_")}}
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    table.to_csv(out / "decision_grid.csv", index=False)
    log_experiment(cfg, run_id, "lightgbm", f"{len(cols)} features", best["f05"],
                   notes=f"t={best['threshold']} m={best['margin']} {meta['per_country']} {args.notes}".strip())
    print(run_id)


if __name__ == "__main__":
    main()
