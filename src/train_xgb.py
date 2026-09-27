"""XGBoost (GPU) pair classifier on the same folds and feature folders as src/train_matcher.py.

A second model family for averaging with the LightGBM matchers: same S1-grouped folds (crc32), same
train/valid sampling, out-of-fold probabilities for every train pair and fold-averaged test
probabilities, saved in a run folder like train_matcher's (oof.npy, test_prob.npy, meta.json).

    python -m src.train_xgb --config configs/base.yaml configs/stage3_big.yaml \
        --extra features_la features_rel features_s3_<stage-2 run> --drop s1_n_cands --name stage3xgb
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import xgboost as xgb

from src.blocking import pruned_path
from src.data import build_labels
from src.decide import tune
from src.features import features_dir, load_features, load_part, part_columns
from src.normalize import records_path
from src.train_matcher import run_dir, s1_hash
from src.utils import add_config_arg, get_logger, load_config, log_experiment, make_run_id, seed_everything, timer

log = get_logger("train_xgb")

PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist", "device": "cuda",
          "max_depth": 0, "grow_policy": "lossguide", "max_leaves": 255, "eta": 0.05, "subsample": 0.8,
          "colsample_bytree": 0.7, "min_child_weight": 20, "lambda": 1.0, "max_bin": 256}


def main():
    ap = argparse.ArgumentParser()
    add_config_arg(ap)
    ap.add_argument("--name", default="xgb")
    ap.add_argument("--drop", nargs="*", default=[])
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--rounds", type=int, default=6000)
    args = ap.parse_args()
    cfg = load_config(args.config)
    mcfg = cfg["model"]
    seed_everything(cfg["seed"])
    run_id = make_run_id(args.name)
    out = run_dir(cfg, run_id)

    cands = pd.read_parquet(pruned_path(cfg, "train"), columns=["s1", "rec"])
    s1, rec = cands["s1"].to_numpy(np.int64), cands["rec"].to_numpy(np.int64)
    s1_rec = pd.read_parquet(records_path(cfg, "train", "s1"), columns=["entity_id", "country"])
    rec_ids = pd.read_parquet(records_path(cfg, "train", "s23"), columns=["entity_id"])["entity_id"].to_numpy()
    true_s1, n_true = build_labels(cfg, s1_rec["entity_id"].to_numpy(), rec_ids)
    del rec_ids
    y = (true_s1[rec] == s1).astype(np.float32)
    h = s1_hash(s1_rec["entity_id"].to_numpy())
    n_folds = mcfg["n_folds"]
    fold = (h % n_folds)[s1]
    u = ((h // n_folds) % 10_000 / 10_000.0)[s1]
    need = np.flatnonzero(u < mcfg["train_frac"] + mcfg["valid_frac"])
    drop = set(args.drop)
    cols_by_dir = {d: [c for c in part_columns(cfg, "train", d) if c not in drop] for d in ["features", *args.extra]}
    cols = [c for cs in cols_by_dir.values() for c in cs]
    with timer(f"load {len(need):,} sampled rows", log):
        X_need = pd.concat([load_features(cfg, "train", rows=need, columns=cs, subdir=d)
                            for d, cs in cols_by_dir.items()], axis=1)[cols].to_numpy(np.float32)
    boosters = []
    for f in range(n_folds):
        tr = (fold[need] != f) & (u[need] < mcfg["train_frac"])
        va = (fold[need] != f) & (u[need] >= mcfg["train_frac"])
        with timer(f"fold {f}: fit on {tr.sum():,} pairs", log):
            dtr = xgb.QuantileDMatrix(X_need[tr], label=y[need][tr], max_bin=PARAMS["max_bin"])
            dva = xgb.QuantileDMatrix(X_need[va], label=y[need][va], ref=dtr)
            b = xgb.train({**PARAMS, "seed": cfg["seed"] + f}, dtr, num_boost_round=args.rounds,
                          evals=[(dva, "valid")], early_stopping_rounds=100, verbose_eval=500)
            del dtr, dva
        log.info("fold %d best_iteration=%d", f, b.best_iteration)
        b.save_model(str(out / f"model_fold{f}.json"))
        boosters.append(b)
    del X_need

    chunk = cfg["features"]["chunk_pairs"]
    oof = np.zeros(len(y), dtype=np.float32)
    with timer("OOF prediction", log):
        for i, _ in enumerate(sorted(features_dir(cfg, "train").glob("part_*.parquet"))):
            X = load_part(cfg, "train", i, cols_by_dir)[cols].to_numpy(np.float32)
            lo = i * chunk
            fp = fold[lo:lo + len(X)]
            for f, b in enumerate(boosters):
                m = fp == f
                if m.any():
                    oof[lo:lo + len(X)][m] = b.inplace_predict(X[m], iteration_range=(0, b.best_iteration + 1))
    np.save(out / "oof.npy", oof)
    probs = []
    with timer("test prediction", log):
        for i, _ in enumerate(sorted(features_dir(cfg, "test").glob("part_*.parquet"))):
            X = load_part(cfg, "test", i, cols_by_dir)[cols].to_numpy(np.float32)
            probs.append(np.mean([b.inplace_predict(X, iteration_range=(0, b.best_iteration + 1)) for b in boosters],
                                 axis=0).astype(np.float32))
    np.save(out / "test_prob.npy", np.concatenate(probs))

    countries = s1_rec["country"].to_numpy()
    best, table = tune(s1, rec, oof, true_s1, n_true, cfg["decision"]["thresholds"], cfg["decision"]["margins"],
                       groups={c: countries == c for c in sorted(set(countries))})
    log.info("OOF macro F0.5 = %.5f at threshold=%.2f margin=%.2f", best["f05"], best["threshold"], best["margin"])
    meta = {"run_id": run_id, "features": cols, "feature_dirs": cols_by_dir, "decision": best, "n_folds": n_folds,
            "params": PARAMS, "best_iterations": [b.best_iteration for b in boosters],
            "per_country": {k: float(v) for k, v in table.iloc[0].items() if k.startswith("f05_")}}
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    table.to_csv(out / "decision_grid.csv", index=False)
    log_experiment(cfg, run_id, "xgboost", f"{len(cols)} features", best["f05"],
                   notes=f"t={best['threshold']} m={best['margin']} {meta['per_country']}")
    print(run_id)


if __name__ == "__main__":
    main()
