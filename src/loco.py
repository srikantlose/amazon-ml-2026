"""Leave-one-country-out check: how well does the matcher transfer to a country it never saw?

The test set contains France, which never appears in training. As a proxy, a stage-2 model is
trained on one training country only and scored on the other; its macro F0.5 is compared with
the out-of-fold score that country gets when it is part of training.

    python -m src.loco --config configs/base.yaml configs/stage2.yaml --train-country US --eval-country India \
        [--reference-run <stage-2 run id>]
"""
from __future__ import annotations

import argparse
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.blocking import pruned_path
from src.data import build_labels
from src.decide import tune
from src.features import features_dir, load_features, part_columns
from src.normalize import records_path
from src.train_matcher import run_dir, s1_hash
from src.utils import add_config_arg, get_logger, load_config, seed_everything, timer

log = get_logger("loco")


def main():
    ap = argparse.ArgumentParser()
    add_config_arg(ap)
    ap.add_argument("--train-country", required=True)
    ap.add_argument("--eval-country", required=True)
    ap.add_argument("--reference-run", help="stage-2 run trained on all countries, for comparison")
    ap.add_argument("--drop", nargs="*", default=[], help="feature columns to leave out")
    ap.add_argument("--extra", nargs="*", default=[], help="additional pair-aligned feature folders")
    args = ap.parse_args()
    cfg = load_config(args.config)
    mcfg = cfg["model"]
    seed_everything(cfg["seed"])

    cands = pd.read_parquet(pruned_path(cfg, "train"), columns=["s1", "rec"])
    s1, rec = cands["s1"].to_numpy(np.int64), cands["rec"].to_numpy(np.int64)
    s1_rec = pd.read_parquet(records_path(cfg, "train", "s1"), columns=["entity_id", "country"])
    rec_ids = pd.read_parquet(records_path(cfg, "train", "s23"), columns=["entity_id"])["entity_id"].to_numpy()
    true_s1, n_true = build_labels(cfg, s1_rec["entity_id"].to_numpy(), rec_ids)
    y = (true_s1[rec] == s1).astype(np.uint8)
    country = s1_rec["country"].to_numpy()
    pair_country = country[s1]
    u = ((s1_hash(s1_rec["entity_id"].to_numpy()) // mcfg["n_folds"]) % 10_000 / 10_000.0)[s1]

    cols_by_dir = {d: [c for c in part_columns(cfg, "train", d) if c not in set(args.drop)]
                   for d in ["features", *args.extra]}

    def load(rows):
        return pd.concat([load_features(cfg, "train", rows=rows, columns=cs, subdir=d)
                          for d, cs in cols_by_dir.items()], axis=1)

    tr = np.flatnonzero((pair_country == args.train_country) & (u < mcfg["train_frac"]))
    va = np.flatnonzero((pair_country == args.train_country) & (u >= mcfg["train_frac"])
                        & (u < mcfg["train_frac"] + mcfg["valid_frac"]))
    ev = np.flatnonzero(pair_country == args.eval_country)
    with timer("load features", log):
        X = load(np.sort(np.r_[tr, va]))
    order = np.sort(np.r_[tr, va])
    is_tr = np.isin(order, tr)
    with timer(f"fit on {args.train_country}: {is_tr.sum():,} pairs", log):
        model = lgb.LGBMClassifier(**mcfg["lightgbm"], random_state=cfg["seed"], n_jobs=-1)
        model.fit(X[is_tr], y[order][is_tr], eval_set=[(X[~is_tr], y[order][~is_tr])],
                  callbacks=[lgb.early_stopping(mcfg["early_stopping_rounds"], verbose=False)])
    del X
    with timer(f"score {args.eval_country}: {len(ev):,} pairs", log):
        prob = model.booster_.predict(load(ev))

    # evaluate on the held-out country's S1 entities only (its records never compete across countries)
    keep_s1 = country == args.eval_country
    n_true_eval = np.where(keep_s1, n_true, 0)
    best, table = tune(s1[ev], rec[ev], prob.astype(np.float32), true_s1, n_true_eval,
                       cfg["decision"]["thresholds"], cfg["decision"]["margins"], groups={"eval": keep_s1})
    row = table.sort_values("f05_eval", ascending=False).iloc[0]
    at_default = table[(table.threshold == 0.6) & (table.margin == 0.5)]
    msg = {"train_country": args.train_country, "eval_country": args.eval_country, "dropped": args.drop,
           "extra": args.extra,
           "best_f05": float(row.f05_eval), "best_threshold": float(row.threshold), "best_margin": float(row.margin),
           "f05_at_0.6_0.5": float(at_default.f05_eval.iloc[0]) if len(at_default) else None}
    if args.reference_run:
        oof = np.load(run_dir(cfg, args.reference_run) / "oof.npy")
        ref, ref_table = tune(s1[ev], rec[ev], oof[ev], true_s1, n_true_eval, [0.6], [0.5], groups={"eval": keep_s1})
        msg["reference_oof_f05_at_0.6_0.5"] = float(ref_table.f05_eval.iloc[0])
    log.info("LOCO result: %s", json.dumps(msg))
    print(json.dumps(msg))


if __name__ == "__main__":
    main()
