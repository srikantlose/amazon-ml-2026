"""Average the probabilities of several matcher runs trained on the same candidate pairs.

Members differ in seed and tree shape. Out-of-fold probabilities are averaged and the decision
(threshold, margin) is re-tuned on the average; test probabilities are averaged the same way.
The result is saved as a run of its own, so `python -m src.predict --run <ensemble> --reuse-probs`
writes the submission files.

    python -m src.ensemble --config configs/base.yaml --runs <run_a> <run_b> <run_c> [--name ens]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.blocking import pruned_path
from src.data import build_labels
from src.decide import tune
from src.normalize import records_path
from src.train_matcher import run_dir
from src.utils import add_config_arg, get_logger, load_config, log_experiment, make_run_id

log = get_logger("ensemble")


def main():
    ap = argparse.ArgumentParser()
    add_config_arg(ap)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--name", default="ens")
    args = ap.parse_args()
    cfg = load_config(args.config)

    cands = pd.read_parquet(pruned_path(cfg, "train"), columns=["s1", "rec"])
    s1, rec = cands["s1"].to_numpy(np.int64), cands["rec"].to_numpy(np.int64)
    s1_rec = pd.read_parquet(records_path(cfg, "train", "s1"), columns=["entity_id", "country"])
    rec_ids = pd.read_parquet(records_path(cfg, "train", "s23"), columns=["entity_id"])["entity_id"].to_numpy()
    true_s1, n_true = build_labels(cfg, s1_rec["entity_id"].to_numpy(), rec_ids)
    countries = s1_rec["country"].to_numpy()
    groups = {c: countries == c for c in sorted(set(countries))}

    oofs, tests = [], []
    for r in args.runs:
        d = run_dir(cfg, r)
        oofs.append(np.load(d / "oof.npy"))
        path = d / "test_prob.npy"
        if not path.exists():
            from src.predict import predict_test

            predict_test(cfg, r)
        tests.append(np.load(path))
        single, _ = tune(s1, rec, oofs[-1], true_s1, n_true, cfg["decision"]["thresholds"],
                         cfg["decision"]["margins"])
        log.info("member %s: OOF F0.5 %.5f", r, single["f05"])
    oof = np.mean(oofs, axis=0).astype(np.float32)
    test = np.mean(tests, axis=0).astype(np.float32)
    best, table = tune(s1, rec, oof, true_s1, n_true, cfg["decision"]["thresholds"], cfg["decision"]["margins"],
                       groups=groups)
    log.info("ensemble of %d: OOF macro F0.5 = %.5f at threshold=%.2f margin=%.2f", len(args.runs), best["f05"],
             best["threshold"], best["margin"])

    run_id = make_run_id(args.name)
    out = run_dir(cfg, run_id)
    np.save(out / "oof.npy", oof)
    np.save(out / "test_prob.npy", test)
    meta = {"run_id": run_id, "members": args.runs, "decision": best, "features": [],
            "per_country": {k: float(v) for k, v in table.iloc[0].items() if k.startswith("f05_")}}
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    table.to_csv(out / "decision_grid.csv", index=False)
    log_experiment(cfg, run_id, "ensemble", args.runs, best["f05"],
                   notes=f"t={best['threshold']} m={best['margin']} {meta['per_country']}")
    print(run_id)


if __name__ == "__main__":
    main()
