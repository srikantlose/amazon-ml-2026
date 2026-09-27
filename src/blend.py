"""Final decision from several runs' test probabilities.

Pairs of S1 entities whose country label occurs in training use the mean probability of --seen-runs and
the given threshold/margin; pairs of labels never seen in training use --unseen-run and
`decision.unseen_country` from the config. Writes both output files and runs the validators.

    python -m src.blend --config configs/base.yaml --seen-runs <A_big stage 3> <B_big stage 3> \
        --unseen-run <B_big stage 3> --threshold 0.60 --margin 0.50 --out output
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.blocking import pruned_path
from src.data import CANDIDATE_HEADER, MATCHING_HEADER
from src.decide import select_per_pair
from src.normalize import records_path
from src.predict import write_pairs_as_lists
from src.train_matcher import run_dir
from src.utils import add_config_arg, get_logger, load_config, resolve
from src.validate import validate

log = get_logger("blend")


def main():
    ap = argparse.ArgumentParser()
    add_config_arg(ap)
    ap.add_argument("--seen-runs", nargs="+", required=True)
    ap.add_argument("--unseen-runs", nargs="+", required=True)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--margin", type=float, required=True)
    ap.add_argument("--out", default="output")
    args = ap.parse_args()
    cfg = load_config(args.config)
    cands = pd.read_parquet(pruned_path(cfg, "test"), columns=["s1", "rec"])
    s1, rec = cands["s1"].to_numpy(np.int64), cands["rec"].to_numpy(np.int64)
    test_s1 = pd.read_parquet(records_path(cfg, "test", "s1"), columns=["entity_id", "country"])
    seen = set(pd.read_parquet(records_path(cfg, "train", "s1"), columns=["country"])["country"].unique())
    unseen = (~np.isin(test_s1["country"].to_numpy(), list(seen)))[s1]
    p_seen = np.mean([np.load(run_dir(cfg, r) / "test_prob.npy") for r in args.seen_runs], axis=0)
    p_unseen = np.mean([np.load(run_dir(cfg, r) / "test_prob.npy") for r in args.unseen_runs], axis=0)
    prob = np.where(unseen, p_unseen, p_seen).astype(np.float32)
    ucfg = cfg["decision"]["unseen_country"]
    thr = np.where(unseen, ucfg["threshold"], args.threshold).astype(np.float32)
    mar = np.where(unseen, ucfg["margin"], args.margin).astype(np.float32)
    keep = select_per_pair(rec, prob, thr, mar)
    log.info("seen labels: mean of %s at t=%.2f m=%.2f; unseen labels: %s at t=%.2f m=%.2f -> %s matches",
             args.seen_runs, args.threshold, args.margin, args.unseen_runs, ucfg["threshold"], ucfg["margin"],
             f"{keep.sum():,}")
    out = resolve(args.out)
    ids = test_s1["entity_id"].to_numpy(object)
    rec_ids = pd.read_parquet(records_path(cfg, "test", "s23"), columns=["entity_id"])["entity_id"].to_numpy(object)
    write_pairs_as_lists(out / "matching_results.tsv", MATCHING_HEADER, ids, rec_ids, s1[keep], rec[keep])
    write_pairs_as_lists(out / "candidate_pairs.tsv", CANDIDATE_HEADER, ids, rec_ids, s1, rec)
    validate(cfg, out / "matching_results.tsv", out / "candidate_pairs.tsv")


if __name__ == "__main__":
    main()
