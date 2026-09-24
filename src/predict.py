"""Test inference: average the fold models, apply the tuned decision, write both output files.

    python -m src.predict --config configs/base.yaml --run <run_id>

Writes output/matching_results.tsv (leaderboard file) and output/candidate_pairs.tsv (every
pair the model scored), then runs the validators.
"""
from __future__ import annotations

import argparse
import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.blocking import pruned_path
from src.data import CANDIDATE_HEADER, MATCHING_HEADER
from src.decide import select
from src.features import features_dir
from src.normalize import records_path
from src.train_matcher import run_dir
from src.utils import add_config_arg, get_logger, load_config, resolve, timer
from src.validate import validate

log = get_logger("predict")


def write_pairs_as_lists(path, header, s1_ids: np.ndarray, rec_ids: np.ndarray,
                         s1: np.ndarray, rec: np.ndarray) -> None:
    """One row per S1 (in s1_ids order) with its record ids comma-joined; "\\n" line endings.

    Works on index arrays (sorted by S1) to avoid building tens of millions of string pairs.
    """
    order = np.lexsort((rec, s1))
    s1_sorted, rec_sorted = s1[order], rec[order]
    bounds = np.searchsorted(s1_sorted, np.arange(len(s1_ids) + 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(header) + "\n")
        for i, sid in enumerate(s1_ids):
            a, b = bounds[i], bounds[i + 1]
            f.write(f"{sid}\t{','.join(rec_ids[rec_sorted[a:b]]) if b > a else ''}\n")


def predict_test(cfg: dict, run_id: str) -> np.ndarray:
    d = run_dir(cfg, run_id)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    boosters = [lgb.Booster(model_file=str(p)) for p in sorted(d.glob("model_fold*.txt"))]
    parts = sorted(features_dir(cfg, "test").glob("part_*.parquet"))
    probs = []
    with timer(f"predict test with {len(boosters)} models", log):
        for p in parts:
            X = pd.read_parquet(p, columns=meta["features"])
            probs.append(np.mean([b.predict(X) for b in boosters], axis=0).astype(np.float32))
    prob = np.concatenate(probs)
    np.save(d / "test_prob.npy", prob)
    return prob


def main():
    ap = argparse.ArgumentParser()
    add_config_arg(ap)
    ap.add_argument("--run", required=True)
    ap.add_argument("--threshold", type=float, help="override the tuned threshold")
    ap.add_argument("--margin", type=float, help="override the tuned margin")
    args = ap.parse_args()
    cfg = load_config(args.config)
    meta = json.loads((run_dir(cfg, args.run) / "meta.json").read_text(encoding="utf-8"))
    t = args.threshold if args.threshold is not None else meta["decision"]["threshold"]
    m = args.margin if args.margin is not None else meta["decision"]["margin"]

    prob = predict_test(cfg, args.run)
    cands = pd.read_parquet(pruned_path(cfg, "test"), columns=["s1", "rec"])
    s1, rec = cands["s1"].to_numpy(np.int64), cands["rec"].to_numpy(np.int64)
    s1_ids = pd.read_parquet(records_path(cfg, "test", "s1"), columns=["entity_id"])["entity_id"].to_numpy(object)
    rec_ids = pd.read_parquet(records_path(cfg, "test", "s23"), columns=["entity_id"])["entity_id"].to_numpy(object)
    keep = select(rec, prob, t, m)
    log.info("threshold=%.2f margin=%.2f -> %s matches", t, m, f"{keep.sum():,}")

    out = resolve(cfg["paths"]["output_dir"])
    with timer("write outputs", log):
        write_pairs_as_lists(out / "matching_results.tsv", MATCHING_HEADER, s1_ids, rec_ids, s1[keep], rec[keep])
        write_pairs_as_lists(out / "candidate_pairs.tsv", CANDIDATE_HEADER, s1_ids, rec_ids, s1, rec)
    ok = validate(cfg, out / "matching_results.tsv", out / "candidate_pairs.tsv")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
