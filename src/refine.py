"""Stage-3 context features from a trained stage-2 matcher's probabilities.

Stage-2 probabilities (p2) are much sharper than the stage-1 retrieval ranker's, so the
competition context is recomputed from them:
  - rank / gap / runner-up of the pair among its record's candidates and among its S1's candidates
  - how many records already point to the S1 with high p2 (all sources, and the pair's source)
  - similarity of the record to the S1's confident p2-linked records ("siblings")
Train uses the run's out-of-fold p2; test uses the run's averaged test p2 (computed if missing).
A new matcher trained with these extra columns (train_matcher --extra) is the stage-3 model.

    python -m src.refine --config configs/base.yaml --run <stage-2 run id>
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.blocking import pruned_path
from src.features import (_group_stats, country_segments, features_dir, record_arrays, sibling_features,
                          write_parts)
from src.normalize import records_path
from src.train_matcher import run_dir
from src.utils import add_config_arg, get_logger, load_config, timer

log = get_logger("refine")


def extra_dir_name(run_id: str) -> str:
    return f"features_s3_{run_id}"


def context_features(cands: pd.DataFrame, p2: np.ndarray, src: np.ndarray) -> pd.DataFrame:
    s1 = cands["s1"].to_numpy(np.int64)
    rec = cands["rec"].to_numpy(np.int64)
    p2 = p2.astype(np.float32)
    out = {"p2": p2}
    for key, tag in ((rec, "rec"), (s1, "s1")):
        rank, gmax, gsecond, _ = _group_stats(key, p2)
        out[f"p2_{tag}_rank"] = rank
        out[f"p2_{tag}_gap"] = gmax - p2
        out[f"p2_{tag}_margin"] = np.where(rank == 0, p2 - np.nan_to_num(gsecond, nan=0.0), 0.0).astype(np.float32)
    rec_best = out["p2_rec_rank"] == 0
    strong = rec_best & (p2 >= 0.5)
    n_s1 = int(s1.max()) + 1 if len(s1) else 0
    cnt = np.bincount(s1, weights=strong, minlength=n_s1)
    out["p2_s1_n_linked"] = (cnt[s1] - strong).astype(np.float32)          # other records linked to this S1
    same_src = np.zeros(len(s1), np.float32)
    for v in np.unique(src):
        m = src == v
        c = np.bincount(s1[m], weights=strong[m], minlength=n_s1)
        same_src[m] = c[s1[m]] - strong[m]
    out["p2_s1_n_linked_same_src"] = same_src
    out["p2_s1_sum"] = np.bincount(s1, weights=p2, minlength=n_s1)[s1].astype(np.float32)
    return pd.DataFrame(out)


def build(cfg: dict, split: str, run_id: str) -> None:
    d = run_dir(cfg, run_id)
    if split == "train":
        p2 = np.load(d / "oof.npy")
    else:
        path = d / "test_prob.npy"
        if not path.exists():
            from src.predict import predict_test

            predict_test(cfg, run_id)
        p2 = np.load(path)
    cands = pd.read_parquet(pruned_path(cfg, split), columns=["s1", "rec"])
    if len(p2) != len(cands):
        raise ValueError(f"{split}: {len(p2)} probabilities for {len(cands)} pairs")
    s23 = pd.read_parquet(records_path(cfg, split, "s23"), columns=["src", "name_n", "name_ns", "addr_n", "name_core",
                                                                     "addr_nums", "house", "postal", "is_domain",
                                                                     "name_non_latin", "addr_empty", "entity_id"])
    s1_country = pd.read_parquet(records_path(cfg, split, "s1"), columns=["country"])["country"].to_numpy()
    src = s23["src"].to_numpy()[cands["rec"].to_numpy()]
    with timer(f"stage-3 context {split}", log):
        parts = []
        for seg in country_segments(cands["s1"].to_numpy(), s1_country):
            parts.append(context_features(cands.iloc[seg].reset_index(drop=True), p2[seg], src[seg]))
        ctx = pd.concat(parts, ignore_index=True)
        B = record_arrays(s23)
        sib = sibling_features(cands, B, min_p1=0.9, prob=p2, prefix="p2sib")
        ctx = pd.concat([ctx, sib], axis=1)
    out = features_dir(cfg, split, extra_dir_name(run_id))
    write_parts(ctx, out, cfg["features"]["chunk_pairs"])
    log.info("%s: wrote %d stage-3 columns for %s pairs -> %s", split, ctx.shape[1], f"{len(ctx):,}", out)


def main():
    ap = argparse.ArgumentParser()
    add_config_arg(ap)
    ap.add_argument("--run", required=True, help="stage-2 run whose probabilities feed the context")
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    args = ap.parse_args()
    cfg = load_config(args.config)
    for split in args.splits:
        build(cfg, split, args.run)


if __name__ == "__main__":
    main()
