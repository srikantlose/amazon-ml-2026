"""Candidate counts relative to their split and country.

Test has more distractor records per S1 than train (5.5-5.8 vs 4.7 records per S1, 6.8 vs 5.5
pruned candidates per S1), so the raw count of candidates an S1 has is inflated on test for
reasons unrelated to matching; a stage-2 model re-scored with s1_n_cands raised by 22% accepts
~25-30% more false pairs. Dividing by the mean over the S1 entities of the same split and country
keeps the within-country information and removes the shift. Used with `--drop s1_n_cands`.

    python -m src.relcounts --config configs/base.yaml --split train
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.blocking import pruned_path
from src.features import features_dir, load_features, write_parts
from src.normalize import records_path
from src.utils import add_config_arg, get_logger, load_config

log = get_logger("relcounts")

SUBDIR = "features_rel"


def build(cfg: dict, split: str) -> None:
    s1 = pd.read_parquet(pruned_path(cfg, split), columns=["s1"])["s1"].to_numpy(np.int64)
    country = pd.read_parquet(records_path(cfg, split, "s1"), columns=["country"])["country"].to_numpy()[s1]
    n = load_features(cfg, split, columns=["s1_n_cands"])["s1_n_cands"].to_numpy(np.float64)
    rel = np.empty(len(n), np.float32)
    for c in np.unique(country):
        m = country == c
        per_s1 = len(np.unique(s1[m]))
        mean = m.sum() / per_s1                      # candidates per S1 entity in this split and country
        rel[m] = n[m] / mean
        log.info("%s %s: %.2f candidates per S1", split, c, mean)
    write_parts(pd.DataFrame({"s1_n_cands_rel": rel}), features_dir(cfg, split, SUBDIR), cfg["features"]["chunk_pairs"])


def main():
    ap = argparse.ArgumentParser()
    add_config_arg(ap)
    ap.add_argument("--split", required=True, choices=["train", "test"])
    args = ap.parse_args()
    build(load_config(args.config), args.split)


if __name__ == "__main__":
    main()
