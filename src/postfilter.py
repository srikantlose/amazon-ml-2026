"""Drop accepted pairs whose record swaps one of the S1's name words for another ordinary word.

The pattern (see src/lookalike.py): the record keeps the S1's other words but one S1 word is
missing and an ordinary vocabulary word of 4+ letters with a different initial appears instead
("maisons ecole ei" -> "maisons comite ei"). Among pairs the stage-2 matcher accepts out of fold,
these are 12% correct (US 26%, India 4%); removing them raises OOF macro F0.5 by 0.0003, and the
test set has 5-60x more of them per S1 than train.

    python -m src.postfilter --config configs/base.yaml --matches output/matching_results.tsv \
        --out output/matching_results_filtered.tsv
"""
from __future__ import annotations

import argparse
import csv

import numpy as np
import pandas as pd

from src.data import MATCHING_HEADER
from src.lookalike import COLUMNS, _init, pair_row, token_stats
from src.normalize import records_path
from src.utils import add_config_arg, get_logger, load_config

log = get_logger("postfilter")

_SWAP, _SAME_INITIAL, _LEN = (COLUMNS.index(c) for c in ("la_swap_vocab", "la_swap_same_initial",
                                                         "la_extra_vocab_len_min"))


def is_vocab_swap(row: tuple) -> bool:
    return row[_SWAP] == 1 and row[_SAME_INITIAL] == 0 and row[_LEN] >= 4


def filter_file(cfg: dict, split: str, matches, out) -> dict:
    s1 = pd.read_parquet(records_path(cfg, split, "s1"), columns=["entity_id", "name_n", "country"])
    s23 = pd.read_parquet(records_path(cfg, split, "s23"), columns=["entity_id", "name_n", "country"])
    _init(token_stats(s1, s23))
    s1_info = dict(zip(s1["entity_id"], zip(s1["name_n"], s1["country"])))
    rec_name = dict(zip(s23["entity_id"], s23["name_n"]))
    dropped, kept = {}, 0
    csv.field_size_limit(10**9)
    with open(matches, encoding="utf-8", newline="") as fin, open(out, "w", encoding="utf-8", newline="\n") as fout:
        r = csv.reader(fin, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(r)
        if tuple(header) != tuple(MATCHING_HEADER):
            raise ValueError(f"unexpected header {header}")
        fout.write("\t".join(header) + "\n")
        for sid, ids in r:
            name, country = s1_info[sid]
            keep = [i for i in (ids.split(",") if ids else []) if not is_vocab_swap(pair_row(name, rec_name[i], country))]
            n_drop = (len(ids.split(",")) if ids else 0) - len(keep)
            if n_drop:
                dropped[country] = dropped.get(country, 0) + n_drop
            kept += len(keep)
            fout.write(f"{sid}\t{','.join(keep)}\n")
    log.info("kept %s pairs; dropped %s", f"{kept:,}", {k: f"{v:,}" for k, v in sorted(dropped.items())})
    return dropped


def main():
    ap = argparse.ArgumentParser()
    add_config_arg(ap)
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--matches", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    filter_file(load_config(args.config), args.split, args.matches, args.out)


if __name__ == "__main__":
    main()
