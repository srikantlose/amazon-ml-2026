"""Combine two predictions of the same candidate set per country label.

Rows of S1 entities whose country label occurs in training come from --seen; rows of labels that
never occur in training (France in the test set) come from --unseen, optionally after the
vocabulary-swap post-filter. Both inputs must have been written for the same candidate pairs.

    python -m src.combine --config configs/base.yaml --seen output_la --unseen output_rebuild \
        [--unseen-postfilter] --out output
"""
from __future__ import annotations

import argparse
import csv
import filecmp
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from src.normalize import records_path
from src.postfilter import filter_file
from src.utils import add_config_arg, get_logger, load_config, resolve
from src.validate import validate

log = get_logger("combine")


def rows(path):
    csv.field_size_limit(10**9)
    with open(path, encoding="utf-8", newline="") as f:
        r = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(r)
        return header, list(r)


def combine(cfg: dict, seen_dir, unseen_dir, out_dir, unseen_postfilter: bool = False) -> None:
    seen_dir, unseen_dir, out_dir = resolve(seen_dir), resolve(unseen_dir), resolve(out_dir)
    if not filecmp.cmp(seen_dir / "candidate_pairs.tsv", unseen_dir / "candidate_pairs.tsv", shallow=False):
        raise ValueError("the two predictions were written for different candidate pairs")
    train_countries = set(pd.read_parquet(records_path(cfg, "train", "s1"), columns=["country"])["country"].unique())
    test_s1 = pd.read_parquet(records_path(cfg, "test", "s1"), columns=["entity_id", "country"])
    unseen_ids = set(test_s1.loc[~test_s1["country"].isin(train_countries), "entity_id"])
    unseen_file = unseen_dir / "matching_results.tsv"
    with tempfile.TemporaryDirectory() as tmp:
        if unseen_postfilter:
            filtered = Path(tmp) / "unseen_filtered.tsv"
            filter_file(cfg, "test", unseen_file, filtered)
            unseen_file = filtered
        header, seen_rows = rows(seen_dir / "matching_results.tsv")
        _, unseen_rows = rows(unseen_file)
        unseen_map = {r[0]: r for r in unseen_rows}
        out_dir.mkdir(parents=True, exist_ok=True)
        n_unseen = 0
        with open(out_dir / "matching_results.tsv", "w", encoding="utf-8", newline="\n") as f:
            f.write("\t".join(header) + "\n")
            for r in seen_rows:
                if r[0] in unseen_ids:
                    r = unseen_map[r[0]]
                    n_unseen += 1
                f.write("\t".join(r) + "\n")
    if out_dir != seen_dir:
        shutil.copy2(seen_dir / "candidate_pairs.tsv", out_dir / "candidate_pairs.tsv")
    log.info("%d rows from %s (unseen country labels), %d from %s", n_unseen, unseen_dir.name,
             len(seen_rows) - n_unseen, seen_dir.name)
    validate(cfg, out_dir / "matching_results.tsv", out_dir / "candidate_pairs.tsv")


def main():
    ap = argparse.ArgumentParser()
    add_config_arg(ap)
    ap.add_argument("--seen", required=True, help="output folder used for country labels seen in training")
    ap.add_argument("--unseen", required=True, help="output folder used for labels never seen in training")
    ap.add_argument("--unseen-postfilter", action="store_true", help="apply src.postfilter to the unseen rows")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    combine(load_config(args.config), args.seen, args.unseen, args.out, args.unseen_postfilter)


if __name__ == "__main__":
    main()
