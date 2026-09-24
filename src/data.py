"""Reading the competition TSVs and writing the two submission files.

All files are tab-separated. Reading uses QUOTE_NONE and disables NA parsing so that
business names like "NA" or ones containing quote characters survive intact.
Writing uses "\n" line endings explicitly: the official validator only strips "\n",
so Windows "\r\n" endings would leave a stray "\r" on the last ID of every list.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import resolve

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
MATCHING_HEADER = ("source1_entity_id", "matched_entity_ids")
CANDIDATE_HEADER = ("source1_entity_id", "candidate_entity_ids")


def _read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False,
                       quoting=csv.QUOTE_NONE, encoding="utf-8")


def fix_mojibake(s: str) -> str:
    """Undo UTF-8 text that was decoded as Latin-1/cp1252 ("â€“" -> "–"); leave others alone."""
    for enc in ("cp1252", "latin-1"):
        try:
            return s.encode(enc).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return s


def source_path(cfg: dict, split: str, n: int) -> Path:
    return resolve(cfg["data_dir"]) / split / f"{split}_source{n}.tsv"


def read_source(cfg: dict, split: str, n: int) -> pd.DataFrame:
    df = _read_tsv(source_path(cfg, split, n))
    missing = set(SOURCE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{split}_source{n}.tsv is missing columns {missing}")
    df = df[SOURCE_COLUMNS].copy()
    for col in ("business_name", "business_address"):
        bad = df[col].str.contains("â|Ã", regex=True)
        if bad.any():
            df.loc[bad, col] = df.loc[bad, col].map(fix_mojibake)
    return df


def read_ground_truth(cfg: dict) -> pd.DataFrame:
    return _read_tsv(resolve(cfg["data_dir"]) / "train" / "train_ground_truth.tsv")


def ground_truth_pairs(cfg: dict) -> pd.DataFrame:
    """Exploded ground truth: one row per (source1_entity_id, matched id)."""
    gt = read_ground_truth(cfg)
    ids = gt["matched_entity_ids"].str.split(",")
    ex = pd.DataFrame({"s1_id": gt["source1_entity_id"].repeat(ids.str.len()).to_numpy(),
                       "rec_id": np.concatenate(ids.to_numpy()) if len(ids) else []})
    return ex[ex["rec_id"] != ""].reset_index(drop=True)


def build_labels(cfg: dict, s1_ids: np.ndarray, rec_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ground truth as arrays over record/S1 row positions.

    Returns (true_s1_of_rec, n_true): true_s1_of_rec[r] is the S1 row that record row r belongs
    to (-1 if unmatched); n_true[s] is the number of true matches of S1 row s.
    """
    pairs = ground_truth_pairs(cfg)
    si = pd.Index(s1_ids).get_indexer(pairs["s1_id"])
    ri = pd.Index(rec_ids).get_indexer(pairs["rec_id"])
    if (si < 0).any() or (ri < 0).any():
        raise ValueError("ground truth references ids missing from the records")
    true_s1 = np.full(len(rec_ids), -1, dtype=np.int64)
    true_s1[ri] = si
    n_true = np.bincount(si, minlength=len(s1_ids))
    return true_s1, n_true


def write_id_lists(path: Path, header: tuple[str, str], s1_ids, lists: dict) -> None:
    """Write one row per S1 id (in the given order); ids are de-duplicated and sorted."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(header) + "\n")
        for s1 in s1_ids:
            ids = lists.get(s1)
            f.write(f"{s1}\t{','.join(sorted(set(ids))) if ids else ''}\n")
