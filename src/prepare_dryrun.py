"""Build the dry-run split from the 2025 labeled train set, and score dry-run submissions.

    python -m src.prepare_dryrun --src data/raw/2025 --n-train 5000 --n-test 2000
    python -m src.prepare_dryrun --score submissions/dryrun/ens_xxx.csv

The held-out "test" rows have known prices, so --score acts as a private leaderboard and shows
whether CV tracks held-out performance.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.metrics import smape
from src.utils import ROOT, get_logger

log = get_logger("prepare_dryrun")
OUT = ROOT / "data" / "raw" / "dryrun"
ID, TARGET = "sample_id", "price"


def find_train_csv(src: Path) -> Path:
    if src.is_file():
        return src
    hits = sorted(p for p in src.rglob("train.csv"))
    if not hits:
        raise FileNotFoundError(f"no train.csv under {src}")
    return hits[0]


def build(src: Path, n_train: int, n_test: int, seed: int) -> None:
    path = find_train_csv(src)
    df = pd.read_csv(path, dtype={ID: str})
    log.info("read %s: %s, columns=%s", path, df.shape, list(df.columns))
    df = df.dropna(subset=[TARGET])
    df = df.sample(n=min(n_train + n_test, len(df)), random_state=seed).reset_index(drop=True)
    train, test = df.iloc[:n_train], df.iloc[n_train:]

    OUT.mkdir(parents=True, exist_ok=True)
    train.to_csv(OUT / "train.csv", index=False)
    test.drop(columns=[TARGET]).to_csv(OUT / "test.csv", index=False)
    test[[ID, TARGET]].to_csv(OUT / "test_labels.csv", index=False)
    test[[ID]].assign(**{TARGET: 1.0}).to_csv(OUT / "sample_test_out.csv", index=False)
    log.info("wrote %s: train=%d test=%d", OUT, len(train), len(test))


def score(sub_path: Path) -> float:
    sub = pd.read_csv(sub_path, dtype={ID: str})
    labels = pd.read_csv(OUT / "test_labels.csv", dtype={ID: str})
    m = labels.merge(sub, on=ID, suffixes=("_true", "_pred"))
    s = smape(m[f"{TARGET}_true"], m[f"{TARGET}_pred"])
    print(f"held-out SMAPE = {s:.4f} on {len(m)} rows ({sub_path.name})")
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data" / "raw" / "2025"), help="2025 train.csv or a dir containing it")
    ap.add_argument("--n-train", type=int, default=5000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--score", help="score a dry-run submission against held-out labels")
    args = ap.parse_args()
    if args.score:
        score(Path(args.score))
    else:
        build(Path(args.src), args.n_train, args.n_test, args.seed)
