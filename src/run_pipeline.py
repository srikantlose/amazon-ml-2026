"""End-to-end run: raw TSVs -> blocking -> matching -> output/*.tsv.

    python -m src.run_pipeline --config configs/base.yaml

Stages (each caches its output under data/cache/, so a rerun resumes cheaply):
  1. normalize      learned transliteration map (train pairs) + normalized records
  2. blocking       union of retrieval views for train and test
  3. prune          stage-1 ranker keeps the top S1 candidates per record (fit on train, applied to test)
  4. features       pair features for train and test
  5. train_matcher  LightGBM, 3 S1-grouped folds, OOF-tuned decision (prints the run id)
  6. predict        test inference -> output/matching_results.tsv + output/candidate_pairs.tsv
"""
from __future__ import annotations

import argparse
import subprocess
import sys

from src.utils import ROOT, get_logger

log = get_logger("pipeline")


def run(args: list[str]) -> str:
    cmd = [sys.executable, "-m", *args]
    log.info("$ %s", " ".join(cmd[1:]))
    res = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise SystemExit(f"stage failed: {' '.join(args)}")
    return res.stdout.strip().splitlines()[-1] if res.stdout.strip() else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", default=[str(ROOT / "configs" / "base.yaml")])
    args = ap.parse_args()
    cfg = ["--config", *args.config]
    run(["src.normalize", *cfg])
    for split in ("train", "test"):
        run(["src.blocking", *cfg, "--split", split])
    for split in ("train", "test"):
        run(["src.prune", *cfg, "--split", split])
        run(["src.features", *cfg, "--split", split])
    run_id = run(["src.train_matcher", *cfg])
    run(["src.predict", *cfg, "--run", run_id])
    log.info("done: run %s -> output/matching_results.tsv, output/candidate_pairs.tsv", run_id)


if __name__ == "__main__":
    main()
