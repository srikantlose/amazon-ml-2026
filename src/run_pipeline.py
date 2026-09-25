"""End-to-end run: raw TSVs -> blocking -> matching -> output/*.tsv.

    python -m src.run_pipeline --config configs/base.yaml

Stages (each caches its output under data/cache/, so a rerun resumes cheaply):
  1. normalize      learned transliteration map (train pairs) + normalized records
  2. blocking       union of retrieval views for train and test
  3. prune          stage-1 ranker keeps the top S1 candidates per record (fit on train, applied to test)
  4. features       pair features for train and test
  5. stage 2        LightGBM matcher, 3 S1-grouped folds, out-of-fold probabilities (configs/stage2.yaml)
  6. refine         context features recomputed from stage-2 probabilities (train OOF, test average)
  7. stage 3        LightGBM matcher on features + stage-2 context, OOF-tuned decision (configs/stage3.yaml)
  8. predict        test inference -> output/matching_results.tsv + output/candidate_pairs.tsv
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
    ap.add_argument("--config", default=str(ROOT / "configs" / "base.yaml"))
    args = ap.parse_args()
    base = ["--config", args.config]
    run(["src.normalize", *base])
    for split in ("train", "test"):
        run(["src.blocking", *base, "--split", split])
    for split in ("train", "test"):
        run(["src.prune", *base, "--split", split])
        run(["src.features", *base, "--split", split])
    stage2 = run(["src.train_matcher", "--config", args.config, str(ROOT / "configs" / "stage2.yaml"),
                  "--name", "stage2"])
    run(["src.refine", *base, "--run", stage2])
    stage3 = run(["src.train_matcher", "--config", args.config, str(ROOT / "configs" / "stage3.yaml"),
                  "--extra", f"features_s3_{stage2}", "--name", "stage3"])
    run(["src.predict", *base, "--run", stage3])
    log.info("done: stage-2 %s, stage-3 %s -> output/matching_results.tsv, output/candidate_pairs.tsv",
             stage2, stage3)


if __name__ == "__main__":
    main()
