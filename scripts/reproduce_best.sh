#!/usr/bin/env bash
# Rebuild the final submission (25, public F0.5 0.983405) from the raw competition data.
#
#   bash scripts/reproduce_best.sh            # writes output/matching_results.tsv + output/candidate_pairs.tsv
#
# Needs student_resource/dataset/{train,test} next to src/, the .venv (see README), ~8 h on 12 CPU threads +
# an 8 GB CUDA GPU, 31 GB RAM and ~60 GB free disk. GPU fp16 retrieval and multi-threaded tree training make
# a rebuild agree with the uploaded file on ~99% of pairs rather than byte for byte.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python}
[ -x "$PY" ] || PY=python
export PYTHONIOENCODING=utf-8

# 1. normalize -> blocking -> prune -> features -> stage 2 -> refine -> stage 3 -> predict.
#    Produces the candidate pairs and base features, and the "rebuild" stage-3 model used for France below.
$PY -m src.run_pipeline --config configs/base.yaml
REBUILD=$(ls -td data/cache/runs/lgb_stage3_* | head -1 | xargs basename)

# 2. look-alike name columns and relative candidate counts, per split (statistics from each split's own names)
for split in train test; do
  $PY -m src.lookalike --config configs/base.yaml --split "$split"
  $PY -m src.relcounts --config configs/base.yaml --split "$split"
done

# 3. model A (all look-alike columns) and model B (categorical look-alike columns only), each stage 2 + refine +
#    stage 3 trained on 90% / 95% of the other folds' S1 groups (configs/stage2_big.yaml, stage3_big.yaml)
bash scripts/train_big.sh A
bash scripts/train_big.sh B
A3=$(ls -td data/cache/runs/lgb_stage3labig_* | head -1 | xargs basename)
B3=$(ls -td data/cache/runs/lgb_stage3lacatbig_* | head -1 | xargs basename)

# 4. decision: US/India (labels seen in training) = mean of A and B at t=0.60, m=0.50;
#    France (unseen label) = mean of B and the rebuild model at t=0.75, m=0.50 (configs/base.yaml)
$PY -m src.blend --config configs/base.yaml --seen-runs "$A3" "$B3" --unseen-runs "$B3" "$REBUILD" \
    --threshold 0.60 --margin 0.50 --out output_blend
# 5. vocabulary-swap post-filter on the unseen-label rows -> output/
$PY -m src.combine --config configs/base.yaml --seen output_blend --unseen output_blend --unseen-postfilter --out output
echo "models: A $A3, B $B3, rebuild $REBUILD; final files in output/"
