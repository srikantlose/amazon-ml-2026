#!/usr/bin/env bash
# Rebuild the best leaderboard submission (16, public F0.5 0.982629) from the raw competition data.
#
#   bash scripts/reproduce_best.sh            # writes output/matching_results.tsv + output/candidate_pairs.tsv
#
# Needs student_resource/dataset/{train,test} next to src/, the .venv (see README), ~7 h on 12 CPU threads +
# an 8 GB CUDA GPU, 31 GB RAM and ~60 GB free disk. GPU fp16 retrieval and multi-threaded tree training make
# a rebuild agree with the uploaded file on ~99% of pairs rather than byte for byte.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python}
[ -x "$PY" ] || PY=python
export PYTHONIOENCODING=utf-8

# 1. normalize -> blocking -> prune -> features -> stage 2 -> refine -> stage 3 -> predict.
#    Produces the pruned candidate pairs and base features every later step uses (its own models are the
#    "rebuild" runs of submissions 14's France rows; they are not needed for submission 16 itself).
$PY -m src.run_pipeline --config configs/base.yaml

# 2. look-alike name columns and relative candidate counts, per split (statistics from each split's own names)
for split in train test; do
  $PY -m src.lookalike --config configs/base.yaml --split "$split"
  $PY -m src.relcounts --config configs/base.yaml --split "$split"
done

# 3. model A, used for country labels seen in training (US, India): all 13 look-alike columns
A2=$($PY -m src.train_matcher --config configs/base.yaml configs/stage2.yaml --extra features_la features_rel \
     --drop s1_n_cands --name stage2la | tail -1)
$PY -m src.refine --config configs/base.yaml --run "$A2"
A3=$($PY -m src.train_matcher --config configs/base.yaml configs/stage3.yaml \
     --extra features_la features_rel "features_s3_$A2" --drop s1_n_cands --name stage3la | tail -1)
$PY -m src.predict --config configs/base.yaml configs/output_la.yaml --run "$A3"

# 4. model B, used for labels never seen in training (France): categorical look-alike columns only
#    (the three word-frequency values transfer badly to a new country); unseen-label threshold 0.75 from base.yaml
DROP="s1_n_cands la_extra_lratio_min la_extra_lshare_max la_missing_lshare_min"
B2=$($PY -m src.train_matcher --config configs/base.yaml configs/stage2.yaml --extra features_la features_rel \
     --drop $DROP --name stage2lacat | tail -1)
$PY -m src.refine --config configs/base.yaml --run "$B2"
B3=$($PY -m src.train_matcher --config configs/base.yaml configs/stage3.yaml \
     --extra features_la features_rel "features_s3_$B2" --drop $DROP --name stage3lacat | tail -1)
$PY -m src.predict --config configs/base.yaml configs/output_lacat.yaml --run "$B3"

# 5. per-country combination: seen labels from model A, unseen labels from model B -> output/
$PY -m src.combine --config configs/base.yaml --seen output_la --unseen output_lacat --out output
echo "model A: $A2 -> $A3; model B: $B2 -> $B3; final files in output/"
