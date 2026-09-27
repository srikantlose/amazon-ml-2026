#!/usr/bin/env bash
# Models A and B of the final configuration retrained on more S1 groups per fold (stage2_big / stage3_big).
#   bash scripts/train_big.sh A     # or B
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python}
export PYTHONIOENCODING=utf-8
if [ "$1" = "A" ]; then DROP="s1_n_cands"; NAME=la; OUT=configs/output_la_big.yaml
else DROP="s1_n_cands la_extra_lratio_min la_extra_lshare_max la_missing_lshare_min"; NAME=lacat; OUT=configs/output_lacat_big.yaml; fi
R2=$($PY -m src.train_matcher --config configs/base.yaml configs/stage2_big.yaml --extra features_la features_rel \
     --drop $DROP --name "stage2${NAME}big" | tail -1)
echo "stage 2: $R2" >&2
$PY -m src.refine --config configs/base.yaml --run "$R2"
R3=$($PY -m src.train_matcher --config configs/base.yaml configs/stage3_big.yaml \
     --extra features_la features_rel "features_s3_$R2" --drop $DROP --name "stage3${NAME}big" | tail -1)
echo "stage 3: $R3" >&2
$PY -m src.predict --config configs/base.yaml "$OUT" --run "$R3"
echo "done $1: $R2 $R3" >&2
