#!/usr/bin/env bash
# End-to-end dry run on a subset of the 2025 price data.
# Run from the repo root on the SageMaker instance, inside the conda_pytorch_p310 env:
#   source activate pytorch_p310 && bash scripts/dry_run.sh 2>&1 | tee logs/dry_run.log
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
CFG="configs/base.yaml configs/dryrun.yaml"

echo "== 0. metric self-checks"
python -m src.metrics

echo "== 1. dataset (Kaggle mirror of the 2025 problem)"
if [ ! -f data/raw/dryrun/train.csv ]; then
  if ! find data/raw/2025 -name train.csv 2>/dev/null | grep -q .; then
    kaggle datasets download -d suvroo/amazon-ml -p data/raw/2025 --unzip
  fi
  python -m src.prepare_dryrun --src data/raw/2025 --n-train "${N_TRAIN:-5000}" --n-test "${N_TEST:-2000}"
fi

echo "== 2. images"
python -m src.download_images --config $CFG --split train
python -m src.download_images --config $CFG --split test

echo "== 3. folds + text features"
python -m src.cv --config $CFG
python -m src.features_text --config $CFG

echo "== 4. embeddings"
python -m src.embed --config $CFG --what text image

echo "== 5. GBM"
GBM_BASE=$(python -m src.train_gbm --config $CFG --name base | tail -n 1)
GBM_ALL=$(python -m src.train_gbm --config $CFG --features hand svd txt img has_image --name all | tail -n 1)

echo "== 6. fusion"
FUSION=$(python -m src.train_fusion --config $CFG | tail -n 1)

echo "== 7. ensemble + submission"
ENS=$(python -m src.ensemble --config $CFG --runs "$GBM_BASE" "$GBM_ALL" "$FUSION" --write | tail -n 1)
SUB="submissions/dryrun/${ENS}.csv"

echo "== 8. validate (must pass), then broken copies (must fail)"
python -m src.validate_submission --config $CFG --sub "$SUB"
head -n -1 "$SUB" > logs/broken_rows.csv
python - "$SUB" <<'EOF'
import sys, pandas as pd
s = pd.read_csv(sys.argv[1], dtype={"sample_id": str})
s.loc[0, "price"] = float("nan"); s.to_csv("logs/broken_nan.csv", index=False)
s = pd.read_csv(sys.argv[1], dtype={"sample_id": str})
s.iloc[::-1].to_csv("logs/broken_order.csv", index=False)
s[["price", "sample_id"]].to_csv("logs/broken_cols.csv", index=False)
EOF
for f in logs/broken_rows.csv logs/broken_nan.csv logs/broken_order.csv logs/broken_cols.csv; do
  if python -m src.validate_submission --config $CFG --sub "$f" > /dev/null; then
    echo "VALIDATOR BUG: $f passed"; exit 1
  else
    echo "rejected as expected: $f"
  fi
done

echo "== 9. held-out score (CV should be close to this)"
python -m src.prepare_dryrun --score "$SUB"
tail -n 5 experiments_dryrun.csv
echo "DRY RUN OK"
