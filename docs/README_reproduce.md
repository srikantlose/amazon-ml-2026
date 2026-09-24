# Business Entity Resolution: reproduction guide

This folder regenerates `output/matching_results.tsv` and `output/candidate_pairs.tsv` from the
competition data using only the code here. No external data or services are used.

## 1. Environment
- Python 3.11+ (developed on 3.13, Windows 11; also runs on Linux).
- NVIDIA GPU recommended for blocking (developed on an RTX 4060 8 GB). CPU works but is much slower.

```bash
python -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128   # Windows: .venv\Scripts\pip
.venv/bin/pip install -r requirements.txt
```

## 2. Data
Put the organizers' dataset folder here, so the layout is:
```
student_resource/dataset/train/train_source{1,2,3}.tsv, train_ground_truth.tsv
student_resource/dataset/test/test_source{1,2,3}.tsv
student_resource/utils/validate_submission.py
```
Alternatively, set `data_dir` / `paths.validator` in `configs/base.yaml`.

## 3. Run everything
```bash
python -m src.run_pipeline --config configs/base.yaml
```
Or run it stage by stage (each stage caches its output under `data/cache/`):
```bash
python -m src.normalize     --config configs/base.yaml                  # ~5 min
python -m src.blocking      --config configs/base.yaml --split train
python -m src.blocking      --config configs/base.yaml --split test
python -m src.features      --config configs/base.yaml --split train
python -m src.features      --config configs/base.yaml --split test
python -m src.train_matcher --config configs/base.yaml                  # prints <run_id>
python -m src.predict       --config configs/base.yaml --run <run_id>   # writes output/, runs validators
```

## 4. Code map (`src/`)
| File | Purpose |
|---|---|
| `data.py` | TSV reading (tab, no quoting, no NA parsing, mojibake repair), label arrays, output writer (`\n` endings) |
| `normalize.py` | transliteration (learned from train pairs + anyascii), canonical legal forms, address terms, state codes |
| `blocking.py` | char 3-gram TF-IDF → SVD vectors; exact GPU top-k S1 per S2/S3 record within each country label |
| `features.py` | rapidfuzz string similarities, numeric/address agreement, competition features |
| `train_matcher.py` | LightGBM pair classifier, S1-grouped 3-fold OOF predictions, decision tuning |
| `decide.py` | one best S1 per record + threshold/margin, tuned for macro F0.5 |
| `predict.py` | test inference and output files |
| `validate.py` | official validator + subset/format checks |
| `metrics.py` | macro F0.5 (exact competition definition) and blocking diagnostics |
