# Business Entity Resolution: reproduction guide

This folder regenerates `output/matching_results.tsv` and `output/candidate_pairs.tsv` from the
competition data using only the code here. No external data or services are used; the only
model family is LightGBM (MIT license).

## 1. Environment
- Python 3.11+ (developed on 3.13, Windows 11; also runs on Linux).
- Hardware used: 12-thread CPU, 31 GB RAM, NVIDIA RTX 4060 8 GB (the GPU is used for
  candidate retrieval only; CPU works but blocking is much slower). About 60 GB of free disk for caches.

```bash
python -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128   # Windows: .venv\Scripts\pip
.venv/bin/pip install -r requirements.txt
```

## 2. Data
Put the organizers' dataset folder next to `src/`, so the layout is:
```
student_resource/dataset/train/train_source{1,2,3}.tsv, train_ground_truth.tsv
student_resource/dataset/test/test_source{1,2,3}.tsv
student_resource/utils/validate_submission.py
```
Or set `data_dir` / `paths.validator` in `configs/base.yaml`.

## 3. Run everything (about 5 hours on the hardware above)
```bash
python -m src.run_pipeline --config configs/base.yaml
```
Or stage by stage (each stage caches its output under `data/cache/`):
```bash
python -m src.normalize  --config configs/base.yaml                        # ~5 min
python -m src.blocking   --config configs/base.yaml --split train          # ~35 min
python -m src.blocking   --config configs/base.yaml --split test           # ~30 min
python -m src.prune      --config configs/base.yaml --split train          # ~45 min
python -m src.prune      --config configs/base.yaml --split test           # ~30 min
python -m src.features   --config configs/base.yaml --split train          # ~4 min
python -m src.features   --config configs/base.yaml --split test           # ~4 min
python -m src.train_matcher --config configs/base.yaml configs/stage2.yaml --name stage2       # prints <s2>
python -m src.refine     --config configs/base.yaml --run <s2>             # stage-2 context, train + test
python -m src.train_matcher --config configs/base.yaml configs/stage3.yaml --extra features_s3_<s2> --name stage3  # prints <s3>
python -m src.predict    --config configs/base.yaml --run <s3>             # writes output/, runs validators
```

## 4. Code map (`src/`)
| File | Purpose |
|---|---|
| `data.py` | TSV reading (tab, no quoting, no NA parsing, mojibake repair), label arrays, `\n` line endings |
| `normalize.py` | transliteration (map learned from train pairs + anyascii), legal forms, address terms, state codes |
| `blocking.py` | char 3-gram TF-IDF → SVD vectors with exact GPU top-k; word-token TF-IDF inverted index; exact name and brand-token+house keys; all within each country label |
| `prune.py` | stage-1 LightGBM ranker over retrieval scores + cheap string similarities; keeps the top S1 per record |
| `features.py` | rapidfuzz similarities, numeric/state agreement, name uniqueness, competition and sibling-record features |
| `train_matcher.py` | LightGBM pair classifier, 3 S1-grouped folds, out-of-fold predictions, decision tuning |
| `refine.py` | stage-3 context features from stage-2 probabilities |
| `decide.py` | one best S1 per record, threshold + margin tuned for macro F0.5 |
| `predict.py` | test inference and the two output files |
| `validate.py` | official validator + subset/format checks |
| `metrics.py` | macro F0.5 (exact competition definition) and blocking diagnostics |
| `run_pipeline.py` | runs all stages in order |
