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

## 3. Run everything (about 7 hours on the hardware above)
```bash
bash scripts/reproduce_best.sh
```
This is the configuration of the final submission (public leaderboard 0.983405):
1. **Base pipeline:** `python -m src.run_pipeline --config configs/base.yaml`. It produces the candidates, the base features and the "rebuild" stage-3 model.
2. **Look-alike and relative-count columns for both splits:** `src.lookalike` and `src.relcounts`.
3. **Model A and model B:** `bash scripts/train_big.sh A` and `bash scripts/train_big.sh B`. Each is stage 2 + refine + stage 3 on 90% / 95% of the other folds' S1 groups. Model A uses all look-alike columns; model B drops the three word-frequency columns.
4. **Decision:** `python -m src.blend`.
   - Labels seen in training (US, India): mean of A and B at threshold 0.60, margin 0.50.
   - The label never seen in training (France): mean of B and the rebuild model at threshold 0.75, margin 0.50.
5. **Post-filter:** `python -m src.combine --seen output_blend --unseen output_blend --unseen-postfilter --out output` drops vocabulary-swap look-alikes from the France rows.

The base pipeline alone, stage by stage (each stage caches its output under `data/cache/`):
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
| `lookalike.py` | look-alike name columns: word alignment, filler / ordinary-vocabulary / rare classes from per-country S2/S3-vs-S1 name frequencies, swap flag and shape |
| `relcounts.py` | candidate count of an S1 relative to its split/country mean (replaces the raw count) |
| `blend.py` | final decision: mean of several runs per country group, thresholds, both output files |
| `train_xgb.py` | XGBoost matcher on the same folds (tried; not in the final blend) |
| `combine.py` | rows of seen country labels from one prediction, unseen labels from another (same candidates) |
| `postfilter.py` | optional rule that drops accepted vocabulary-swap pairs (used for models without the look-alike columns) |
| `loco.py` | leave-one-country-out check (train on one country, score the other) |
| `decide.py` | one best S1 per record, threshold + margin tuned for macro F0.5 |
| `predict.py` | test inference and the two output files |
| `validate.py` | official validator + subset/format checks |
| `metrics.py` | macro F0.5 (exact competition definition) and blocking diagnostics |
| `run_pipeline.py` | runs all stages in order |
