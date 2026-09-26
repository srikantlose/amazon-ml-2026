# Handoff: Amazon ML Challenge 2026, Business Entity Resolution

This is the full state of the project at handoff, **27 Sep 2026, ~01:00 IST**:
- what the task is
- how the pipeline works
- every experiment and leaderboard result, with its numbers
- what was learned, what failed
- what is still open, and what must happen before the deadline

Read sections 0 and 11 first if you are short on time.

---

## 0. Status at handoff

| Item | Value |
|---|---|
| Best public leaderboard (LB) score | **0.982629**, submission **16** (git tag `sub-16`, folder `submissions/16_hybrid_la_us_in_lacat075_fr/`); it was also the last upload of 26 Sep |
| Latest upload | submission 18 = **0.98234** (worse; see §7) |
| Leader when last checked (25 Sep) | 0.985884; not re-checked since |
| Deadline | **27 Sep 2026 23:59 IST** (window opened 25 Sep 00:00 IST) |
| Uploads left | 5 per day; **4 left on 27 Sep** (one used by submission 18 after midnight) |
| Final ranking | private LB (the part of test not in the public LB) **plus documentation quality**; top teams' packages are reviewed |
| Out-of-fold (OOF) macro F0.5 of the best models | model A 0.98741 (used for US/India), model B 0.98705 (used for France) |
| Test performance split, from probe submission 17 | **US+India 0.9844** (OOF level 0.9874), **France 0.9727** |

**Must do before the deadline:**
1. Keep submission 16's file (or something that beats it) as the **last upload** of 27 Sep.
2. Build the official final zip with `scripts/make_submission_zip.py`. It needs the **team name**, which the owner has not provided yet (ask them).
3. Fill the placeholders in `docs/Documentation_template.md` (§11).

---

## 1. The competition

**Task.** For each Source-1 (S1) business entity in test, list the Source-2/3 (S2/S3) records that refer to the same business.
- Each record has `entity_id`, `business_name`, `business_address` and `country`.
- Train has country labels **US and India**. Test adds **France**, which never appears in train (259,452 of 1,732,544 test S1 = **14.9752%**).
- Treat `country` as an open set: no hard-coding or one-hot of US/India, and every test S1 must appear in the output.

**Metric.** Macro F0.5 over all S1 entities: F0.5 = 1.25·P·R / (0.25·P + R) per S1, then averaged.
- A singleton (no true matches) scores 1 for an empty prediction and 0 for any prediction.
- An entity with true matches scores 0 for an empty prediction.
- Per S1 with a true positives, b false positives and T true matches: F = 1.25a / (a + b + 0.25T).
- The metric is precision-heavy. Adding one false positive to a 3–4-match S1 costs about 0.17–0.21. Dropping one true match costs about 0.07. Dropping a pair therefore pays off when its precision is below ≈ 0.74.

**Output files.** Both are TSV with **`\n` line endings**; the official validator only strips `\n`, so a Windows `\r` corrupts the last id.
- `matching_results.tsv`: `source1_entity_id` and `matched_entity_ids` (comma-joined, no quotes), one row per test S1, empty string for no match. **This is the only file scored**, and the only file uploaded to the portal.
- `candidate_pairs.tsv`: `source1_entity_id` and `candidate_entity_ids`, the exact candidate set the model scored. The matches must be a subset of it. It is required in the final zip but not scored.

**Validator.**
```bash
python student_resource/utils/validate_submission.py --matching <file> [--candidate <file>] \
    --test-dir student_resource/dataset/test --check-ids
```
It must print `PASS`. Rejected: ids not in test, S1 ids in the lists, duplicate ids, duplicate or missing S1 rows.

**Rules.**
- Final model must be MIT or Apache 2.0 licensed and ≤ 8B parameters. We only use LightGBM (MIT); there are no pretrained models.
- **No external data**: no geocoding, registries or entity-resolution APIs.
- We interpret "using only the provided training data" as **no pseudo-labelling on test**. Unlabelled test statistics (IDF, word frequencies) are fine and used.

**Leaderboard.** Public LB = a subset of test; private LB = the rest. Every upload is predictions for the full test set. There are 5 uploads per day.

**Final package.** `<team>_submission.zip` containing:
- `output/{matching_results.tsv,candidate_pairs.tsv}`
- `code/business_entity_resolution/{src/, README.md, requirements.txt, …}`
- the filled `Documentation_template.md` (`.md` or `.pdf`)

`scripts/make_submission_zip.py --team "<name>" --outputs <folder with the two TSVs>` builds it.

**Data facts (verified on all training pairs):**
- **Train:** 2,206,821 S1 (US 1,323,633 / India 883,188); S2 5,034,616; S3 5,285,603 (S2+S3: US 6,186,873 / India 4,133,346).
- **Test:** 1,732,544 S1 (US 663,106 / India 809,986 / France 259,452); S2 4,887,273; S3 5,082,316 (S2+S3: US 3,817,031 / India 4,717,565 / France 1,434,993).
- **Each S2/S3 record matches at most one S1.** Matched records **always share the S1's country label**.
- **About 26% of train S2/S3 records match nothing** (distractors).
- **Singletons are 5.58% (US) / 5.59% (India)**. Exactly-one-match S1 are 5.42% / 5.37%. The mean is 3.46 matches per S1; the maximum is 11 (≤ 5 from S2, ≤ 6 from S3).
- **Records per S1:** train 4.67 (US) / 4.68 (India); test 5.76 (US) / 5.82 (India) / 5.53 (France). Test has about 1.9× more distractors per S1.
- **Name overlap:** 39% of S1 names are shared by another S1 of the same country, and only 26% of true pairs have identical normalized names.
- **Noise types:**
  - Indian scripts in about 4% of records
  - website-style names (`cardiologyphysicians.com`)
  - junk prefixes (`--`, `<<`)
  - legal-form variants
  - typos, and injected accents in France (`Çulture`, `Àmicale`)
  - reordered address components
  - state code vs name (US: S1/S2 codes, S3 names; India: S1 names, S3 codes or native script)
  - French region vs department
  - trade names / DBA / "formerly known as" prefixes
  - acronyms (`PBS` for `Puits Bouge Sport`)
  - word-order shuffles
  - contractions (`gaming` → `gg`)
  - garbled words (`Ptorfgrma` for `Program`)
  - house-number noise (`008`, `N°`, `9B` = `9 bis`, dropped or changed digits)
  - empty addresses (4.4% of matched records vs 0.3% of unmatched)
  - Latin-1 mojibake (`â€“`, repaired at read time)
  - injected filler words (§6)
  - **planted look-alike distractors** (§6)

---

## 2. Repository map

- **Remote:** `https://github.com/srikantlose/amazon-ml-2026.git` (private), branch `master`.
- **Tags:** `sub-01`…`sub-18`. Each maps a leaderboard file to the exact code state.
- **What git ignores:**
  - `student_resource/`, `data/`, `output/`, `output_*/`, `logs/`, `*.parquet`, `*.npy`, `*.zip`
  - `submissions/*/*.tsv`: the TSVs exist locally, the `NOTE.md` files are tracked.

| Path | Role |
|---|---|
| `src/data.py` | TSV reading (tab, `QUOTE_NONE`, no NA parsing), mojibake repair, labels (`build_labels`), output headers |
| `src/normalize.py` | name/address normalization; learned non-Latin → Latin map (from train pairs) + anyascii; legal forms; street terms; state/region codes (incl. French departments → region); number canonicalization; `records_path`, `load_records` |
| `src/blocking.py` | candidate generation per country label (char 3-gram TF-IDF → SVD 256 fp16, exact GPU top-k; word-token TF-IDF inverted index; exact keys); `pruned_path` |
| `src/prune.py` | stage-1 LightGBM ranker, keeps the top 8 S1 per record with p ≥ 0.001; writes `cands_pruned.parquet` |
| `src/features.py` | pair features (rapidfuzz), competition/group features, sibling features, name frequency/ambiguity; `features_dir`, `load_features`, `write_parts` |
| `src/train_matcher.py` | LightGBM with 3 S1-grouped folds (crc32 of the S1 id), OOF for all pairs, decision tuning; `--extra` folders, `--drop` columns; prints the run id on stdout (logs go to stderr) |
| `src/refine.py` | stage-3 context features from a stage-2 run's probabilities (train OOF, test fold-average) → `features_s3_<run>` |
| `src/decide.py` | best S1 per record; accept if p ≥ t and margin ≥ m; grid tuning for exact macro F0.5 |
| `src/predict.py` | test inference (average of fold models), unseen-country thresholds, writes both TSVs to `paths.output_dir`, runs validators; `--reuse-probs`, `--threshold`, `--margin` |
| `src/validate.py` | official validator + subset/format checks |
| `src/metrics.py` | exact macro F0.5 and blocking diagnostics |
| `src/run_pipeline.py` | normalize → blocking → prune → features → stage 2 → refine → stage 3 → predict |
| `src/loco.py` | leave-one-country-out check (`--extra`, `--drop`, `--reference-run`) |
| **`src/lookalike.py`** | **new (26 Sep):** look-alike name columns → `features_la` (§6) |
| **`src/relcounts.py`** | **new:** `s1_n_cands_rel` = S1 candidate count / split-country mean → `features_rel` |
| **`src/postfilter.py`** | **new:** drops accepted "vocabulary swap" pairs from a matching file (rule, no model) |
| **`src/combine.py`** | **new:** seen-country rows from one prediction, unseen-country rows from another (same candidates), optional post-filter on the unseen rows; validates |
| `src/ensemble.py`, `src/utils.py` | probability averaging (tested, unused); config/log/timer/run-id helpers. `load_config` takes a **list** of YAML paths; later files override earlier ones |
| `configs/base.yaml` | all defaults (paths, blocking views, prune, model, decision grid, `decision.unseen_country` = 0.75 / 0.5) |
| `configs/stage2.yaml`, `stage3.yaml` | stage-2 (train_frac 0.45, lr 0.05) and stage-3 (train_frac 0.75, lr 0.05) overrides |
| `configs/unseen_065/085/095.yaml` | unseen-country (France) threshold overlays |
| `configs/output_la.yaml`, `output_lacat.yaml`, `output_la080.yaml` | send `predict` output to another folder (so runs don't overwrite `output/`) |
| `configs/loco_grid.yaml` | LOCO decision grid up to 0.99 |
| `configs/exp_*.yaml`, `stage3_b/c.yaml` | old experiment overlays (ensemble members, big-train) |
| `scripts/save_submission.py` | archives an output folder as `submissions/NN_<run>/` (TSVs + meta + configs + NOTE.md), `git add -A`, commit, tag `sub-NN`, push |
| `scripts/make_submission_zip.py` | official final zip (now includes `output_la.yaml`, `output_lacat.yaml`, `scripts/reproduce_best.sh`) |
| **`scripts/reproduce_best.sh`** | **one command to rebuild submission 16** (§10) |
| `docs/Documentation_template.md` | methodology write-up for the final zip (has placeholders, §11) |
| `docs/approach_notes.md` | running lab notebook with all experiments and numbers |
| `docs/README_reproduce.md` | becomes `code/business_entity_resolution/README.md` in the zip |
| `docs/error_analysis.txt` | OOF error analysis (25 Sep) |
| `analysis/` | scripts behind the findings in §6 to §8 (see `analysis/README.md`) |
| `submissions/NN_*/NOTE.md` | one note per submission: run, change, OOF, LB score |
| `experiments.csv` | every training run: timestamp, run id, features, OOF, notes |
| `notebooks/01_eda.ipynb` | EDA |

**Cache layout (`data/cache/`, about 45 GB on disk; C: had about 45 GB free):**
- **Per split, `train/` and `test/`:**
  - `records_s1.parquet`, `records_s23.parquet` (normalized fields: `name_n`, `name_core`, `name_ns`, `addr_n`, `addr_nums`, `house`, `postal`, flags, `src`, `country`)
  - `vec_{name,addr}_{s1,s23}.npy`
  - `cands.parquet`: the blocking union, about 9 GB. **Never load it whole.**
  - `cands_pruned.parquet`: `s1`, `rec` row indices, retrieval scores, `p1`, `views`
  - `features/part_XXX.parquet`: 5M-row parts
  - `features_la/`, `features_rel/`, `features_s3_<stage2 run>/`, all aligned row by row with `cands_pruned`
- **Per run, `runs/<run_id>/`:** `model_fold{0,1,2}.txt`, `oof.npy` (train pairs), `test_prob.npy`, `meta.json` (features, feature_dirs, decision, per-country OOF), `decision_grid.csv`, `importance.csv`
- **Run folders present:** `lgb_stage2_0925_175903`, `lgb_stage3_0925_183300` (models of submissions 07–10), `lgb_stage2_0926_021300`, `lgb_stage3_0926_025245` (rebuild), `lgb_stage2la_0926_023743`, `lgb_stage3la_0926_033632` (model A), `lgb_stage2lacat_0926_043451`, `lgb_stage3lacat_0926_051023` (model B).
- **Caveat:** the `oof.npy`/`test_prob.npy` of the 25 Sep runs belong to the old candidate set, which the rebuild overwrote, so they no longer line up with `cands_pruned.parquet`. Older runs are in `data/runs_archive/`.

**Output folders:**

| Folder | Contents |
|---|---|
| `output_la/` | model A (US/India rows used by 13–17; its US/India rows do not depend on the France threshold) |
| `output_lacat/` | model B @ France 0.75 = the France rows of 16. It held 0.85 for 15 until it was re-predicted at handoff to verify 16 |
| `output/` | model B @ France 0.75 (same France rows); `reproduce_best.sh` writes the final combination here |
| `output_la080/` | model A @ US/India 0.80 (submission 18) |
| `output_rebuild/` | rebuild @ France 0.95 + post-filter (France rows of 14) |
| `output_rebuild_raw/` | the same before the filter |

---

## 3. Environment and runtime

- **Machine:** Windows 11, Ryzen 5 7600 (12 threads), RTX 4060 8 GB, 31 GB RAM.
- **Python and packages:** Python 3.13.7 in `.venv`: torch 2.11.0+cu128, lightgbm 4.7.0, rapidfuzz 3.14.6, pandas 3.0.6, pyarrow 25.0.1, scikit-learn 1.9.1, numpy 2.5.2, anyascii, pyyaml, scipy, tqdm.
- **Timings:**
  - Full base pipeline about 4.5 h (the last run went 23:09 → 03:42):
    - normalize 5 min
    - blocking about 45 + 30 min
    - prune train 60 min (group stats 20 min, 4 ranker fits 13 min, OOF scoring 25 min) and prune test 45 min
    - features 4 min per split
    - stage 2 about 20 min
    - refine about 17 min (incl. 13–14 min test scoring)
    - stage 3 about 20 min
    - predict 6 min
  - Look-alike columns about 1 min per split (8 workers); relcounts about 1 min per split.
  - Model A or model B chain (stage 2 → refine → stage 3 → predict) about 50–75 min.
  - One LOCO run about 8–12 min.
- **Windows gotchas:**
  - **Do not edit `src/` while a job is running.** Worker pools (spawn) re-import `src/` modules; editing one mid-run crashed a 30-minute blocking job once.
  - **Memory is tight.**
    - Pruning holds about 19 GB.
    - Two trainings at once are fine; more is not.
    - `blocking.token_workers` = 8, because 12 ran out of memory.
    - Keep ad-hoc analyses under about 3 GB while the pipeline runs (read only the columns you need).
  - Encoding: set `PYTHONIOENCODING=utf-8` for console output with accents.
  - LightGBM prints `LGBMDeprecationWarning: eval_set` warnings; they are harmless.
- **Log quirks:**
  - `prune` logs the test split as `test: kept N of M pairs`, without a path. An earlier chain script waited for a path and hung.
  - `features` logs `wrote K parts to …\test\features`.

---

## 4. Pipeline in detail (final configuration)

### 4.1 Normalization (`src/normalize.py`)
- **Latin conversion:**
  - Mojibake repair.
  - A learned map from train pairs converts non-Latin tokens and address components to Latin: 4,935 tokens and 84 components with `min_count` 5 and `min_share` 0.5, e.g. `प्राइवेट`→private, `தமிழ்நாடு`→tamil nadu.
  - anyascii handles the rest, including accents.
- **Tokens:** lowercase; `&`→and; dots/apostrophes deleted (`l.l.c.`→llc); other ASCII punctuation → space.
- **Legal forms:** llc, inc, corp, co, ltd, pvt, llp, lp, plc, pc, opc, sarl, sas, sasu, sa, sci, eurl, snc, selarl, cie; **`compagnie`→`cie`** (see §9). Name filler: the, and, of, india.
- **Address terms:** street→st, road→rd, avenue/av→ave, bd→blvd, r→rue, all→allee, …; north→n etc.; near→nr.
- **States:** state names → codes (US, India). French regions and departments → region code:
  - `hauts de france`/`nord`/`pas de calais`→hdf
  - `nouvelle aquitaine`/`gironde`→naq
  - `pays de la loire`/`loire atlantique`→pdl
  - S1 always writes the region; S2/S3 write the region or the department.
- **Numbers (added 25–26 Sep):**
  - `N°`/`Nº` → `no `; glued `no169`/`ndeg14`/`n14` → digits; leading zeros stripped (`008`→8).
  - The house number is the digit run of the first token that starts with a digit (`9b`, `9 bis` → 9).
  - This removed most of the formatting "house conflicts" in true pairs (27% of them had been format artifacts).
- **Derived fields:**
  - `name_core`: legal and filler words removed
  - `name_ns`: core without spaces
  - `addr_nums`: digit tokens
  - `house`, `postal` (5–6 digit runs)
  - `is_domain`, `name_non_latin`, `addr_empty`

### 4.2 Blocking (`src/blocking.py`)
Every S2/S3 record queries S1 records of its own country label only.

| View | What | Train pair recall |
|---|---|---|
| `full` | char 3-gram TF-IDF (fit on up to 1M rows per split), TruncatedSVD 256 per field, fp16, L2; score 0.5·cos(name) + 0.5·cos(address); exact GPU top-**30** | 0.9630 |
| `tfull` | word-token TF-IDF inverted index (tokens in > 2,000 S1 excluded, min score 0.02), top 10 by name+address | 0.8973 |
| `tname` / `taddr` | same index, top 5 by name / by address | 0.3554 / 0.7650 |
| `key` | exact `name_core` join (skipped if > 20 S1 share it) | 0.4276 |
| `hkey` | exact (first name token, house number) join (skipped if > 50 S1) | 0.5707 |
| **union** | 441,333,170 train pairs (200 per S1); test 436,043,021 | **0.9881** (oracle F0.5 0.9964) |

### 4.3 Stage-1 pruning (`src/prune.py`)
- **Model:** a LightGBM ranker over retrieval scores/ranks, per-record and per-S1 group statistics (rank, gap, counts; computed per country segment), and three rapidfuzz features (core-name ratio, address token-set, digit-token overlap).
- **Training:** OOF for train using 3 fold models (5% of S1 groups each), plus `stage1_full.txt` for test.
- **Kept:** top 8 per record with p ≥ 0.001 → **12,223,957 train pairs** (1.18 per record; recall **0.9845**, oracle 0.9953; 61.5% positives) and **11,749,613 test pairs** (1.18 per record).
- **Recall at other settings:** top 3 0.9812 · top 5 0.9833 · top 10 0.9851 · top 12 0.9854.

### 4.4 Pair features (`src/features.py`), 83 columns
- **Names:** rapidfuzz ratio / partial / token-sort / token-set / Jaro-Winkler on `name_n`, `name_core`, `name_ns`; legal-form and first-token agreement; brand token of one name inside the other; S1 name frequency and record-name frequency (count of S1 sharing the core name); near-identical names among the record's candidates.
- **Addresses:** ratio / partial / token-sort / token-set; digit-token overlap; house / postal / state agreement (state from the first/last alphabetic token, "la" excluded); empty-address flags.
- **Other:**
  - retrieval cosines, `p1`, view bits
  - competition features: rank / gap / margin of the pair among its record's candidates and its S1's candidates for full, token and p1 scores; `s1_n_cands`, `rec_n_cands`, `s1_sum_p1`, `rec_sum_p1`
  - sibling features: best similarity to the S1's other confident records
  - length differences, source flag, non-Latin / website flags
- **No country feature.**

### 4.5 Stage 2 and stage 3 (`src/train_matcher.py`, `src/refine.py`)
- **LightGBM binary:** num_leaves 127, min_child_samples 100, feature/bagging fraction 0.8, λ2 1, early stopping 100.
- **Stage 2:** lr 0.05, up to 5,000 trees, each fold trained on 45% of the other folds' S1 groups (hash-sampled; all their pairs) with a 2% early-stopping slice.
- **Stage 3:** stage-2 features + 14 context columns from stage-2 probabilities:
  - `p2`, `p2_rec_rank/gap/margin`, `p2_s1_rank/gap/margin`
  - `p2_s1_n_linked`, `p2_s1_n_linked_same_src`, `p2_s1_sum`
  - `p2sib_*` (siblings by p2)
  - Trained on 75% of S1 groups, up to 6,000 trees.
- **Folds and test:** 3 folds by crc32(S1 id) % 3. Every train pair gets an OOF probability; test uses the average of the three fold models.

### 4.6 Decision (`src/decide.py`, `src/predict.py`)
- **Rule:** each record keeps only its highest-probability S1 (one-to-one). It is accepted if p ≥ t and (p − runner-up) ≥ m.
- **Tuning:** (t, m) are grid-searched on OOF for exact macro F0.5 over all 2.2M train S1 (singletons and blocking misses included).
  - Model A: t = 0.65, m = 0.6.
  - Model B: t = 0.65, m = 0.5.
- **Unseen labels:** pairs of S1 whose country label never occurs in train (France) use `decision.unseen_country` (0.75 / 0.5 in base.yaml).

### 4.7 Final configuration (submission 16)
| Rows | Source | Settings |
|---|---|---|
| US, India (1,473,092 S1) | **model A** `lgb_stage2la_0926_023743` → `lgb_stage3la_0926_033632` | 13 look-alike columns + `s1_n_cands_rel`, raw `s1_n_cands` dropped; 96 / 110 features; OOF 0.98619 / **0.98741** (US 0.98760, India 0.98713); t = 0.65, m = 0.6 |
| France (259,452 S1) | **model B** `lgb_stage2lacat_0926_043451` → `lgb_stage3lacat_0926_051023` | same, but also drops `la_extra_lratio_min`, `la_extra_lshare_max`, `la_missing_lshare_min`; 93 / 107 features; OOF 0.98576 / **0.98705**; France t = 0.75, m = 0.5 |
| Combination | `python -m src.combine --seen output_la --unseen output --out <dir>` | both predictions share the same candidate file (checked byte for byte) |

**Output profile of 16 (pairs per S1):** US 2,245,399 (3.386), India 2,699,987 (3.333), France 852,570 (3.286). Train truth is 3.46.

---

## 5. Chronology

- **25 Sep:**
  - Normalization, blocking, pruning and the stage-2/3 cascade built.
  - First upload was submission 05 (0.974431).
  - Out-of-fold 0.986, but LB 0.974–0.976: an unexplained gap of about 0.011.
  - Tried a stricter France threshold (0.65 → 0.75 → 0.85 → 0.95, each about +0.0004–0.0005 LB) and French department→region normalization (+0.0005).
- **25 Sep night:**
  - Rebuild with house-number canonicalization.
  - Investigation of the gap. One lens (decision/metric) finished before a usage limit stopped the rest. It showed the France threshold route was exhausted, the LOCO optimum had been capped by the grid, and US/India predicted more pairs per S1 on test than OOF.
- **26 Sep 00:50–05:30:**
  - Read France pairs by eye → found **planted look-alikes**.
  - Built the filler/vocabulary statistics and `src/lookalike.py`, validated on train labels.
  - Independent audit. Fixes: scale-free statistics, `relcounts`, contraction rule.
  - Model A, then LOCO and prediction explanations showed model A over-rejects French filler matches → model B (categorical) → per-country combination.
- **26 Sep day:**
  - Uploads 14 (+0.0059), 15, 16 (best).
  - Probe 17 (US+India 0.9844 / France 0.9727).
  - Re-upload of 16.
- **27 Sep 00:3x:** upload 18 (US/India threshold 0.80 → worse).

---

## 6. Main finding: planted look-alike records and injected filler

**1. Leaderboard evidence.**
- Submissions 06/07/08/10 changed only the France rows (US/India byte-identical).
- Turning the LB deltas into the precision of the France pairs removed at each step (using the metric's closed form; noise ≈ 3e-5):
  - [0.65, 0.75): false-positive rate 0.67–0.71
  - [0.75, 0.85): 0.53–0.56
  - [0.85, 0.95): 0.41–0.43, i.e. those France pairs were only about 58% correct, against about 90% for US/India pairs in that band
- Script: `analysis/band.awk`, plus a streaming diff of the submission files.

**2. Reading those France pairs.** About half keep the S1's address and first word but **swap the descriptor for another ordinary word**:
- `Mauges Amis SAS` → `Mauges Collectif SAS` / `Mauges Gestion SAS`
- `Livre Federation` → `Livre Sportive`
- `Developpement Comite` → `Developpement Pharmacie`
- `Tourcoing Fetes` → `Tourcoing Compagnie`
- sometimes with a nearby house number

Such names occur only as a single S2/S3 record, never as an S1. They are planted distractors.

**3. True matches look similar.** In train, true matches often **drop a word and add a filler word injected by the sources**: `pediatric clinic inc` → `pediatric inc services`, `womens health care` → `womens services health`.

Filler differs by country:
- **US:** center / services / service / partners are 92% of the added words in true matches.
- **India:** com / center / services / dr / mr / partners.
- **France:** fils / groupe / services / france / developpement / & associés.

So the model had learned "drop + add a word is still a match" and accepted the swaps.

**4. Label-free separation.** Injected filler is over-represented in S2/S3 names relative to S1 names. Ratio = (document frequency in S2/S3 / document frequency in S1) / (records per S1):
- **Filler (France):** developpement 10.7, groupe 4.7, services 2.4, fils 1.55, france 1.21.
- **Filler (US):** services 4.7, service 3.9, center 2.0, partners 2.0, holdings 3.8, downtown 3.1, north 2.3.
- **Ordinary vocabulary is about 0.85 everywhere:** club, ecole, amis, comite, union, pharmacie, care, associates, first names.

The statistics come from names only, per split and per country label, like IDF (`analysis/noise_ratio.py`).

**5. Train labels confirm it** (`analysis/swap_label_rates.py`; pairs with p1 ≥ 0.5):

| Pattern | US true-match rate | India true-match rate |
|---|---|---|
| All pairs | 0.968 | 0.948 |
| Swap to an ordinary 4+-letter word, different initial | **0.015** (20,757 pairs) | **0.005** (16,343) |
| … same first word and same address (the France case) | 0.422 (287) | **0.008** (5,822) |
| Swap sharing the initial (garbled abbreviation, `care`→`cea`) | 0.503 | 0.654 |
| Swap with an extra word < 4 letters | 0.526 | 0.522 |
| Drop a word + add filler | 0.957 | 0.943 |
| Same name, different house number | 0.70–0.92 | 0.81–0.94 |

Contractions (`gaming`→`gg`, `homes`→`hs`, `power`→`pr`) are treated as the same word.
- US remaining "true swaps" are garbles and a synonym table (`systems`→`technologies`, `motors`→`auto`, `laxmi`→`lakshmi`, `jay`→`jai`).
- US false swaps are numerals (`first`→`fourth`, `ii`→`iv`) and suffixes (`dmd`→`md`).
- India false swaps are France-style descriptor swaps (`solutions`→`products`, `galaxy seven international`→`… global`).

**6. How much test has.** Submission 10 accepted this swap pattern for **8.75% of France S1** (23.5K pairs), **1.04% of India S1** (8.6K) and **0.51% of US S1** (3.5K). Out of fold the same kind of model does it for only 0.09% (US) / 0.21% (India) of S1, so **test US/India also carry about 5–6× more look-alikes than train**.

**7. `src/lookalike.py` columns (13, `features_la`).** Names are tokenized from `name_n`, with legal forms and filler words skipped. Tokens are aligned when they are:
- equal
- prefixes of each other (≥ 3 letters)
- a contraction: same initial, letters in order, and either (short ≤ 3 and long ≤ 7) or short ≥ 0.6 × long
- close by Jaro-Winkler (≥ 0.85)
- glued inside the other name

Glued names and acronyms count as matches.

| Column | Meaning |
|---|---|
| `la_n_missing`, `la_n_extra` | S1 words missing from the record; extra words in the record |
| `la_extra_vocab` / `_filler` / `_rare` | extra words classed by the ratio: filler if log ratio ≥ log 1.15 and df(S2/S3) ≥ 20; vocabulary if df(S1) ≥ max(2, 2.3e-6·#S1); else rare |
| `la_swap_vocab` | an S1 word is missing and an ordinary word appears |
| `la_extra_lratio_min` | smallest log ratio of the extra words (clipped to [−1.5, 1.1]); **word-identifying, model A only** |
| `la_extra_lshare_max`, `la_missing_lshare_min` | log share of S1 names containing the word, per million; **word-identifying, model A only** |
| `la_acronym` | record name is the acronym of the S1 name |
| `la_swap_jw`, `la_swap_same_initial`, `la_extra_vocab_len_min` | shape of the swap |

**8. Relative candidate counts (`src/relcounts.py`).**
- Candidates per S1 are 5.42 (US) / 5.72 (India) in train vs 6.36 / 6.91 / 7.47 (France) in test.
- In the audit, a stage-2 model re-scored with `s1_n_cands` raised by 22% accepted about 25–30% more false pairs (`analysis/audit/shift_sensitivity*.py`).
- Replaced by `s1_n_cands_rel` = count / (pairs in split-country / S1 in split-country).
- LOCO: neutral (0.97177 vs 0.97178).

**9. Results.**

| Run | OOF |
|---|---|
| Rebuild stage 2 → 3 | 0.98509 → 0.98646 |
| Model A stage 2 → 3 | 0.98619 → **0.98741** (+0.00095) |
| Model B stage 2 → 3 | 0.98576 → 0.98705 |

Swap pairs still accepted on test:
- **Rebuild:** US 3,498 / India 7,868 / France 33,452 (France at 0.65).
- **Model A:** 90 / 92 / 655.
- **Model B:** 124 / 93 / 948 at 0.65; 658 at France 0.75; 401 at 0.85.

**10. Why model A is not used for France.**
- Against the rebuild, model A dropped about 28K France pairs that are *not* swaps; by eye most are true "drop + French filler" matches (`cascades (france) club` → `cascades club et fils`, p 1.00 → 0.04).
- Prediction explanations (`analysis/explain_pairs.py`):
  - In stage 2, `la_extra_lratio_min` / `la_extra_lshare_max` pull these pairs down by 1.3–2.7 logits: French filler (`fils` 0.44, `france` 0.19) sits in the range of US *distractor* words (`group` 0.38, `north` 0.81).
  - `s1_n_cands_rel` pulls them down too.
  - Stage 3 then follows `p2_rec_margin`.
- LOCO agrees: the full look-alike columns give US→India +0.0063 but India→US −0.0024. The categorical version gives **+0.0037 / +0.0015** (§8).
- Model B keeps 60,788 France filler-addition pairs (model A 49,054; rebuild + filter 63,819).

**11. Post-filter rule (`src/postfilter.py`).** It drops accepted pairs with a vocabulary swap, different initial and extra word of ≥ 4 letters.
- Stage-2 rebuild OOF 0.98509 → 0.98536 (dropped pairs 12.4% correct); stage-3 rebuild 0.98646 → 0.98665.
- **No gain for model A** (its 342 accepted swaps are 98% correct).
- Used only for submission 12 and the France rows of 14.
- Extending it to same-initial swaps with Jaro-Winkler < 0.6–0.8 is flat (those pairs are 57–79% correct).

**12. Addresses: no counterpart.** With the same name and house number, address-word swaps are still 98–99.6% true in train (alternate city names are true noise), so no address rule was added.

---

## 7. Leaderboard history (all submissions)

The public LB is shown as the score and the delta vs the previous comparable submission. "OOF" is the run's own OOF.

| # | Folder / run | Change | OOF | Public LB |
|---|---|---|---|---|
| 01 | `lgb_v1_0925_032005` | hybrid blocking + stage-1 + LightGBM | 0.98031 | none recorded |
| 02 | `lgb_sib_0925_040217` | 4.5× train data + sibling features | 0.98279 | none recorded |
| 03 | `lgb_s3_0925_052339` | stage 3 | 0.98495 | none recorded |
| 04 | `lgb_s3b_0925_085328` | blocking v2 (char top-30 + house key) | 0.98589 | none recorded |
| 05 | `lgb_s3c_0925_115542` | pruning v2 (string features, top-8) | 0.98601 | **0.974431** (first upload) |
| 06 | same | France threshold 0.75 | 0.98601 | 0.974962 (+0.000531) |
| 07 | `lgb_stage3_0925_183300` | clean rerun + French department→region | 0.98596 | 0.975440 (+0.00048) |
| 08 | same | France 0.85 | | 0.975878 (+0.00044) |
| 09 | same | France 0.65 | | not uploaded |
| 10 | same | France 0.95 | | 0.976312 (+0.000434) |
| 11 | probe | France rows zeroed (wrong ids) on 10 | | not uploaded |
| 12 | 10 + `postfilter` | swap pairs removed | | not uploaded |
| 13 | model A @ France 0.85 | look-alike + rel counts everywhere | 0.98741 | not uploaded (France over-rejection found first) |
| 14 | A (US/IN) + rebuild @ 0.95 + filter (FR) | per-country hybrid | | **0.982205** (+0.005893 vs 10) |
| 15 | A (US/IN) + B @ 0.85 (FR) | France from model B | 0.98705 (B) | **0.982594** (+0.000389 vs 14; France rows only) |
| **16** | A (US/IN) + B @ **0.75** (FR) | France threshold 0.75 | | **0.982629** (+0.000035, noise level); **best**; re-uploaded as the last of 26 Sep |
| 17 | probe | France rows zeroed on 16 | | 0.836972 → **US+India 0.98439, France 0.97266** |
| 18 | A @ US/IN **t = 0.80** + B @ 0.75 | stricter US/India | | 0.98234 (**−0.000289**) → keep 0.65 |

**Probe arithmetic.** Every France S1 in the probe gets one distinct real US S2 id, so all France S1 score exactly 0 (matches always share the country label). Then:
- F(US+India) = LB / 0.850248
- F(France) = (LB of the base submission − probe LB) / 0.149752

The public split's France share may differ slightly from 14.98%, which gives ±0.0006 on US+India and ±0.004 on France.

---

## 8. Other measured facts

**LOCO** (stage-2 models trained on one country and scored on the other, grid up to 0.99; `logs/loco_*.log`):

| Features | US → India (best t) | India → US (best t) |
|---|---|---|
| Rebuild features | 0.95491 (0.80) | 0.97178 (0.85) |
| + all look-alike + rel counts | 0.96122 (0.65) | 0.96938 (0.85) |
| + all look-alike only | | 0.96948 (0.85) |
| + rel counts only | | 0.97177 (0.90) |
| **+ categorical look-alike + rel counts (model B)** | **0.95860 (0.75)** | **0.97330 (0.90)** |
| In-distribution reference (t = 0.6, m = 0.5) | 0.98432 | 0.98559 |

Older LOCO (25 Sep, old candidates, grid capped at 0.75): US→India 0.9509, India→US 0.9677, both at the grid edge. That is why the earlier "0.75 for an unseen country" rule looked optimal. "Rare-token" features lowered US→India to 0.9454.

**Test vs OOF density** (model A, best-per-record pairs with margin ≥ 0.6, per 1000 S1; `analysis/test_band_density.py`):
- Test has **1.4–2.6×** the OOF density in every band from 0.6 to 0.99, for both US and India; the ≥ 0.99 band is 0.97–0.99×.
- OOF precision by band: 0.63 [0.60, 0.65) · 0.68 · 0.74 · 0.78 · 0.84 · 0.88 · 0.93 · 0.98 [0.95, 0.99).
- A stricter threshold (submission 18) still lost, so much of the extra test mass is true matches that score lower on test, not only false pairs.

**France entities with 0–1 predictions** (`analysis/france_missed_matches.py`, 4,000 sampled):
- Unaccepted records at the same number+street that share a name word: 677 were candidates scored low, 671 were not candidates, 66 lost to another S1.
- The low-scored ones are mostly correctly rejected look-alikes (`vaillante ecole` ← `vaillante comite`, p = 0.000). Only a few true misses were seen (`union du detudes` ← `… & fils`, p = 0.29).
- The "not candidates" are mostly key collisions (other streets).
- France has fewer predictions per S1 (3.29 vs about 3.46 true in train) and slightly more 0/1-prediction S1: 6.1% / 6.7% vs truth 5.6% / 5.4%.

**Averaging models (OOF):**

| Ensemble | OOF |
|---|---|
| Model A + model B | 0.98744 (+0.00003) |
| A + B + rebuild | 0.98740 |
| 3-member stage-3 ensemble (25 Sep) | 0.98605 (+0.00004) |

None of these were used.

**Other checks:**
- France S1 names follow the pattern `<place or word> <descriptor> <legal form>`. Some are very common (`tourcoing ecole` 114 S1, `nantes primaire` 102).
- Campus addresses host dozens of businesses (`60 bd vauban, lille`).

---

## 9. Tried and rejected, or not worth it

| Idea | Result |
|---|---|
| Stage 4 (context from stage-3 probabilities) | OOF 0.98576 < 0.98589 |
| Model ensembles (see §8) | ≤ +0.00004 OOF |
| "Rare-token" (country-frequent-token-free) similarities | +0.0003 OOF, but US→India LOCO −0.0034; off (`features.rare_tokens: false`) |
| Address look-alike rule | no training signal (address swaps are true in train) |
| Post-filter on same-initial swaps | flat |
| All look-alike columns for France (13) | over-rejects French filler matches; model B instead |
| France threshold 0.75 vs 0.85 with model B | +0.000035 (flat) |
| France threshold with the old model | 0.65 → 0.95 gave +0.0013 in total; exhausted (next step estimated ≤ +0.0001) |
| US/India threshold 0.80 (submission 18) | −0.000289 |
| Per-source caps (≤ 5 S2, ≤ 6 S3 per S1) | violated by about 100 S1 in total; worth about +1e-5; not applied |
| Pseudo-labelling on test | not done (rules) |

---

## 10. How to reproduce and verify

- **Full rebuild of submission 16:** `bash scripts/reproduce_best.sh`. It runs all steps in §4.7 and writes `output/`. It takes about 7 h and needs `student_resource/` in place.
  - GPU fp16 ties and multi-threaded training mean a rebuild matches the uploaded file on about 99% of pairs. The 25 Sep clean rerun matched 99.2% of pairs, with OOF 0.98596 vs 0.98601.
- **From existing caches:** `python -m src.predict --config configs/base.yaml configs/output_lacat.yaml --run lgb_stage3lacat_0926_051023 --reuse-probs`. Then `python -m src.combine --config configs/base.yaml --seen output_la --unseen output_lacat --out <dir>`.
  - This reproduces **submission 16 byte for byte**, including `candidate_pairs.tsv`. It was checked with `cmp` at handoff. `output_la/` must hold a model A prediction; its US/India rows do not depend on the France threshold.
  - `src.combine` reproduced 14 (with `--unseen-postfilter` on the rebuild output) and 15 exactly.
- **Checks:**
  - validator `PASS` with `--check-ids`
  - `cmp` against `submissions/16_hybrid_la_us_in_lacat075_fr/matching_results.tsv`
  - per-country pair counts as in §4.7

---

## 11. What to do next (ranked)

**A. Required before 23:59 IST 27 Sep:**
1. Final upload: submission 16's file, or a new file that beats 0.982629 on the public LB **and** is backed by OOF/LOCO.
   - Slots used so far on 27 Sep: 1 (submission 18).
   - The final package's `output/` must contain the same file that is uploaded.
2. Get the **team name and members** from the owner.
3. Fill `docs/Documentation_template.md`:
   - Team / members / date.
   - Executive summary: "[0.98589 → final]" → final OOF (A 0.98741 / B 0.98705) and public LB 0.982629.
   - §3: "Candidate pairs generated (test): [12.5M → final]" → 11,749,613 (1.18 per record, 6.78 per S1).
   - §3: "Pruning … [98.26% → final] kept" → 98.45%.
   - §4: "[final value]" → 0.75 for unseen labels (model B).
   - §5 bullets: final OOF per country and the public LB.
   - The blocking recall table was written for the 25 Sep candidates; the rebuild numbers are in §4.2 of this file.
4. Build the zip:
   ```bash
   python scripts/make_submission_zip.py --team "<name>" --outputs submissions/16_hybrid_la_us_in_lacat075_fr
   ```
   - It takes the TSVs from that folder, all `src/*.py`, base/stage2/stage3/output_la/output_lacat configs, `scripts/reproduce_best.sh`, `docs/README_reproduce.md` as README, pinned requirements and the documentation.
   - Unzip it and check the layout; optionally run the validator on its `output/`.

**B. If there is time: the US+India shortfall** (test 0.9844 vs OOF 0.9874, worth up to about 0.0025 LB). Test has 1.9× more distractors and about 5–6× more look-alikes than train, so OOF-tuned models under-reject on test.
1. **Train on a test-like composition.** Drop about 19% of train S1 entities and keep their records as orphan distractors, which matches test's distractor share: (1.21 + 3.46·d) / (1 − d) = 2.30 distractors per S1 gives d ≈ 0.19. Then:
   - Remove the dropped S1s' pairs from train `cands_pruned`; their records keep their candidates to other S1s, which are negatives.
   - Recompute the candidate-set features (group/competition/sibling, `relcounts`, stage-3 context).
   - Retrain stages 2/3 and tune (t, m) on OOF of the new composition.

   Roughly 2–3 h. Untested; check it on LOCO too. The aim is a model whose priors match test, where look-alikes and distractors are more common.
2. **Audit item not yet fixed:** `s1_name_freq` / `rec_name_s1_freq` are raw counts; the mean is 22.8 in train vs 12.1 in test US. Use a rate per 100K S1 of the country. Needs a feature rebuild and retraining (about 1.5 h).
3. **Audit item not yet fixed:** test `p1` comes from `stage1_full` while train uses fold models. Scoring test with the fold model given by `s1_hash % 3` makes them consistent (re-prune test about 45 min, then test features + refine + predict).
4. **More look-alike detectors** for the patterns seen in US train distractors:
   - numeral swaps (`first`/`fourth`, `ii`/`iv`)
   - professional suffix swaps (`md`/`do`/`dmd`)
   - unit/building/fraction changes (`unit#8` vs `unit#17`, `1/2` vs `1/15`)
   - geo-word additions (`north`, `valley`, `downtown`, `midtown`, `greater`)
5. Do **not** raise the US/India threshold again (submission 18 lost).

**C. France** (0.9727 on test; worth up to about 0.002 LB).
1. `normalize.py` maps `compagnie`→`cie`, a legal form that is then skipped. In France "Compagnie" is a descriptor (`Tourcoing Compagnie` has 40 S1), so swaps to or from it are invisible to the look-alike columns. The fix needs re-normalization, i.e. a full rebuild of about 4.5 h plus the two model chains.
2. Model B still sometimes rejects `… & Fils` / `… et Fils` filler matches (p ≈ 0.3).
3. Untested ideas from the decision analysis:
   - an "n-aware" France rule: be less strict for an S1's only prediction, stricter when it already has confident matches
   - comparing stage-2 vs stage-3 France ranking at a matched pair count
4. The France threshold itself is exhausted: 0.75 ≈ 0.85 with model B.

**D. Leaderboard hygiene.**
- Adopt a change only if its paired delta is ≥ 1e-4 **and** it has an OOF or LOCO rationale.
- Small differences (< 1e-4) are noise plus winner's curse, and the private LB decides.

---

## 12. Working conventions (keep them)

- **Commits:** plain human engineering voice, matching the existing history. **No co-author trailers, no tool-credit footers, and no mention of any AI tool** anywhere: code, docs, file names or commit messages.
  - A local pre-push hook (`.git/hooks/pre-push`, not versioned) rejects pushes whose history contains such mentions or trailers. Read it before pushing.
  - Local tooling files are kept out of git via `.git/info/exclude`.
- **Per submission:**
  - Archive it (`scripts/save_submission.py --config … --run … --note "…"`, or by hand for hybrids and probes).
  - Write `NOTE.md` with the change, OOF and validator result.
  - Commit, tag `sub-NN` and push.
  - Fill in the LB score in `NOTE.md` when known (`experiments.csv` has an `lb_score` column that is mostly empty).
- **Before trusting a new model on test:**
  1. OOF with its own tuned decision.
  2. LOCO both directions (`src/loco.py … --extra … --drop …` with `configs/loco_grid.yaml`).
  3. Pair counts per country and swap/filler counts on test (`analysis/test_profile.py`, `analysis/france_model_diff.py`).
  4. Read about 30–50 pairs of the uncertain band by eye.
- **Never** overwrite `output/` while another job needs it; use an `output_*.yaml` overlay. `predict` always writes to `paths.output_dir`.
- `train_matcher` prints the run id as the last stdout line; logs go to stderr.
- Don't read files the pipeline is writing (Windows locks); wait for the stage's "wrote …" log line.
