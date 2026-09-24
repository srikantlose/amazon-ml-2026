# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** [Date]

---

## 1. Executive Summary
We treat the task as record-centric matching: every Source 2/3 record belongs to at most one Source 1 entity. Records are first normalized with transliteration learned from training pairs. Candidates are then retrieved within each country label by combining GPU dense char-n-gram retrieval with an exact rare-word inverted index, and pruned by a light learned ranker. A LightGBM pair classifier scores the surviving pairs. Each record is then assigned to its single best S1 entity only when the probability and its margin over the runner-up clear thresholds tuned directly for macro F0.5 on out-of-fold predictions.

---

## 2. Methodology

### 2.1 Problem Analysis
- **Scale:** train has 2.21M S1 / 5.03M S2 / 5.29M S3 records; test has 1.73M / 4.89M / 5.08M, plus the unseen country France (15% of test S1).
- **Structure found in the ground truth:**
  - Every S2/S3 record matches **at most one** S1 entity (0 exceptions in 7.64M pairs).
  - About 26% of S2/S3 records match nothing.
  - Matched pairs **always share the country label**.
  - 5.6% of S1 entities are singletons; the rest average 3.5 matches (max 11: ≤5 from S2, ≤6 from S3).
- **Why it is hard:**
  - 39% of S1 names are shared by another S1 of the same country, so the address must disambiguate.
  - Only 26% of true pairs have identical normalized names.
- **Noise patterns:**
  - names written in six Indian scripts
  - website-style names (`cardiologyphysicians.com`)
  - junk prefixes (`--`, `<<`)
  - legal-suffix variants and moves (`L.L.C.`, `Pvt Ltd`, `S.A.S`)
  - brand token kept while the descriptive words are swapped (`prock allstate llc` → `prock llc services`)
  - typos and injected accents
  - reordered address components
  - state codes vs full names vs native script
  - empty addresses (4.4% of matched records vs 0.3% of unmatched)
  - Latin-1 mojibake

### 2.2 Solution Strategy
**Approach Type:** Blocking + two-stage learned ranking/classification + constrained assignment.  
**Core Innovation:**
- Record-centric one-to-one assignment optimized for macro F0.5.
- Hybrid retrieval: a GPU dense char-3-gram view plus an exact rare-token view.
- Transliteration maps learned from training pairs, so the pipeline stays country-agnostic.

---

## 3. Candidate Generation (Blocking)
- **Normalization:**
  - Unicode repair; non-Latin tokens mapped via a table learned from train pairs, with anyascii as fallback.
  - Canonical legal forms and street terms.
  - State names → codes.
  - Digit-token and house-number extraction.
- **Blocking keys / views (all within the same `country` label, handled generically):**
  1. Char 3-gram TF-IDF → TruncatedSVD(256) per field; exact top-10 S1 by 0.5·cos(name)+0.5·cos(address) on the GPU.
  2. Word-token TF-IDF inverted index (tokens in >2000 S1 records excluded): top-10 by name+address, top-5 by name, top-5 by address.
  3. Exact normalized-name key (skipping names shared by >20 S1 records).
- **Stage-1 pruning:** LightGBM over retrieval scores and ranks keeps the top-[N] S1 per record (p ≥ [x]).
- **Candidate pairs generated:** [total test pairs] ([per record] per record, [per S1] per S1).
- **How we ensured true matches were not lost:** views were chosen for complementary recall (measured per view on train: [table]); pair recall after union [x%], after pruning [y%]; oracle F0.5 ceiling [z].

---

## 4. Matching Model

**Features used:**
- **Name features:**
  - rapidfuzz ratio, partial ratio, token-sort and token-set ratios, and Jaro-Winkler on the normalized name, the core name (legal/filler removed) and the space-free name
  - legal-form agreement
  - first-token agreement, and brand token contained in the other name
- **Address features:** the same string similarities on the normalized address, digit-token overlap, house-number and postal agreement, and empty-address flags.
- **Other:**
  - char and token cosines from blocking
  - stage-1 probability
  - competition features: rank, gap and margin of this pair among the record's candidates and among the S1's candidates; candidates per record/S1
  - source flag, length differences, non-Latin/domain flags
  - No country feature.

**Model type:** LightGBM binary classifier (MIT license), 3 folds grouped by S1 entity, out-of-fold predictions for every training pair.  
**Threshold selection method:**
- Each record keeps only its best S1.
- The match is accepted if p ≥ t and p − p_second ≥ m.
- (t, m) are grid-searched to maximize exact macro F0.5 on OOF predictions over all 2.2M training S1 entities, singletons and blocking misses included.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** OOF [x] (US [x], India [x]); public LB [x].
- **Common false positives (wrong merges):** [ ]
- **Common false negatives (missed matches):** [ ]

---

## 6. Conclusion
[ ]

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/`: `src/` (data, normalize, blocking, prune, features, train_matcher, decide, predict, validate, metrics, run_pipeline), `configs/base.yaml`, `README.md`, `requirements.txt`. Entry point: `python -m src.run_pipeline --config configs/base.yaml` regenerates `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

### B. Additional Results
[recall per view, pruning curve, decision grid, feature importance]

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
