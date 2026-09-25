# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** [Date]

---

## 1. Executive Summary
Each Source-2/3 record belongs to at most one Source-1 entity, so we solve the task record by record inside each country label:
- **Normalize** all 24M records, using transliteration learned from training pairs.
- **Retrieve** candidate S1 entities with a hybrid of GPU dense char-n-gram search, an exact rare-word inverted index and two exact keys.
- **Prune** with a learned ranker to about 1.3 candidates per record.
- **Score** the pairs with a three-stage LightGBM cascade whose later stages see how each pair compares with its competitors.
- **Assign** each record to its best entity only when probability and margin clear thresholds tuned directly for macro F0.5.

Out-of-fold macro F0.5 on the full training set is **[0.98589 → final]**.

---

## 2. Methodology

### 2.1 Problem Analysis
- **Scale:**
  - Train: 2,206,821 S1 / 5,034,616 S2 / 5,285,603 S3 records.
  - Test: 1,732,544 / 4,887,273 / 5,082,316 records, including 259,452 S1 in France, which never appears in train.
- **Structure we exploit (verified on all 7.64M training pairs):**
  - Every S2/S3 record matches **at most one** S1 entity.
  - Matched records **always share the S1's country label**.
  - About 26% of train S2/S3 records match nothing.
  - 5.6% of S1 entities are singletons; the rest average 3.5 matches (max 11: ≤5 from S2, ≤6 from S3).
  - Test has 5.75 records per S1 (train 4.68) but the same match rate per S1, so ~39% of test records are distractors.
- **Why names alone fail:**
  - 39% of S1 names are shared by another S1 of the same country (chains, generic names), so the address must decide.
  - Only 26% of true pairs have identical normalized names.
- **Name noise:**
  - names written in Indian scripts (Devanagari, Telugu, Kannada, Tamil, Bengali, Gujarati; ~4% of records), sometimes mixed with Latin
  - website-style names (`cardiologyphysicians.com`, `blueengineeringcom`)
  - junk prefixes (`--`, `<<`)
  - legal suffixes moved or rewritten (`L.L.C.`, `Pvt Ltd`, `S.A.S`)
  - the *brand token kept while the other words are replaced* (`prock allstate llc` → `prock llc services`, `tps restaurants co` → `services tps co`)
  - typos and injected accents
  - trade names completely different from the registered name at the same address
- **Address noise:**
  - reordered components
  - abbreviations (`Rd`, `R.`, `AV`)
  - state written as a code, full name or native script (S1 uses `Maharashtra`, S3 uses `MH` or `महाराष्ट्र`)
  - landmark phrases
  - house numbers with suffixes (`726A/20`)
  - empty addresses (4.4% of matched records vs 0.3% of unmatched)
  - Latin-1 mojibake (`â€“`)

### 2.2 Solution Strategy
**Approach Type:** Blocking (hybrid retrieval) → learned pruning → 3-stage gradient-boosted matching with competition context → constrained one-to-one assignment.  
**Core Innovation:**
1. Record-centric one-to-one assignment optimized for the exact macro-F0.5 metric (singletons and missed pairs included).
2. Hybrid retrieval: dense char-n-gram vectors catch typos; an exact rare-token index catches the "brand token kept" pattern that dense vectors dilute; keys catch house-number evidence.
3. A cascade in which each stage recomputes competition features from the previous stage's out-of-fold probabilities: rank/margin against the record's other candidates, how many records already link to the S1, and similarity to those "sibling" records.
4. Country-agnostic design (no country features, per-label processing, learned transliteration), so the unseen France label is handled like any other.

---

## 3. Candidate Generation (Blocking)
- **Normalization (all sources, all countries alike):**
  - Mojibake repair.
  - Non-Latin tokens and address components mapped via a table learned from training pairs (4,935 tokens and 84 components, e.g. `प्राइवेट`→private, `தமிழ்நாடு`→tamil nadu), with anyascii romanization as fallback.
  - Lowercasing; `&`→and; dots and apostrophes deleted (`l.l.c.`→llc); other punctuation → space.
  - Canonical legal forms; canonical street terms (street→st, rue/r→rue, near→nr …); full state names → codes; ordinals → digits.
  - Derived fields: core name (legal/filler words removed), space-free name, digit tokens, house number.
- **Blocking views** (every S2/S3 record queries the S1 records of its own country label):
  1. **Dense char-3-gram view (GPU):** TF-IDF on core name and on address, each reduced to 256 dimensions with TruncatedSVD. Exact top-30 S1 per record by 0.5·cos(name)+0.5·cos(address), as brute-force fp16 matrix products on an RTX 4060.
  2. **Rare-word inverted index (CPU, exact):** word-token TF-IDF, with tokens appearing in >2,000 S1 records excluded from the index. Top-10 by name+address, top-5 by name, top-5 by address.
  3. **Exact keys:** normalized core name (skipped if >20 S1 share it); (first name token, house number) (skipped if >50 S1 share it).
- **Union:** 440M train pairs; pair recall **98.79%**, oracle F0.5 0.9963.
- **Stage-1 pruning:**
  - A LightGBM ranker over retrieval scores, per-record/per-S1 ranks and gaps, and three cheap string similarities (core-name ratio, address token-set, digit-token overlap).
  - Trained out of fold.
  - Keeps the top-[N] S1 per record with p ≥ 0.001.
- **Candidate pairs generated (test):** [12.5M → final] (≈1.3 per record, ≈7.2 per S1), out of 3.8×10¹² possible same-country pairs.
- **How we ensured true matches were not lost:**
  - Each view was added only when it raised union recall on train. Recall by view:

    | View | Recall |
    |---|---|
    | char full top-30 | 96.3% |
    | token full | 89.5% |
    | token address | 75.9% |
    | brand+house key | 53.4% |
    | exact name | 42.8% |
    | token name | 35.5% |
    | **Union** | **98.79%** |

  - Pruning was tuned on out-of-fold recall: [98.26% → final] kept.

---

## 4. Matching Model

**Features used (~83 base + 14 stage-3 context):**
- **Name features:**
  - rapidfuzz ratio / partial / token-sort / token-set and Jaro-Winkler on the normalized, core and space-free names
  - legal-form agreement and first-token agreement
  - brand token of one name contained in the other
  - name uniqueness (how many S1 share this core name)
  - near-identical names among the record's candidates
- **Address features:**
  - ratio / partial / token-sort / token-set
  - digit-token overlap, house-number agreement, postal agreement
  - state-code agreement (first/last alphabetic token only, so French words such as "la"/"de" are not mistaken for states)
  - empty-address flags
- **Other:**
  - char and token cosines from blocking, and the stage-1 probability
  - competition features: rank, gap and margin of the pair among the record's candidates and among the S1's candidates; candidate counts; expected match count of the S1
  - sibling features: best similarity of the record to other records confidently linked to the same S1
  - source flag, length differences, non-Latin/website flags
  - **No country feature.**
- **Stage-3 context:** the same competition and sibling statistics recomputed from stage-2 out-of-fold probabilities.

**Model type:** LightGBM binary classifiers (MIT license, no pretrained language models):
- stage 1 (pruning)
- stage 2 (full features)
- stage 3 (full features + stage-2 context)

All stages use 3 folds grouped by S1 entity. Every training pair gets an out-of-fold probability, and the test set uses the average of the fold models.

**Threshold selection method:**
- Each record keeps only its highest-probability S1 (one-to-one assignment).
- It is accepted if p ≥ t and p − p_runner-up ≥ m.
- (t, m) are grid-searched to maximize the exact macro F0.5 over all 2.2M training S1 entities, using out-of-fold probabilities, singletons and blocking misses included. Final in-distribution setting: t = 0.65, m = 0.6.
- Robustness checks:
  - **Test distractor density:** weighting distractor false positives 1.9× (the test/train ratio) moves the optimum only to t ≈ 0.70, with negligible score difference.
  - **Unseen country:** leave-one-country-out runs (Appendix B) show that a country absent from training scores 2–3.5 points lower, and its best threshold is 0.75 in every run. Being stricter in-distribution costs only ~0.0002. So S1 entities whose country label does not occur in training (France in the test set) use t = 0.75, m = 0.5. This is a generic rule for any unseen label, not a France-specific one.

---

## 5. Results & Error Analysis

| Step | OOF macro F0.5 |
|---|---|
| Hybrid blocking + pruning + stage-2 LightGBM (10% of S1 groups per fold) | 0.9803 |
| + 4.5× training data | 0.9825 |
| + sibling-record features | 0.9828 |
| + name uniqueness, state agreement, name ambiguity | 0.9830 |
| + stage 3 (context from stage-2 probabilities) | 0.9850 |
| + blocking v2 (char top-30, brand+house key) | 0.9859 |
| + pruning v2 (string features in the ranker, top-8 per record) | 0.9860 |
| (tried) a fourth refinement stage (context from stage-3 probabilities) | 0.9858, not used |

- **F_0.5 Score (macro):**
  - Out-of-fold [0.98589 → final] (US [0.9863], India [0.9853]).
  - Public leaderboard [ ].
  - A further refinement round (stage 4) did not help (0.98576).
- **Score by entity type (train OOF):**

  | Entity type | Count | F0.5 |
  |---|---|---|
  | Singletons | 123K | 0.987 |
  | Exactly 1 true match (hardest) | 119K | 0.946 |
  | 2–3 matches | 906K | 0.986 |
  | 4–5 matches | 806K | 0.990 |
  | 6+ matches | 252K | 0.991 |

- **Common false positives (wrong merges):** 23.5K, 0.32% of predicted pairs.
  - 82% are distractor records (businesses with no S1 entity); 18% belong to another S1.
  - 69% have a near-identical name and 26% the same house number. The data contains deliberately planted look-alikes: `jarleus doubleline group | 2638 jefferson ave` vs `jarleuz doubleline group | 2638 jefferson ave` is a non-match, as are `2925` vs `2928 madison ave` and `620` vs `623 3 st`. These are indistinguishable from true noisy duplicates.
  - Only 1,578 of 123K singleton S1 entities receive a wrong match.
- **Common false negatives (missed matches):** 251K true pairs; 131K inside the candidates, 120K never retrieved or pruned away.
  - 62% of in-candidate misses have an **empty record address** (vs 3.4% of all true pairs), with a generic or altered name shared by several S1 entities (`prime grand hall`, `sidhant nidhi services`).
  - 11% are trade names with no name overlap (`brixnexvio` for `alisa j student do`), matchable only through a partial address.
  - S2 and S3 fail at the same rate.

---

## 6. Conclusion
Exploiting the one-to-one structure (each record chooses one entity) and optimizing the decision for the exact metric mattered as much as the matcher. Hybrid retrieval plus a cheap learned pruning stage kept 98%+ of true pairs with about 1.3 candidates per record. The biggest late gain came from letting the model see how a pair compares with its competitors, first with stage-1 scores and then with stage-2 probabilities. The remaining errors are mostly records with no address and generic names, which are ambiguous even for a human.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/` contains:
- `src/`: data, normalize, blocking, prune, features, train_matcher, refine, decide, predict, validate, metrics, run_pipeline, utils
- `configs/`: base, stage2, stage3
- `README.md` and a pinned `requirements.txt`

Entry point:
```bash
python -m src.run_pipeline --config configs/base.yaml
```
It regenerates `output/matching_results.tsv` and `output/candidate_pairs.tsv` (~5 h on a 12-thread CPU, 31 GB RAM and an RTX 4060). Each stage can also be run on its own (see README).

### B. Additional Results
- **Leave-one-country-out (proxy for the unseen France label):** a stage-2 model trained on one country only is scored on the other (`src/loco.py`).

  | Setting | US → India | India → US |
  |---|---|---|
  | All features, t = 0.6 | 0.9488 | 0.9643 |
  | All features, best threshold | 0.9509 (t = 0.75) | 0.9677 (t = 0.75) |
  | Without the non-Latin flag | 0.9487 | 0.9643 |
  | Also without length / name-frequency / legal / state features | 0.9441 | 0.9623 |
  | Country included in training (reference) | 0.9836 | 0.9850 |

  Conclusions: the feature set transfers best as it is; the stricter threshold helps an unseen country.
- Blocking/pruning recall per view and per setting: see Section 3.
- Pruning curve (train OOF, blocking v1): top-3 97.73%, top-5 97.95%, top-8 98.08% of true pairs, versus 98.23% before pruning.
- Test-set checks: 5.7% of test S1 predicted singleton (train singleton rate 5.6%); ~3.4 matches per S1 in every country (France 3.40, India 3.34, US 3.38).

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
