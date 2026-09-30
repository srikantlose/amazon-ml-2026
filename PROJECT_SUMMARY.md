# Amazon ML Challenge 2026: Project Summary

**Team:** ChickenJockey (Srikaanth Gurumurthy, Prathmesh Sayal)
**Task:** Business Entity Resolution
**Window:** 25 Sep 2026, 00:00 IST, to 27 Sep 2026, 23:59 IST

**Final public leaderboard score: 0.983405.** This was submission 25, and it was also our last upload. The first upload scored 0.974431.

**Official package:** `ChickenJockey_submission.zip`, containing the outputs of submission 25, the code, and the filled methodology document.

This document records everything we did, in order: preparation, what we built, every submission and its score, what worked, what didn't, and where every artefact lives. For operational detail (file-by-file map, commands, gotchas) see `HANDOFF.md`. The methodology write-up is `docs/Documentation_template.md`, and the lab notebook is `docs/approach_notes.md`.

---

## 1. Results at a glance

| Milestone | Public LB | What changed |
|---|---|---|
| First upload (submission 05) | 0.974431 | Full pipeline: blocking → pruning → 3-stage LightGBM → one-to-one assignment |
| France threshold tuning + French address fix (06–10) | 0.976312 | Stricter decision for the unseen country; department → region normalization |
| Look-alike discovery (14) | **0.982205** | Largest single jump, **+0.0059** |
| France handled by a transfer-robust model (15, 16) | 0.982629 | Per-country model choice |
| Two models averaged for US/India (19) | 0.983167 | Plus a slightly more permissive decision |
| Models retrained on 2× data (22) | 0.983212 | |
| France from two averaged views + look-alike filter (25) | **0.983405** | **Final** |

- **Out-of-fold macro F0.5** (train, held out by S1 group) rose from 0.9803 to **0.9877** over the same period.
- **Leader when last checked** (25 Sep): 0.985884.

---

## 2. Preparation before the problem was released

- **Scaffold for the expected problem.** Previous years' challenges were price/image regression, so we first prepared that kind of scaffold: config-driven pipeline, experiment log, submission archiving, validators and smoke tests.
- **Compute.** We weighed AWS SageMaker, including a GPU quota request. We decided to run everything **on the local PC**:
  - Windows 11, Ryzen 5 7600 (12 threads), RTX 4060 8 GB, 31 GB RAM
  - Python 3.13 virtualenv with torch (CUDA 12.8), LightGBM, rapidfuzz, anyascii, pandas/pyarrow, scikit-learn
- **When the real problem arrived (Business Entity Resolution):**
  - We committed the old scaffold, replaced it with an entity-resolution pipeline, and moved the organizers' data into the repo folder (git-ignored).
- **Repository conventions:**
  - Private GitHub repo `srikantlose/amazon-ml-2026`. Every leaderboard file maps to a git commit and tag (`sub-NN`), with a `NOTE.md` per submission.
  - Every model run is logged in `experiments.csv`.
  - Commits are pushed automatically.
  - Commit messages are written in a plain engineering voice.

---

## 3. The problem, and the facts that shaped the design

**Task.** For each Source-1 (S1) business entity, list the Source-2/3 records that describe the same business. Records have only a name, an address and a country label.

- **Train:** 2.21M S1, 5.03M S2 and 5.29M S3 records, US and India only.
- **Test:** 1.73M S1, 4.89M S2 and 5.08M S3 records, and adds **France** (15% of test S1), which never appears in train.

**Metric.** Macro F0.5 per S1, which is precision-heavy.
- A true singleton scores 1 for an empty prediction and 0 for any match.
- Adding one false match to a typical entity costs about 0.2. Missing one true match costs about 0.07.

**Structure we verified on all 7.6M training pairs, and exploited:**
- **Each S2/S3 record matches at most one S1.** So we solve the task record by record, with one-to-one assignment.
- **Matches always share the S1's country label.** So all blocking runs within a country label, treated as an open set.
- **Distractors:**
  - 26% of train records match nothing.
  - Test has about 1.9× more distractors per entity (5.5–5.8 records per S1 vs 4.7).
- **Entity sizes:** 5.6% of S1 are singletons, and the mean is 3.46 matches per S1.
- **Names alone are not enough:** 39% of S1 names are shared by another S1, and only 26% of true pairs have identical normalized names.
- **Noise:**
  - names in Indian scripts
  - website-style names
  - legal-suffix variants
  - typos and injected accents
  - trade names and DBAs
  - acronyms
  - word shuffles
  - state codes vs names
  - French regions vs departments
  - garbled house numbers
  - empty addresses
  - mojibake

---

## 4. What we built (the pipeline)

1. **Normalization** (`src/normalize.py`)
   - Mojibake repair.
   - A non-Latin → Latin map learned from train pairs (4,935 tokens, 84 address components), with anyascii fallback.
   - Canonical legal forms, street terms, and state/region codes. French departments map to their region.
   - Number canonicalization: `N°`, leading zeros, `9B` = `9 bis`.
   - Derived fields: core name, space-free name, house number, postal code.

2. **Blocking** (`src/blocking.py`): every S2/S3 record queries S1 records of its own country only.
   - Char 3-gram TF-IDF → SVD 256, exact GPU top-30 (recall 96.3%).
   - Rare-word inverted index (top 10/5/5).
   - Exact name key and (first word, house number) key.
   - Union recall **98.81%**.

3. **Stage-1 pruning** (`src/prune.py`): a LightGBM ranker keeps the top 8 S1 per record.
   - Recall **98.45%**.
   - **1.18 candidates per record, 6.78 per S1** (11.7M test pairs). The organizers later said smaller candidate sets rank higher, and ours is small.

4. **Pair features** (`src/features.py`, 83 columns)
   - rapidfuzz name/address similarities.
   - House, postal, state and legal-form agreement.
   - Name frequency.
   - Competition features: how a pair ranks against the record's and the S1's other candidates.
   - Sibling-record similarity.
   - No country feature.

5. **Stage 2 and stage 3 LightGBM** (`src/train_matcher.py`, `src/refine.py`)
   - 3 folds grouped by S1, with out-of-fold probabilities for every train pair.
   - Stage 3 adds context recomputed from stage-2 probabilities.

6. **Decision** (`src/decide.py`)
   - Each record keeps its best S1 if probability ≥ t and the margin over the runner-up ≥ m.
   - (t, m) are tuned for the exact metric on out-of-fold predictions.
   - Country labels never seen in training get their own threshold.

7. **Additions from the final day and a half:**
   - look-alike features (`src/lookalike.py`)
   - relative candidate counts (`src/relcounts.py`)
   - look-alike post-filter (`src/postfilter.py`)
   - per-country combination (`src/combine.py`)
   - final probability blend (`src/blend.py`)
   - leave-one-country-out check (`src/loco.py`)
   - an XGBoost matcher (`src/train_xgb.py`, tried and not used)

**Final configuration (submission 25):**

| Rows | Model | Decision |
|---|---|---|
| US, India | Mean of **model A** (all look-alike features) and **model B** (categorical look-alike features only), both retrained on 90%/95% of S1 groups per fold | t = 0.60, m = 0.50 (OOF 0.98771) |
| France | Mean of model B and the pre-look-alike "rebuild" model, then the look-alike post-filter | t = 0.75, m = 0.50 |

`scripts/reproduce_best.sh` rebuilds this from raw data in about 8 hours. From the cached models, `src.blend` + `src.combine` regenerate submission 25 **byte for byte** (verified).

---

## 5. Timeline of the work

### 25 Sep: build the pipeline, first uploads
- Built normalization, blocking and the LightGBM cascade.
- **Out-of-fold improvements, step by step:**

| Change | Out-of-fold F0.5 |
|---|---|
| First model (10% of S1 groups per fold) | 0.9803 |
| 4.5× training data | 0.9825 |
| Sibling features | 0.9828 |
| Name uniqueness / state / ambiguity features | 0.9830 |
| Stage 3 | 0.9850 |
| Blocking v2 (char top-30, house key) | 0.9859 |
| Pruning v2 | 0.9860 |

- **First upload (05): 0.974431.** Out-of-fold said 0.986, so there was an unexplained gap of about 0.011.
- **France was the prime suspect.** Stricter France thresholds each helped by about 0.0004–0.0005: 0.75 (06), 0.85 (08), 0.95 (10). A French department → region address fix added +0.0005 (07). Best after day 1: **0.976312**.
- A clean end-to-end rerun reproduced out-of-fold within 0.00005 and 99.2% of predicted pairs.
- The repo was pushed to GitHub, and a code zip of submission 10 was shared with the team.

### Night of 25–26 Sep: diagnose the gap
- **Rebuild with house-number canonicalization:** out-of-fold 0.98596 → 0.98646.
- **Decision analysis.** Turning the France-only leaderboard deltas into precision showed that France pairs scored 0.85–0.95 were only about 58% correct, vs about 90% for US/India. It also showed that more France threshold steps were nearly exhausted.
- **Reading those France pairs by eye revealed planted look-alikes.** These records copy an entity's address and first word but swap the descriptor for another ordinary word: `Mauges Amis SAS` → `Mauges Collectif SAS`, `Livre Federation` → `Livre Sportive`. Such names never exist as an S1.
- **Why the model accepted them.** True matches often drop a word and add a filler word injected by the sources:
  - US: `services`, `center`, `partners`
  - France: `& fils`, `groupe`, `développement`
- **Label-free separation.** Filler is 2–10× over-represented in S2/S3 names vs S1 names, while ordinary vocabulary sits at about 0.85×. So filler and real words can be told apart from the unlabelled test data itself.
- **Train labels confirmed it.** A swap to an ordinary 4+-letter word with a different first letter is only 1.5% (US) / 0.5% (India) true. Submission 10 had accepted this pattern for 8.75% of France entities, and test US/India had 5–6× more of it than train.
- **Built `src/lookalike.py`** (13 columns). An independent audit then:
  - made the statistics scale-free
  - tightened the contraction rule
  - flagged that test's larger candidate counts inflate an important feature, so `src/relcounts.py` replaced it
- **Model A** (look-alike + relative counts): out-of-fold **0.98741**; it rejects about 98% of test look-alikes.
- **A catch.** Model A also rejected true French filler matches (`cascades club et fils`: p 1.00 → 0.04). Score explanations and leave-one-country-out checks pinned this on the continuous word-frequency columns, which effectively memorise US words.
- **Model B** keeps only the categorical look-alike columns. It transfers better in both directions (US→India +0.0037, India→US +0.0015) with out-of-fold 0.98705.
- **Per-country combination:** model A for US/India, a France-appropriate model for France.

### 26 Sep: the jump
- **Submission 14 (hybrid): 0.982205, +0.0059.** 15 (model B for France): 0.982594. 16 (France threshold 0.75): 0.982629.
- **Probe (17).** France rows were deliberately zeroed to split the score: **US+India 0.9844 on test** (out-of-fold 0.9874), **France 0.9727**.
- **Other checks:**
  - Label-free checks on 24 error patterns found no second planted-record family.
  - The France recall check found that the model mostly rejects look-alikes correctly.
- **Handoff (early 27 Sep, ~01:00).** `HANDOFF.md`, `analysis/` scripts and a one-command rebuild were written for review, then zipped.

### 27 Sep: final day
- **Submission 18:** a stricter US/India threshold lost (0.98234).
- **Submission 19:** the mean of models A and B, with a slightly more permissive decision: **0.983167**.
- **The pattern.** Test true matches score lower than out-of-fold, so more permissive decisions win.
- **Submission 22:** models retrained with twice the data per fold (out-of-fold A 0.98769, B 0.98736, mean 0.98771): 0.983212.
- **Submission 23:** adding an XGBoost member (out-of-fold 0.98768) *lost* 0.0003. XGBoost is stricter on test.
- **Submission 24:** threshold 0.55 was slightly worse (0.983155).
- **Submission 25:** France from the mean of model B and the rebuild model, then the look-alike filter: **0.983405, final**.
- **Official package.** `src/blend.py` was added so the code reproduces submission 25 exactly. The documentation placeholders and team details were filled, and `ChickenJockey_submission.zip` was built and verified.

---

## 6. Every submission

Scores are public leaderboard; "–" means not uploaded, or no score recorded.

| # | Change | OOF | Public LB |
|---|---|---|---|
| 01 | First pipeline | 0.98031 | – |
| 02 | 4.5× data + sibling features | 0.98279 | – |
| 03 | Stage 3 | 0.98495 | – |
| 04 | Blocking v2 | 0.98589 | – |
| 05 | Pruning v2 | 0.98601 | 0.974431 |
| 06 | France threshold 0.75 | | 0.974962 |
| 07 | Clean rerun + French department → region | 0.98596 | 0.975440 |
| 08 | France 0.85 | | 0.975878 |
| 09 | France 0.65 | | – |
| 10 | France 0.95 | | 0.976312 |
| 11 | Probe (France zeroed) on 10 | | – |
| 12 | 10 + look-alike filter | | – |
| 13 | Model A everywhere | 0.98741 | – |
| 14 | Model A (US/IN) + rebuild + filter (FR) | | 0.982205 |
| 15 | Model A (US/IN) + model B @0.85 (FR) | 0.98705 (B) | 0.982594 |
| 16 | Model A (US/IN) + model B @0.75 (FR) | | 0.982629 |
| 17 | Probe on 16 | | 0.836972 → US+IN 0.9844 / FR 0.9727 |
| 18 | US/India threshold 0.80 | | 0.98234 |
| 19 | Mean(A, B) for US/IN, t 0.60, m 0.5 | 0.98744 | 0.983167 |
| 20 | 19 with US/IN t 0.55 | | – |
| 21 | 19 with France from mean(B, rebuild) + filter | | – |
| 22 | 19 with models retrained on 2× data | 0.98771 | 0.983212 |
| 23 | 22 + XGBoost member | 0.98778 | 0.98289 |
| 24 | 22 with US/IN t 0.55 | | 0.983155 |
| **25** | 22 with France from mean(B, rebuild) + filter | | **0.983405 (final)** |

---

## 7. Key findings

1. **Planted look-alikes were the main hidden error.** They were rare in train but common in test, especially France. Separating injected filler from ordinary vocabulary, using S2/S3-vs-S1 word frequencies computed label-free per country, fixed most of it: **+0.0059 in one step**.
2. **Transfer to an unseen country needs features that don't memorise words.** Continuous word-frequency values helped in-distribution but misjudged French filler. Keeping only the categorical signal fixed it.
3. **Test differs from train in more than the unseen country.**
   - Test has more distractors and look-alikes, and its true matches score lower.
   - Stricter decisions lost on test (18, 23), and permissive ones won (19).
   - Averaging models with different feature views helped (19, 25); a same-view model of a different family (XGBoost) did not.
4. **Out-of-fold gains don't always carry over.** The 3-model average was best out-of-fold but lost on the leaderboard. We only adopted changes backed by both an out-of-fold/transfer rationale and a leaderboard delta above noise (~0.0001).
5. **The leaderboard probe was very informative.** Zeroing France rows split the score exactly into US/India vs France and redirected the remaining effort.

---

## 8. What didn't work, or wasn't worth it

| Idea | Result |
|---|---|
| Fourth refinement stage | Out-of-fold lower (0.98576) |
| Same-family ensembles (LightGBM seeds/params; A + B + old models) | ≤ +0.00004 out-of-fold |
| XGBoost member | −0.0003 on the leaderboard |
| "Rare-token" similarities | Hurt transfer to an unseen country |
| Address-level look-alike rule | Address word swaps are true matches in train (alternate city names) |
| Extending the post-filter to same-initial swaps | Flat |
| Stricter US/India thresholds | Lost on the leaderboard |
| Further France threshold steps | Exhausted after 0.95 with the old model; flat (0.75 ≈ 0.85) with model B |
| Per-source caps, per-country thresholds | Negligible |

---

## 9. Ideas not attempted (for a future round)

- **Train on a test-like composition** (drop about 19% of S1 entities so their records become distractors), so the model learns test's priors.
- **Name-frequency features as rates** rather than counts (test has half the S1 per country).
- **Score test stage-1 with the fold models** instead of the full model, for train/test consistency.
- **France normalization:** `compagnie` is canonicalized to the legal form `cie`, which hides some French descriptor swaps. Fixing it needs a full rebuild.
- **More look-alike detectors:** numeral swaps (`first`/`fourth`, `ii`/`iv`), professional suffixes (`md`/`do`), unit-number changes.

---

## 10. Deliverables and where they are

All paths are under `C:\Users\user\Desktop\PROJECTS 2026\Amazon ML challenge 2026\`.

| Item | Location |
|---|---|
| **Official package** (submission 25 outputs, code, documentation) | `ChickenJockey_submission.zip` |
| Code repository (all tags `sub-01`…`sub-25`, `sub-25-final`) | `amazon-ml-2026\`, and GitHub `srikantlose/amazon-ml-2026` (private) |
| Methodology document | `amazon-ml-2026\docs\Documentation_template.md` |
| Full handoff (file map, commands, gotchas, final-day update) | `amazon-ml-2026\HANDOFF.md` |
| Lab notebook with all experiment numbers | `amazon-ml-2026\docs\approach_notes.md` |
| Analysis scripts behind every finding | `amazon-ml-2026\analysis\` |
| One-command rebuild of the final submission | `amazon-ml-2026\scripts\reproduce_best.sh` |
| Every submission's file and note | `amazon-ml-2026\submissions\NN_*\` |
| Earlier packages (outdated) | `amazon-ml-2026_best_sub10.zip` (day 1), `amazon-ml-2026_handoff_sub16.zip` (27 Sep, 01:00) |

---

## 11. Lessons for next time

- **Read the data, not just the metrics.** The breakthrough came from reading 30 borderline France pairs by eye, then verifying the pattern with train labels.
- **Budget leaderboard slots for diagnosis.** One probe told us more than several threshold tweaks.
- **Treat "unseen country" as a transfer problem,** and validate with leave-one-country-out on a grid wide enough not to cap the optimum.
- **Keep every upload reproducible:** commit, tag and note per submission. That made the final package straightforward.
- **Track wall-clock time carefully on the last day,** and keep the final upload and the official package ready well before the deadline.
