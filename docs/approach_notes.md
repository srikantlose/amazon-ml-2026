# Working notes (feed into Documentation_template.md at the end)

## EDA findings (train)
- 2,206,821 S1 / 5,034,616 S2 / 5,285,603 S3; test 1,732,544 / 4,887,273 / 5,082,316 (+ France).
- Every S2/S3 record matches at most one S1; ~26% of S2/S3 records match nothing.
- Matched pairs always share the country label.
- 5.6% of S1 are singletons; mean 3.46 matches per S1 (max 11: ≤5 from S2, ≤6 from S3).
- 39% of S1 names are shared with another S1 in the same country; only 26% of true pairs have identical normalized names.
- Scripts: Devanagari, Telugu, Kannada, Tamil, Bengali, Gujarati, Odia, Gurmukhi, Malayalam in names and state components.
- US: S1/S2 write state codes, S3 spells them out. India: S1 spells state names, S3 uses codes or native script.
- France (test only): regions, departments and cities appear interchangeably as the last address component.
- Empty addresses: 4.4% of matched records vs 0.3% of unmatched ones.
- Latin-1 mojibake in some addresses (`â€“`) → repaired at read time.

## Pipeline decisions
- Record-centric blocking inside each country label (open set, generic groupby).
- Learned transliteration map: 4,935 tokens + 84 address components, from train pairs only.

## Blocking results (full train, 25 Sep)
- Union: 221,784,003 pairs (21.5 per record, 100.5 per S1), pair recall 0.9823, oracle F0.5 0.9944.
- Per view recall: char full@10 0.9484, token full@10 0.8953, token addr@5 0.7586, exact name key 0.4276, token name@5 0.3554.
- Early 10% sample with SVD views only (full/name/addr @10/5/5): union 0.9741 → token views added +0.8 pt.
- Missed pairs, typical cases: brand token kept while the descriptive words change, plus an empty address; trade names with partial address overlap.
- Test union: 218,065,014 pairs. Runtime: train 31 min, test 25 min (RTX 4060 + 12 CPU workers).

## Stage-1 pruning (train, OOF)
- top_n 5 / p ≥ 0.001: 221.8M → 12.25M pairs (1.19 per record); recall 0.9823 → 0.9795; oracle 0.9944 → 0.9936.
- Test: 218.1M → 12.14M pairs (1.22 per record).

## Error decomposition (v1, OOF, t=0.65 m=0.2)
- FP 42.6K (34K distractor records, 8.7K records of another entity) → fixing all: +0.0048.
- FN inside candidates 180K (176K records left unassigned, 4.7K assigned elsewhere) → fixing all: +0.0085.
- Blocking/pruning misses 156.5K pairs.
- Decision variants (wider margin grid, expected-F0.5 decoding) gave only +0.0002 → the decision rule is not the bottleneck.

## Experiments
| run | change | blocking recall | OOF F0.5 | LB |
|---|---|---|---|---|
| lgb_v1_0925_032005 (sub-01) | 72 features, 10% S1 groups per fold, lr 0.08 | 0.9795 | 0.98031 (US 0.9820, IN 0.9778) | |
| lgb_bigtrain_0925_033136 | 45% S1 groups, lr 0.05 (~3.3K trees) | 0.9795 | 0.98250 (US 0.9838, IN 0.9806) | |
| lgb_sib_0925_040217 (sub-02) | + sibling features (similarity to the S1's confident stage-1 records) | 0.9795 | 0.98279 (US 0.9840, IN 0.9810) | |
| lgb_v3_0925_044935 | + name uniqueness (S1 name frequency), state agreement (first/last alphabetic token), name ambiguity within record | 0.9795 | 0.98298 (US 0.9841, IN 0.9812) | |

| lgb_s3_0925_052339 (sub-03) | stage 3: + context from stage-2 OOF probabilities (ranks/gaps/margins, S1 linked counts, p2 siblings), 75% S1 groups | 0.9795 | 0.98495 (US 0.9861, IN 0.9832) | |
| lgb_s2b_0925_081848 | stage 2 on blocking v2 (char full k=30 + brand-token/house key) | 0.9826 | 0.98406 | |
| lgb_s3b_0925_085328 (sub-04) | stage 3 on blocking v2 | 0.9826 | 0.98589 (US 0.9863, IN 0.9853) | |

## Blocking v2 (25 Sep, 06:00-07:00)
- Char full view k 10 → 30, new exact (first name token, house number) key (cap 50 S1 per key).
- Union: 439.8M train pairs, recall 0.9823 → 0.9879, oracle 0.9944 → 0.9963; house key alone recall 0.534.
- After pruning (top 5, p ≥ 0.001): 12.6M pairs, recall 0.9795 → 0.9826; pruning now costs 0.53 pt (top 8 would keep 0.9840).
- Runtime: blocking 64 min, pruning 75 min (group stats over 440M pairs dominate), rest ~70 min.

## Pruning v2 (25 Sep, 09:40-10:40)
- Stage-1 ranker + three rapidfuzz features (core-name ratio, address token-set, digit-token overlap); stage-1 probabilities of all pairs saved.
- Recall on 440M-pair union (0.9879 before pruning), train OOF:
  top 3: 0.9809 · top 5: 0.9830 (v1 ranker 0.9826) · top 8: 0.9843 · top 10: 0.9849 · top 12: 0.9852 (all p ≥ 0.001).
- String features add little (+0.04 pt at top 5); most of the gain is keeping 8 candidates (12.37M pairs, 1.20/record).
- Runtime: group stats 20 min, 4 ranker fits 10 min, OOF scoring with string features 22 min.

## Leave-one-country-out (proxy for unseen France), stage 2, current candidates
| setting | US → India | India → US |
|---|---|---|
| all features, t=0.6 m=0.5 | 0.9488 | 0.9643 |
| all features, best (t=0.75) | 0.9509 | 0.9677 |
| drop name_non_latin | 0.9487 | 0.9643 |
| drop length/frequency/legal/state too | 0.9441 | 0.9623 |
| country in training (reference OOF) | 0.9836 | 0.9850 |
- Keep all features. For unseen country labels use t=0.75, m=0.5 (submission 06); in-distribution cost ~0.0002.

## Stage-3 ensemble (25 Sep, 12:57-14:10)
- Members: lgb_s3c (0.98601), wider trees + feature_fraction 0.6 (0.98601), smaller trees + lr 0.03 (0.98599).
- Average of 3: 0.98605 (+0.00004). The members are too correlated; not worth the extra test scoring. Not used.

## Test-set composition
- Test has 5.75 records per S1 vs 4.68 in train while predicted matches per S1 stay ~3.4 → ~39% distractor records in test vs 26% in train.
- Simulating 1.9x distractor false positives on OOF moves the best threshold 0.60 → 0.70 but changes F0.5 by only 0.0001; cost of the extra distractors ≈ 0.001.

## Error analysis after sub-02 (OOF)
- False negatives: 55% have an empty record address (vs 3.3% of all true pairs); probabilities spread 0.1–0.6.
- False positives: similar names in another city/state; wrong S1 chosen when the true S1 was pruned away.

## Look-alike records (26 Sep, 00:50-01:45)
- Leaderboard deltas of the France threshold steps (0.65 → 0.75 → 0.85 → 0.95, same model, US/India rows identical)
  imply that France pairs scored 0.85-0.95 are only ~58% correct, against ~90% for US/India pairs in that band.
- Reading France pairs dropped between 0.85 and 0.95: about half are records that keep the S1's address and first
  word but swap the descriptor ("mauges amis sas" → "mauges collectif sas", "livre federation" → "livre sportive").
  These names exist as a single S2/S3 record and never as an S1: planted look-alike distractors.
- Words the sources inject into names ("services"/"center"/"partners" in the US, "com"/"dr" in India,
  "developpement"/"groupe"/"fils" in France) are 2-10x more frequent in S2/S3 names than in S1 names; ordinary
  vocabulary ("club", "ecole", "amis", first names) sits at ~0.85x in every country. True matches often drop a
  word and add filler ("pediatric clinic inc" → "pediatric inc services", 96% true in train), so a swap to
  ordinary vocabulary looked like a match to the model.
- Train labels, pairs the stage-1 ranker already likes (p1 ≥ 0.5): a swap to an ordinary word of 4+ letters with
  a different initial is 1.5% (US) / 0.5% (India) true; same pattern with the same first word and same address
  (the France case) 0.8% true (India, 5.8K pairs). Swaps sharing the initial are garbled abbreviations
  ("care" → "cea", 50-65% true); contractions ("gaming" → "gg", "homes" → "hs") are treated as the same word.
- Submission 10 accepted 23.5K France pairs of that pattern (8.8% of France S1), 8.6K India (1.0%), 3.5K US (0.5%).
- Address word swaps with the same name and house number are 98-99.6% true in train (alternate city names), so
  no address counterpart.
- `src/lookalike.py`: 13 columns per pair (missing/extra words, extra words split into filler / vocabulary /
  rare by the per-country S2/S3-vs-S1 frequency ratio, swap flag, swap similarity, shared initial, word length),
  statistics from each split's own names, no labels. Used as an extra feature folder for stages 2 and 3.

## Country-frequent-token features (25 Sep, 14:10-15:00) — not used
- Address/name similarities after removing tokens found in >1% of the country's records (admin areas, generic words).
- Stage-2 OOF 0.98444 → 0.98470, but leave-one-country-out US→India 0.9488 → 0.9454 (India→US 0.9643 → 0.9645).
- Worse transfer to an unseen country outweighs the small in-distribution gain (France = 15% of test). Kept behind `features.rare_tokens: false`.
