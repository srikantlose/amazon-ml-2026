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

## Test-set composition
- Test has 5.75 records per S1 vs 4.68 in train while predicted matches per S1 stay ~3.4 → ~39% distractor records in test vs 26% in train.
- Simulating 1.9x distractor false positives on OOF moves the best threshold 0.60 → 0.70 but changes F0.5 by only 0.0001; cost of the extra distractors ≈ 0.001.

## Error analysis after sub-02 (OOF)
- False negatives: 55% have an empty record address (vs 3.3% of all true pairs); probabilities spread 0.1–0.6.
- False positives: similar names in another city/state; wrong S1 chosen when the true S1 was pruned away.
