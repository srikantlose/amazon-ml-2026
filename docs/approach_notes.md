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
