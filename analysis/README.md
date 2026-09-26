# Analysis scripts

Scripts behind the findings in `docs/approach_notes.md` and `HANDOFF.md`. They are not part of the pipeline.
Run them from the repo root with the project virtualenv after the pipeline has filled `data/cache/`. Most need
the look-alike folders (`python -m src.lookalike --split train|test`) and the run folders named in the
docstrings.

| Script | What it shows |
|---|---|
| `swap_label_rates.py` | Train-label true-match rates of the look-alike patterns (vocabulary swap, different/same initial, drop + filler, same name / other house number) among pairs with p1 ≥ 0.5 |
| `postfilter_oof.py` | Out-of-fold effect of the vocabulary-swap post-filter for given runs |
| `test_band_density.py` | Best-per-record pair density per probability band, out of fold vs test (US/India) |
| `explain_pairs.py` | Per-feature contributions (LightGBM pred_contrib) for chosen test pairs; traced the France over-rejection |
| `france_model_diff.py` | France pairs dropped/added between two decisions; swaps and filler additions kept |
| `france_missed_matches.py` | For France S1 with 0-1 matches: same-address records nobody accepted, and why |
| `test_profile.py` | Accepted test pairs per country and France threshold, swap pairs still accepted, for given runs |
| `fr_forensics.py` | France S1 whose match lists changed between two submissions (read the pairs by eye) |
| `sub_sample.py` | Random S1 of one country with the records a submission matched to them |
| `namediff_train.py` | Same-address train pairs by label and kind of name difference (first look at swaps) |
| `swap_rate.py`, `swap_words.py` | Swap/add-word rates per 1000 S1 and the most common swapped/added words, train truth vs test acceptance |
| `noise_ratio.py` | Word frequency in S2/S3 names vs S1 names per country: filler (>1.15) vs ordinary vocabulary (~0.85) |
| `band.awk` | France pairs removed between threshold submissions, used to turn leaderboard deltas into band precision |
| `audit/` | Independent audit: feature-shift sensitivity (s1_n_cands +22% → +25-30% false pairs), token statistics train vs test, rule gains, edge cases of `src/lookalike.py` |
