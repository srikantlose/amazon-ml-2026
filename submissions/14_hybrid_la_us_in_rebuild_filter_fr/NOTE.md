# Submission 14 (per-country hybrid)

- US and India rows: submission 13 (`lgb_stage3la_0926_033632`, look-alike columns + relative candidate counts;
  OOF F0.5 0.98741, US 0.98760, India 0.98713)
- France rows: house-number rebuild `lgb_stage3_0926_025245` at France threshold 0.95, then `src/postfilter.py`
  (24,086 vocabulary-swap pairs dropped); 846,872 France pairs, 3.264 per S1
- why: on France the look-alike model also drops ~28K pairs that are not swaps; most read as true matches that
  add French filler ("cascades club et fils", "centre de chloe france"). The continuous word-frequency columns
  (la_extra_lratio_min, la_extra_lshare_max) place French filler in ranges that are distractor words in the US
  ("group", "north"), and stage 2 scores those pairs 0.36-0.58 where the rebuild scores 0.98-1.00.
- same candidate pairs as submissions 13 (identical candidate_pairs.tsv); validator PASS (--check-ids)
- public LB: pending
