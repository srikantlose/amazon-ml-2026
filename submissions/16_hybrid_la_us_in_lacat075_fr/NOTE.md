# Submission 16 (per-country hybrid)

- US and India rows: identical to submissions 13-15 (`lgb_stage3la_0926_033632`)
- France rows: `lgb_stage3lacat_0926_051023` (categorical look-alike columns) at France threshold 0.75 instead of
  0.85 (submission 15): 852,570 France pairs (+8,006), 3.286 per S1
- why: submission 15 is the best so far (0.982594). France predicts 3.26 matches per S1 against ~3.46 true per S1
  in train (US 3.39, India 3.33), and has more S1 with 0 or 1 prediction (6.1% / 6.7%, train truth 5.6% / 5.4%),
  so France is short on recall; leave-one-country-out puts the transfer optimum of this model at 0.75-0.90
- produced with `python -m src.combine --seen output_la --unseen output` after predicting the categorical run
  with configs/base.yaml (unseen-country threshold 0.75); validator PASS
- public LB: pending
