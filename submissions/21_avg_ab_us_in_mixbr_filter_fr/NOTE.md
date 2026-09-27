# Submission 21 (France from the average of model B and the rebuild, post-filtered)

- US and India rows: identical to submission 19 (A+B average, t=0.60, m=0.50)
- France rows: mean of model B (`lgb_stage3lacat_0926_051023`) and the house-number rebuild
  (`lgb_stage3_0926_025245`) probabilities at France threshold 0.75, margin 0.5, then `src.postfilter`
  (vocabulary swaps removed); 855,015 France pairs (+2,445 vs 19), 64,604 filler additions kept (19: 63,201),
  0 swaps (19: 658); about 8K France pairs differ
- why: B rejects look-alikes but loses some French filler matches; the rebuild keeps them but accepts swaps,
  which the filter removes; averaging two different views helped for US/India (19)
- built with an inline average -> output_frmix/, then `src.combine --seen output_avg --unseen output_frmix --unseen-postfilter`
- validator PASS
- public LB: pending
