# Submission 25 (France: retrained model B averaged with the rebuild, post-filtered)

- US and India rows: identical to submission 22 (mean of A_big and B_big at t=0.60, m=0.50)
- France rows: mean of B_big (`lgb_stage3lacatbig_0927_181932`) and the house-number rebuild
  (`lgb_stage3_0926_025245`) at France threshold 0.75, margin 0.5, then `src.postfilter` (1,172 swap pairs
  dropped); 854,937 France pairs (22: 852,597); 7,576 France rows differ from 22
- why: B rejects look-alikes but also drops some French "& fils"/"groupe" filler matches that the rebuild keeps;
  averaging two views helped for US/India (19), and the filter removes the swaps the rebuild accepts
- built with an inline average -> output_frmixbig/, then
  `src.combine --seen submissions/22_bigavg_us_in_bigB075_fr --unseen output_frmixbig --unseen-postfilter`
- validator PASS
- public LB: 0.983405 (+0.000193 vs submission 22): best; final leaderboard submission
