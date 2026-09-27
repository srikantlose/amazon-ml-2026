# Submission 22 (submission 19's recipe on models retrained with more data)

- models retrained with configs/stage2_big.yaml (90% of the other folds' S1 groups) and stage3_big.yaml (95%)
  via scripts/train_big.sh:
  - model A: `lgb_stage2labig_0927_164730` (OOF 0.98664) -> `lgb_stage3labig_0927_181722` (OOF 0.98769)
  - model B: `lgb_stage2lacatbig_0927_164730` (OOF 0.98623) -> `lgb_stage3lacatbig_0927_181932` (OOF 0.98736)
- US and India rows: mean of A_big and B_big at t=0.60, m=0.50 (OOF of the mean 0.98771; US 0.98787,
  India 0.98748; submission 19's mean 0.98744)
- France rows: B_big at France threshold 0.75, margin 0.50 (852,597 pairs)
- 19,414 US/India and 6,834 France rows differ from 19; same candidate pairs; validator PASS
- public LB: pending
