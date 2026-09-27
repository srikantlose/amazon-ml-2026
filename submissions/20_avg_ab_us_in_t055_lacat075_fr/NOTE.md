# Submission 20 (US/India threshold 0.55 on the A+B average)

- identical to submission 19 except the US/India decision threshold: 0.55 instead of 0.60 (margin 0.50);
  about 10K more US/India pairs; OOF of the average at (0.55, 0.50) 0.98738 vs 0.98744 at (0.60, 0.50)
- why: 19 (more permissive than 16) gained 0.00054; the added pairs were ~89% correct by the LB delta, so test
  true matches score lower than OOF. This checks whether one more small step in that direction still pays.
- France rows unchanged (model B at 0.75); validator PASS
- public LB: pending
