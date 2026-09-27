# Submission 24 (submission 22 with a more permissive US/India threshold)

- identical to submission 22 except the US/India decision threshold of the A_big/B_big mean: 0.55 instead of
  0.60 (margin 0.50); France rows unchanged
- why: on the public LB, more permissive US/India decisions won (19: +0.00054) and stricter ones lost
  (18: -0.00029; 23, whose XGBoost member accepts ~3K fewer US/India pairs on test: -0.00032). Test true matches
  score lower than out of fold. OOF cost of 0.55 vs 0.60 on the mean is ~0.00006.
- validator PASS
- public LB: 0.983155 (-0.000057 vs submission 22): 0.60 is the better US/India threshold for these models
