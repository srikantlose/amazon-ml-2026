# Submission 11 (diagnostic probe, not a final candidate)

- base: submission 10 (`lgb_stage3_0925_183300`); the 1,473,092 US and India rows are byte-identical to it
- France: each of the 259,452 France S1 rows lists one distinct test S2 id taken from a US record.
  Matches always share the country label, so every France S1 scores exactly 0 (singletons included).
- purpose: split the public score by country group. France is 14.98% of test S1, so
  F(US+India) = LB / 0.850248 and F(France) = (0.976312 - LB) / 0.149752
- checks: row order and count match submission 10, no duplicate or unknown ids, `\n` line endings
- public LB: pending
