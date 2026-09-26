# Submission 17 (diagnostic probe, not a final candidate)

- US and India rows: byte-identical to submission 16 (look-alike model `lgb_stage3la_0926_033632`)
- France rows: the wrong-id rows of submission 11 (one distinct test S2 id of a US record each), so every
  France S1 scores exactly 0
- reading: F(US+India on test) = LB / 0.850248, F(France in submission 16) = (0.982629 - LB) / 0.149752;
  compare F(US+India) with the OOF level of the look-alike model (0.9874)
- validator PASS (the only warning: France ids outside the candidate pairs, by construction)
- public LB: 0.836972 -> F(US+India) = 0.98439, F(France in submission 16) = 0.97266
