# Submission 23 (three-model average for US/India)

- US and India rows: mean of A_big (`lgb_stage3labig_0927_181722`, LightGBM), B_big
  (`lgb_stage3lacatbig_0927_181932`, LightGBM) and XGBoost model A (`stage3xgbA_0927_185908`, `src/train_xgb.py`
  on model A's stage-3 features, GPU, OOF 0.98768), at t=0.60, m=0.50; OOF of the mean 0.98778 (US 0.98794,
  India 0.98754) vs 0.98771 for submission 22's two-model mean
- France rows: identical to submission 22 (B_big at 0.75)
- 12,852 US/India rows differ from 22; validator PASS
- public LB: pending
