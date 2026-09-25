# Submission 03

- run: `lgb_s3_0925_052339`
- note: stage-3 matcher: context re-computed from stage-2 (lgb_v3) probabilities
- OOF macro F0.5: 0.98495 (threshold 0.65, margin 0.5)
- per country: {'f05_India': 0.983180898782892, 'f05_US': 0.9861362155358412}
- public LB: (fill in after upload)
- blocking: candidates for this submission came from the earlier run (char full view k=10, no house key); the k=30 / house_key settings in this commit's base.yaml were added for the next run.
- stage-2 model feeding the context: lgb_v3_0925_044935 (configs/stage2.yaml settings).
