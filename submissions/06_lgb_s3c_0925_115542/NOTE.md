# Submission 06

- run: `lgb_s3c_0925_115542`
- note: submission 05 model + stricter decision for country labels unseen in training (France: threshold 0.75, margin 0.5); leave-one-country-out analysis in src/loco.py
- OOF macro F0.5: 0.98601 (threshold 0.65, margin 0.6)
- per country: {'f05_India': 0.9853397746206582, 'f05_US': 0.9864524162991707}
- public LB: 0.974962 (+0.000531 vs submission 05; only change: France threshold 0.75 instead of 0.65)
