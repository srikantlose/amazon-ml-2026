# Submission 18 (US/India threshold test)

- identical to submission 16 except the US and India decision threshold: 0.80 instead of the OOF-tuned 0.65
  (margin 0.6); France rows unchanged; 31,715 US/India rows differ
- why: the probe (submission 17) puts US+India on test at 0.9844 against 0.9874 out of fold. On test, best-per-record
  pairs scored 0.6-0.99 are 1.4-2.6x as dense per S1 as out of fold while the >= 0.99 band is unchanged, so part
  of the extra mass is false pairs that a stricter threshold removes (estimated effect -0.0005 to +0.0015)
- produced with `src.predict --threshold 0.80` into output_la080/ and `src.combine --seen output_la080 --unseen output`
- validator PASS
- public LB: pending (upload on 27 Sep; the 26 Sep slots ended with a re-upload of submission 16)
