# Submission 15 (per-country hybrid)

- US and India rows: identical to submissions 13 and 14 (`lgb_stage3la_0926_033632`)
- France rows: `lgb_stage3lacat_0926_051023` at France threshold 0.85 — look-alike columns without the three
  word-frequency values (la_extra_lratio_min, la_extra_lshare_max, la_missing_lshare_min) + relative candidate
  counts; OOF F0.5 0.98705 (US 0.98728, India 0.98670); leave-one-country-out +0.0037 US→India, +0.0015 India→US
  over the rebuild features (the full look-alike columns: +0.0063 / −0.0024)
- France: 844,564 pairs, 401 vocabulary swaps, 60,788 filler-addition pairs kept (13: 49,054; 14: 63,819)
- 13 / 14 / 15 differ only in the France rows: look-alike model at 0.85 / rebuild + post-filter at 0.95 /
  categorical look-alike model at 0.85
- validator PASS (--check-ids)
- public LB: pending
