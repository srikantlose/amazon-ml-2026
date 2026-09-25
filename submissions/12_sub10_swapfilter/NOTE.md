# Submission 12

- base: submission 10 (`lgb_stage3_0925_183300`, France threshold 0.95)
- change: accepted pairs whose record swaps one S1 name word for another ordinary word (4+ letters, different
  initial) are dropped by `src/postfilter.py` (statistics from `src/lookalike.py`, test names only, no labels)
- dropped: France 23,584 / India 8,706 / US 3,520 pairs
- train check (stage-2 OOF of the house-number rebuild): the dropped pattern is 12.4% correct among accepted pairs
  (US 26%, India 3.8%); OOF macro F0.5 0.98509 -> 0.98536. Test has 5-60x more of these pairs per S1 than train.
- validator: PASS (--check-ids, matches within submission 10's candidate pairs)
- public LB: pending
