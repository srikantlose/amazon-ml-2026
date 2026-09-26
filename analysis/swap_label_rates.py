"""Train-label rates of the look-alike patterns (supports src/lookalike.py and src/postfilter.py).

Among pruned train pairs the stage-1 ranker already likes (p1 >= 0.5), per country:
share of true matches for vocabulary swaps (different / same initial, short words, low similarity),
"drop a word + add filler", same-name pairs with a different house number.
Run from the repo root after the pipeline and `python -m src.lookalike --split train`.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.blocking import pruned_path  # noqa: E402
from src.data import build_labels  # noqa: E402
from src.features import load_features  # noqa: E402
from src.normalize import records_path  # noqa: E402
from src.utils import load_config  # noqa: E402

cfg = load_config([str(ROOT / "configs" / "base.yaml")])
c = pd.read_parquet(pruned_path(cfg, "train"), columns=["s1", "rec", "p1"])
s1r = pd.read_parquet(records_path(cfg, "train", "s1"), columns=["entity_id", "country"])
rid = pd.read_parquet(records_path(cfg, "train", "s23"), columns=["entity_id"])["entity_id"].to_numpy()
true_s1, _ = build_labels(cfg, s1r["entity_id"].to_numpy(), rid)
y = true_s1[c["rec"].to_numpy()] == c["s1"].to_numpy()
la = load_features(cfg, "train", subdir="features_la")
f = load_features(cfg, "train", columns=["first_eq", "a_tset", "house_eq", "n_tset"])
cty = s1r["country"].to_numpy()[c["s1"].to_numpy()]
hi = c["p1"].to_numpy() >= 0.5
sw = la.la_swap_vocab.to_numpy() == 1
si = la.la_swap_same_initial.to_numpy() == 1
ln = la.la_extra_vocab_len_min.to_numpy() >= 4
jw = la.la_swap_jw.to_numpy()
fe = f.first_eq.to_numpy() == 1
same_addr = (f.a_tset.to_numpy() >= 90) & (f.house_eq.to_numpy() != 0)
drop_fill = (la.la_n_missing.to_numpy() > 0) & (la.la_extra_filler.to_numpy() > 0) & (la.la_extra_vocab.to_numpy() == 0)
same_name_diff_house = (f.n_tset.to_numpy() >= 95) & (f.house_eq.to_numpy() == 0)
patterns = {
    "all pairs": np.ones(len(y), bool),
    "vocabulary swap": sw,
    "swap, different initial, 4+ letters (post-filter rule)": sw & ~si & ln,
    "  ... & same first word & same address (France case)": sw & ~si & ln & fe & same_addr,
    "swap, same initial (garbled abbreviations)": sw & si,
    "swap, extra word < 4 letters": sw & ~ln,
    "swap, similarity < 0.6": sw & (jw < 0.6),
    "drop a word + add filler": drop_fill,
    "same name, different house number": same_name_diff_house,
}
for country in ("US", "India"):
    k = (cty == country) & hi
    print(f"\n[{country}] pairs with p1 >= 0.5")
    for name, m in patterns.items():
        mm = k & m
        print(f"  {name:<58} n={mm.sum():>10,}  true-match rate {y[mm].mean() if mm.any() else float('nan'):.3f}")
