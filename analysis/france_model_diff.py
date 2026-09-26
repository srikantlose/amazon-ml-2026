"""Compare two France decisions on the same test candidates: pairs dropped / added, swaps, filler additions.

    python analysis/france_model_diff.py lgb_stage3_0926_025245:0.95:filter lgb_stage3lacat_0926_051023:0.85

Each argument is run:france_threshold[:filter]; US/India use each run's tuned threshold/margin.
26 Sep results: rebuild@0.95+filter keeps 846,872 France pairs (63,819 filler additions, 0 swaps);
look-alike model@0.85 830,034 (49,054 filler additions); categorical model@0.85 844,564 (60,788, 401 swaps).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.blocking import pruned_path  # noqa: E402
from src.decide import select_per_pair  # noqa: E402
from src.features import load_features  # noqa: E402
from src.normalize import records_path  # noqa: E402
from src.utils import load_config  # noqa: E402

cfg = load_config([str(ROOT / "configs" / "base.yaml")])
c = pd.read_parquet(pruned_path(cfg, "test"), columns=["s1", "rec"])
s1, rec = c["s1"].to_numpy(np.int64), c["rec"].to_numpy(np.int64)
S1 = pd.read_parquet(records_path(cfg, "test", "s1"), columns=["country", "name_n"])
n23 = pd.read_parquet(records_path(cfg, "test", "s23"), columns=["name_n"])["name_n"].to_numpy()
fr = (S1["country"].to_numpy() == "France")[s1]
la = load_features(cfg, "test", subdir="features_la",
                   columns=["la_swap_vocab", "la_swap_same_initial", "la_extra_vocab_len_min",
                            "la_extra_filler", "la_extra_vocab"])
swap = ((la.la_swap_vocab.to_numpy() == 1) & (la.la_swap_same_initial.to_numpy() == 0)
        & (la.la_extra_vocab_len_min.to_numpy() >= 4))
filler_add = (la.la_extra_filler.to_numpy() > 0) & (la.la_extra_vocab.to_numpy() == 0)


def decision(spec):
    parts = spec.split(":")
    run, t_fr = parts[0], float(parts[1])
    meta = json.loads((ROOT / "data" / "cache" / "runs" / run / "meta.json").read_text(encoding="utf-8"))
    p = np.load(ROOT / "data" / "cache" / "runs" / run / "test_prob.npy")
    t, m = meta["decision"]["threshold"], meta["decision"]["margin"]
    keep = select_per_pair(rec, p, np.where(fr, t_fr, t).astype(np.float32), np.where(fr, 0.5, m).astype(np.float32))
    if len(parts) > 2 and parts[2] == "filter":
        keep &= ~swap
    return keep


keeps = [decision(a) for a in sys.argv[1:3]]
for spec, k in zip(sys.argv[1:3], keeps):
    kf = k & fr
    print(f"{spec:<45} France pairs {kf.sum():>9,} | filler additions {int((kf & filler_add).sum()):>7,} "
          f"| swaps {int((kf & swap).sum()):>6,}")
a, b = keeps
dropped, added = a & ~b & fr, b & ~a & fr
print(f"\nin first, not second: {dropped.sum():,} (filler additions {int((dropped & filler_add).sum()):,}); "
      f"in second, not first: {added.sum():,}")
n1 = S1["name_n"].to_numpy()[s1]
rng = np.random.default_rng(11)
for title, m in (("in first only", dropped), ("in second only", added)):
    print(f"\n--- sample {title}")
    for i in rng.choice(np.flatnonzero(m), size=min(20, int(m.sum())), replace=False):
        print(f"   {n1[i]:<40} -> {n23[rec[i]]}")
