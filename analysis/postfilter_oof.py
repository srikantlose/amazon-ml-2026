"""Effect of the vocabulary-swap post-filter on out-of-fold macro F0.5 for given runs.

    python analysis/postfilter_oof.py lgb_stage2_0926_021300 lgb_stage3_0926_025245 lgb_stage3la_0926_033632

Results on 26 Sep: stage-2 rebuild 0.98509 -> 0.98536 (3,056 accepted swap pairs, 12.4% correct);
stage-3 rebuild 0.98646 -> 0.98665 (2,183, 17.4% correct); look-alike stage 2 0.98619 -> 0.98618
(342 pairs, 98% correct: the model already handles the pattern).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.blocking import pruned_path  # noqa: E402
from src.data import build_labels  # noqa: E402
from src.decide import select_per_pair  # noqa: E402
from src.features import load_features  # noqa: E402
from src.normalize import records_path  # noqa: E402
from src.utils import load_config  # noqa: E402

cfg = load_config([str(ROOT / "configs" / "base.yaml")])
c = pd.read_parquet(pruned_path(cfg, "train"), columns=["s1", "rec"])
s1, rec = c["s1"].to_numpy(np.int64), c["rec"].to_numpy(np.int64)
s1r = pd.read_parquet(records_path(cfg, "train", "s1"), columns=["entity_id", "country"])
rid = pd.read_parquet(records_path(cfg, "train", "s23"), columns=["entity_id"])["entity_id"].to_numpy()
true_s1, n_true = build_labels(cfg, s1r["entity_id"].to_numpy(), rid)
y = true_s1[rec] == s1
la = load_features(cfg, "train", subdir="features_la",
                   columns=["la_swap_vocab", "la_swap_same_initial", "la_extra_vocab_len_min"])
flag = ((la.la_swap_vocab.to_numpy() == 1) & (la.la_swap_same_initial.to_numpy() == 0)
        & (la.la_extra_vocab_len_min.to_numpy() >= 4))


def macro(keep):
    a = np.bincount(s1[keep & y], minlength=len(n_true))
    b = np.bincount(s1[keep & ~y], minlength=len(n_true))
    f = np.where(n_true == 0, (a + b == 0).astype(float), 1.25 * a / np.maximum(a + b + 0.25 * n_true, 1e-9))
    return f.mean()


for run in sys.argv[1:]:
    d = ROOT / "data" / "cache" / "runs" / run
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    t, m = meta["decision"]["threshold"], meta["decision"]["margin"]
    p = np.load(d / "oof.npy")
    keep = select_per_pair(rec, p, np.full(len(p), t, np.float32), np.full(len(p), m, np.float32))
    print(f"{run} (t={t}, m={m}): accepted {keep.sum():,}; swap pairs accepted {(keep & flag).sum():,} "
          f"(correct {y[keep & flag].mean():.3f}); OOF {macro(keep):.5f} -> filtered {macro(keep & ~flag):.5f}")
