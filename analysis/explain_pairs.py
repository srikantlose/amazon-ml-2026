"""Per-feature contributions (LightGBM pred_contrib, mean of the fold models) for chosen test pairs.

    python analysis/explain_pairs.py lgb_stage2la_0926_023743 "cascades france club|cascades club et fils" ...

Pairs are given as "S1 name_n|record name_n" (normalized names from data/cache/test/records_*.parquet).
This is how the look-alike model's France over-rejection was traced on 26 Sep: for French "drop a word +
add filler" matches, stage 2 was pulled down by la_extra_lratio_min / la_extra_lshare_max (-1.3 to -2.7 logits)
and s1_n_cands_rel, and stage 3 then by p2_rec_margin; the rebuild model scored the same pairs 0.98-1.00.
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.blocking import pruned_path  # noqa: E402
from src.features import load_features  # noqa: E402
from src.normalize import records_path  # noqa: E402
from src.utils import load_config  # noqa: E402

cfg = load_config([str(ROOT / "configs" / "base.yaml")])
run = sys.argv[1]
want = {tuple(a.split("|", 1)) for a in sys.argv[2:]}
c = pd.read_parquet(pruned_path(cfg, "test"), columns=["s1", "rec"])
s1, rec = c["s1"].to_numpy(np.int64), c["rec"].to_numpy(np.int64)
n1 = pd.read_parquet(records_path(cfg, "test", "s1"), columns=["name_n"])["name_n"].to_numpy()[s1]
n2 = pd.read_parquet(records_path(cfg, "test", "s23"), columns=["name_n"])["name_n"].to_numpy()[rec]
cand = np.flatnonzero(np.isin(n1, [a for a, _ in want]))
idx = np.array(sorted(i for i in cand if (n1[i], n2[i]) in want))
d = ROOT / "data" / "cache" / "runs" / run
meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
p = np.load(d / "test_prob.npy")
X = pd.concat([load_features(cfg, "test", rows=idx, columns=cols, subdir=sd)
               for sd, cols in meta["feature_dirs"].items()], axis=1)[meta["features"]]
boosters = [lgb.Booster(model_file=str(f)) for f in sorted(d.glob("model_fold*.txt"))]
contrib = np.mean([b.predict(X, pred_contrib=True) for b in boosters], axis=0)
cols = meta["features"] + ["bias"]
for k, i in enumerate(idx):
    s = pd.Series(contrib[k], index=cols).sort_values()
    print(f"\n{n1[i]} -> {n2[i]}  p={p[i]:.3f}")
    print("   most negative:", ", ".join(f"{col}={X.iloc[k][col]:.3g}({v:+.2f})" for col, v in s.head(7).items()
                                        if col in X))
    print("   most positive:", ", ".join(f"{col}={X.iloc[k][col] if col in X else 0:.3g}({v:+.2f})"
                                        for col, v in s.tail(4).items()))
