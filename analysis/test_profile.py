"""Profile a run's test probabilities: accepted pairs per country, France threshold sweep,
look-alike swap pairs still accepted. Usage: test_profile.py <run> [<run> ...]"""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.blocking import pruned_path  # noqa: E402
from src.decide import select_per_pair  # noqa: E402
from src.features import load_features  # noqa: E402
from src.normalize import records_path  # noqa: E402
from src.utils import load_config  # noqa: E402

cfg = load_config([str(Path(__file__).resolve().parents[1] / "configs" / "base.yaml")])
c = pd.read_parquet(pruned_path(cfg, "test"), columns=["s1", "rec"])
s1, rec = c["s1"].to_numpy(np.int64), c["rec"].to_numpy(np.int64)
country = pd.read_parquet(records_path(cfg, "test", "s1"), columns=["country"])["country"].to_numpy()
pc = country[s1]
n_s1 = {ct: int((country == ct).sum()) for ct in ("US", "India", "France")}
la = load_features(cfg, "test", subdir="features_la", columns=["la_swap_vocab", "la_swap_same_initial", "la_extra_vocab_len_min"])
flag = (la.la_swap_vocab.to_numpy() == 1) & (la.la_swap_same_initial.to_numpy() == 0) & (la.la_extra_vocab_len_min.to_numpy() >= 4)
for run in sys.argv[1:]:
    d = str(Path(__file__).resolve().parents[1] / "data" / "cache" / "runs" / run)
    meta = json.load(open(f"{d}/meta.json"))
    t, m = meta["decision"]["threshold"], meta["decision"]["margin"]
    p = np.load(f"{d}/test_prob.npy")
    print(f"\n== {run}: OOF {meta['decision']['f05']:.5f} at t={t} m={m}; per country {meta['per_country']}")
    for fr_t in (0.65, 0.75, 0.85, 0.9, 0.95):
        thr = np.where(pc == "France", fr_t, t).astype(np.float32)
        mar = np.where(pc == "France", 0.5, m).astype(np.float32)
        keep = select_per_pair(rec, p, thr, mar)
        line = []
        for ct in ("US", "India", "France"):
            k = keep & (pc == ct)
            line.append(f"{ct} {k.sum():>9,} ({k.sum() / n_s1[ct]:.3f}/S1, swaps {int((k & flag).sum()):>6,})")
        print(f"  France t={fr_t:.2f}: " + " | ".join(line))
