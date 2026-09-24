"""Archive the current output/ files as the next numbered submission, then commit and tag.

    python scripts/save_submission.py --run <run_id> --note "baseline lgb"
    python scripts/save_submission.py --run <run_id> --note "..." --no-git

Creates submissions/NN_<run_id>/ with both TSVs (gitignored, kept locally for version history),
the run's meta.json, the config used and a short note, then commits the code state and tags it
sub-NN so every leaderboard upload maps to an exact commit.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import load_config, resolve  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", default=[str(ROOT / "configs" / "base.yaml")])
    ap.add_argument("--run", required=True)
    ap.add_argument("--note", required=True)
    ap.add_argument("--no-git", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)

    subs = resolve(cfg["paths"]["submissions_dir"])
    subs.mkdir(exist_ok=True)
    taken = [int(p.name[:2]) for p in subs.iterdir() if p.is_dir() and p.name[:2].isdigit()]
    n = max(taken, default=0) + 1
    dest = subs / f"{n:02d}_{args.run}"
    dest.mkdir()
    out = resolve(cfg["paths"]["output_dir"])
    for name in ("matching_results.tsv", "candidate_pairs.tsv"):
        shutil.copy2(out / name, dest / name)
    run_dir = resolve(cfg["paths"]["cache_dir"]) / "runs" / args.run
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    shutil.copy2(run_dir / "meta.json", dest / "meta.json")
    for c in args.config:
        shutil.copy2(c, dest / Path(c).name)
    score = meta["decision"]["f05"]
    (dest / "NOTE.md").write_text(
        f"# Submission {n:02d}\n\n- run: `{args.run}`\n- note: {args.note}\n"
        f"- OOF macro F0.5: {score:.5f} (threshold {meta['decision']['threshold']}, "
        f"margin {meta['decision']['margin']})\n- per country: {meta.get('per_country')}\n"
        f"- public LB: (fill in after upload)\n", encoding="utf-8")
    print(f"saved {dest}")

    if not args.no_git:
        msg = f"Submission {n:02d}: {args.note} (OOF F0.5 {score:.4f})"
        subprocess.run(["git", "add", "-A"], cwd=ROOT, check=True)
        subprocess.run(["git", "commit", "-q", "-m", msg], cwd=ROOT, check=True)
        subprocess.run(["git", "tag", f"sub-{n:02d}"], cwd=ROOT, check=True)
        print(f"committed and tagged sub-{n:02d}")


if __name__ == "__main__":
    main()
