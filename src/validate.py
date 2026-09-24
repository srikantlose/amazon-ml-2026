"""Submission checks: the organizers' validator plus our own consistency checks.

    python -m src.validate --config configs/base.yaml [--matching output/matching_results.tsv]
"""
from __future__ import annotations

import argparse
import subprocess
import sys

from src.utils import add_config_arg, get_logger, load_config, resolve

log = get_logger("validate")


def _read_lists(path) -> dict:
    out = {}
    with open(path, encoding="utf-8", newline="") as f:
        next(f)
        for line in f:
            if "\r" in line:
                raise ValueError(f"{path}: carriage return found; write files with '\\n' line endings")
            s1, _, rest = line.rstrip("\n").partition("\t")
            out[s1] = rest.split(",") if rest else []
    return out


def validate(cfg: dict, matching, candidate) -> bool:
    """Returns True when the official validator passes and our checks hold."""
    test_dir = resolve(cfg["data_dir"]) / "test"
    cmd = [sys.executable, str(resolve(cfg["paths"]["validator"])), "--matching", str(matching),
           "--candidate", str(candidate), "--test-dir", str(test_dir), "--check-ids"]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(res.stdout.strip())
    if res.returncode != 0:
        log.error("official validator failed (exit %d)", res.returncode)
        return False
    m, c = _read_lists(matching), _read_lists(candidate)
    not_subset = [s for s, ids in m.items() if set(ids) - set(c.get(s, []))]
    if not_subset:
        log.error("%d S1 rows have matches outside their candidates, e.g. %s", len(not_subset), not_subset[:3])
        return False
    sizes = [len(v) for v in m.values()]
    log.info("matching: %d rows, %.2f%% empty, %.3f matches per S1; candidates: %.2f per S1",
             len(m), 100 * sum(s == 0 for s in sizes) / len(sizes), sum(sizes) / len(sizes),
             sum(len(v) for v in c.values()) / len(c))
    return True


def main():
    ap = argparse.ArgumentParser()
    add_config_arg(ap)
    ap.add_argument("--matching")
    ap.add_argument("--candidate")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = resolve(cfg["paths"]["output_dir"])
    ok = validate(cfg, args.matching or out / "matching_results.tsv", args.candidate or out / "candidate_pairs.tsv")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
