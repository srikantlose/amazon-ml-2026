"""Zip the repo (code, configs, notebooks, docs; no data or envs).

    python scripts/make_zip.py                 # -> ../amazon-ml-2026_code.zip
    python scripts/make_zip.py --out code.zip

Use it for the final code submission, or to move the repo onto a SageMaker instance
through the Jupyter upload button when git isn't an option.
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE_DIRS = {".git", ".venv", "__pycache__", ".ipynb_checkpoints", "data", "logs", "catboost_info", "submissions"}
EXCLUDE_SUFFIXES = {".npy", ".parquet", ".pt", ".pth", ".ckpt", ".zip", ".pyc"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT.parent / f"{ROOT.name}_code.zip"))
    args = ap.parse_args()

    n = 0
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(ROOT.rglob("*")):
            rel = p.relative_to(ROOT)
            if any(part in EXCLUDE_DIRS for part in rel.parts[:-1]) or rel.parts[0] in EXCLUDE_DIRS:
                continue
            if p.is_dir() or p.suffix in EXCLUDE_SUFFIXES or p.name == "kaggle.json":
                continue
            zf.write(p, Path(ROOT.name) / rel)
            n += 1
    print(f"wrote {args.out} ({n} files)")


if __name__ == "__main__":
    main()
