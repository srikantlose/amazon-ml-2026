"""Submission checker. Run before every upload; exits 1 on any failure.

    python -m src.validate_submission --sub submissions/x.csv --test data/raw/test.csv \
        --id-col sample_id --target-col price --task regression [--sample data/raw/sample_test_out.csv]

    # or pull everything from the config:
    python -m src.validate_submission --config configs/base.yaml --sub submissions/x.csv

Extraction tasks (2024 style) check the "<number> <unit>" format; with --entity-col the unit must
be allowed for that row's entity. ENTITY_UNITS is the 2024 list: replace it with the official one on Day 1.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ENTITY_UNITS = {
    "width": {"centimetre", "foot", "inch", "metre", "millimetre", "yard"},
    "depth": {"centimetre", "foot", "inch", "metre", "millimetre", "yard"},
    "height": {"centimetre", "foot", "inch", "metre", "millimetre", "yard"},
    "item_weight": {"gram", "kilogram", "microgram", "milligram", "ounce", "pound", "ton"},
    "maximum_weight_recommendation": {"gram", "kilogram", "microgram", "milligram", "ounce", "pound", "ton"},
    "voltage": {"kilovolt", "millivolt", "volt"},
    "wattage": {"kilowatt", "watt"},
    "item_volume": {"centilitre", "cubic foot", "cubic inch", "cup", "decilitre", "fluid ounce", "gallon",
                    "imperial gallon", "litre", "microlitre", "millilitre", "pint", "quart"},
}
ALL_UNITS = set().union(*ENTITY_UNITS.values())
_EXTRACTION_RE = re.compile(r"^\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s+([A-Za-z][A-Za-z ]*?)\s*$")


def validate(sub_path, test_path, id_col: str, target_col: str, task: str,
             sample_path=None, entity_col: str | None = None) -> list[str]:
    errors: list[str] = []
    sub = pd.read_csv(sub_path, dtype={id_col: str}, keep_default_na=False, na_values=[""])
    test = pd.read_csv(test_path, dtype={id_col: str})

    # columns
    if sample_path is not None and Path(sample_path).exists():
        expected = list(pd.read_csv(sample_path, nrows=0).columns)
    else:
        expected = [id_col, target_col]
    if list(sub.columns) != expected:
        errors.append(f"columns {list(sub.columns)} != expected {expected}")
    if id_col not in sub.columns or target_col not in sub.columns:
        return errors + [f"missing {id_col!r} or {target_col!r}; cannot check further"]

    # rows + ids
    if len(sub) != len(test):
        errors.append(f"row count {len(sub)} != test row count {len(test)}")
    sub_ids, test_ids = sub[id_col].astype(str), test[id_col].astype(str)
    n_dup = int(sub_ids.duplicated().sum())
    if n_dup:
        errors.append(f"{n_dup} duplicate ids")
    missing, extra = set(test_ids) - set(sub_ids), set(sub_ids) - set(test_ids)
    if missing:
        errors.append(f"{len(missing)} test ids missing, e.g. {sorted(missing)[:5]}")
    if extra:
        errors.append(f"{len(extra)} ids not in test, e.g. {sorted(extra)[:5]}")
    if not missing and not extra and len(sub) == len(test) and not (sub_ids.values == test_ids.values).all():
        errors.append("ids match as a set but not in test order")

    # values
    vals = sub[target_col]
    if task == "regression":
        num = pd.to_numeric(vals, errors="coerce")
        n_bad = int(num.isna().sum())
        if n_bad:
            errors.append(f"{n_bad} missing/non-numeric predictions")
        arr = num.to_numpy(dtype=np.float64)
        n_inf = int(np.isinf(arr).sum())
        if n_inf:
            errors.append(f"{n_inf} infinite predictions")
        n_nonpos = int((arr[np.isfinite(arr)] <= 0).sum())
        if n_nonpos:
            errors.append(f"{n_nonpos} predictions <= 0 (must be positive)")
    elif task == "classification":
        n_bad = int(vals.isna().sum())
        if n_bad:
            errors.append(f"{n_bad} missing predictions")
    elif task == "extraction":
        # empty string = "no prediction" and is allowed
        entities = test[entity_col].tolist() if entity_col and entity_col in test.columns else [None] * len(vals)
        bad = []
        for i, (v, ent) in enumerate(zip(vals.fillna(""), entities)):
            v = str(v)
            if not v.strip():
                continue
            m = _EXTRACTION_RE.match(v)
            allowed = ENTITY_UNITS.get(ent, ALL_UNITS) if ent is not None else ALL_UNITS
            if not m or m.group(2).strip() not in allowed:
                bad.append((i, v, ent))
        if bad:
            errors.append(f"{len(bad)} values not '<number> <allowed unit>', e.g. {bad[:5]}")
    else:
        errors.append(f"unknown task {task!r}")
    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", required=True)
    ap.add_argument("--config", nargs="+", help="fill test/id/target/task/sample from config")
    ap.add_argument("--test")
    ap.add_argument("--id-col")
    ap.add_argument("--target-col")
    ap.add_argument("--task", choices=["regression", "classification", "extraction"])
    ap.add_argument("--sample", help="official sample output CSV (defines exact columns)")
    ap.add_argument("--entity-col", help="extraction: test column holding entity names")
    args = ap.parse_args()

    if args.config:
        from src.predict import submission_columns
        from src.utils import get_path, load_config

        cfg = load_config(args.config)
        sample = get_path(cfg, "sample_sub")
        args.test = args.test or str(get_path(cfg, "test_csv"))
        args.id_col = args.id_col or cfg["columns"]["id"]
        args.target_col = args.target_col or submission_columns(cfg)[1]
        args.task = args.task or cfg["task"]
        args.sample = args.sample or (str(sample) if sample else None)
    missing = [k for k in ("test", "id_col", "target_col", "task") if not getattr(args, k)]
    if missing:
        ap.error(f"missing {missing} (pass them or --config)")

    errors = validate(args.sub, args.test, args.id_col, args.target_col, args.task, args.sample, args.entity_col)
    if errors:
        print(f"INVALID: {args.sub}")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print(f"OK: {args.sub}")


if __name__ == "__main__":
    main()
