"""Turn a saved run's test predictions into a submission CSV, then validate it.

    python -m src.predict --config configs/base.yaml --run ens_0926_203000

Columns and their order come from paths.sample_sub when set, otherwise [id, target].
Rows follow the test CSV order. Output: {submissions_dir}/{run_id}.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import add_config_arg, get_logger, get_path, load_config, load_preds, read_split
from src.validate_submission import validate

log = get_logger("predict")


def submission_columns(cfg: dict) -> tuple[list[str], str]:
    """(ordered columns, prediction column name)."""
    id_col, target_col = cfg["columns"]["id"], cfg["columns"]["target"]
    sample = get_path(cfg, "sample_sub")
    if sample is not None and sample.exists():
        cols = list(pd.read_csv(sample, nrows=0).columns)
        pred_cols = [c for c in cols if c != id_col]
        if len(pred_cols) == 1:
            return cols, pred_cols[0]
        log.warning("sample submission has columns %s; using %r as the prediction column", cols, target_col)
        return cols, target_col
    return [id_col, target_col], target_col


def write_submission(cfg: dict, test_pred: np.ndarray, run_id: str, classes=None, out: Path | None = None) -> Path:
    id_col = cfg["columns"]["id"]
    test = read_split(cfg, "test")
    test_pred = np.asarray(test_pred)
    if len(test_pred) != len(test):
        raise ValueError(f"{len(test_pred)} predictions for {len(test)} test rows")

    if cfg["task"] == "regression":
        values = np.clip(test_pred.astype(np.float64), cfg.get("min_pred", 0.0), None)
    elif cfg["task"] == "classification" and test_pred.ndim == 2:
        values = np.asarray(classes)[test_pred.argmax(axis=1)]
    else:
        values = test_pred

    cols, pred_col = submission_columns(cfg)
    sub = pd.DataFrame({id_col: test[id_col].values, pred_col: values})[cols]
    out = out or get_path(cfg, "submissions_dir") / f"{run_id}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False)
    log.info("wrote %s (%d rows)", out, len(sub))

    errors = validate(out, get_path(cfg, "test_csv"), id_col, pred_col, cfg["task"],
                      sample_path=get_path(cfg, "sample_sub"))
    if errors:
        for e in errors:
            log.error("INVALID: %s", e)
        sys.exit(1)
    log.info("submission valid: %s", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    add_config_arg(ap)
    ap.add_argument("--run", required=True, help="run id with saved test preds")
    ap.add_argument("--out", help="output path (default: submissions_dir/{run}.csv)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    _, test, meta = load_preds(cfg, args.run)
    write_submission(cfg, test, args.run, meta.get("classes"), Path(args.out) if args.out else None)


if __name__ == "__main__":
    main()
