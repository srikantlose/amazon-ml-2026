"""Fold assignment and generic OOF loop.

Folds are written once to {cache_dir}/folds.csv and reused by every model so OOF
predictions are comparable and blendable. Delete the file (or pass --force) to re-split.

    python -m src.cv --config configs/base.yaml
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

from src.utils import add_config_arg, get_logger, get_path, load_config, read_split

log = get_logger("cv")


def _resolve_scheme(cfg: dict) -> str:
    scheme = cfg["cv"].get("scheme", "auto")
    if scheme != "auto":
        return scheme
    if cfg["columns"].get("group"):
        return "group"
    if cfg["task"] in ("regression", "classification"):
        return "stratified"
    return "kfold"


def make_folds(df: pd.DataFrame, cfg: dict, force: bool = False) -> np.ndarray:
    """Return an int array of fold ids aligned with df rows (cached on disk)."""
    id_col = cfg["columns"]["id"]
    path = get_path(cfg, "cache_dir") / "folds.csv"
    if path.exists() and not force:
        saved = pd.read_csv(path, dtype={id_col: str})
        if len(saved) == len(df) and (saved[id_col].values == df[id_col].astype(str).values).all():
            return saved["fold"].to_numpy()
        raise ValueError(f"{path} does not match the current train rows; re-run with --force")

    n_folds = cfg["cv"]["n_folds"]
    seed = cfg["seed"]
    scheme = _resolve_scheme(cfg)
    target = cfg["columns"].get("target")
    folds = np.full(len(df), -1, dtype=int)
    idx = np.arange(len(df))

    if scheme == "group":
        splitter = GroupKFold(n_splits=n_folds)
        splits = splitter.split(idx, groups=df[cfg["columns"]["group"]])
    elif scheme == "stratified":
        y = df[target]
        if cfg["task"] == "regression":
            # rank first so ties don't collapse quantile edges
            y = pd.qcut(y.rank(method="first"), q=cfg["cv"]["n_bins"], labels=False)
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = splitter.split(idx, y)
    elif scheme == "kfold":
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = splitter.split(idx)
    else:
        raise ValueError(f"unknown cv scheme {scheme!r}")

    for k, (_, va) in enumerate(splits):
        folds[va] = k
    assert (folds >= 0).all()

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({id_col: df[id_col].astype(str), "fold": folds}).to_csv(path, index=False)
    log.info("wrote %s (%s, %d folds, sizes %s)", path, scheme, n_folds, np.bincount(folds).tolist())
    return folds


def _take(X, mask):
    return X.iloc[mask] if hasattr(X, "iloc") else X[mask]


def run_cv(fit_predict, X, y, X_test, folds: np.ndarray):
    """Generic OOF loop.

    fit_predict(X_tr, y_tr, X_va, y_va, X_test, fold) -> (va_pred, test_pred)
    Returns (oof, test_pred averaged over folds). Preds may be 1-D or 2-D (class probs).
    """
    oof, test = None, None
    fold_ids = np.unique(folds)
    for k in fold_ids:
        tr, va = folds != k, folds == k
        va_pred, te_pred = fit_predict(_take(X, tr), y[tr], _take(X, va), y[va], X_test, int(k))
        va_pred, te_pred = np.asarray(va_pred), np.asarray(te_pred)
        if oof is None:
            oof = np.zeros((len(folds),) + va_pred.shape[1:], dtype=np.float64)
            test = np.zeros_like(te_pred, dtype=np.float64)
        oof[va] = va_pred
        test += te_pred / len(fold_ids)
    return oof, test


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    add_config_arg(ap)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    make_folds(read_split(cfg, "train"), cfg, force=args.force)
