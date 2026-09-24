"""Weighted blend of saved OOF predictions; weights optimized on the OOF metric, applied to test.

    python -m src.ensemble --config configs/base.yaml --runs gbm_lightgbm_0925_101500 fusion_l1_log_0925_110000
    python -m src.ensemble --config configs/base.yaml --runs <ids...> --space log --write

--space log blends log1p(pred) (a geometric-style mean), which usually suits SMAPE/MAPE targets.
Weights are non-negative and sum to 1. The blend is saved as its own run so it can be re-blended.
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy.optimize import minimize

from src.metrics import get_metric
from src.utils import (add_config_arg, get_logger, load_config, load_preds, log_experiment, make_run_id,
                       read_split, save_preds, score_preds)

log = get_logger("ensemble")


def blend(preds: list[np.ndarray], w: np.ndarray, space: str) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    w = w / w.sum() if w.sum() > 0 else np.full(len(w), 1 / len(w))
    if space == "log":
        return np.expm1(sum(wi * np.log1p(np.clip(p, 0, None)) for wi, p in zip(w, preds)))
    return sum(wi * p for wi, p in zip(w, preds))


def optimize_weights(oofs, y, cfg, classes, space: str) -> np.ndarray:
    _, greater = get_metric(cfg["metric"])
    sign = -1.0 if greater else 1.0
    objective = lambda w: sign * score_preds(cfg, y, blend(oofs, w, space), classes)
    n = len(oofs)
    best = None
    # a few starts: equal weights plus each single model, since the surface can be flat/non-convex
    starts = [np.full(n, 1 / n)] + [np.eye(n)[i] * 0.7 + 0.3 / n for i in range(n)]
    for x0 in starts:
        res = minimize(objective, x0, method="SLSQP", bounds=[(0, 1)] * n,
                       constraints=({"type": "eq", "fun": lambda w: w.sum() - 1},),
                       options={"maxiter": 200, "ftol": 1e-9})
        if best is None or res.fun < best.fun:
            best = res
    w = np.clip(best.x, 0, None)
    return w / w.sum()


def main():
    ap = argparse.ArgumentParser()
    add_config_arg(ap)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--space", choices=["linear", "log"], default="log")
    ap.add_argument("--write", action="store_true", help="also write + validate the submission CSV")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    cfg = load_config(args.config)
    space = args.space if cfg["task"] == "regression" else "linear"
    y = read_split(cfg, "train")[cfg["columns"]["target"]].to_numpy()

    oofs, tests, classes = [], [], None
    for rid in args.runs:
        oof, test, meta = load_preds(cfg, rid)
        classes = meta.get("classes", classes)
        oofs.append(oof)
        tests.append(test)
        log.info("%-45s CV %s = %.5f", rid, cfg["metric"], score_preds(cfg, y, oof, classes))

    w = optimize_weights(oofs, y, cfg, classes, space)
    oof_b, test_b = blend(oofs, w, space), blend(tests, w, space)
    if cfg["task"] == "regression":
        oof_b = np.clip(oof_b, cfg.get("min_pred", 0.0), None)
        test_b = np.clip(test_b, cfg.get("min_pred", 0.0), None)
    cv_score = score_preds(cfg, y, oof_b, classes)
    for rid, wi in zip(args.runs, w):
        log.info("weight %.3f  %s", wi, rid)
    log.info("blend (%s space) CV %s = %.5f", space, cfg["metric"], cv_score)

    run_id = make_run_id("ens")
    save_preds(cfg, run_id, oof_b, test_b, {
        "model": "ensemble", "runs": args.runs, "weights": w.tolist(), "space": space,
        "cv_score": cv_score, "metric": cfg["metric"], "classes": classes,
    })
    log_experiment(cfg, run_id, "ensemble", args.runs, cv_score,
                   notes=f"space={space} weights={np.round(w, 3).tolist()} {args.notes}".strip())
    print(run_id)

    if args.write:
        from src.predict import write_submission

        write_submission(cfg, test_b, run_id, classes)


if __name__ == "__main__":
    main()
