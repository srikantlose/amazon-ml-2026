"""GBM on cached feature blocks, using the shared folds.

    python -m src.train_gbm --config configs/base.yaml
    python -m src.train_gbm --config configs/base.yaml --features hand svd txt img has_image --model catboost

Saves {preds_dir}/oof_{run_id}.npy, test_{run_id}.npy and meta_{run_id}.json (regression preds on the
original scale, classification preds as class probabilities) and appends to experiments.csv.
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from src.cv import make_folds, run_cv
from src.utils import (add_config_arg, decode_pred, encode_target, get_logger, load_config, load_features,
                       log_experiment, make_run_id, read_split, save_preds, score_preds, seed_everything, timer)

log = get_logger("train_gbm")
# lightgbm >= 4.6 renamed eval_set; the old name still works and keeps older installs (SageMaker) happy
warnings.filterwarnings("ignore", message=".*'eval_set' is deprecated.*")


def make_fit_predict(cfg: dict, model_name: str, n_classes: int | None):
    gcfg = cfg["gbm"]
    es = gcfg["early_stopping_rounds"]
    is_cls = n_classes is not None

    def fit_predict(X_tr, y_tr, X_va, y_va, X_te, fold):
        seed = cfg["seed"] + fold
        if model_name == "lightgbm":
            import lightgbm as lgb

            params = dict(gcfg["lightgbm"], random_state=seed)
            if is_cls:
                params["objective"] = "multiclass" if n_classes > 2 else "binary"
                model = lgb.LGBMClassifier(**params)
            else:
                model = lgb.LGBMRegressor(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                      callbacks=[lgb.early_stopping(es, verbose=False), lgb.log_evaluation(0)])
            best = model.best_iteration_
        elif model_name == "catboost":
            import catboost as cb

            params = dict(gcfg["catboost"], random_seed=seed, early_stopping_rounds=es)
            if is_cls:
                params["loss_function"] = "MultiClass" if n_classes > 2 else "Logloss"
                model = cb.CatBoostClassifier(**params)
            else:
                model = cb.CatBoostRegressor(**params)
            model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
            best = model.get_best_iteration()
        else:
            raise ValueError(f"unknown gbm model {model_name!r}")

        predict = model.predict_proba if is_cls else model.predict
        va_pred, te_pred = predict(X_va), predict(X_te)
        log.info("fold %d: best_iter=%s", fold, best)
        return va_pred, te_pred

    return fit_predict


def main():
    ap = argparse.ArgumentParser()
    add_config_arg(ap)
    ap.add_argument("--features", nargs="+", help="feature blocks (default: gbm.features)")
    ap.add_argument("--model", choices=["lightgbm", "catboost"])
    ap.add_argument("--name", default="", help="suffix for the run id")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    model_name = args.model or cfg["gbm"]["model"]
    features = args.features or cfg["gbm"]["features"]
    run_id = make_run_id(f"gbm_{model_name}" + (f"_{args.name}" if args.name else ""))

    train = read_split(cfg, "train")
    target_col = cfg["columns"]["target"]
    folds = make_folds(train, cfg)
    y_fit, classes = encode_target(cfg, train[target_col])

    X_tr, cols = load_features(cfg, features, "train")
    X_te, _ = load_features(cfg, features, "test")
    X_tr = pd.DataFrame(X_tr, columns=cols)
    X_te = pd.DataFrame(X_te, columns=cols)
    log.info("run %s: %s, X_train=%s, X_test=%s", run_id, model_name, X_tr.shape, X_te.shape)

    with timer(run_id, log):
        oof, test = run_cv(make_fit_predict(cfg, model_name, len(classes) if classes else None),
                           X_tr, y_fit, X_te, folds)

    if cfg["task"] == "regression":
        oof, test = decode_pred(cfg, oof), decode_pred(cfg, test)
    cv_score = score_preds(cfg, train[target_col].to_numpy(), oof, classes)
    log.info("CV %s = %.5f", cfg["metric"], cv_score)

    save_preds(cfg, run_id, oof, test, {
        "model": model_name, "features": features, "cv_score": cv_score,
        "metric": cfg["metric"], "classes": classes,
    })
    log_experiment(cfg, run_id, model_name, features, cv_score, notes=args.notes)
    print(run_id)


if __name__ == "__main__":
    main()
