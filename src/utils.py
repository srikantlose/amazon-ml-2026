"""Shared helpers: config loading, seeding, logging, timing, feature loading, experiment log."""
from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import getpass
import json
import logging
import os
import random
import re
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "base.yaml"

EXPERIMENT_COLUMNS = [
    "timestamp", "run_id", "owner", "model", "features", "cv_score", "lb_score", "notes",
]


# ---------------------------------------------------------------- config

def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(paths=None) -> dict:
    """Load one or more YAML files; later files override earlier ones."""
    paths = paths or [DEFAULT_CONFIG]
    cfg: dict = {}
    for p in paths:
        p = Path(p)
        if not p.is_absolute():
            p = ROOT / p
        with open(p, encoding="utf-8") as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f) or {})
    return cfg


def add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", nargs="+", default=[str(DEFAULT_CONFIG)],
        help="YAML config(s); later files override earlier ones",
    )


def get_path(cfg: dict, key: str) -> Path:
    """Resolve cfg['paths'][key] against the repo root."""
    value = cfg["paths"].get(key)
    if value is None:
        return None
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def read_split(cfg: dict, split: str) -> pd.DataFrame:
    """Read the train or test CSV, keeping the id column as string so ids compare exactly."""
    id_col = cfg["columns"]["id"]
    return pd.read_csv(get_path(cfg, f"{split}_csv"), dtype={id_col: str})


# ---------------------------------------------------------------- runtime

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def get_logger(name: str = "aml") -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


@contextmanager
def timer(name: str, logger: logging.Logger | None = None):
    t0 = time.perf_counter()
    yield
    msg = f"[{name}] {time.perf_counter() - t0:.1f}s"
    (logger.info if logger else print)(msg)


def make_run_id(prefix: str) -> str:
    return f"{prefix}_{dt.datetime.now():%m%d_%H%M%S}"


# ---------------------------------------------------------------- cached features

def cache_file(cfg: dict, name: str, split: str, ext: str = "npy") -> Path:
    return get_path(cfg, "cache_dir") / f"{name}_{split}.{ext}"


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s.split("/")[-1])


def text_cache_name(cfg: dict) -> str:
    return "txt_" + _slug(cfg["embed"]["text_model"])


def image_cache_name(cfg: dict) -> str:
    e = cfg["embed"]
    return "img_" + _slug(f"{e['image_model']}_{e['image_pretrained']}")


def resolve_feature_names(cfg: dict, names: list[str]) -> list[str]:
    """Expand the shorthands "txt"/"img" to the cache names of the configured embedding models."""
    alias = {"txt": text_cache_name(cfg), "img": image_cache_name(cfg)}
    return [alias.get(n, n) for n in names]


def load_features(cfg: dict, names: list[str], split: str) -> tuple[np.ndarray, list[str]]:
    """Concatenate cached feature blocks column-wise.

    "hand" -> hand_{split}.parquet (handcrafted text features)
    "txt" / "img" -> configured text / image embedding
    anything else -> {name}_{split}.npy (svd, has_image, other embeddings)
    """
    blocks, cols = [], []
    for name in resolve_feature_names(cfg, names):
        if name == "hand":
            df = pd.read_parquet(cache_file(cfg, "hand", split, "parquet"))
            blocks.append(df.to_numpy(dtype=np.float32))
            cols += list(df.columns)
        else:
            arr = np.load(cache_file(cfg, name, split))
            arr = arr.reshape(len(arr), -1).astype(np.float32)
            blocks.append(arr)
            cols += [f"{name}_{i}" for i in range(arr.shape[1])] if arr.shape[1] > 1 else [name]
    return np.hstack(blocks), cols


# ---------------------------------------------------------------- targets

def encode_target(cfg: dict, y_raw: pd.Series):
    """Returns (y_fit, classes). Regression: log1p if target_transform == 'log1p'."""
    task = cfg["task"]
    if task == "regression":
        y = y_raw.astype(float).to_numpy()
        if cfg.get("target_transform") == "log1p":
            y = np.log1p(np.clip(y, 0, None))
        return y, None
    if task == "classification":
        classes, codes = np.unique(y_raw.astype(str).to_numpy(), return_inverse=True)
        return codes, classes.tolist()
    raise NotImplementedError(
        "extraction tasks need a generative model; the GBM/fusion trainers cover regression and classification")


def decode_pred(cfg: dict, pred: np.ndarray, classes=None) -> np.ndarray:
    """Model space -> submission space. Regression preds come back on the original scale, clipped."""
    if cfg["task"] == "regression":
        p = np.expm1(pred) if cfg.get("target_transform") == "log1p" else pred
        return np.clip(p, cfg.get("min_pred", 0.0), None)
    return np.asarray(classes)[np.asarray(pred).argmax(axis=1)]


def score_preds(cfg: dict, y_true, pred, classes=None) -> float:
    """Metric on stored predictions (original-scale values, or class probabilities)."""
    from src.metrics import get_metric

    fn, _ = get_metric(cfg["metric"])
    if cfg["task"] == "classification":
        pred = np.asarray(classes)[np.asarray(pred).argmax(axis=1)]
        y_true = np.asarray(y_true).astype(str)
    return fn(y_true, pred)


# ---------------------------------------------------------------- predictions + experiment log

def save_preds(cfg: dict, run_id: str, oof: np.ndarray, test: np.ndarray, meta: dict) -> None:
    out = get_path(cfg, "preds_dir")
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"oof_{run_id}.npy", oof)
    np.save(out / f"test_{run_id}.npy", test)
    with open(out / f"meta_{run_id}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)


def load_preds(cfg: dict, run_id: str) -> tuple[np.ndarray, np.ndarray, dict]:
    d = get_path(cfg, "preds_dir")
    meta_path = d / f"meta_{run_id}.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return np.load(d / f"oof_{run_id}.npy"), np.load(d / f"test_{run_id}.npy"), meta


def log_experiment(cfg: dict, run_id: str, model: str, features, cv_score: float,
                   notes: str = "", lb_score: str = "") -> None:
    path = get_path(cfg, "experiments_csv")
    new_file = not path.exists() or path.stat().st_size == 0
    row = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "owner": os.environ.get("EXP_OWNER") or getpass.getuser(),
        "model": model,
        "features": "+".join(features) if isinstance(features, (list, tuple)) else features,
        "cv_score": f"{cv_score:.5f}",
        "lb_score": lb_score,
        "notes": notes,
    }
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=EXPERIMENT_COLUMNS)
        if new_file:
            w.writeheader()
        w.writerow(row)
