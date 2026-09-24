"""Shared helpers: config loading, seeding, logging, timing, experiment log."""
from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import getpass
import logging
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "base.yaml"

EXPERIMENT_COLUMNS = [
    "timestamp", "run_id", "owner", "model", "features", "cv_score", "lb_score", "notes",
]


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
    cfg: dict = {}
    for p in paths or [DEFAULT_CONFIG]:
        p = Path(p)
        if not p.is_absolute():
            p = ROOT / p
        with open(p, encoding="utf-8") as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f) or {})
    return cfg


def add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", nargs="+", default=[str(DEFAULT_CONFIG)],
                        help="YAML config(s); later files override earlier ones")


def resolve(path_like) -> Path:
    """Resolve a config path against the repo root."""
    p = Path(path_like)
    return p if p.is_absolute() else ROOT / p


def cache_dir(cfg: dict, split: str | None = None) -> Path:
    d = resolve(cfg["paths"]["cache_dir"])
    if split:
        d = d / split
    d.mkdir(parents=True, exist_ok=True)
    return d


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


def get_logger(name: str = "er") -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s",
                        datefmt="%H:%M:%S")
    return logging.getLogger(name)


@contextmanager
def timer(name: str, logger: logging.Logger | None = None):
    t0 = time.perf_counter()
    yield
    msg = f"[{name}] {time.perf_counter() - t0:.1f}s"
    (logger.info if logger else print)(msg)


def make_run_id(prefix: str) -> str:
    return f"{prefix}_{dt.datetime.now():%m%d_%H%M%S}"


def log_experiment(cfg: dict, run_id: str, model: str, features, cv_score: float,
                   notes: str = "", lb_score: str = "") -> None:
    path = resolve(cfg["paths"]["experiments_csv"])
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
