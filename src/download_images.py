"""Parallel, resumable image downloader.

    python -m src.download_images --config configs/base.yaml --split train
    python -m src.download_images --csv data/raw/test.csv --out data/images/test --url-col image_link --id-col sample_id

Files are named {id}.jpg (or sha1(url).jpg without an id column), so re-running skips
anything already on disk. Writes manifest_{name}.csv (every row) and failures_{name}.csv
into the output dir. Missing images are handled downstream (zero embedding + has_image=0).
"""
from __future__ import annotations

import argparse
import hashlib
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

from src.utils import add_config_arg, get_logger, get_path, load_config

log = get_logger("download")
_local = threading.local()
_HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko)"}


def image_path(images_dir: Path, row_id, url) -> Path:
    """Canonical on-disk path for a row's image; embed.py uses the same rule."""
    if row_id is not None and str(row_id) != "nan":
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(row_id))
    else:
        name = hashlib.sha1(str(url).encode()).hexdigest()
    return Path(images_dir) / f"{name}.jpg"


def _session() -> requests.Session:
    if not hasattr(_local, "s"):
        s = requests.Session()
        s.headers.update(_HEADERS)
        adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=4)
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        _local.s = s
    return _local.s


def fetch_one(url, dest: Path, max_side: int | None, timeout: float, retries: int) -> tuple[str, str]:
    """Returns (status, error). status: ok | skip | no_url | fail."""
    if dest.exists() and dest.stat().st_size > 0:
        return "skip", ""
    if not isinstance(url, str) or not url.startswith("http"):
        return "no_url", ""
    err = ""
    for attempt in range(retries + 1):
        try:
            r = _session().get(url, timeout=timeout)
            if r.status_code == 404:
                return "fail", "404"
            r.raise_for_status()
            img = Image.open(BytesIO(r.content)).convert("RGB")
            if max_side:
                img.thumbnail((max_side, max_side))
            tmp = dest.with_suffix(".part")
            img.save(tmp, format="JPEG", quality=90)
            tmp.replace(dest)
            return "ok", ""
        except Exception as e:  # network errors, bad bytes, truncated images
            err = f"{type(e).__name__}: {e}"[:200]
            if attempt < retries:
                time.sleep(0.5 * 2 ** attempt + random.random() * 0.5)
    return "fail", err


def download(df: pd.DataFrame, url_col: str, id_col: str | None, out_dir: Path, name: str,
             workers: int = 64, max_side: int | None = 512, timeout: float = 15, retries: int = 3) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = df[id_col].tolist() if id_col else [None] * len(df)
    urls = df[url_col].tolist()
    dests = [image_path(out_dir, i, u) for i, u in zip(ids, urls)]

    results = [None] * len(df)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_one, u, d, max_side, timeout, retries): k for k, (u, d) in enumerate(zip(urls, dests))}
        for fut in tqdm(as_completed(futs), total=len(futs), desc=f"images:{name}", mininterval=2):
            results[futs[fut]] = fut.result()
    elapsed = time.perf_counter() - t0

    manifest = pd.DataFrame({
        "id": ids, "url": urls, "path": [str(d) for d in dests],
        "status": [r[0] for r in results], "error": [r[1] for r in results],
    })
    manifest.to_csv(out_dir / f"manifest_{name}.csv", index=False)
    fails = manifest[manifest.status.isin(["fail", "no_url"])]
    fails.to_csv(out_dir / f"failures_{name}.csv", index=False)

    counts = manifest.status.value_counts().to_dict()
    fetched = counts.get("ok", 0)
    log.info("%s: %s in %.0fs (%.1f new img/s); failures logged to %s",
             name, counts, elapsed, fetched / max(elapsed, 1e-9), out_dir / f"failures_{name}.csv")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    add_config_arg(ap)
    ap.add_argument("--split", choices=["train", "test"], help="use the train/test CSV from config")
    ap.add_argument("--csv", help="explicit CSV path (overrides --split)")
    ap.add_argument("--out", help="output dir (default: {images_dir}/{split})")
    ap.add_argument("--url-col")
    ap.add_argument("--id-col")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--max-side", type=int, help="0 disables resizing")
    ap.add_argument("--timeout", type=float)
    ap.add_argument("--retries", type=int)
    ap.add_argument("--limit", type=int, help="only the first N rows (quick connectivity test)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    icfg = cfg["images"]
    if args.csv:
        csv_path, name = Path(args.csv), Path(args.csv).stem
    elif args.split:
        csv_path, name = get_path(cfg, f"{args.split}_csv"), args.split
    else:
        ap.error("pass --split or --csv")
    out_dir = Path(args.out) if args.out else get_path(cfg, "images_dir") / name
    id_col = args.id_col or cfg["columns"]["id"]
    url_col = args.url_col or cfg["columns"]["image_url"]

    df = pd.read_csv(csv_path, dtype={id_col: str})
    if args.limit:
        df = df.head(args.limit)
    max_side = args.max_side if args.max_side is not None else icfg["max_side"]
    download(
        df, url_col, id_col if id_col in df.columns else None, out_dir, name,
        workers=args.workers or icfg["workers"], max_side=max_side or None,
        timeout=args.timeout or icfg["timeout"],
        retries=args.retries if args.retries is not None else icfg["retries"],
    )


if __name__ == "__main__":
    main()
