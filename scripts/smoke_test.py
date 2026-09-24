"""Pipeline smoke test on synthetic 2025-style data. No external data, runs in minutes.

    python scripts/smoke_test.py            # from the repo root
    python scripts/smoke_test.py --clean    # wipe previous smoke outputs first

Generates catalog-like text + prices, renders images and serves them from a local HTTP
server (with some 404s and blank URLs to exercise the failure paths), then runs every stage:
download -> folds -> text features -> embeddings -> GBM (LightGBM + CatBoost) -> fusion ->
ensemble -> submission -> validator (including broken copies that must be rejected).
Model weights (MiniLM, CLIP) are fetched from the Hugging Face hub on first use.
"""
from __future__ import annotations

import argparse
import functools
import http.server
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CFG = ["--config", "configs/base.yaml", "configs/smoke.yaml"]
RAW = ROOT / "data" / "raw" / "smoke"

BRANDS = ["kraft", "nestle", "colgate", "dove", "pepsi", "heinz", "lipton", "oreo", "tide", "gillette"]
PRODUCTS = ["pasta sauce", "green tea", "toothpaste", "body wash", "cola", "ketchup", "cookies", "detergent"]
UNITS = [("oz", "Ounce", 28.35), ("fl oz", "Fl Oz", 29.57), ("g", "Grams", 1.0), ("lb", "Pound", 453.6),
         ("ml", "Millilitre", 1.0), ("count", "Count", 1.0)]


def make_data(n_train: int, n_test: int, port: int, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    img_src = RAW / "img_src"
    img_src.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        b = rng.integers(len(BRANDS))
        p = rng.integers(len(PRODUCTS))
        u = rng.integers(len(UNITS))
        qty = float(np.round(rng.uniform(1, 40), 1))
        pack = int(rng.choice([1, 1, 1, 2, 4, 6, 12]))
        unit_short, unit_long, factor = UNITS[u]
        price = (1.5 + b * 0.8) * (qty * factor) ** 0.35 * pack ** 0.8 * rng.lognormal(0, 0.25)
        pack_txt = f" (Pack of {pack})" if pack > 1 else ""
        text = (f"Item Name: {BRANDS[b].title()} {PRODUCTS[p].title()} {qty} {unit_short}{pack_txt}\n"
                f"Bullet Point 1: Premium {PRODUCTS[p]} from {BRANDS[b].title()}\n"
                + (f"Product Description: Great {PRODUCTS[p]} for everyday use.\n" if rng.random() < 0.6 else "")
                + f"Value: {qty * pack}\nUnit: {unit_long}")
        sid = str(100000 + i)
        # image colour carries some price signal so image embeddings aren't pure noise
        shade = int(np.clip(40 + 20 * np.log(price), 0, 255))
        Image.new("RGB", (320, 240), (shade, 255 - shade, (b * 25) % 255)).save(img_src / f"{sid}.jpg")
        r = rng.random()
        if r < 0.05:
            url = f"http://127.0.0.1:{port}/missing_{sid}.jpg"   # 404
        elif r < 0.08:
            url = ""                                             # no URL
        else:
            url = f"http://127.0.0.1:{port}/{sid}.jpg"
        rows.append({"sample_id": sid, "catalog_content": text, "image_link": url, "price": round(price, 2)})

    df = pd.DataFrame(rows)
    train, test = df.iloc[:n_train], df.iloc[n_train:]
    train.to_csv(RAW / "train.csv", index=False)
    test.drop(columns=["price"]).to_csv(RAW / "test.csv", index=False)
    test[["sample_id", "price"]].to_csv(RAW / "test_labels.csv", index=False)
    test[["sample_id"]].assign(price=1.0).to_csv(RAW / "sample_test_out.csv", index=False)
    print(f"synthetic data: train={len(train)} test={len(test)} -> {RAW}")


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def serve(directory: Path) -> http.server.ThreadingHTTPServer:
    handler = functools.partial(_QuietHandler, directory=str(directory))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def run(*args: str, expect_fail: bool = False) -> str:
    cmd = [sys.executable, *args]
    print("\n$", " ".join(args), flush=True)
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if res.stderr.strip():
        print("\n".join(res.stderr.strip().splitlines()[-12:]))
    if res.stdout.strip():
        print("\n".join(res.stdout.strip().splitlines()[-6:]))
    if expect_fail:
        if res.returncode == 0:
            raise SystemExit(f"FAIL: expected non-zero exit from {args}")
        return ""
    if res.returncode != 0:
        raise SystemExit(f"FAIL ({res.returncode}): {' '.join(args)}")
    return res.stdout.strip().splitlines()[-1] if res.stdout.strip() else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=150)
    args = ap.parse_args()

    if args.clean:
        for p in [RAW, ROOT / "data/images/smoke", ROOT / "data/cache/smoke", ROOT / "submissions/smoke",
                  ROOT / "experiments_smoke.csv"]:
            shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)

    RAW.mkdir(parents=True, exist_ok=True)
    srv = serve(RAW / "img_src")
    port = srv.server_address[1]
    # URLs embed the port, so regenerate the data for every run and clear stale images/caches
    for p in [ROOT / "data/images/smoke", ROOT / "data/cache/smoke"]:
        shutil.rmtree(p, ignore_errors=True)
    make_data(args.n_train, args.n_test, port)

    try:
        run("-m", "src.metrics")
        run("-m", "src.download_images", *CFG, "--split", "train")
        run("-m", "src.download_images", *CFG, "--split", "test")
        run("-m", "src.download_images", *CFG, "--split", "train")  # second pass must skip everything
        run("-m", "src.cv", *CFG)
        run("-m", "src.features_text", *CFG)
        run("-m", "src.features_text", *CFG)  # cached
        run("-m", "src.embed", *CFG)
        run("-m", "src.embed", *CFG)  # cached
        gbm_base = run("-m", "src.train_gbm", *CFG, "--name", "base")
        gbm_all = run("-m", "src.train_gbm", *CFG, "--features", "hand", "svd", "txt", "img", "has_image",
                      "--name", "all")
        cat = run("-m", "src.train_gbm", *CFG, "--model", "catboost", "--name", "cat")
        fusion = run("-m", "src.train_fusion", *CFG)
        fusion_smape = run("-m", "src.train_fusion", *CFG, "--loss", "smape_surrogate", "--features",
                           "txt", "img", "has_image", "hand", "--seeds", "0", "--name", "smape")
        ens = run("-m", "src.ensemble", *CFG, "--runs", gbm_base, gbm_all, cat, fusion, fusion_smape, "--write")
        run("-m", "src.predict", *CFG, "--run", gbm_base)

        sub = ROOT / "submissions" / "smoke" / f"{ens}.csv"
        run("-m", "src.validate_submission", *CFG, "--sub", str(sub))
        s = pd.read_csv(sub, dtype={"sample_id": str})
        broken = ROOT / "data" / "cache" / "smoke" / "broken"
        broken.mkdir(parents=True, exist_ok=True)
        s.iloc[:-1].to_csv(broken / "rows.csv", index=False)
        s.assign(price=s.price.where(s.index != 0)).to_csv(broken / "nan.csv", index=False)
        s.assign(price=-s.price).to_csv(broken / "neg.csv", index=False)
        s.iloc[::-1].to_csv(broken / "order.csv", index=False)
        s[["price", "sample_id"]].to_csv(broken / "cols.csv", index=False)
        pd.concat([s.iloc[:-1], s.iloc[:1]]).to_csv(broken / "dup.csv", index=False)
        for f in sorted(broken.glob("*.csv")):
            run("-m", "src.validate_submission", *CFG, "--sub", str(f), expect_fail=True)
        print("validator rejected all broken submissions")

        labels = pd.read_csv(RAW / "test_labels.csv", dtype={"sample_id": str})
        from src.metrics import smape  # noqa: E402  (repo root is cwd)
        m = labels.merge(s, on="sample_id", suffixes=("_true", "_pred"))
        print(f"\nheld-out SMAPE of ensemble: {smape(m.price_true, m.price_pred):.3f}")
        print((ROOT / "experiments_smoke.csv").read_text(encoding="utf-8"))
        print("SMOKE TEST OK")
    finally:
        srv.shutdown()


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
