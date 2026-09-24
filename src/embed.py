"""Text and image embeddings, cached by row order.

    python -m src.embed --config configs/base.yaml --what text image

Outputs (row order == CSV row order):
    {cache_dir}/txt_{model}_{split}.npy
    {cache_dir}/img_{model}_{pretrained}_{split}.npy
    {cache_dir}/has_image_{split}.npy
Existing caches are never recomputed unless --force. Throughput is logged along with a
projection for the full test set so heavy models can be ruled out early.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.download_images import image_path
from src.utils import (add_config_arg, cache_file, get_device, get_logger, get_path, image_cache_name,
                       load_config, read_split, text_cache_name)

log = get_logger("embed")


def _report(kind: str, n: int, seconds: float, n_test: int) -> None:
    rate = n / max(seconds, 1e-9)
    log.info("%s: %d rows in %.1fs = %.1f rows/s -> full test (%d rows) ~ %.1f min",
             kind, n, seconds, rate, n_test, n_test / rate / 60)


# ---------------------------------------------------------------- text

def embed_text(texts: list[str], model_name: str, batch_size: int, max_length: int, device: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = max_length
    if device == "cuda":
        model.half()
    return model.encode(
        texts, batch_size=batch_size, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32)


# ---------------------------------------------------------------- images

class _ImageDataset(Dataset):
    def __init__(self, paths, preprocess):
        self.paths = paths
        self.preprocess = preprocess
        self.blank = preprocess(Image.new("RGB", (224, 224)))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            with Image.open(self.paths[i]) as im:
                return self.preprocess(im.convert("RGB")), 1
        except Exception:  # missing / corrupt file -> placeholder, flagged via has_image=0
            return self.blank, 0


def embed_images(paths, model_name: str, pretrained: str, batch_size: int, num_workers: int, device: str):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    model.eval()
    loader = DataLoader(_ImageDataset(paths, preprocess), batch_size=batch_size,
                        num_workers=num_workers, pin_memory=device == "cuda")
    feats, flags = [], []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
        for x, ok in loader:
            f = model.encode_image(x.to(device, non_blocking=True)).float()
            f = torch.nn.functional.normalize(f, dim=-1)
            f[(ok == 0).to(f.device)] = 0.0
            feats.append(f.cpu().numpy())
            flags.append(ok.numpy())
    return np.concatenate(feats).astype(np.float32), np.concatenate(flags).astype(np.float32)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    add_config_arg(ap)
    ap.add_argument("--what", nargs="+", default=["text", "image"], choices=["text", "image"])
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ecfg, cols = cfg["embed"], cfg["columns"]
    device = get_device()
    log.info("device=%s", device)
    n_test = len(read_split(cfg, "test"))

    for split in args.splits:
        df = read_split(cfg, split)
        cache_file(cfg, "x", split).parent.mkdir(parents=True, exist_ok=True)

        if "text" in args.what:
            out = cache_file(cfg, text_cache_name(cfg), split)
            if out.exists() and not args.force:
                log.info("cached: %s", out)
            else:
                texts = df[cols["text"]].fillna("").astype(str).tolist()
                t0 = time.perf_counter()
                emb = embed_text(texts, ecfg["text_model"], ecfg["text_batch_size"], ecfg["text_max_length"], device)
                _report(f"text/{split}", len(texts), time.perf_counter() - t0, n_test)
                np.save(out, emb)
                log.info("wrote %s %s", out, emb.shape)

        if "image" in args.what:
            out = cache_file(cfg, image_cache_name(cfg), split)
            flag_out = cache_file(cfg, "has_image", split)
            if out.exists() and flag_out.exists() and not args.force:
                log.info("cached: %s", out)
            else:
                img_dir = get_path(cfg, "images_dir") / split
                paths = [image_path(img_dir, i, u) for i, u in zip(df[cols["id"]], df[cols["image_url"]])]
                t0 = time.perf_counter()
                emb, has = embed_images(paths, ecfg["image_model"], ecfg["image_pretrained"],
                                        ecfg["image_batch_size"], ecfg["num_workers"], device)
                _report(f"image/{split}", len(paths), time.perf_counter() - t0, n_test)
                np.save(out, emb)
                np.save(flag_out, has)
                log.info("wrote %s %s; has_image=%.1f%%", out, emb.shape, 100 * has.mean())


if __name__ == "__main__":
    main()
