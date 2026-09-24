"""MLP head over cached embeddings (text + image + flags), multi-seed, shared folds.

    python -m src.train_fusion --config configs/base.yaml
    python -m src.train_fusion --config configs/base.yaml --features txt img has_image hand --loss smape_surrogate --seeds 0 1 2 3 4

Everything fits in GPU memory, so batches are sliced from device tensors (no DataLoader).
Test predictions are averaged over folds and seeds. Outputs match train_gbm.py.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from src.cv import make_folds, run_cv
from src.metrics import get_metric
from src.utils import (add_config_arg, decode_pred, encode_target, get_device, get_logger, load_config,
                       load_features, log_experiment, make_run_id, read_split, save_preds, score_preds,
                       seed_everything, timer)

log = get_logger("train_fusion")


class MLP(nn.Module):
    def __init__(self, d_in: int, hidden: list[int], dropout: float, d_out: int):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def make_loss(name: str, cfg: dict, is_cls: bool):
    if is_cls:
        return nn.CrossEntropyLoss()
    if name == "l1_log":
        return nn.L1Loss()
    if name == "huber":
        return nn.HuberLoss(delta=cfg["fusion"]["huber_delta"])
    if name == "smape_surrogate":
        log_target = cfg.get("target_transform") == "log1p"

        def smape_loss(pred, target):
            p = torch.expm1(pred.clamp(max=20)) if log_target else pred
            a = torch.expm1(target) if log_target else target
            p = p.clamp(min=cfg.get("min_pred", 0.0))
            return (2 * (p - a).abs() / (p.abs() + a.abs() + 1e-6)).mean()

        return smape_loss
    raise ValueError(f"unknown loss {name!r}")


def make_fit_predict(cfg: dict, loss_name: str, seed: int, classes, device: str):
    fcfg = cfg["fusion"]
    is_cls = classes is not None
    d_out = len(classes) if is_cls else 1

    def fit_predict(X_tr, y_tr, X_va, y_va, X_te, fold):
        seed_everything(seed * 100 + fold)
        clean = lambda a: np.nan_to_num(np.asarray(a, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        scaler = StandardScaler().fit(clean(X_tr))
        to_t = lambda a: torch.tensor(scaler.transform(clean(a)), dtype=torch.float32, device=device)
        xtr, xva, xte = to_t(X_tr), to_t(X_va), to_t(X_te)
        ydt = torch.long if is_cls else torch.float32
        ytr = torch.tensor(y_tr, dtype=ydt, device=device)

        model = MLP(xtr.shape[1], fcfg["hidden"], fcfg["dropout"], d_out).to(device)
        if not is_cls:
            # start at the target mean so early epochs aren't spent learning the offset
            nn.init.constant_(model.net[-1].bias, float(np.mean(y_tr)))
        opt = torch.optim.AdamW(model.parameters(), lr=fcfg["lr"], weight_decay=fcfg["weight_decay"])
        steps_per_epoch = int(np.ceil(len(xtr) / fcfg["batch_size"]))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=fcfg["lr"], epochs=fcfg["epochs"],
                                                    steps_per_epoch=steps_per_epoch)
        loss_fn = make_loss(loss_name, cfg, is_cls)

        def predict(x):
            model.eval()
            with torch.no_grad():
                out = torch.cat([model(x[i:i + 8192]) for i in range(0, len(x), 8192)])
            out = out.softmax(-1) if is_cls else out.squeeze(-1)
            return out.cpu().numpy()

        # early stopping on the competition metric in submission space
        y_va_true = y_va if is_cls else decode_pred(cfg, y_va)
        best, best_state, bad = None, None, 0
        _, greater = get_metric(cfg["metric"])
        for epoch in range(fcfg["epochs"]):
            model.train()
            perm = torch.randperm(len(xtr), device=device)
            for i in range(0, len(perm), fcfg["batch_size"]):
                idx = perm[i:i + fcfg["batch_size"]]
                out = model(xtr[idx])
                loss = loss_fn(out if is_cls else out.squeeze(-1), ytr[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
            va = predict(xva)
            s = score_preds(cfg, np.asarray(classes)[y_va_true] if is_cls else y_va_true,
                            va if is_cls else decode_pred(cfg, va), classes)
            if best is None or (s > best if greater else s < best):
                best, bad = s, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= fcfg["patience"]:
                    break
        model.load_state_dict(best_state)
        log.info("seed %d fold %d: best %s=%.5f (stopped at epoch %d)", seed, fold, cfg["metric"], best, epoch + 1)
        return predict(xva), predict(xte)

    return fit_predict


def main():
    ap = argparse.ArgumentParser()
    add_config_arg(ap)
    ap.add_argument("--features", nargs="+", help="feature blocks (default: fusion.features)")
    ap.add_argument("--loss", choices=["l1_log", "huber", "smape_surrogate"])
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--name", default="")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    cfg = load_config(args.config)
    fcfg = cfg["fusion"]
    features = args.features or fcfg["features"]
    loss_name = args.loss or fcfg["loss"]
    seeds = args.seeds or fcfg["seeds"]
    device = get_device()
    run_id = make_run_id(f"fusion_{loss_name}" + (f"_{args.name}" if args.name else ""))

    train = read_split(cfg, "train")
    target_col = cfg["columns"]["target"]
    folds = make_folds(train, cfg)
    y_fit, classes = encode_target(cfg, train[target_col])
    X_tr, _ = load_features(cfg, features, "train")
    X_te, _ = load_features(cfg, features, "test")
    log.info("run %s on %s: X_train=%s, seeds=%s", run_id, device, X_tr.shape, seeds)

    oofs, tests, seed_scores = [], [], []
    with timer(run_id, log):
        for seed in seeds:
            oof, test = run_cv(make_fit_predict(cfg, loss_name, seed, classes, device), X_tr, y_fit, X_te, folds)
            if cfg["task"] == "regression":
                oof, test = decode_pred(cfg, oof), decode_pred(cfg, test)
            s = score_preds(cfg, train[target_col].to_numpy(), oof, classes)
            log.info("seed %d CV %s = %.5f", seed, cfg["metric"], s)
            oofs.append(oof)
            tests.append(test)
            seed_scores.append(s)

    oof, test = np.mean(oofs, axis=0), np.mean(tests, axis=0)
    cv_score = score_preds(cfg, train[target_col].to_numpy(), oof, classes)
    log.info("CV %s (seed avg) = %.5f; per seed %s", cfg["metric"], cv_score, [round(s, 4) for s in seed_scores])

    save_preds(cfg, run_id, oof, test, {
        "model": "fusion_mlp", "features": features, "loss": loss_name, "seeds": seeds,
        "seed_scores": seed_scores, "cv_score": cv_score, "metric": cfg["metric"], "classes": classes,
    })
    notes = f"loss={loss_name} seeds={seeds} per_seed={[round(s, 4) for s in seed_scores]} {args.notes}".strip()
    log_experiment(cfg, run_id, "fusion_mlp", features, cv_score, notes=notes)
    print(run_id)


if __name__ == "__main__":
    main()
