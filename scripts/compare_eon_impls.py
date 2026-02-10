#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extremonet.model import ExtremeLearning, ExtremONet
from extremonet.train import train_ridge


def _load_external_modules(base_dir: Path):
    eon_path = base_dir / "EON.py"
    train_path = base_dir / "EON_train.py"
    if not eon_path.exists() or not train_path.exists():
        raise FileNotFoundError(f"external implementation not found under: {base_dir}")

    spec_eon = importlib.util.spec_from_file_location("ext_eon", str(eon_path))
    ext_eon = importlib.util.module_from_spec(spec_eon)
    assert spec_eon.loader is not None
    spec_eon.loader.exec_module(ext_eon)

    spec_train = importlib.util.spec_from_file_location("ext_train", str(train_path))
    ext_train = importlib.util.module_from_spec(spec_train)
    assert spec_train.loader is not None
    spec_train.loader.exec_module(ext_train)
    return ext_eon, ext_train


def _mean_rel_l2(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    num = np.linalg.norm(y_true - y_pred, axis=1)
    den = np.linalg.norm(y_true, axis=1) + 1e-12
    rel = num / den
    return float(np.mean(rel)), float(np.median(rel))


def _make_synthetic_data(seed: int, n_total: int, n_train: int, sensors: int, outdim: int, width: int):
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 1, size=(n_total, 1)).astype(np.float32)
    u = rng.normal(size=(n_total, sensors)).astype(np.float32)

    # Teacher mapping with multiplicative interactions (EON-favorable structure).
    wb = rng.normal(scale=0.5, size=(sensors, width)).astype(np.float32)
    wt = rng.normal(scale=1.0, size=(1, width)).astype(np.float32)
    a = rng.normal(scale=0.2, size=(width, outdim)).astype(np.float32)
    b = rng.normal(scale=0.1, size=(1, outdim)).astype(np.float32)

    bf = np.tanh(0.1 * (u @ wb))
    tf = np.tanh(10.0 * (t @ wt))
    y = (bf * tf) @ a + b + 0.01 * rng.normal(size=(n_total, outdim)).astype(np.float32)

    idx = rng.permutation(n_total)
    tr_idx, te_idx = idx[:n_train], idx[n_train:]
    return t[tr_idx], u[tr_idx], y[tr_idx], t[te_idx], u[te_idx], y[te_idx]


def _run_once(
    seed: int,
    ext_eon,
    ext_train,
    n_total: int,
    n_train: int,
    sensors: int,
    outdim: int,
    width: int,
    ridge_iters: int,
    val_frac: float,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    ttr, utr, ytr, tte, ute, yte = _make_synthetic_data(seed, n_total, n_train, sensors, outdim, width)
    device = torch.device("cpu")

    trunk_ext = ext_eon.ExtremeLearning(
        1, width, c=1, s=10, acfunc=nn.Tanh(), norm=[np.mean(ttr), np.std(ttr)], device=device
    ).to(device)
    branch_ext = ext_eon.ExtremeLearning(
        sensors, width, c=1, s=0.1, acfunc=nn.Tanh(), norm=[np.mean(utr), np.std(utr)], device=device
    ).to(device)
    model_ext = ext_eon.ExtremONet(outdim, width, trunk_ext, branch_ext, device=device).to(device)

    ts = time.time()
    ext_train.train_EON(
        model_ext,
        ttr,
        utr,
        ytr,
        bounds=[-10, 10],
        iters=ridge_iters,
        tp=val_frac,
        verbose=False,
        rnn=False,
    )
    ext_train_s = time.time() - ts
    ts = time.time()
    yhat_ext = model_ext.predict(tte, ute)
    ext_pred_s = time.time() - ts
    ext_mean, ext_med = _mean_rel_l2(yte, yhat_ext)

    trunk = ExtremeLearning(
        1,
        width,
        c=1,
        s=10.0,
        norm_mean=float(np.mean(ttr)),
        norm_std=float(np.std(ttr) + 1e-6),
        seed=seed,
    )
    branch = ExtremeLearning(
        sensors,
        width,
        c=1,
        s=0.1,
        norm_mean=float(np.mean(utr)),
        norm_std=float(np.std(utr) + 1e-6),
        seed=seed + 1,
    )
    model = ExtremONet(trunk=trunk, branch=branch, outdim=outdim).to(device)
    xtr = torch.tensor(ttr, dtype=torch.float32).unsqueeze(1)
    xte = torch.tensor(tte, dtype=torch.float32).unsqueeze(1)
    utr_t = torch.tensor(utr, dtype=torch.float32)
    ute_t = torch.tensor(ute, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32).unsqueeze(1)

    ts = time.time()
    train_ridge(
        model,
        xtr,
        utr_t,
        ytr_t,
        bounds=(-10, 10),
        iters=ridge_iters,
        val_frac=val_frac,
        seed=seed,
        batch_size=0,
    )
    cur_train_s = time.time() - ts
    ts = time.time()
    with torch.no_grad():
        yhat_cur = model.predict_tensor(xte, ute_t).squeeze(1).cpu().numpy()
    cur_pred_s = time.time() - ts
    cur_mean, cur_med = _mean_rel_l2(yte, yhat_cur)

    return {
        "seed": seed,
        "ext_mean": ext_mean,
        "ext_med": ext_med,
        "ext_train_s": ext_train_s,
        "ext_pred_s": ext_pred_s,
        "cur_mean": cur_mean,
        "cur_med": cur_med,
        "cur_train_s": cur_train_s,
        "cur_pred_s": cur_pred_s,
    }


def main():
    p = argparse.ArgumentParser(description="Compare current EON implementation against external reference.")
    p.add_argument("--external-dir", default="external/ExtremONet-MLDE/CODE")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--n-total", type=int, default=1200)
    p.add_argument("--n-train", type=int, default=900)
    p.add_argument("--sensors", type=int, default=64)
    p.add_argument("--outdim", type=int, default=32)
    p.add_argument("--width", type=int, default=384)
    p.add_argument("--ridge-iters", type=int, default=80)
    p.add_argument("--val-frac", type=float, default=0.2)
    args = p.parse_args()

    ext_eon, ext_train = _load_external_modules(Path(args.external_dir))
    rows = []
    for seed in args.seeds:
        rows.append(
            _run_once(
                seed,
                ext_eon,
                ext_train,
                n_total=args.n_total,
                n_train=args.n_train,
                sensors=args.sensors,
                outdim=args.outdim,
                width=args.width,
                ridge_iters=args.ridge_iters,
                val_frac=args.val_frac,
            )
        )

    print("seed ext_mean cur_mean ext_med cur_med ext_train_s cur_train_s ext_pred_s cur_pred_s")
    for r in rows:
        print(
            f"{r['seed']} {r['ext_mean']:.6e} {r['cur_mean']:.6e} {r['ext_med']:.6e} {r['cur_med']:.6e} "
            f"{r['ext_train_s']:.3f} {r['cur_train_s']:.3f} {r['ext_pred_s']:.3f} {r['cur_pred_s']:.3f}"
        )

    arr = {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0] if k != "seed"}
    print("--- aggregate ---")
    print(
        "external: relL2_mean(avg)=%.6e, relL2_median(avg)=%.6e, train_s(avg)=%.3f, pred_s(avg)=%.3f"
        % (arr["ext_mean"].mean(), arr["ext_med"].mean(), arr["ext_train_s"].mean(), arr["ext_pred_s"].mean())
    )
    print(
        "current : relL2_mean(avg)=%.6e, relL2_median(avg)=%.6e, train_s(avg)=%.3f, pred_s(avg)=%.3f"
        % (arr["cur_mean"].mean(), arr["cur_med"].mean(), arr["cur_train_s"].mean(), arr["cur_pred_s"].mean())
    )
    print(
        "delta(current-external): relL2_mean=%.6e, train_s=%.3f, pred_s=%.3f"
        % (
            (arr["cur_mean"] - arr["ext_mean"]).mean(),
            (arr["cur_train_s"] - arr["ext_train_s"]).mean(),
            (arr["cur_pred_s"] - arr["ext_pred_s"]).mean(),
        )
    )


if __name__ == "__main__":
    main()
