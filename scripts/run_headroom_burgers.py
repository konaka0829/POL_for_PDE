#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import scipy.io
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pol.model123_1d.readouts import FourierDiagonalReadout, evaluate_readout, target_variance_l2h


def _zeta_grid(raw: str) -> list[float]:
    return [float(item) for item in raw.split(",") if item.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Estimate Burgers linear headroom with Fourier diagonal readout.")
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--out-dir", default="outputs/headroom_burgers")
    parser.add_argument("--ntrain", type=int, default=800)
    parser.add_argument("--nval", type=int, default=200)
    parser.add_argument("--ntest", type=int, default=200)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--zeta-grid", default="1e-12,1e-10,1e-8,1e-6,1e-4,1e-2,1e0")
    args = parser.parse_args(argv)

    data = scipy.io.loadmat(args.data_file)
    x = torch.as_tensor(data["a"], dtype=torch.float64)
    y = torch.as_tensor(data["u"], dtype=torch.float64)
    gen = torch.Generator().manual_seed(args.split_seed)
    idx = torch.randperm(x.shape[0], generator=gen)
    train_idx = idx[: args.ntrain]
    val_idx = idx[args.ntrain : args.ntrain + args.nval]
    test_idx = idx[args.ntrain + args.nval : args.ntrain + args.nval + args.ntest]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    best = None
    for zeta in _zeta_grid(args.zeta_grid):
        readout = FourierDiagonalReadout(zeta=zeta).fit(x[train_idx], y[train_idx])
        pred_val = readout.predict(x[val_idx])
        pred_test = readout.predict(x[test_idx])
        val = evaluate_readout(pred_val, y[val_idx])
        test = evaluate_readout(pred_test, y[test_idx])
        row = {
            "zeta": zeta,
            "val_absL2h": val["absL2h"],
            "test_absL2h": test["absL2h"],
            "test_relL2_mean": test["relL2_mean"],
            "test_relL2_agg": test["relL2_agg"],
        }
        rows.append(row)
        if best is None or row["val_absL2h"] < best["val_absL2h"]:
            best = row
    sigma2 = target_variance_l2h(y[test_idx])
    dlin2 = float(best["test_absL2h"]) ** 2
    summary = {
        "selected_zeta": best["zeta"],
        "selection_metric": "val_absL2h",
        "test_Dlin2": dlin2,
        "sigma_T2": sigma2,
        "headroom_H": dlin2 / sigma2 if sigma2 > 0 else None,
        "linear_explained_variance": 1.0 - dlin2 / sigma2 if sigma2 > 0 else None,
    }
    with (out_dir / "headroom_zeta_path.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "headroom_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
