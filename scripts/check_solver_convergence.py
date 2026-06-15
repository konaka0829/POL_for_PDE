#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pol.model123_1d.metrics import dataset_abs_l2h_rmse
from pol.spectral_etdrk4_1d import dealias_mask_2_3, simulate_burgers_etdrk4


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Small ETDRK4 Burgers convergence check.")
    parser.add_argument("--out-dir", default="outputs/solver_checks/etdrk4_smoke")
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--T", type=float, default=0.05)
    parser.add_argument("--nu", type=float, default=0.01)
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x = torch.linspace(0.0, 1.0, args.nx + 1, dtype=torch.float64)[:-1]
    u0 = torch.sin(2.0 * torch.pi * x).unsqueeze(0)
    ref = simulate_burgers_etdrk4(u0, nu=args.nu, T=args.T, dt=args.T / 200.0)
    rows = []
    for dt in [args.T / 25.0, args.T / 50.0, args.T / 100.0]:
        pred = simulate_burgers_etdrk4(u0, nu=args.nu, T=args.T, dt=dt)
        rows.append({"nx": args.nx, "dt": dt, "error_to_reference": dataset_abs_l2h_rmse(pred, ref)})
    with (out_dir / "convergence_table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "dealias_mask_shape": list(dealias_mask_2_3(args.nx).shape),
        "finite": bool(torch.isfinite(ref).all()),
        "rows": rows,
    }
    (out_dir / "convergence_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
