#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import scipy.io
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pol.model123_1d import DatasetConfig, build_dataset, save_dataset_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate 1D periodic Burgers data for Model123")
    parser.add_argument("--out-file", required=True)
    parser.add_argument("--format", choices=("mat", "pt"), default="")
    parser.add_argument("--num-samples", type=int, default=1200)
    parser.add_argument("--total-samples", type=int, default=0)
    parser.add_argument("--ntrain", type=int, default=0)
    parser.add_argument("--ntest", type=int, default=0)
    parser.add_argument("--grid-size", "--nx", dest="nx", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nu", type=float, default=0.05)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument("--fine-dt", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    return parser


def _resolve_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    total = args.total_samples if args.total_samples > 0 else args.num_samples
    if args.ntrain <= 0 and args.ntest <= 0:
        ntrain = total
        ntest = 0
    else:
        ntrain = args.ntrain
        ntest = args.ntest
    if ntest == 0:
        cfg_total = total + 1
        cfg_ntrain = total
        cfg_ntest = 1
    else:
        cfg_total = total
        cfg_ntrain = ntrain
        cfg_ntest = ntest

    cfg = DatasetConfig(
        total_samples=cfg_total,
        ntrain=cfg_ntrain,
        ntest=cfg_ntest,
        seed=args.seed,
        nx=args.nx,
        target_nu=args.nu,
        T=args.T,
        dt=args.dt,
        fine_dt=args.fine_dt,
        batch_size=args.batch_size,
        dtype=args.dtype,
    )
    bundle = build_dataset(cfg, device=_resolve_device(args.device))
    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    output_format = args.format or out_file.suffix.lower().lstrip(".") or "mat"

    if ntest == 0:
        a = bundle.u0_train[:total]
        u = bundle.y_train[:total]
    else:
        a = torch.cat([bundle.u0_train, bundle.u0_test], dim=0)
        u = torch.cat([bundle.y_train, bundle.y_test], dim=0)

    if output_format == "pt":
        save_dataset_bundle(bundle, out_file)
    else:
        scipy.io.savemat(
            out_file,
            {
                "a": a.detach().cpu().numpy(),
                "u": u.detach().cpu().numpy(),
                "T": float(args.T),
                "dt": float(args.dt),
                "nu": float(args.nu),
                "nx": int(args.nx),
                "num_samples": int(a.shape[0]),
            },
        )
    print(f"saved dataset: {out_file}")
    print(f"a: {tuple(a.shape)} u: {tuple(u.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
