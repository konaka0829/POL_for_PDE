#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--config", default="")
    parser.add_argument("--out-file", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--format", choices=("mat", "pt"), default="")
    parser.add_argument("--num-samples", type=int, default=1200)
    parser.add_argument("--total-samples", type=int, default=0)
    parser.add_argument("--ntrain", type=int, default=0)
    parser.add_argument("--nval", type=int, default=0)
    parser.add_argument("--ntest", type=int, default=0)
    parser.add_argument("--grid-size", "--nx", dest="nx", type=int, default=256)
    parser.add_argument("--domain-length", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-seed", type=int, default=None)
    parser.add_argument("--nu", type=float, default=0.05)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument("--fine-dt", type=float, default=1e-4)
    parser.add_argument("--solver", choices=("split_step", "semi_implicit", "etdrk4", "fourier_pseudospectral_etdrk4"), default="split_step")
    parser.add_argument("--dealias", action="store_true")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--ic-type", choices=("grf", "fourier"), default="fourier")
    parser.add_argument("--grf-gamma", type=float, default=2.0)
    parser.add_argument("--grf-tau", type=float, default=5.0)
    parser.add_argument("--grf-sigma", type=float, default=25.0)
    parser.add_argument("--grf-mean", type=float, default=0.0)
    parser.add_argument("--fourier-num-modes", type=int, default=8)
    parser.add_argument("--fourier-amplitude", type=float, default=0.5)
    return parser


def _set_if_default(parser: argparse.ArgumentParser, args: argparse.Namespace, name: str, value) -> None:
    if value is None or not hasattr(args, name):
        return
    if getattr(args, name) == parser.get_default(name):
        setattr(args, name, value)
        args._config_applied[name] = value


def _apply_config_defaults(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.config:
        return
    args._config_applied = {}
    with Path(args.config).open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    domain = cfg.get("domain", {})
    target = cfg.get("target", {})
    data = cfg.get("data", {})
    _set_if_default(parser, args, "nx", domain.get("nx"))
    _set_if_default(parser, args, "domain_length", domain.get("length"))
    _set_if_default(parser, args, "nu", target.get("nu"))
    _set_if_default(parser, args, "T", target.get("T"))
    _set_if_default(parser, args, "dt", target.get("dt"))
    _set_if_default(parser, args, "solver", target.get("solver"))
    if target.get("dealias") is not None and args.dealias == parser.get_default("dealias"):
        args.dealias = bool(target.get("dealias"))
    _set_if_default(parser, args, "total_samples", data.get("total_samples"))
    _set_if_default(parser, args, "ntrain", data.get("ntrain"))
    _set_if_default(parser, args, "nval", data.get("nval"))
    _set_if_default(parser, args, "ntest", data.get("ntest"))
    _set_if_default(parser, args, "data_seed", data.get("data_seed"))
    _set_if_default(parser, args, "ic_type", data.get("ic_type"))
    _set_if_default(parser, args, "grf_gamma", data.get("grf_gamma"))
    _set_if_default(parser, args, "grf_tau", data.get("grf_tau"))
    _set_if_default(parser, args, "grf_sigma", data.get("grf_sigma"))
    _set_if_default(parser, args, "grf_mean", data.get("grf_mean"))
    _set_if_default(parser, args, "dtype", data.get("sim_dtype", data.get("data_dtype")))


def _resolve_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _apply_config_defaults(parser, args)
    if not hasattr(args, "_config_applied"):
        args._config_applied = {}
    total = args.total_samples if args.total_samples > 0 else args.num_samples
    if total <= 0:
        raise ValueError("--num-samples/--total-samples must be positive")
    if args.ntrain < 0 or args.nval < 0 or args.ntest < 0:
        raise ValueError("--ntrain, --nval, and --ntest must be nonnegative")
    if args.ntrain <= 0 and args.nval <= 0 and args.ntest <= 0:
        ntrain = total
        nval = 0
        ntest = 0
    elif args.ntrain <= 0:
        nval = args.nval
        ntest = args.ntest
        ntrain = total - nval - ntest
    elif args.ntest <= 0:
        ntrain = args.ntrain
        nval = args.nval
        ntest = total - ntrain - nval
    else:
        ntrain = args.ntrain
        nval = args.nval
        ntest = args.ntest
    if ntrain <= 0:
        raise ValueError("resolved ntrain must be positive")
    if ntest < 0:
        raise ValueError("resolved ntest must be nonnegative")
    if ntrain + nval + ntest != total:
        raise ValueError("resolved ntrain + nval + ntest must equal total samples")

    cfg = DatasetConfig(
        total_samples=total,
        ntrain=ntrain,
        nval=nval,
        ntest=ntest,
        seed=args.seed,
        data_seed=args.data_seed,
        nx=args.nx,
        domain_length=args.domain_length,
        target_nu=args.nu,
        T=args.T,
        dt=args.dt,
        fine_dt=args.fine_dt,
        solver=args.solver,
        dealias=bool(args.dealias),
        batch_size=args.batch_size,
        dtype=args.dtype,
        ic_type=args.ic_type,
        grf_gamma=args.grf_gamma,
        grf_tau=args.grf_tau,
        grf_sigma=args.grf_sigma,
        grf_mean=args.grf_mean,
        fourier_num_modes=args.fourier_num_modes,
        fourier_amplitude=args.fourier_amplitude,
    )
    bundle = build_dataset(cfg, device=_resolve_device(args.device))
    if not args.out_file:
        if not args.output_dir:
            raise ValueError("--out-file or --output-dir is required")
        suffix = "pt" if args.format == "pt" else "mat"
        args.out_file = str(Path(args.output_dir) / f"burgers_model123.{suffix}")
    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    output_format = args.format or out_file.suffix.lower().lstrip(".") or "mat"

    a = torch.cat([bundle.u0_train, bundle.u0_val, bundle.u0_test], dim=0)
    u = torch.cat([bundle.y_train, bundle.y_val, bundle.y_test], dim=0)

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
                "target_nu": float(args.nu),
                "nx": int(args.nx),
                "domain_length": float(args.domain_length),
                "num_samples": int(a.shape[0]),
                "equation": "burgers",
                "target_equation": "burgers",
                "solver": args.solver,
                "time_integrator": args.solver,
                "burgers_scheme": args.solver,
                "dealias": bool(args.dealias),
                "ic_type": args.ic_type,
                "ic_coordinate_convention": "normalized_periodic_coordinate_x_over_L",
                "grf_gamma": float(args.grf_gamma),
                "grf_tau": float(args.grf_tau),
                "grf_sigma": float(args.grf_sigma),
                "grf_mean": float(args.grf_mean),
                "fourier_num_modes": int(args.fourier_num_modes),
                "fourier_amplitude": float(args.fourier_amplitude),
            },
        )
    print(f"saved dataset: {out_file}")
    print(f"a: {tuple(a.shape)} u: {tuple(u.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
