from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import scipy.io
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pol.burgers_spectral_1d import simulate_burgers_split_step
from pol.model123_1d.initial_conditions import (
    evaluate_initial_conditions,
    sample_gaussian_random_field_initial_conditions,
    sample_initial_condition_coefficients,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate fixed-seed Burgers dataset with 8-mode Fourier initial conditions"
    )
    parser.add_argument("--out-file", required=True)
    parser.add_argument(
        "--initial-condition-type",
        choices=("fourier", "gaussian_rf"),
        default="fourier",
    )
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--num-modes", type=int, default=8)
    parser.add_argument("--amplitude", type=float, default=0.5)
    parser.add_argument("--grf-mean", type=float, default=0.0)
    parser.add_argument("--grf-gamma", type=float, default=2.0)
    parser.add_argument("--grf-tau", type=float, default=5.0)
    parser.add_argument("--grf-sigma", type=float, default=25.0)
    parser.add_argument("--grid-size", type=int, default=1024)
    parser.add_argument("--nu", type=float, default=0.05)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=1e-2)
    parser.add_argument("--fine-dt", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--save-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--dealias", type=int, choices=(0, 1), default=1)
    args = parser.parse_args()

    if args.num_samples <= 0:
        parser.error("--num-samples must be positive")
    if args.num_modes <= 0:
        parser.error("--num-modes must be positive")
    if args.amplitude <= 0.0:
        parser.error("--amplitude must be positive")
    if args.grf_gamma <= 0.0:
        parser.error("--grf-gamma must be positive")
    if args.grf_tau < 0.0:
        parser.error("--grf-tau must be non-negative")
    if args.grf_sigma < 0.0:
        parser.error("--grf-sigma must be non-negative")
    if args.grid_size <= 1:
        parser.error("--grid-size must be >= 2")
    if args.nu < 0.0:
        parser.error("--nu must be non-negative")
    if args.T <= 0.0 or args.dt <= 0.0 or args.fine_dt <= 0.0:
        parser.error("--T, --dt, and --fine-dt must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device=cuda was requested but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    return torch.float64


@torch.no_grad()
def generate_dataset(args: argparse.Namespace) -> dict[str, np.ndarray]:
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    coeffs = None
    if args.initial_condition_type == "fourier":
        coeffs = sample_initial_condition_coefficients(
            args.num_samples,
            seed=args.seed,
            num_modes=args.num_modes,
            dtype=dtype,
        )
        a = evaluate_initial_conditions(
            coeffs,
            args.grid_size,
            amplitude=args.amplitude,
            device=device,
            dtype=dtype,
        )
        am_coeff = coeffs.a.detach().cpu().numpy()
        bm_coeff = coeffs.b.detach().cpu().numpy()
    else:
        a = sample_gaussian_random_field_initial_conditions(
            args.num_samples,
            args.grid_size,
            seed=args.seed,
            gamma=args.grf_gamma,
            tau=args.grf_tau,
            sigma=args.grf_sigma,
            mean=args.grf_mean,
            device=device,
            dtype=dtype,
        )
        am_coeff = np.zeros((args.num_samples, 0), dtype=np.float64)
        bm_coeff = np.zeros((args.num_samples, 0), dtype=np.float64)

    obs_step = int(round(args.T / args.dt))
    outputs: list[torch.Tensor] = []
    for start in range(0, args.num_samples, args.batch_size):
        a_batch = a[start : start + args.batch_size]
        states = simulate_burgers_split_step(
            a_batch,
            dt=args.dt,
            Tr=args.T,
            obs_steps=[obs_step],
            nu=args.nu,
            fine_dt=args.fine_dt,
            b=1.0,
            forcing=None,
            forcing_steps=None,
            dealias=bool(args.dealias),
        )
        outputs.append(states[-1].detach().cpu())

    u = torch.cat(outputs, dim=0).cpu()
    save_dtype = np.float32 if args.save_dtype == "float32" else np.float64
    x_grid = np.linspace(0.0, 1.0, args.grid_size, endpoint=False, dtype=save_dtype)
    return {
        "a": a.detach().cpu().numpy().astype(save_dtype),
        "u": u.numpy().astype(save_dtype),
        "x_grid": x_grid,
        "nu": np.array([[args.nu]], dtype=save_dtype),
        "T": np.array([[args.T]], dtype=save_dtype),
        "dt": np.array([[args.dt]], dtype=save_dtype),
        "fine_dt": np.array([[args.fine_dt]], dtype=save_dtype),
        "grid_size": np.array([[args.grid_size]], dtype=np.int32),
        "seed": np.array([[args.seed]], dtype=np.int32),
        "num_modes": np.array([[args.num_modes]], dtype=np.int32),
        "num_samples": np.array([[args.num_samples]], dtype=np.int32),
        "initial_condition_type": np.array([args.initial_condition_type]),
        "grf_mean": np.array([[args.grf_mean]], dtype=save_dtype),
        "grf_gamma": np.array([[args.grf_gamma]], dtype=save_dtype),
        "grf_tau": np.array([[args.grf_tau]], dtype=save_dtype),
        "grf_sigma": np.array([[args.grf_sigma]], dtype=save_dtype),
        "am_coeff": am_coeff.astype(save_dtype),
        "bm_coeff": bm_coeff.astype(save_dtype),
    }


def save_dataset(payload: dict[str, np.ndarray], out_file: str | Path) -> None:
    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(out_path, payload)


def main() -> None:
    args = parse_args()
    payload = generate_dataset(args)
    save_dataset(payload, args.out_file)
    print(f"saved dataset: {args.out_file}")
    print(f"a shape: {payload['a'].shape}, u shape: {payload['u'].shape}")


if __name__ == "__main__":
    main()
