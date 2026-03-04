from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np
import scipy.io
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pol.burgers_spectral_1d import simulate_burgers_split_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 1D Burgers dataset (a -> u(T))")
    parser.add_argument("--out-file", required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--grid-size", type=int, required=True)
    parser.add_argument("--nu", type=float, required=True)

    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=1e-2)
    parser.add_argument("--fine-dt", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--save-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--dealias", action="store_true")

    parser.add_argument("--grf-gamma", type=float, default=2.0)
    parser.add_argument("--grf-tau", type=float, default=5.0)
    parser.add_argument("--grf-sigma", type=float, default=25.0)
    parser.add_argument("--grf-mean", type=float, default=0.0)

    args = parser.parse_args()
    if args.num_samples <= 0:
        parser.error("--num-samples must be positive")
    if args.grid_size <= 1:
        parser.error("--grid-size must be >= 2")
    if args.nu < 0.0:
        parser.error("--nu must be non-negative")
    if args.T <= 0.0:
        parser.error("--T must be positive")
    if args.dt <= 0.0:
        parser.error("--dt must be positive")
    if args.fine_dt <= 0.0:
        parser.error("--fine-dt must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.grf_gamma <= 0.0:
        parser.error("--grf-gamma must be positive")
    if args.grf_tau < 0.0:
        parser.error("--grf-tau must be non-negative")
    if args.grf_sigma < 0.0:
        parser.error("--grf-sigma must be non-negative")
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
def sample_periodic_grf(
    num_samples: int,
    grid_size: int,
    *,
    gamma: float,
    tau: float,
    sigma: float,
    mean: float,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    nfreq = grid_size // 2 + 1
    n = torch.arange(nfreq, dtype=dtype, device=device)
    omega = 2.0 * torch.pi * n
    spectrum = (sigma**2) * ((omega.pow(2) + tau**2).pow(-gamma))

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    real = torch.randn((num_samples, nfreq), generator=gen, dtype=torch.float32)
    imag = torch.randn((num_samples, nfreq), generator=gen, dtype=torch.float32)
    real = real.to(device=device, dtype=dtype)
    imag = imag.to(device=device, dtype=dtype)

    coeff = (real + 1j * imag) / math.sqrt(2.0)
    coeff = coeff * spectrum.sqrt().unsqueeze(0).to(dtype=dtype)

    coeff[:, 0] = coeff[:, 0].real + 0j
    if grid_size % 2 == 0:
        coeff[:, -1] = coeff[:, -1].real + 0j

    u0 = torch.fft.irfft(coeff, n=grid_size, dim=-1)
    if mean != 0.0:
        u0 = u0 + mean
    return u0


@torch.no_grad()
def generate_dataset(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)

    a_all = []
    u_all = []

    n_outer = int(round(args.T / args.dt))
    if n_outer < 1:
        raise ValueError("T/dt must result in at least one outer step")

    produced = 0
    while produced < args.num_samples:
        b = min(args.batch_size, args.num_samples - produced)
        a_batch = sample_periodic_grf(
            b,
            args.grid_size,
            gamma=args.grf_gamma,
            tau=args.grf_tau,
            sigma=args.grf_sigma,
            mean=args.grf_mean,
            seed=args.seed + produced,
            device=device,
            dtype=dtype,
        )
        states = simulate_burgers_split_step(
            a_batch,
            dt=args.dt,
            Tr=args.T,
            obs_steps=[n_outer],
            nu=args.nu,
            fine_dt=args.fine_dt,
            forcing=None,
            forcing_steps=None,
            dealias=args.dealias,
        )
        u_batch = states[-1]

        a_all.append(a_batch.detach().cpu())
        u_all.append(u_batch.detach().cpu())
        produced += b

    a = torch.cat(a_all, dim=0).numpy()
    u = torch.cat(u_all, dim=0).numpy()
    return a, u


def main() -> None:
    args = parse_args()
    a, u = generate_dataset(args)

    save_dtype = np.float32 if args.save_dtype == "float32" else np.float64
    a = a.astype(save_dtype)
    u = u.astype(save_dtype)
    x_grid = np.linspace(0.0, 1.0, args.grid_size, endpoint=False, dtype=save_dtype)

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(
        out_path,
        {
            "a": a,
            "u": u,
            "x_grid": x_grid,
            "nu": np.array([[args.nu]], dtype=save_dtype),
            "T": np.array([[args.T]], dtype=save_dtype),
            "dt": np.array([[args.dt]], dtype=save_dtype),
            "fine_dt": np.array([[args.fine_dt]], dtype=save_dtype),
            "grid_size": np.array([[args.grid_size]], dtype=np.int32),
            "generator": np.array(["burgers_split_step_v1"], dtype=object),
        },
    )
    print(f"saved dataset: {out_path}")
    print(f"a shape: {a.shape}, u shape: {u.shape}")


if __name__ == "__main__":
    main()
