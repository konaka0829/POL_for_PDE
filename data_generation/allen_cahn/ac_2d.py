"""Generate 2D Allen-Cahn trajectories (time-last) in MAT format.

PDE: u_t = epsilon^2 * Delta u + u - u^3, periodic.
Output:
  u: [N, S, S, T_total]
  t: [T_total]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import scipy.io
import torch

NS_DIR = Path(__file__).resolve().parents[1] / "navier_stokes"
if str(NS_DIR) not in sys.path:
    sys.path.append(str(NS_DIR))
from random_fields import GaussianRF  # type: ignore  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="2D Allen-Cahn time-series generator")
    p.add_argument("--out", type=str, default="data/allen_cahn_2d_ts.mat", help="Output .mat file path")
    p.add_argument("--N", type=int, default=200, help="Number of trajectories")
    p.add_argument("--S", type=int, default=64, help="Grid size")
    p.add_argument("--T-final", type=float, default=1.0, help="Final time")
    p.add_argument("--dt", type=float, default=1e-3, help="Internal time step")
    p.add_argument("--record-steps", type=int, default=200, help="Number of snapshots")
    p.add_argument("--epsilon", type=float, default=0.01, help="Allen-Cahn epsilon")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--alpha", type=float, default=2.5, help="GaussianRF alpha")
    p.add_argument("--tau", type=float, default=7.0, help="GaussianRF tau")
    p.add_argument("--scale", type=float, default=1.0, help="Scale factor for initial condition")
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto", help="Sampling device")
    return p


def _pick_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("--device=cuda was requested, but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _integrate_allen_cahn(u0: np.ndarray, t_final: float, dt: float, record_steps: int, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
    s = u0.shape[0]
    steps = int(math.ceil(t_final / dt))
    if steps < 1:
        raise ValueError("T-final must be positive")
    record_every = max(1, steps // record_steps)

    k = 2.0 * np.pi * np.fft.fftfreq(s, d=1.0 / s)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    k2 = kx * kx + ky * ky

    u_hat = np.fft.fft2(u0)

    out = np.zeros((s, s, record_steps), dtype=np.float32)
    t_out = np.zeros((record_steps,), dtype=np.float32)
    t = 0.0
    c = 0

    for n in range(steps):
        u = np.fft.ifft2(u_hat).real
        nonlinear = u - u * u * u
        nonlinear_hat = np.fft.fft2(nonlinear)

        # IMEX Euler: implicit diffusion + explicit reaction.
        denom = 1.0 + dt * (epsilon**2) * k2
        u_hat = (u_hat + dt * nonlinear_hat) / denom

        t += dt
        if ((n + 1) % record_every == 0) and (c < record_steps):
            out[:, :, c] = np.fft.ifft2(u_hat).real.astype(np.float32)
            t_out[c] = float(min(t, t_final))
            c += 1

    final_u = np.fft.ifft2(u_hat).real.astype(np.float32)
    while c < record_steps:
        out[:, :, c] = final_u
        t_out[c] = float(t_final)
        c += 1

    return out, t_out


def main() -> None:
    args = _build_parser().parse_args()
    if (
        args.N <= 0
        or args.S <= 1
        or args.record_steps <= 1
        or args.dt <= 0
        or args.T_final <= 0
        or args.epsilon <= 0
    ):
        raise ValueError("Invalid arguments")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = _pick_device(args.device)
    grf = GaussianRF(2, args.S, alpha=args.alpha, tau=args.tau, device=device)

    u = np.zeros((args.N, args.S, args.S, args.record_steps), dtype=np.float32)
    t_ref: np.ndarray | None = None

    for i in range(args.N):
        u0 = grf.sample(1)[0].detach().cpu().numpy().astype(np.float64)
        u0 *= float(args.scale)
        traj, t = _integrate_allen_cahn(
            u0=u0,
            t_final=args.T_final,
            dt=args.dt,
            record_steps=args.record_steps,
            epsilon=args.epsilon,
        )
        u[i] = traj
        if t_ref is None:
            t_ref = t
        if (i + 1) % max(1, args.N // 10) == 0:
            print(f"[progress] {i+1}/{args.N}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(out_path, {"u": u, "t": t_ref})
    print(f"[done] saved {out_path} with u.shape={u.shape}")


if __name__ == "__main__":
    main()
