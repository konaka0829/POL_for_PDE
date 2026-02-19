"""Generate 1D Kuramoto-Sivashinsky trajectories (time-last) with ETDRK4.

PDE: u_t + u u_x + u_xx + u_xxxx = 0 (periodic)
Output:
  u: [N, S, T_total]
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
    p = argparse.ArgumentParser(description="1D Kuramoto-Sivashinsky time-series generator (ETDRK4)")
    p.add_argument("--out", type=str, default="data/ks_1d_ts.mat", help="Output .mat file path")
    p.add_argument("--N", type=int, default=200, help="Number of trajectories")
    p.add_argument("--S", type=int, default=1024, help="Number of spatial points")
    p.add_argument("--T-final", type=float, default=50.0, help="Final time")
    p.add_argument("--dt", type=float, default=2.5e-3, help="Internal time step")
    p.add_argument("--record-steps", type=int, default=200, help="Number of snapshots")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--alpha", type=float, default=2.0, help="GaussianRF alpha")
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


def _etdrk4_coeffs(
    L: np.ndarray, dt: float, m: int = 16
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    E = np.exp(dt * L)
    E2 = np.exp(dt * L / 2.0)

    r = np.exp(1j * np.pi * (np.arange(1, m + 1) - 0.5) / m)
    LR = dt * L[:, None] + r[None, :]
    Q = dt * np.real(np.mean((np.exp(LR / 2.0) - 1.0) / LR, axis=1))
    f1 = dt * np.real(np.mean((-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR**2)) / (LR**3), axis=1))
    f2 = dt * np.real(np.mean((2.0 + LR + np.exp(LR) * (-2.0 + LR)) / (LR**3), axis=1))
    f3 = dt * np.real(np.mean((-4.0 - 3.0 * LR - LR**2 + np.exp(LR) * (4.0 - LR)) / (LR**3), axis=1))
    return E, E2, Q, f1, f2, f3


def _integrate_ks(u0: np.ndarray, t_final: float, dt: float, record_steps: int) -> tuple[np.ndarray, np.ndarray]:
    s = u0.shape[0]
    steps = int(math.ceil(t_final / dt))
    if steps < 1:
        raise ValueError("T-final must be positive")
    record_every = max(1, steps // record_steps)

    # Periodic domain [0, 2pi); derivative is ik with integer k.
    k = np.fft.fftfreq(s, d=1.0 / s)
    ik = 1j * k
    L = k**2 - k**4
    E, E2, Q, f1, f2, f3 = _etdrk4_coeffs(L=L, dt=dt, m=16)

    v = np.fft.fft(u0)

    def N(vh: np.ndarray) -> np.ndarray:
        u = np.fft.ifft(vh).real
        return -0.5 * ik * np.fft.fft(u * u)

    out = np.zeros((s, record_steps), dtype=np.float32)
    t_out = np.zeros((record_steps,), dtype=np.float32)
    t = 0.0
    c = 0

    for n in range(steps):
        Nv = N(v)
        a = E2 * v + Q * Nv
        Na = N(a)
        b = E2 * v + Q * Na
        Nb = N(b)
        ctmp = E2 * a + Q * (2.0 * Nb - Nv)
        Nc = N(ctmp)
        v = E * v + f1 * Nv + 2.0 * f2 * (Na + Nb) + f3 * Nc

        t += dt
        if ((n + 1) % record_every == 0) and (c < record_steps):
            out[:, c] = np.fft.ifft(v).real.astype(np.float32)
            t_out[c] = float(min(t, t_final))
            c += 1

    final_u = np.fft.ifft(v).real.astype(np.float32)
    while c < record_steps:
        out[:, c] = final_u
        t_out[c] = float(t_final)
        c += 1

    return out, t_out


def main() -> None:
    args = _build_parser().parse_args()
    if args.N <= 0 or args.S <= 1 or args.record_steps <= 1 or args.dt <= 0 or args.T_final <= 0:
        raise ValueError("Invalid arguments")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = _pick_device(args.device)
    grf = GaussianRF(1, args.S, alpha=args.alpha, tau=args.tau, device=device)

    u = np.zeros((args.N, args.S, args.record_steps), dtype=np.float32)
    t_ref: np.ndarray | None = None

    for i in range(args.N):
        u0 = grf.sample(1)[0].detach().cpu().numpy().astype(np.float64)
        u0 *= float(args.scale)
        traj, t = _integrate_ks(u0=u0, t_final=args.T_final, dt=args.dt, record_steps=args.record_steps)
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
