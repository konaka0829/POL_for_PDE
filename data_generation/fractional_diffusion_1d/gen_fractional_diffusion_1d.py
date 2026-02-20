import argparse
import os

import numpy as np
import scipy.io


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate 1D fractional diffusion data (FFT exact).")
    parser.add_argument("--out-file", type=str, required=True, help="Output .mat path.")
    parser.add_argument("--N", type=int, default=1200, help="Number of samples.")
    parser.add_argument("--S", type=int, default=1024, help="Spatial grid size.")
    parser.add_argument("--T", type=int, default=11, help="Number of time points.")
    parser.add_argument("--t-max", type=float, default=1.0, help="Final time.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Fractional exponent in psi(lam)=lam^alpha.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--grf-beta", type=float, default=3.0, help="Spectral decay exponent for GRF.")
    parser.add_argument("--grf-scale", type=float, default=1.0, help="Amplitude scale for GRF.")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.N <= 0 or args.S <= 1 or args.T <= 1:
        raise ValueError("Require N>0, S>1, T>1.")
    if args.t_max <= 0:
        raise ValueError("Require t-max > 0.")
    if args.alpha <= 0:
        raise ValueError("Require alpha > 0.")
    if args.grf_scale <= 0:
        raise ValueError("Require grf-scale > 0.")


def _generate_grf_initial_conditions(
    N: int,
    S: int,
    beta: float,
    scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    k = np.fft.rfftfreq(S, d=1.0 / S)
    w = 2.0 * np.pi * k
    lam = w**2

    # Complex Gaussian with unit variance per complex coeff.
    z = (rng.standard_normal((N, k.size)) + 1j * rng.standard_normal((N, k.size))) / np.sqrt(2.0)
    z[:, 0] = 0.0 + 0.0j  # zero mean in physical space
    if S % 2 == 0:
        z[:, -1] = rng.standard_normal(N)  # Nyquist coefficient must be real

    amp = scale * (1.0 + lam) ** (-beta / 2.0)
    a_hat = z * amp[None, :]
    a = np.fft.irfft(a_hat, n=S, axis=-1).real

    # Per-sample normalization for stable scale.
    mean = a.mean(axis=1, keepdims=True)
    std = a.std(axis=1, keepdims=True)
    a = (a - mean) / (std + 1e-6)
    return a.astype(np.float32)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(args)

    rng = np.random.default_rng(args.seed)
    a = _generate_grf_initial_conditions(args.N, args.S, args.grf_beta, args.grf_scale, rng)

    k = np.fft.rfftfreq(args.S, d=1.0 / args.S)
    w = 2.0 * np.pi * k
    lam = w**2
    psi_true = lam**args.alpha

    t = np.linspace(0.0, args.t_max, args.T, dtype=np.float32)
    mult = np.exp(-t[:, None] * psi_true[None, :])  # (T, K)

    a_hat = np.fft.rfft(a, axis=-1)  # (N, K)
    u_hat = a_hat[:, None, :] * mult[None, :, :]  # (N, T, K)
    u = np.fft.irfft(u_hat, n=args.S, axis=-1).real  # (N, T, S)
    u = np.transpose(u, (0, 2, 1)).astype(np.float32)  # (N, S, T)

    out_dir = os.path.dirname(args.out_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    scipy.io.savemat(
        args.out_file,
        {
            "a": a.astype(np.float32),
            "u": u.astype(np.float32),
            "t": t.astype(np.float32),
            "alpha": np.array(args.alpha, dtype=np.float32),
            "S": np.array(args.S, dtype=np.int32),
            "T": np.array(args.T, dtype=np.int32),
            "t_max": np.array(args.t_max, dtype=np.float32),
            "grf_beta": np.array(args.grf_beta, dtype=np.float32),
            "grf_scale": np.array(args.grf_scale, dtype=np.float32),
        },
    )

    print(f"Saved: {args.out_file}")
    print(f"a shape={a.shape}, u shape={u.shape}, t shape={t.shape}, alpha={args.alpha}")


if __name__ == "__main__":
    main()
