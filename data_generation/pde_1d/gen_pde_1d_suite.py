import argparse
import os

import numpy as np
import scipy.io


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate 1D periodic PDE datasets for hybrid subordination tests.")
    parser.add_argument(
        "--pde",
        type=str,
        required=True,
        choices=("burgers", "advection_diffusion", "reaction_diffusion", "variable_diffusion", "ks"),
        help="PDE type to generate.",
    )
    parser.add_argument("--out-file", type=str, required=True, help="Output .mat path.")

    parser.add_argument("--N", type=int, default=1200, help="Number of samples.")
    parser.add_argument("--S", type=int, default=256, help="Spatial grid size.")
    parser.add_argument("--T", type=int, default=21, help="Number of saved time points.")
    parser.add_argument("--t-max", type=float, default=1.0, help="Final time.")
    parser.add_argument(
        "--steps-per-save",
        type=int,
        default=20,
        help="Internal RK4 substeps between consecutive saved times.",
    )
    parser.add_argument(
        "--u-clip",
        type=float,
        default=50.0,
        help="Clip |u| after each internal step for numerical robustness (<=0 disables clipping).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")

    parser.add_argument("--grf-beta", type=float, default=3.0, help="Spectral decay exponent for GRF initial conditions.")
    parser.add_argument("--grf-scale", type=float, default=0.2, help="Amplitude scale for GRF initial conditions.")

    # Burgers: u_t + u u_x = nu u_xx
    parser.add_argument("--nu", type=float, default=0.05, help="Viscosity for Burgers diffusion term.")

    # Advection-diffusion: u_t + c u_x = kappa u_xx
    parser.add_argument("--adv-c", type=float, default=1.0, help="Advection speed c.")
    parser.add_argument("--diff-kappa", type=float, default=0.05, help="Constant diffusion kappa.")

    # Reaction-diffusion: u_t = kappa u_xx + r u - beta u^3
    parser.add_argument("--reaction-r", type=float, default=0.5, help="Linear reaction coefficient r.")
    parser.add_argument("--reaction-beta", type=float, default=1.0, help="Cubic damping coefficient beta.")

    # Variable diffusion: u_t = d_x( a(x) d_x u )
    parser.add_argument("--vd-kappa0", type=float, default=0.1, help="Baseline positive diffusion coefficient.")
    parser.add_argument("--vd-kappa1", type=float, default=0.05, help="Amplitude for spatially varying diffusion.")

    # KS: u_t + u u_x + c2 u_xx + c4 u_xxxx = 0
    parser.add_argument("--ks-c2", type=float, default=1.0, help="Second-order coefficient c2 for KS.")
    parser.add_argument("--ks-c4", type=float, default=1.0, help="Fourth-order coefficient c4 for KS.")

    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.N <= 0 or args.S <= 1 or args.T <= 1:
        raise ValueError("Require N>0, S>1, T>1.")
    if args.t_max <= 0:
        raise ValueError("Require --t-max > 0.")
    if args.steps_per_save <= 0:
        raise ValueError("Require --steps-per-save > 0.")
    if args.u_clip < 0:
        raise ValueError("Require --u-clip >= 0.")
    if args.grf_scale <= 0:
        raise ValueError("Require --grf-scale > 0.")

    if args.pde == "burgers" and args.nu < 0:
        raise ValueError("Burgers requires --nu >= 0.")
    if args.pde == "advection_diffusion" and args.diff_kappa < 0:
        raise ValueError("Advection-diffusion requires --diff-kappa >= 0.")
    if args.pde == "reaction_diffusion" and args.diff_kappa < 0:
        raise ValueError("Reaction-diffusion requires --diff-kappa >= 0.")
    if args.pde == "variable_diffusion" and args.vd_kappa0 <= 0:
        raise ValueError("Variable diffusion requires --vd-kappa0 > 0.")
    if args.pde == "ks":
        if args.ks_c2 < 0 or args.ks_c4 <= 0:
            raise ValueError("KS requires --ks-c2 >= 0 and --ks-c4 > 0.")


def _generate_grf_initial_conditions(
    N: int,
    S: int,
    beta: float,
    scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Periodic GRF by spectral synthesis, output shape (N,S)."""
    k = np.fft.rfftfreq(S, d=1.0 / S)
    w = 2.0 * np.pi * k
    lam = w**2

    z = (rng.standard_normal((N, k.size)) + 1j * rng.standard_normal((N, k.size))) / np.sqrt(2.0)
    z[:, 0] = 0.0 + 0.0j
    if S % 2 == 0:
        z[:, -1] = rng.standard_normal(N)

    amp = (1.0 + lam) ** (-beta / 2.0)
    a_hat = z * amp[None, :]
    a = np.fft.irfft(a_hat, n=S, axis=-1).real

    mean = a.mean(axis=1, keepdims=True)
    std = a.std(axis=1, keepdims=True)
    a = (a - mean) / (std + 1e-6)
    a = a * float(scale)
    return a.astype(np.float64)


def _spectral_derivatives(u: np.ndarray, ik: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (u_hat, u_x, u_xx, u_xxxx) for u shape (N,S)."""
    u_hat = np.fft.rfft(u, axis=-1)
    ux = np.fft.irfft(ik[None, :] * u_hat, n=u.shape[-1], axis=-1).real
    uxx = np.fft.irfft((ik[None, :] ** 2) * u_hat, n=u.shape[-1], axis=-1).real
    uxxxx = np.fft.irfft((ik[None, :] ** 4) * u_hat, n=u.shape[-1], axis=-1).real
    return u_hat, ux, uxx, uxxxx


def _build_variable_diffusion_coeff(S: int, kappa0: float, kappa1: float, rng: np.random.Generator) -> np.ndarray:
    x = np.arange(S, dtype=np.float64) / float(S)
    phi1 = rng.uniform(0.0, 2.0 * np.pi)
    phi2 = rng.uniform(0.0, 2.0 * np.pi)

    kappa = kappa0 + kappa1 * (np.sin(2.0 * np.pi * x + phi1) + 0.5 * np.sin(4.0 * np.pi * x + phi2))
    min_k = float(np.min(kappa))
    if min_k <= 1e-6:
        kappa = kappa + (1e-6 - min_k)
    return kappa.astype(np.float64)


def _rhs(
    u: np.ndarray,
    ik: np.ndarray,
    pde: str,
    args: argparse.Namespace,
    kappa_x: np.ndarray | None,
) -> np.ndarray:
    """Compute du/dt for batch u with shape (N,S)."""
    _, ux, uxx, uxxxx = _spectral_derivatives(u, ik)

    if pde == "burgers":
        # u_t + u u_x = nu u_xx
        return -u * ux + args.nu * uxx

    if pde == "advection_diffusion":
        # u_t + c u_x = kappa u_xx
        return -args.adv_c * ux + args.diff_kappa * uxx

    if pde == "reaction_diffusion":
        # u_t = kappa u_xx + r u - beta u^3
        return args.diff_kappa * uxx + args.reaction_r * u - args.reaction_beta * (u**3)

    if pde == "variable_diffusion":
        if kappa_x is None:
            raise RuntimeError("Internal error: kappa_x is required for variable_diffusion.")
        # u_t = d_x( kappa(x) d_x u )
        flux = kappa_x[None, :] * ux
        flux_hat = np.fft.rfft(flux, axis=-1)
        return np.fft.irfft(ik[None, :] * flux_hat, n=u.shape[-1], axis=-1).real

    if pde == "ks":
        # u_t + u u_x + c2 u_xx + c4 u_xxxx = 0
        return -u * ux - args.ks_c2 * uxx - args.ks_c4 * uxxxx

    raise ValueError(f"Unsupported pde: {pde}")


def _integrate_dataset(
    a0: np.ndarray,
    t: np.ndarray,
    pde: str,
    args: argparse.Namespace,
    kappa_x: np.ndarray | None,
) -> tuple[np.ndarray, int]:
    """Integrate PDE for all samples (periodic, pseudo-spectral IMEX/explicit)."""
    N, S = a0.shape
    T = t.size

    dt_save = float(t[1] - t[0])
    step_multiplier = {
        "advection_diffusion": 1,
        "burgers": 2,
        "reaction_diffusion": 3,
        "variable_diffusion": 4,
        "ks": 8,
    }[pde]
    effective_steps_per_save = int(args.steps_per_save * step_multiplier)
    dt = dt_save / float(effective_steps_per_save)

    k = 2.0 * np.pi * np.fft.rfftfreq(S, d=1.0 / S)  # angular frequency
    ik = 1j * k
    k2 = k**2
    k4 = k**4

    # 2/3 de-aliasing mask for nonlinear term in Fourier space.
    cutoff_mode = S // 3
    mode_ids = np.arange(k.size)
    dealias_mask = mode_ids <= cutoff_mode

    u = a0.copy()
    u_hist = np.empty((N, S, T), dtype=np.float64)
    u_hist[:, :, 0] = u

    for j in range(1, T):
        for _ in range(effective_steps_per_save):
            u_hat = np.fft.rfft(u, axis=-1)

            if pde == "advection_diffusion":
                # u_t + c u_x = kappa u_xx
                L = -1j * args.adv_c * k - args.diff_kappa * k2
                u_hat_new = np.exp(dt * L)[None, :] * u_hat
                u = np.fft.irfft(u_hat_new, n=S, axis=-1).real

            elif pde == "burgers":
                # IMEX Euler: (I - dt L) u_{n+1} = u_n + dt N(u_n), L = nu * d_xx
                ux = np.fft.irfft(ik[None, :] * u_hat, n=S, axis=-1).real
                N_hat = np.fft.rfft(-u * ux, axis=-1)
                N_hat[:, ~dealias_mask] = 0.0
                L = -args.nu * k2
                u_hat_new = (u_hat + dt * N_hat) / (1.0 - dt * L)[None, :]
                u = np.fft.irfft(u_hat_new, n=S, axis=-1).real

            elif pde == "reaction_diffusion":
                # u_t = kappa u_xx + r u - beta u^3
                N_hat = np.fft.rfft(-args.reaction_beta * (u**3), axis=-1)
                N_hat[:, ~dealias_mask] = 0.0
                L = -args.diff_kappa * k2 + args.reaction_r
                u_hat_new = (u_hat + dt * N_hat) / (1.0 - dt * L)[None, :]
                u = np.fft.irfft(u_hat_new, n=S, axis=-1).real

            elif pde == "variable_diffusion":
                if kappa_x is None:
                    raise RuntimeError("Internal error: kappa_x is required for variable_diffusion.")
                ux = np.fft.irfft(ik[None, :] * u_hat, n=S, axis=-1).real
                flux = kappa_x[None, :] * ux
                flux_hat = np.fft.rfft(flux, axis=-1)
                rhs = np.fft.irfft(ik[None, :] * flux_hat, n=S, axis=-1).real
                u = u + dt * rhs

            elif pde == "ks":
                # u_t + u u_x + c2 u_xx + c4 u_xxxx = 0
                # => u_t = -(u u_x) + c2 k^2 u_hat - c4 k^4 u_hat in Fourier linear part
                ux = np.fft.irfft(ik[None, :] * u_hat, n=S, axis=-1).real
                N_hat = np.fft.rfft(-u * ux, axis=-1)
                N_hat[:, ~dealias_mask] = 0.0
                L = args.ks_c2 * k2 - args.ks_c4 * k4
                u_hat_new = (u_hat + dt * N_hat) / (1.0 - dt * L)[None, :]
                u = np.fft.irfft(u_hat_new, n=S, axis=-1).real

            else:
                raise ValueError(f"Unsupported pde: {pde}")

            if args.u_clip > 0:
                np.clip(u, -args.u_clip, args.u_clip, out=u)

        if not np.isfinite(u).all():
            raise FloatingPointError(
                "Numerical instability encountered (NaN/Inf). "
                "Try smaller --t-max, larger --steps-per-save, smaller --grf-scale, "
                "smoother initial data (larger --grf-beta), "
                "or milder PDE coefficients."
            )
        u_hist[:, :, j] = u

    return u_hist, effective_steps_per_save


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(args)

    rng = np.random.default_rng(args.seed)

    a = _generate_grf_initial_conditions(
        N=args.N,
        S=args.S,
        beta=args.grf_beta,
        scale=args.grf_scale,
        rng=rng,
    )

    t = np.linspace(0.0, args.t_max, args.T, dtype=np.float64)

    kappa_x = None
    if args.pde == "variable_diffusion":
        kappa_x = _build_variable_diffusion_coeff(args.S, args.vd_kappa0, args.vd_kappa1, rng)

    u, effective_steps_per_save = _integrate_dataset(a, t, args.pde, args, kappa_x)

    out_dir = os.path.dirname(args.out_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    mdict = {
        "a": a.astype(np.float32),
        "u": u.astype(np.float32),
        "t": t.astype(np.float32),
        "pde": np.array([args.pde]),
        "S": np.array(args.S, dtype=np.int32),
        "T": np.array(args.T, dtype=np.int32),
        "t_max": np.array(args.t_max, dtype=np.float32),
        "steps_per_save": np.array(args.steps_per_save, dtype=np.int32),
        "effective_steps_per_save": np.array(effective_steps_per_save, dtype=np.int32),
        "u_clip": np.array(args.u_clip, dtype=np.float32),
        "seed": np.array(args.seed, dtype=np.int32),
        "grf_beta": np.array(args.grf_beta, dtype=np.float32),
        "grf_scale": np.array(args.grf_scale, dtype=np.float32),
        "nu": np.array(args.nu, dtype=np.float32),
        "adv_c": np.array(args.adv_c, dtype=np.float32),
        "diff_kappa": np.array(args.diff_kappa, dtype=np.float32),
        "reaction_r": np.array(args.reaction_r, dtype=np.float32),
        "reaction_beta": np.array(args.reaction_beta, dtype=np.float32),
        "vd_kappa0": np.array(args.vd_kappa0, dtype=np.float32),
        "vd_kappa1": np.array(args.vd_kappa1, dtype=np.float32),
        "ks_c2": np.array(args.ks_c2, dtype=np.float32),
        "ks_c4": np.array(args.ks_c4, dtype=np.float32),
    }
    if kappa_x is not None:
        mdict["kappa_x"] = kappa_x.astype(np.float32)

    scipy.io.savemat(args.out_file, mdict)

    print(f"Saved: {args.out_file}")
    print(
        f"pde={args.pde}, a shape={a.shape}, u shape={u.shape}, t shape={t.shape}, "
        f"effective_steps_per_save={effective_steps_per_save}"
    )


if __name__ == "__main__":
    main()
