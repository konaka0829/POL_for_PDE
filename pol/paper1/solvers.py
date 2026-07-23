from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch

from pol.burgers_spectral_1d import simulate_burgers_split_step
from pol.spectral_etdrk4_1d import simulate_burgers_etdrk4


def normalize_burgers_solver_name(solver: str) -> str:
    """Return the canonical E0 Burgers solver name."""
    if solver in {"etdrk4", "fourier_pseudospectral_etdrk4"}:
        return "etdrk4"
    if solver in {"split_step", "semi_implicit"}:
        return "split_step"
    raise ValueError(f"unsupported Burgers solver: {solver}")


def burgers_step_metadata(*, solver: str, dt: float, fine_dt: float | None) -> tuple[float, int]:
    """Return ``(effective_inner_step, substeps_per_outer)``."""
    if dt <= 0:
        raise ValueError("dt must be positive")
    normalized = normalize_burgers_solver_name(solver)
    if normalized == "etdrk4":
        return float(dt), 1
    if fine_dt is None or fine_dt <= 0:
        raise ValueError("split_step requires positive fine_dt")
    substeps = max(1, int(math.ceil(dt / fine_dt)))
    return float(dt) / substeps, substeps


def effective_inner_step(*, solver: str, dt: float, fine_dt: float | None) -> float:
    """Return the actual numerical step used by the selected solver."""
    return burgers_step_metadata(solver=solver, dt=dt, fine_dt=fine_dt)[0]


@dataclass(frozen=True)
class BurgersSolverMetadata:
    solver: str
    requested_dt: float
    requested_fine_dt: float | None
    effective_inner_step: float
    outer_steps: int
    substeps_per_outer: int
    dealias: bool
    domain_length: float
    dtype: str
    device: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BurgersFinalStateResult:
    values: torch.Tensor
    metadata: BurgersSolverMetadata


@torch.no_grad()
def solve_burgers_final_state(
    u0: torch.Tensor,
    *,
    nu: float,
    T: float,
    dt: float,
    fine_dt: float | None,
    solver: str,
    dealias: bool,
    domain_length: float,
) -> BurgersFinalStateResult:
    """Solve a batch of endpoint-free periodic Burgers fields to ``T``.

    ``u0`` has shape ``(batch, nx)`` and is preserved on its dtype/device.
    Split-step uses the existing ``ceil(dt/fine_dt)`` inner-step convention;
    ETDRK4 uses ``dt`` directly.
    """
    if u0.ndim != 2:
        raise ValueError(f"u0 must have shape (batch, nx), got {tuple(u0.shape)}")
    if not u0.dtype.is_floating_point:
        raise TypeError("u0 must be real floating point")
    if T <= 0 or dt <= 0:
        raise ValueError("T and dt must be positive")
    outer_steps = int(round(T / dt))
    if abs(outer_steps * dt - T) > 1e-10 * max(1.0, abs(T)):
        raise ValueError(f"T={T} must be aligned with dt={dt}")
    normalized = normalize_burgers_solver_name(solver)
    effective, substeps = burgers_step_metadata(solver=solver, dt=dt, fine_dt=fine_dt)
    if normalized == "etdrk4":
        values = simulate_burgers_etdrk4(u0, nu=nu, T=T, dt=dt, dealias=dealias, domain_length=domain_length)
        requested_fine = None if fine_dt is None else float(fine_dt)
    elif normalized == "split_step":
        if fine_dt is None or fine_dt <= 0:
            raise ValueError("split_step requires positive fine_dt")
        values = simulate_burgers_split_step(
            u0, dt=dt, Tr=T, obs_steps=[outer_steps], nu=nu,
            fine_dt=fine_dt, dealias=dealias, domain_length=domain_length,
        )[-1]
        requested_fine = float(fine_dt)
    else:
        raise ValueError(f"unsupported Burgers solver: {solver}")
    finite = torch.isfinite(values).reshape(values.shape[0], -1).all(dim=1)
    if not bool(finite.all()):
        bad = torch.nonzero(~finite, as_tuple=False).flatten().tolist()
        raise FloatingPointError(
            f"Burgers solver produced NaN/Inf for batch sample indices {bad}; "
            f"solver={solver}, nx={u0.shape[-1]}, T={T}, dt={dt}, fine_dt={fine_dt}"
        )
    metadata = BurgersSolverMetadata(
        solver=normalized, requested_dt=float(dt), requested_fine_dt=requested_fine,
        effective_inner_step=float(effective), outer_steps=outer_steps,
        substeps_per_outer=substeps, dealias=bool(dealias), domain_length=float(domain_length),
        dtype=str(u0.dtype).removeprefix("torch."), device=str(u0.device),
    )
    return BurgersFinalStateResult(values=values.detach(), metadata=metadata)


@dataclass(frozen=True)
class ReactionDiffusionFinalStateResult:
    values: torch.Tensor
    metadata: dict[str, object]


@torch.no_grad()
def solve_reaction_diffusion_final_state(
    u0: torch.Tensor, *, nu: float, alpha: float, beta: float, T: float, dt: float,
    domain_length: float, nonlinear_filter: str = "two_thirds", context: str = "",
) -> ReactionDiffusionFinalStateResult:
    """Semi-implicit spectral Euler for ``r_t=nu*r_xx+alpha*r-beta*r^3``.

    The update is ``rhat[n+1]=(rhat[n]+dt*FFT(alpha*r-beta*r^3))/
    (1+dt*nu*k^2)``.  ``two_thirds`` masks the sampled cubic term; this is a
    stabilizing 2/3 filter, not exact cubic de-aliasing.
    """
    if u0.ndim != 2 or not u0.dtype.is_floating_point:
        raise ValueError("reaction-diffusion u0 must be real (batch,nx)")
    if nu <= 0 or T <= 0 or dt <= 0 or domain_length <= 0:
        raise ValueError("reaction-diffusion nu,T,dt,L must be positive")
    if nonlinear_filter not in {"none", "two_thirds"}:
        raise ValueError("nonlinear_filter must be none or two_thirds")
    steps = int(round(T / dt))
    if abs(steps * dt - T) > 1e-10 * max(1.0, abs(T)):
        raise ValueError(f"T={T} must be aligned with dt={dt}")
    nx = u0.shape[-1]
    k = 2 * math.pi * torch.fft.rfftfreq(nx, d=domain_length / nx, device=u0.device, dtype=u0.dtype)
    denominator = 1.0 + dt * nu * k.square()
    mask = (torch.arange(k.numel(), device=u0.device) <= nx // 3).to(u0.dtype)
    values = u0.clone()
    for _ in range(steps):
        nonlinear_hat = torch.fft.rfft(alpha * values - beta * values.pow(3), dim=-1)
        if nonlinear_filter == "two_thirds":
            nonlinear_hat = nonlinear_hat * mask
        values = torch.fft.irfft(
            (torch.fft.rfft(values, dim=-1) + dt * nonlinear_hat) / denominator,
            n=nx, dim=-1,
        )
        if not bool(torch.isfinite(values).all()):
            raise FloatingPointError(
                f"reaction-diffusion produced NaN/Inf; context={context or 'unspecified'}, "
                f"nx={nx}, nu={nu}, T={T}, dt={dt}")
    metadata = {
        "solver": "semi_implicit_spectral_euler", "requested_dt": float(dt),
        "effective_inner_step": float(dt), "step_count": steps, "nu_tilde": float(nu),
        "alpha": float(alpha), "beta": float(beta), "nonlinear_filter": nonlinear_filter,
        "dealiasing": False, "domain_length": float(domain_length),
        "dtype": str(u0.dtype).removeprefix("torch."), "device": str(u0.device),
    }
    return ReactionDiffusionFinalStateResult(values.detach(), metadata)
