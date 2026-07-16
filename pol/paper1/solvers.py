from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch

from pol.burgers_spectral_1d import simulate_burgers_split_step
from pol.spectral_etdrk4_1d import simulate_burgers_etdrk4


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
    normalized = "etdrk4" if solver in {"etdrk4", "fourier_pseudospectral_etdrk4"} else "split_step" if solver in {"split_step", "semi_implicit"} else solver
    if normalized == "etdrk4":
        substeps = 1
        effective = dt
        values = simulate_burgers_etdrk4(u0, nu=nu, T=T, dt=dt, dealias=dealias, domain_length=domain_length)
        requested_fine = None if fine_dt is None else float(fine_dt)
    elif normalized == "split_step":
        if fine_dt is None or fine_dt <= 0:
            raise ValueError("split_step requires positive fine_dt")
        substeps = max(1, int(math.ceil(dt / fine_dt)))
        effective = dt / float(substeps)
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
