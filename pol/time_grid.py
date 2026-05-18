from __future__ import annotations

import math


def require_time_aligned(
    t: float,
    dt: float,
    name: str = "time",
    *,
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-12,
) -> int:
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    step_float = float(t) / float(dt)
    step = int(round(step_float))
    if step < 0:
        raise ValueError(f"{name} must be nonnegative")
    if not math.isclose(step * float(dt), float(t), rel_tol=rel_tol, abs_tol=abs_tol):
        raise ValueError(
            f"{name}={t} is not aligned with dt={dt}; "
            f"expected {name} to be an integer multiple of dt. "
            "Use an aligned value or change dt."
        )
    return step
