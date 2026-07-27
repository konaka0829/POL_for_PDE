"""Typed E2 convergence and explicit rerun decisions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ConvergenceDecision:
    """Outcome consumed by orchestration before any test evaluation."""

    status: Literal["accept", "rerun", "reject"]
    selected_n_sur: int | None
    reason: str | None = None


def decide_convergence(
    *, pilot_n_sur: int, selected_base: int | None, reruns_remaining: int
) -> ConvergenceDecision:
    if selected_base is None:
        return ConvergenceDecision("reject", None, "no acceptable n_sur")
    if selected_base <= pilot_n_sur:
        return ConvergenceDecision("accept", selected_base)
    if reruns_remaining <= 0:
        return ConvergenceDecision("reject", selected_base, "rerun limit reached")
    return ConvergenceDecision("rerun", selected_base, "selected base exceeds pilot")
