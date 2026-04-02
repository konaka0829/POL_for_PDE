from __future__ import annotations

from dataclasses import dataclass

import torch

from pol.reservoir_1d import Reservoir1DSolver, ReservoirConfig


@dataclass(frozen=True)
class SurrogateSpec:
    family: str
    dt: float
    T: float
    fine_dt: float = 1e-4
    rd_nu: float = 1e-3
    rd_alpha: float = 1.0
    rd_beta: float = 1.0
    burgers_nu: float = 0.05
    burgers_b: float = 1.0
    ks_dealias: bool = False
    ks_b: float = 1.0
    ks_eta: float = 1.0
    ks_kappa: float = 1.0

    def make_solver(self) -> Reservoir1DSolver:
        if self.family == "reaction_diffusion":
            cfg = ReservoirConfig(
                reservoir="reaction_diffusion",
                rd_nu=self.rd_nu,
                rd_alpha=self.rd_alpha,
                rd_beta=self.rd_beta,
            )
            return Reservoir1DSolver(cfg)
        if self.family == "ks":
            cfg = ReservoirConfig(
                reservoir="ks",
                ks_dealias=self.ks_dealias,
                ks_b=self.ks_b,
                ks_eta=self.ks_eta,
                ks_kappa=self.ks_kappa,
            )
            return Reservoir1DSolver(cfg)
        if self.family == "burgers":
            cfg = ReservoirConfig(
                reservoir="burgers",
                res_burgers_nu=self.burgers_nu,
                res_burgers_b=self.burgers_b,
                burgers_scheme="split_step",
                burgers_fine_dt=self.fine_dt,
            )
            return Reservoir1DSolver(cfg)
        raise ValueError(f"Unsupported family: {self.family}")


def simulate_surrogate_batches(
    u0: torch.Tensor,
    *,
    spec: SurrogateSpec,
    obs_steps: list[int],
    batch_size: int,
) -> list[torch.Tensor]:
    solver = spec.make_solver()
    all_states: list[list[torch.Tensor]] = [[] for _ in obs_steps]
    for start in range(0, u0.shape[0], batch_size):
        batch = u0[start : start + batch_size]
        states = solver.simulate(batch, dt=spec.dt, Tr=spec.T, obs_steps=obs_steps)
        for idx, state in enumerate(states):
            all_states[idx].append(state.detach().cpu())
    return [torch.cat(chunks, dim=0) for chunks in all_states]
