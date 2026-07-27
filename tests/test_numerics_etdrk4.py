import torch

from pol.numerics.etdrk4 import (
    cox_matthews_coefficients,
    cox_matthews_etdrk4_step,
    simulate_burgers_etdrk4,
)


def dataset_abs_l2h_rmse(
    prediction: torch.Tensor, reference: torch.Tensor
) -> torch.Tensor:
    return torch.sqrt(torch.mean((prediction - reference) ** 2))


def test_l_zero_etdrk4_matches_classical_rk4_one_step():
    y0 = torch.tensor([0.2], dtype=torch.float64)
    dt = 0.05

    def nonlinear(y):
        return y * y

    got = cox_matthews_etdrk4_step(y0, L=torch.zeros_like(y0), dt=dt, nonlinear=nonlinear)
    k1 = nonlinear(y0)
    k2 = nonlinear(y0 + 0.5 * dt * k1)
    k3 = nonlinear(y0 + 0.5 * dt * k2)
    k4 = nonlinear(y0 + dt * k3)
    expected = y0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    assert torch.allclose(got, expected, atol=1e-14, rtol=1e-14)


def test_coefficients_have_rk4_l_zero_limits():
    L = torch.zeros(3, dtype=torch.float64)
    dt = 0.125
    _, _, Q, f1, f2, f3 = cox_matthews_coefficients(L, dt)
    assert torch.allclose(Q, torch.full_like(Q, dt / 2.0))
    assert torch.allclose(f1, torch.full_like(f1, dt / 6.0))
    assert torch.allclose(2.0 * f2, torch.full_like(f2, dt / 3.0))
    assert torch.allclose(f3, torch.full_like(f3, dt / 6.0))


def test_linear_n_zero_is_exact_exponential_step():
    v0 = torch.tensor([1.0, 2.0], dtype=torch.float64)
    L = torch.tensor([-0.5, -2.0], dtype=torch.float64)
    dt = 0.1
    got = cox_matthews_etdrk4_step(v0, L=L, dt=dt, nonlinear=lambda v: torch.zeros_like(v))
    assert torch.allclose(got, torch.exp(dt * L) * v0, atol=1e-14, rtol=1e-14)


def test_burgers_etdrk4_smoke_is_finite():
    x = torch.linspace(0.0, 1.0, 33, dtype=torch.float64)[:-1]
    u0 = torch.sin(2.0 * torch.pi * x).unsqueeze(0)
    out = simulate_burgers_etdrk4(u0, nu=0.01, T=0.02, dt=0.005)
    assert out.shape == u0.shape
    assert torch.isfinite(out).all()


def test_burgers_dt_refinement_improves_against_reference():
    x = torch.linspace(0.0, 1.0, 65, dtype=torch.float64)[:-1]
    u0 = (0.4 * torch.sin(2.0 * torch.pi * x)).unsqueeze(0)
    ref = simulate_burgers_etdrk4(u0, nu=0.05, T=0.05, dt=0.0005)
    coarse = simulate_burgers_etdrk4(u0, nu=0.05, T=0.05, dt=0.01)
    fine = simulate_burgers_etdrk4(u0, nu=0.05, T=0.05, dt=0.005)
    assert dataset_abs_l2h_rmse(fine, ref) < dataset_abs_l2h_rmse(coarse, ref)
