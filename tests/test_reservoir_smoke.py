import subprocess
import sys
from pathlib import Path

import pytest
import torch

from test_utils import run_cli
from pol.elm import FixedRandomELM
from pol.features_1d import build_sensor_indices, collect_observations, flatten_observations
from pol.reservoir_1d import Reservoir1DSolver, ReservoirConfig
from pol.ridge import (
    fit_ridge_streaming,
    fit_ridge_streaming_standardized,
    predict_linear,
)


def test_reservoir_feature_and_ridge_smoke():
    b, s = 6, 128
    x = torch.randn(b, s)
    y = torch.randn(b, s)

    solver = Reservoir1DSolver(ReservoirConfig(reservoir="reaction_diffusion"))
    obs_steps = [2, 4, 6]
    states = solver.simulate(x, dt=1e-2, Tr=0.1, obs_steps=obs_steps)

    sensor_idx = build_sensor_indices(s, obs="points", J=16, sensor_mode="equispaced", sensor_seed=0)
    obs_list = collect_observations(states, obs="points", sensor_idx=sensor_idx)
    phi = flatten_observations(obs_list)
    assert phi.shape == (b, len(obs_steps) * 16)
    assert torch.isfinite(phi).all()

    elm = FixedRandomELM(in_dim=phi.shape[1], hidden_dim=32, seed=0)

    def feature_fn(x_batch):
        states_local = solver.simulate(x_batch, dt=1e-2, Tr=0.1, obs_steps=obs_steps)
        obs_local = collect_observations(states_local, obs="points", sensor_idx=sensor_idx)
        phi_local = flatten_observations(obs_local)
        return elm(phi_local)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y),
        batch_size=3,
        shuffle=False,
    )
    state = fit_ridge_streaming(loader, feature_fn, ridge_lambda=1e-3, dtype=torch.float64)
    feat = feature_fn(x).to(dtype=state["W"].dtype)
    pred = predict_linear(feat, state["W"])

    assert pred.shape == (b, s)
    assert torch.isfinite(pred).all()


def test_fourier_observation_shapes():
    b, s = 4, 128
    j = 8
    obs_steps = [2, 4, 6]
    x = torch.randn(b, s)
    solver = Reservoir1DSolver(ReservoirConfig(reservoir="reaction_diffusion"))
    states = solver.simulate(x, dt=1e-2, Tr=0.1, obs_steps=obs_steps)

    sensor_idx = build_sensor_indices(s, obs="fourier", J=j, sensor_mode="equispaced", sensor_seed=0)
    obs_list = collect_observations(states, obs="fourier", sensor_idx=sensor_idx)

    assert sensor_idx.dtype == torch.long
    assert sensor_idx.shape == (j,)
    assert obs_list[0].shape == (b, 2 * j)
    phi = flatten_observations(obs_list)
    assert phi.shape == (b, len(obs_steps) * 2 * j)
    assert torch.isfinite(phi).all()


def test_proj_observation_shapes():
    b, s = 4, 128
    j = 8
    obs_steps = [1, 3, 5]
    x = torch.randn(b, s)
    solver = Reservoir1DSolver(ReservoirConfig(reservoir="reaction_diffusion"))
    states = solver.simulate(x, dt=1e-2, Tr=0.1, obs_steps=obs_steps)

    sensor_idx = build_sensor_indices(s, obs="proj", J=j, sensor_mode="equispaced", sensor_seed=1)
    obs_list = collect_observations(states, obs="proj", sensor_idx=sensor_idx)

    assert sensor_idx.dtype.is_floating_point
    assert sensor_idx.shape == (j, s)
    assert obs_list[0].shape == (b, j)
    phi = flatten_observations(obs_list)
    assert phi.shape == (b, len(obs_steps) * j)
    assert torch.isfinite(phi).all()


def test_ridge_standardized_matches_raw_prediction():
    torch.manual_seed(0)
    n, in_dim, out_dim = 20, 16, 7
    x = torch.randn(n, in_dim)
    y = torch.randn(n, out_dim)
    eps = 1e-6

    def feature_fn(x_batch):
        return x_batch[:, :10]

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y),
        batch_size=5,
        shuffle=False,
    )
    state = fit_ridge_streaming_standardized(
        loader,
        feature_fn,
        ridge_lambda=1e-4,
        dtype=torch.float64,
        eps=eps,
    )

    phi = feature_fn(x).to(dtype=state["W"].dtype)
    pred_raw = predict_linear(phi, state["W"])
    mean = state["mean"]
    std = state["std"]
    phi_std = (phi - mean) / (std + eps)
    pred_std = predict_linear(phi_std, state["W_std"])

    assert torch.allclose(pred_raw, pred_std, atol=1e-8, rtol=1e-6)


def test_reservoir_forcing_changes_trajectory():
    b, s = 3, 128
    z0 = torch.randn(b, s)
    forcing = 0.5 * torch.randn(b, s)
    obs_steps = [2, 4, 6]

    solver = Reservoir1DSolver(ReservoirConfig(reservoir="reaction_diffusion"))
    states_no = solver.simulate(z0, dt=1e-2, Tr=0.1, obs_steps=obs_steps)
    states_force = solver.simulate(
        z0,
        dt=1e-2,
        Tr=0.1,
        obs_steps=obs_steps,
        forcing=forcing,
        forcing_steps=None,
    )

    diff = 0.0
    for a, b_state in zip(states_no, states_force):
        diff += torch.linalg.norm(a - b_state).item()
    assert diff > 0.0


def test_burgers_split_step_smoke():
    b, s = 4, 128
    z0 = torch.randn(b, s)
    obs_steps = [1, 2, 3]
    solver = Reservoir1DSolver(
        ReservoirConfig(
            reservoir="burgers",
            burgers_scheme="split_step",
            burgers_fine_dt=1e-3,
            burgers_dealias=True,
        )
    )
    states = solver.simulate(z0, dt=1e-2, Tr=0.1, obs_steps=obs_steps)
    assert len(states) == len(obs_steps)
    assert states[0].shape == (b, s)
    assert torch.isfinite(torch.stack(states, dim=0)).all()


def test_burgers_scheme_coexistence():
    b, s = 2, 64
    z0 = torch.randn(b, s)
    obs_steps = [1, 2]
    solver_semi = Reservoir1DSolver(
        ReservoirConfig(reservoir="burgers", burgers_scheme="semi_implicit")
    )
    solver_split = Reservoir1DSolver(
        ReservoirConfig(
            reservoir="burgers",
            burgers_scheme="split_step",
            burgers_fine_dt=2e-3,
        )
    )
    states_semi = solver_semi.simulate(z0, dt=1e-2, Tr=0.1, obs_steps=obs_steps)
    states_split = solver_split.simulate(z0, dt=1e-2, Tr=0.1, obs_steps=obs_steps)
    assert len(states_semi) == len(obs_steps)
    assert len(states_split) == len(obs_steps)
    assert torch.isfinite(torch.stack(states_semi, dim=0)).all()
    assert torch.isfinite(torch.stack(states_split, dim=0)).all()


def test_reservoir_wavenumbers_scale_with_domain_length():
    solver_l1 = Reservoir1DSolver(ReservoirConfig(domain_length=1.0))
    solver_l2 = Reservoir1DSolver(ReservoirConfig(domain_length=2.0))
    k1 = solver_l1._wavenumbers(16, torch.device("cpu"), torch.float64)
    k2 = solver_l2._wavenumbers(16, torch.device("cpu"), torch.float64)
    assert torch.allclose(k2, 0.5 * k1)


def test_heat_surrogate_single_mode_uses_domain_length():
    s = 64
    domain_length = 2.0
    nu = 0.1
    t = 0.2
    x = torch.arange(s, dtype=torch.float64) * (domain_length / s)
    z0 = torch.sin(2.0 * torch.pi * x / domain_length).unsqueeze(0)
    solver = Reservoir1DSolver(ReservoirConfig(reservoir="heat", heat_nu=nu, domain_length=domain_length))
    out = solver.simulate(z0, dt=t, Tr=t, obs_steps=[1])[0]
    expected_amp = torch.exp(torch.tensor(-nu * (2.0 * torch.pi / domain_length) ** 2 * t, dtype=torch.float64))
    assert torch.allclose(out, expected_amp * z0, atol=1e-8, rtol=1e-8)


def test_advection_surrogate_single_mode_uses_domain_length():
    s = 64
    domain_length = 2.0
    c = 0.25
    t = 0.2
    x = torch.arange(s, dtype=torch.float64) * (domain_length / s)
    z0 = torch.sin(2.0 * torch.pi * x / domain_length).unsqueeze(0)
    solver = Reservoir1DSolver(ReservoirConfig(reservoir="advection", advection_c=c, domain_length=domain_length))
    out = solver.simulate(z0, dt=t, Tr=t, obs_steps=[1])[0]
    expected = torch.sin(2.0 * torch.pi * (x - c * t) / domain_length).unsqueeze(0)
    assert torch.allclose(out, expected, atol=1e-8, rtol=1e-8)


@pytest.mark.slow
def test_dataset_generator_smoke(tmp_path):
    out_file = tmp_path / "burgers_small.mat"
    repo_root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        "scripts/generate_burgers_1d.py",
        "--out-file",
        str(out_file),
        "--num-samples",
        "4",
        "--grid-size",
        "64",
        "--nu",
        "0.1",
        "--T",
        "0.05",
        "--dt",
        "0.01",
        "--fine-dt",
        "0.002",
        "--batch-size",
        "2",
        "--device",
        "cpu",
    ]
    proc = run_cli(cmd, cwd=repo_root)
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    assert out_file.exists()

    from pol.io_mat import MatReader

    reader = MatReader(str(out_file))
    a = reader.read_field("a")
    u = reader.read_field("u")
    assert tuple(a.shape) == (4, 64)
    assert tuple(u.shape) == (4, 64)
