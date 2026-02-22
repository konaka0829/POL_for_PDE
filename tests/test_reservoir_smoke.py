import torch

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
