import torch

from pol.elm import FixedRandomELM
from pol.features_1d import build_sensor_indices, collect_observations, flatten_observations
from pol.reservoir_1d import Reservoir1DSolver, ReservoirConfig
from pol.ridge import fit_ridge_streaming, predict_linear


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
