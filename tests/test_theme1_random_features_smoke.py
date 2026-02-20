import torch

from pol.features_1d import build_sensor_indices
from pol.reservoir_1d import Reservoir1DSolver
from pol.ridge import fit_ridge_streaming, predict_linear
from pol.theme1_random_features_1d import (
    RandomReservoirFeatureMap1D,
    sample_reservoir_configs,
)


def test_theme1_random_features_smoke():
    b, s = 6, 128
    r = 4
    h_per = 12
    x = torch.randn(b, s)
    y = torch.randn(b, s)

    obs_steps = [2, 4, 6]
    sensor_idx = build_sensor_indices(
        s=s,
        obs="points",
        J=16,
        sensor_mode="equispaced",
        sensor_seed=0,
    )
    theta_cfgs = sample_reservoir_configs(
        reservoir="reaction_diffusion",
        R=r,
        theta_seed=0,
    )
    solvers = [Reservoir1DSolver(cfg) for cfg in theta_cfgs]
    fmap = RandomReservoirFeatureMap1D(
        solvers=solvers,
        obs_steps=obs_steps,
        obs="points",
        sensor_idx=sensor_idx,
        Tr=0.1,
        dt=1e-2,
        use_elm=True,
        elm_mode="per_reservoir",
        elm_h_per=h_per,
        elm_seed=0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    feat = fmap(x)
    assert feat.shape == (b, r * h_per)
    assert torch.isfinite(feat).all()

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y),
        batch_size=3,
        shuffle=False,
    )
    ridge_state = fit_ridge_streaming(
        loader,
        fmap,
        ridge_lambda=1e-3,
        dtype=torch.float64,
    )
    pred = predict_linear(fmap(x).to(dtype=torch.float64), ridge_state["W"])
    assert pred.shape == (b, s)
    assert torch.isfinite(pred).all()
