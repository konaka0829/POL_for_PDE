import torch

from pol.burgers_spectral_1d import simulate_burgers_split_step
from pol.model123_1d import Model1Predictor1D, Model123Config
from pol.model123_1d.initial_conditions import evaluate_initial_conditions, sample_initial_condition_coefficients


def test_model1_uses_ttilde_even_when_feature_times_excludes_it():
    torch.manual_seed(0)
    x = torch.randn(3, 64, dtype=torch.float64)
    cfg = Model123Config(
        reservoir="burgers",
        Ttilde=0.1,
        dt=0.01,
        K=1,
        feature_times="0.05",
        input_scale=1.2,
        input_shift=-0.3,
        res_burgers_nu=0.05,
        res_burgers_b=1.0,
        burgers_scheme="split_step",
        burgers_fine_dt=0.002,
        burgers_dealias=False,
        device="cpu",
        dtype=torch.float64,
    )
    model = Model1Predictor1D(s=x.shape[1], config=cfg)

    pred = model.predict(x)

    z0 = model.feature_map.encode(x)
    expected = model.feature_map.reservoir.simulate(
        z0,
        dt=cfg.dt,
        Tr=cfg.Ttilde,
        obs_steps=[int(round(cfg.Ttilde / cfg.dt))],
    )[0]
    assert torch.allclose(pred, expected, atol=1e-10, rtol=1e-10)


def test_model1_same_burgers_is_exact_even_with_custom_feature_times():
    coeffs = sample_initial_condition_coefficients(5, seed=0, dtype=torch.float64)
    u0 = evaluate_initial_conditions(coeffs, 64, device=torch.device("cpu"), dtype=torch.float64)

    T = 0.1
    dt = 0.01
    fine_dt = 0.002
    nu = 0.05
    step = int(round(T / dt))

    target = simulate_burgers_split_step(
        u0,
        dt=dt,
        Tr=T,
        obs_steps=[step],
        nu=nu,
        fine_dt=fine_dt,
        b=1.0,
        dealias=False,
    )[0]

    cfg = Model123Config(
        reservoir="burgers",
        Ttilde=T,
        dt=dt,
        K=1,
        feature_times="0.05",
        res_burgers_nu=nu,
        res_burgers_b=1.0,
        burgers_scheme="split_step",
        burgers_fine_dt=fine_dt,
        burgers_dealias=False,
        device="cpu",
        dtype=torch.float64,
    )
    model = Model1Predictor1D(s=u0.shape[1], config=cfg)
    pred = model.predict(u0).cpu()

    assert torch.allclose(pred, target, atol=1e-8, rtol=1e-8)
