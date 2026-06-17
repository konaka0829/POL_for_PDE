import subprocess
import sys
import json
from pathlib import Path

import pytest
import scipy.io
import torch

from test_utils import run_cli
from pol.model123_1d import (
    Model1Predictor1D,
    Model2Regressor1D,
    Model3Regressor1D,
    Model123Config,
)
from pol.features_1d import build_time_grid
from pol.time_grid import require_time_aligned

REPO_ROOT = Path(__file__).resolve().parents[1]


def make_data(num_samples: int = 8, s: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.randn(num_samples, s)
    grid = torch.linspace(0.0, 1.0, s + 1)[:-1].unsqueeze(0)
    y = 0.7 * torch.roll(x, shifts=2, dims=1) + 0.2 * x.pow(2) + 0.1 * torch.sin(2.0 * torch.pi * grid)
    return x, y


def make_loader(x: torch.Tensor, y: torch.Tensor, batch_size: int = 4):
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=False,
    )


def base_config(**kwargs) -> Model123Config:
    cfg = Model123Config(
        reservoir="reaction_diffusion",
        Ttilde=0.1,
        dt=1e-2,
        K=3,
        feature_times="",
        obs="full",
        J=16,
        ridge_lambda=1e-4,
        elm_hidden_dim=32,
        device="cpu",
        dtype=torch.float32,
    )
    for key, value in kwargs.items():
        cfg = Model123Config(**{**cfg.__dict__, key: value})
    return cfg


def test_model1_smoke():
    x, _ = make_data()
    model = Model1Predictor1D(s=x.shape[1], config=base_config())
    pred = model.predict(x[:3])
    assert pred.shape == (3, x.shape[1])
    assert torch.isfinite(pred).all()


def test_model2_smoke():
    x, y = make_data()
    model = Model2Regressor1D(s=x.shape[1], config=base_config(obs="points", J=12))
    model.fit(make_loader(x, y))
    pred = model.predict(x[:4])
    assert pred.shape == (4, x.shape[1])
    assert torch.isfinite(pred).all()


def test_model3_smoke():
    x, y = make_data()
    model = Model3Regressor1D(s=x.shape[1], config=base_config(obs="fourier", J=8))
    model.fit(make_loader(x, y))
    pred = model.predict(x[:4])
    assert pred.shape == (4, x.shape[1])
    assert torch.isfinite(pred).all()


def test_model2_contains_model1_via_last_full_observation():
    x, _ = make_data()
    cfg = base_config(obs="full", K=4)
    model1 = Model1Predictor1D(s=x.shape[1], config=cfg)
    model2 = Model2Regressor1D(s=x.shape[1], config=cfg)
    phi = model2.features(x[:2])
    last_obs = model2.observation.select_last_observation(phi)
    pred1 = model1.predict(x[:2]).cpu()
    assert torch.allclose(last_obs.cpu(), pred1, atol=1e-6, rtol=1e-6)


def test_model3_contains_model2_via_skip_block():
    x, _ = make_data()
    cfg = base_config(obs="points", J=10)
    model3 = Model3Regressor1D(s=x.shape[1], config=cfg)
    phi = model3.phi(x[:2])
    aug = model3.augment_features(phi)
    assert aug.shape[1] == phi.shape[1] + cfg.elm_hidden_dim
    assert torch.allclose(aug[:, : phi.shape[1]], phi, atol=1e-7, rtol=1e-7)


def test_require_time_aligned_accepts_integer_step():
    assert require_time_aligned(0.05, 0.01) == 5


def test_require_time_aligned_rejects_non_grid_time():
    with pytest.raises(ValueError, match="not aligned"):
        require_time_aligned(0.055, 0.01)


def test_model1_rejects_non_grid_ttilde():
    x, _ = make_data(num_samples=2, s=32)
    with pytest.raises(ValueError, match="not aligned"):
        model = Model1Predictor1D(s=x.shape[1], config=base_config(Ttilde=0.055, dt=0.01))
        model.predict(x)


def test_build_time_grid_rejects_explicit_non_grid_time():
    with pytest.raises(ValueError, match="not aligned"):
        build_time_grid(Tr=0.05, dt=0.01, K=1, feature_times="0.015")


def test_build_time_grid_rejects_non_grid_tr_with_explicit_times():
    with pytest.raises(ValueError, match="not aligned"):
        build_time_grid(Tr=0.055, dt=0.01, K=1, feature_times="0.01,0.03")


def test_build_time_grid_automatic_uses_integer_steps():
    times, steps = build_time_grid(Tr=0.05, dt=0.01, K=3, feature_times="")
    assert steps == [2, 3, 5]
    assert all(require_time_aligned(t, 0.01) == step for t, step in zip(times, steps))


def test_build_time_grid_automatic_spaces_positive_times_evenly():
    times, steps = build_time_grid(Tr=1.0, dt=0.001, K=4, feature_times="")
    assert steps == [250, 500, 750, 1000]
    assert times == pytest.approx([0.25, 0.5, 0.75, 1.0])


def test_model2_progress_output(capsys):
    x, y = make_data()
    model = Model2Regressor1D(s=x.shape[1], config=base_config(obs="points", J=12))
    model.fit(
        make_loader(x, y),
        progress_fn=lambda batch_idx, total_batches: print(
            "[model2 train] batch %d/%d" % (batch_idx, total_batches),
            flush=True,
        ),
        progress_label="model2 train",
    )
    captured = capsys.readouterr()
    assert "[model2 train] ridge feature accumulation start" in captured.out
    assert "[model2 train] batch" in captured.out
    assert "[model2 train] ridge solve done" in captured.out


def test_model123_cli_records_domain_length_dx_and_legacy_lambda(tmp_path):
    nx = 8
    data_file = tmp_path / "split_l2.pt"
    torch.save(
        {
            "u0_train": torch.zeros(2, nx),
            "y_train": torch.zeros(2, nx),
            "u0_val": torch.zeros(1, nx),
            "y_val": torch.zeros(1, nx),
            "u0_test": torch.zeros(1, nx),
            "y_test": torch.zeros(1, nx),
            "metadata": {
                "T": 0.1,
                "dt": 0.01,
                "target_nu": 0.01,
                "nx": nx,
                "domain_length": 2.0,
                "ic_coordinate_convention": "normalized_periodic_coordinate_x_over_L",
            },
        },
        data_file,
    )
    out_dir = tmp_path / "cli_l2"
    cmd = [
        sys.executable,
        "model123_burgers_1d.py",
        "--model",
        "model2",
        "--reservoir",
        "static",
        "--data-file",
        str(data_file),
        "--ntrain",
        "2",
        "--nval",
        "1",
        "--ntest",
        "1",
        "--batch-size",
        "2",
        "--T",
        "0.1",
        "--dt",
        "0.01",
        "--target-nu",
        "0.01",
        "--ridge-zeta",
        "1e-6",
        "--out-dir",
        str(out_dir),
        "--device",
        "cpu",
    ]
    proc = run_cli(cmd, cwd=REPO_ROOT)
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    payload = json.loads((out_dir / "run_config.json").read_text(encoding="utf-8"))
    assert payload["domain_length"] == 2.0
    assert payload["grid"]["dx"] == 2.0 / nx
    assert payload["metrics"]["dx"] == 2.0 / nx
    assert payload["readout"]["dx"] == 2.0 / nx
    assert payload["dataset_metadata"]["ic_coordinate_convention"] == "normalized_periodic_coordinate_x_over_L"
    assert payload["effective_code_lambda_legacy_equivalent"] == pytest.approx(2 * 1e-6 / (2.0 / nx))


@pytest.mark.slow
def test_model123_cli_smoke(tmp_path):
    x, y = make_data(num_samples=8, s=64)
    data_file = tmp_path / "small.mat"
    scipy.io.savemat(data_file, {"a": x.numpy(), "u": y.numpy()})
    out_dir = tmp_path / "cli_out"
    cmd = [
        sys.executable,
        "model123_burgers_1d.py",
        "--model",
        "model2",
        "--reservoir",
        "reaction_diffusion",
        "--data-mode",
        "single_split",
        "--data-file",
        str(data_file),
        "--train-split",
        "0.5",
        "--ntrain",
        "4",
        "--ntest",
        "4",
        "--batch-size",
        "2",
        "--sub",
        "1",
        "--T",
        "0.1",
        "--Ttilde",
        "0.1",
        "--dt",
        "0.01",
        "--K",
        "3",
        "--obs",
        "points",
        "--J",
        "16",
        "--out-dir",
        str(out_dir),
    ]
    proc = run_cli(cmd, cwd=REPO_ROOT)
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    assert (out_dir / "run_config.json").exists()
    payload = json.loads((out_dir / "run_config.json").read_text(encoding="utf-8"))
    assert "train_absL2h" in payload
    assert "test_absL2h" in payload
    assert "train_relL2" in payload
    assert "test_relL2" in payload
    assert "[model2 train]" in proc.stdout
    assert "[model2 eval-test]" in proc.stdout
