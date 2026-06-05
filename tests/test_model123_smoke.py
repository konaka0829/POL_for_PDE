import subprocess
import sys
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.io import savemat

from test_utils import run_cli
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pol.model123_1d.datasets import DatasetConfig, build_dataset
from pol.model123_1d.experiments import ExperimentConfig, run_experiment
from pol.model123_1d.initial_conditions import evaluate_initial_conditions, sample_initial_condition_coefficients


def _write_tiny_model123_mat(path: Path, *, n: int = 10, s: int = 16, T: float = 0.02) -> None:
    x = np.linspace(0.0, 1.0, s, endpoint=False, dtype=np.float64)
    a = []
    u = []
    for idx in range(n):
        values = 0.1 * np.sin(2.0 * np.pi * (idx % 3 + 1) * x)
        a.append(values)
        u.append(values)
    savemat(
        path,
        {
            "a": np.asarray(a, dtype=np.float32),
            "u": np.asarray(u, dtype=np.float32),
            "T": np.asarray([[T]], dtype=np.float64),
            "dt": np.asarray([[0.01]], dtype=np.float64),
            "nu": np.asarray([[0.05]], dtype=np.float64),
        },
    )


def test_initial_conditions_are_reproducible_and_normalized():
    coeffs0 = sample_initial_condition_coefficients(4, seed=3)
    coeffs1 = sample_initial_condition_coefficients(4, seed=3)
    assert torch.allclose(coeffs0.a, coeffs1.a)
    assert torch.allclose(coeffs0.b, coeffs1.b)

    u0 = evaluate_initial_conditions(coeffs0, 64, device=torch.device("cpu"))
    amps = u0.abs().amax(dim=-1)
    assert torch.allclose(amps, 0.5 * torch.ones_like(amps), atol=1e-10, rtol=1e-10)


def test_dataset_builder_shapes():
    cfg = DatasetConfig(total_samples=18, ntrain=12, ntest=6, nx=64, dt=1e-2, fine_dt=2e-3, batch_size=3)
    bundle = build_dataset(cfg)
    assert bundle.u0_train.shape == (12, 64)
    assert bundle.y_train.shape == (12, 64)
    assert bundle.u0_test.shape == (6, 64)
    assert bundle.y_test.shape == (6, 64)


def test_dataset_builder_allows_empty_test_split():
    cfg = DatasetConfig(total_samples=3, ntrain=3, ntest=0, nx=32, dt=1e-2, fine_dt=2e-3, batch_size=3)
    bundle = build_dataset(cfg)
    assert bundle.u0_train.shape == (3, 32)
    assert bundle.y_train.shape == (3, 32)
    assert bundle.u0_test.shape == (0, 32)
    assert bundle.y_test.shape == (0, 32)


@pytest.mark.slow
@pytest.mark.parametrize("burgers_dealias", [0, 1])
def test_model123_defect_smoke_model1_consistent_d1(tmp_path, burgers_dealias):
    data_file = tmp_path / "tiny.mat"
    out_dir = tmp_path / f"model1_dealias_{burgers_dealias}"
    _write_tiny_model123_mat(data_file)
    cmd = [
        sys.executable,
        "model123_burgers_1d.py",
        "--model",
        "model1",
        "--reservoir",
        "burgers",
        "--data-file",
        str(data_file),
        "--train-split",
        "0.5",
        "--ntrain",
        "5",
        "--ntest",
        "5",
        "--T",
        "0.02",
        "--Ttilde",
        "0.02",
        "--dt",
        "0.01",
        "--batch-size",
        "5",
        "--burgers-fine-dt",
        "0.002",
        "--burgers-dealias",
        str(burgers_dealias),
        "--res-burgers-nu",
        "0.05",
        "--res-burgers-b",
        "1.0",
        "--compute-time-scaled-defect",
        "--defect-dtype",
        "float64",
        "--device",
        "cpu",
        "--out-dir",
        str(out_dir),
    ]
    proc = run_cli(cmd, cwd=REPO_ROOT, timeout=90)
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    for name in [
        "time_scaled_defect_metrics.json",
        "time_scaled_defect_per_sample.csv",
        "time_scaled_defect_per_sample.json",
        "error_vs_defect_scatter.png",
        "error_vs_defect_scatter.pdf",
        "error_vs_defect_scatter.svg",
    ]:
        assert (out_dir / name).exists()
    run_config = json.loads((out_dir / "run_config.json").read_text(encoding="utf-8"))
    metrics = json.loads((out_dir / "time_scaled_defect_metrics.json").read_text(encoding="utf-8"))
    for key in [
        "alpha",
        "delta_scale_rms_abs_l2h",
        "corr_error_delta_scale_pearson",
        "corr_error_delta_scale_spearman",
    ]:
        assert key in run_config
    assert metrics["max_abs_difference_model1_D1"] < 1e-4


@pytest.mark.slow
def test_model123_defect_smoke_model1_input_transform_delta_init(tmp_path):
    data_file = tmp_path / "tiny.mat"
    out_dir = tmp_path / "model1_transform"
    _write_tiny_model123_mat(data_file)
    cmd = [
        sys.executable,
        "model123_burgers_1d.py",
        "--model",
        "model1",
        "--reservoir",
        "burgers",
        "--data-file",
        str(data_file),
        "--train-split",
        "0.5",
        "--ntrain",
        "5",
        "--ntest",
        "5",
        "--T",
        "0.02",
        "--Ttilde",
        "0.02",
        "--dt",
        "0.01",
        "--batch-size",
        "5",
        "--burgers-fine-dt",
        "0.002",
        "--burgers-dealias",
        "0",
        "--input-scale",
        "2.0",
        "--res-burgers-nu",
        "0.05",
        "--res-burgers-b",
        "1.0",
        "--compute-time-scaled-defect",
        "--defect-dtype",
        "float64",
        "--device",
        "cpu",
        "--out-dir",
        str(out_dir),
    ]
    proc = run_cli(cmd, cwd=REPO_ROOT, timeout=90)
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    metrics = json.loads((out_dir / "time_scaled_defect_metrics.json").read_text(encoding="utf-8"))
    assert metrics["Delta_init"] > 0.0
    assert metrics["max_abs_difference_model1_D1"] < 1e-4


@pytest.mark.slow
def test_exact_burgers_inclusion_chain_smoke(tmp_path):
    cfg = ExperimentConfig(
        total_samples=18,
        ntrain=12,
        ntest=6,
        nx=64,
        dt=1e-2,
        fine_dt=2e-3,
        batch_size=3,
        obs="full",
        J=16,
        K=4,
        feature_times="0.25,0.5,0.75,1.0",
        reservoir="burgers",
        burgers_nu=0.05,
        model3_m=32,
        out_dir=str(tmp_path),
    )
    metrics = run_experiment(cfg)
    assert metrics["E1_train"] < 1e-8
    assert metrics["E2_train"] <= metrics["E1_train"] + 1e-8
    assert metrics["E3_train"] <= metrics["E2_train"] + 1e-8
    assert metrics["main_metric"] == "abs_l2h"
    assert "E1_train_abs_l2h" in metrics
    assert "E1_train_rel_l2h_mean" in metrics


@pytest.mark.slow
def test_cli_smoke(tmp_path):
    out_dir = tmp_path / "cli_out"
    cmd = [
        sys.executable,
        "scripts/run_model123_synthetic_study.py",
        "--total-samples",
        "18",
        "--ntrain",
        "12",
        "--ntest",
        "6",
        "--nx",
        "64",
        "--dt",
        "0.01",
        "--fine-dt",
        "0.002",
        "--batch-size",
        "3",
        "--obs",
        "full",
        "--J",
        "16",
        "--K",
        "4",
        "--feature-times",
        "0.25,0.5,0.75,1.0",
        "--reservoir",
        "reaction_diffusion",
        "--model3-m",
        "16",
        "--out-dir",
        str(out_dir),
    ]
    proc = run_cli(cmd, cwd=REPO_ROOT)
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    assert (out_dir / "metrics.json").exists()
