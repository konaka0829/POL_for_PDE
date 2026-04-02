import subprocess
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pol.model123_1d.datasets import DatasetConfig, build_dataset
from pol.model123_1d.experiments import ExperimentConfig, run_experiment
from pol.model123_1d.initial_conditions import evaluate_initial_conditions, sample_initial_condition_coefficients


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


def test_cli_smoke(tmp_path):
    out_dir = tmp_path / "cli_out"
    cmd = [
        sys.executable,
        "model123_error_study.py",
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
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    assert (out_dir / "metrics.json").exists()
