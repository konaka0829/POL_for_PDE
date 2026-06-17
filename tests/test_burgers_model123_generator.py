import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.io
import torch

from test_utils import run_cli
from pol.model123_1d.initial_conditions import sample_gaussian_random_field_initial_conditions


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_generator(out_file: Path, seed: int) -> dict:
    cmd = [
        sys.executable,
        "scripts/generate_burgers_1d.py",
        "--out-file",
        str(out_file),
        "--num-samples",
        "4",
        "--grid-size",
        "64",
        "--T",
        "0.1",
        "--dt",
        "0.01",
        "--fine-dt",
        "0.002",
        "--seed",
        str(seed),
        "--device",
        "cpu",
    ]
    proc = run_cli(cmd, cwd=REPO_ROOT)
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    return scipy.io.loadmat(out_file)


@pytest.mark.slow
def test_model123_generator_smoke_metadata_and_reproducibility(tmp_path):
    payload0 = run_generator(tmp_path / "run0.mat", seed=3)
    payload1 = run_generator(tmp_path / "run1.mat", seed=3)
    assert payload0["a"].shape == (4, 64)
    assert payload0["u"].shape == (4, 64)
    assert np.allclose(payload0["a"], payload1["a"])
    assert np.allclose(payload0["u"], payload1["u"])
    assert float(payload0["T"][0, 0]) == 0.1
    assert float(payload0["dt"][0, 0]) == 0.01
    assert int(payload0["nx"][0, 0]) == 64
    assert int(payload0["num_samples"][0, 0]) == 4


@pytest.mark.slow
def test_generator_pt_has_no_extra_sample_when_ntest_zero(tmp_path):
    out_file = tmp_path / "small.pt"
    cmd = [
        sys.executable,
        "scripts/generate_burgers_1d.py",
        "--out-file",
        str(out_file),
        "--format",
        "pt",
        "--num-samples",
        "3",
        "--grid-size",
        "32",
        "--T",
        "0.03",
        "--dt",
        "0.01",
        "--fine-dt",
        "0.002",
        "--batch-size",
        "3",
        "--device",
        "cpu",
    ]
    proc = run_cli(cmd, cwd=REPO_ROOT)
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    payload = torch.load(out_file, map_location="cpu", weights_only=False)
    assert payload["u0_train"].shape == (3, 32)
    assert payload["y_train"].shape == (3, 32)
    assert payload["u0_test"].shape == (0, 32)
    assert payload["y_test"].shape == (0, 32)


@pytest.mark.slow
def test_generator_mat_sample_count_matches_request(tmp_path):
    payload = run_generator(tmp_path / "small.mat", seed=5)
    assert payload["a"].shape[0] == 4
    assert payload["u"].shape[0] == 4


def test_gaussian_rf_zero_mean_matches_matlab_periodic_convention():
    u0 = sample_gaussian_random_field_initial_conditions(
        3,
        64,
        seed=11,
        gamma=2.0,
        tau=5.0,
        sigma=25.0,
        mean=0.0,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    means = u0.mean(dim=-1)
    assert torch.allclose(means, torch.zeros_like(means), atol=1e-10, rtol=1e-10)


def test_generator_domain_length_is_saved_in_pt_and_mat_metadata(tmp_path):
    common = [
        sys.executable,
        "scripts/generate_burgers_1d.py",
        "--total-samples",
        "3",
        "--ntrain",
        "1",
        "--nval",
        "1",
        "--ntest",
        "1",
        "--grid-size",
        "8",
        "--T",
        "0.01",
        "--dt",
        "0.01",
        "--fine-dt",
        "0.01",
        "--domain-length",
        "2.0",
        "--batch-size",
        "3",
        "--device",
        "cpu",
    ]
    pt_path = tmp_path / "domain.pt"
    mat_path = tmp_path / "domain.mat"
    proc = run_cli([*common, "--format", "pt", "--out-file", str(pt_path)], cwd=REPO_ROOT)
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    proc = run_cli([*common, "--format", "mat", "--out-file", str(mat_path)], cwd=REPO_ROOT)
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr

    pt_payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    assert pt_payload["metadata"]["domain_length"] == pytest.approx(2.0)
    assert pt_payload["metadata"]["ic_coordinate_convention"] == "normalized_periodic_coordinate_x_over_L"
    mat_payload = scipy.io.loadmat(mat_path)
    assert float(mat_payload["domain_length"][0, 0]) == pytest.approx(2.0)
    assert str(mat_payload["ic_coordinate_convention"][0]) == "normalized_periodic_coordinate_x_over_L"
