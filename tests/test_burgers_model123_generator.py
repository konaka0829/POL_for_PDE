import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.io
import torch

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
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    return scipy.io.loadmat(out_file)


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
