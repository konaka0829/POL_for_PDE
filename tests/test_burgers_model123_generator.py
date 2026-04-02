import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.io
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_generation.burgers.generate_burgers_1d import sample_periodic_grf
from pol.model123_1d.initial_conditions import sample_gaussian_random_field_initial_conditions


def run_generator(out_file: Path, seed: int, extra_args: list[str] | None = None) -> dict:
    cmd = [
        sys.executable,
        "data_generation/burgers/generate_burgers_1d_model123.py",
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
    if extra_args:
        cmd.extend(extra_args)
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    return scipy.io.loadmat(out_file)


def test_model123_generator_smoke_and_stats(tmp_path):
    payload = run_generator(tmp_path / "small.mat", seed=0)
    a = payload["a"]
    u = payload["u"]
    assert a.shape == (4, 64)
    assert u.shape == (4, 64)
    amps = np.max(np.abs(a), axis=1)
    assert np.allclose(amps, 0.5, atol=1e-5, rtol=1e-5)
    means = np.mean(a, axis=1)
    assert np.allclose(means, 0.0, atol=1e-5, rtol=1e-5)
    assert payload["am_coeff"].shape == (4, 8)
    assert payload["bm_coeff"].shape == (4, 8)


def test_model123_generator_is_reproducible(tmp_path):
    payload0 = run_generator(tmp_path / "run0.mat", seed=3)
    payload1 = run_generator(tmp_path / "run1.mat", seed=3)
    assert np.allclose(payload0["a"], payload1["a"])


def test_model123_generator_gaussian_rf_is_reproducible(tmp_path):
    extra_args = [
        "--initial-condition-type",
        "gaussian_rf",
        "--grf-gamma",
        "2.0",
        "--grf-tau",
        "5.0",
        "--grf-sigma",
        "25.0",
    ]
    payload0 = run_generator(tmp_path / "gauss0.mat", seed=7, extra_args=extra_args)
    payload1 = run_generator(tmp_path / "gauss1.mat", seed=7, extra_args=extra_args)
    assert np.allclose(payload0["a"], payload1["a"])
    assert payload0["am_coeff"].shape == (4, 0)
    assert payload0["bm_coeff"].shape == (4, 0)
    assert float(payload0["grf_gamma"][0, 0]) == 2.0
    assert float(payload0["grf_tau"][0, 0]) == 5.0
    assert float(payload0["grf_sigma"][0, 0]) == 25.0


def test_gaussian_rf_matches_shared_generator():
    kwargs = dict(
        num_samples=2,
        grid_size=64,
        gamma=2.0,
        tau=5.0,
        sigma=25.0,
        mean=0.0,
        seed=7,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    base = sample_periodic_grf(**kwargs)
    shared = sample_gaussian_random_field_initial_conditions(
        kwargs["num_samples"],
        kwargs["grid_size"],
        seed=kwargs["seed"],
        gamma=kwargs["gamma"],
        tau=kwargs["tau"],
        sigma=kwargs["sigma"],
        mean=kwargs["mean"],
        device=kwargs["device"],
        dtype=kwargs["dtype"],
    )
    assert torch.allclose(base, shared)


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
