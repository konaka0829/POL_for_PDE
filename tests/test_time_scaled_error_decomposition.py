import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from test_utils import run_cli
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pol.model123_1d.error_decomposition import (
    ErrorDecompositionConfig,
    aggregate_metric_rows,
    interpolate_trajectory_at_times,
    run_error_decomposition,
)


def test_summary_schema_uses_delta_scale_and_theorem_rhs():
    cfg = ErrorDecompositionConfig(
        num_samples=6,
        nx=32,
        seed=0,
        batch_size=3,
        target_nu=0.05,
        T=0.1,
        Ttilde_values=[0.05, 0.1],
        dt=0.01,
        fine_dt=0.002,
        reservoir="reaction_diffusion",
        beta_mode="zero",
        dtype="float64",
        device="cpu",
    )
    result = run_error_decomposition(cfg)
    assert "beta_details_by_ttilde" in result
    for row in result["summary_rows"]:
        assert "alpha" in row
        assert "Delta_scale" in row
        assert "Delta_time" not in row
        assert abs(row["rhs_beta"] - math.sqrt(cfg.T) * row["Delta_scale"]) < 1e-10
        assert abs(row["rhs_beta0"] - row["rhs_beta"]) < 1e-12


def test_same_burgers_alpha_one_has_near_zero_scaled_defect():
    cfg = ErrorDecompositionConfig(
        num_samples=6,
        nx=32,
        seed=1,
        batch_size=3,
        target_nu=0.05,
        T=0.1,
        Ttilde_values=[0.1],
        dt=0.01,
        fine_dt=0.002,
        reservoir="burgers",
        res_burgers_nu=0.05,
        res_burgers_b=1.0,
        dtype="float64",
        device="cpu",
    )
    result = run_error_decomposition(cfg)
    summary = aggregate_metric_rows(result["rows"])
    assert summary[0]["D1"] < 1e-10
    assert summary[0]["Delta_init"] < 1e-10
    assert summary[0]["Delta_scale"] < 1e-10


def test_ttilde_not_equal_t_uses_rescaled_trajectory_not_native_time():
    dt = 0.1
    states = torch.arange(0, 6, dtype=torch.float64).reshape(6, 1, 1)
    query = torch.tensor([0.0, 0.15, 0.3], dtype=torch.float64)
    got = interpolate_trajectory_at_times(states, query, dt=dt).reshape(-1)
    assert torch.allclose(got, torch.tensor([0.0, 1.5, 3.0], dtype=torch.float64))

    cfg = ErrorDecompositionConfig(
        num_samples=4,
        nx=32,
        seed=2,
        batch_size=2,
        target_nu=0.05,
        T=0.1,
        Ttilde_values=[0.2],
        dt=0.01,
        fine_dt=0.002,
        reservoir="burgers",
        res_burgers_nu=0.025,
        res_burgers_b=0.5,
        beta_mode="zero",
        dtype="float64",
        device="cpu",
    )
    result = run_error_decomposition(cfg)
    row = result["summary_rows"][0]
    assert row["alpha"] == 2.0
    assert row["Delta_scale"] < 1e-10


@pytest.mark.slow
def test_end_to_end_cli_smoke_uses_scaled_output_names(tmp_path):
    out_dir = tmp_path / "error_decomp"
    cmd = [
        sys.executable,
        "model1_error_decomposition_1d.py",
        "--num-samples",
        "4",
        "--nx",
        "32",
        "--batch-size",
        "2",
        "--T",
        "0.1",
        "--ttilde-values",
        "0.05,0.1",
        "--dt",
        "0.01",
        "--fine-dt",
        "0.002",
        "--reservoir",
        "reaction_diffusion",
        "--beta-mode",
        "fixed",
        "--beta-fixed",
        "0.1",
        "--device",
        "cpu",
        "--out-dir",
        str(out_dir),
    ]
    proc = run_cli(cmd, cwd=REPO_ROOT)
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    assert "Delta_scale" in proc.stdout
    assert "Delta_time" not in proc.stdout
    assert (out_dir / "per_sample_metrics.csv").exists()
    assert (out_dir / "summary_metrics.json").exists()
    assert (out_dir / "delta_scale_vs_ttilde.png").exists()
    assert (out_dir / "scaled_bound_vs_ttilde.png").exists()
    assert (out_dir / "scaled_bound_scatter.png").exists()
