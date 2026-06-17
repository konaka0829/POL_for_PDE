# Model 1 の time-scaled error decomposition が、理論どおり Delta_scale を中心に実装されているかを確認
import math
import sys
from pathlib import Path

import pytest
import torch

from test_utils import run_cli
# リポジトリのpathをREPO_ROOT に保存
REPO_ROOT = Path(__file__).resolve().parents[1]
# リポジトリrootがimport探索パスに入っていない場合，先頭に入れる
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pol.model123_1d.error_decomposition import (
    ErrorDecompositionConfig,
    aggregate_metric_rows,
    compute_beta,
    compute_time_scaled_defect_for_dataset,
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
        assert row["rhs_beta_legacy_alias_of"] == "rhs_beta_theorem_components"
        assert "rhs_beta_theorem_components" in row
        assert "rhs_beta_pathwise_rms" in row


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


def test_dataset_defect_helper_returns_pathwise_integrated_delta_scale():
    cfg = ErrorDecompositionConfig(
        num_samples=3,
        nx=16,
        batch_size=2,
        target_nu=0.05,
        T=0.04,
        Ttilde_values=[0.04],
        dt=0.02,
        fine_dt=0.002,
        reservoir="burgers",
        res_burgers_nu=0.05,
        res_burgers_b=1.0,
        beta_mode="zero",
        dtype="float64",
        device="cpu",
    )
    u0 = torch.zeros(3, 16, dtype=torch.float64)
    target_T = torch.zeros(3, 16, dtype=torch.float64)
    result = compute_time_scaled_defect_for_dataset(u0=u0, target_T=target_T, cfg=cfg)
    rows = result["rows"]
    assert len(rows) == 3
    assert "delta_scale_pathwise_abs_l2h" in rows[0]
    values = torch.tensor([row["delta_scale_pathwise_abs_l2h"] for row in rows], dtype=torch.float64)
    expected_rms = float(torch.sqrt(torch.mean(values.pow(2))).item())
    assert result["summary"]["delta_scale_rms_abs_l2h"] == pytest.approx(expected_rms)
    assert result["summary"]["defect_metric"] == "pathwise_integrated_time_scaled_generator_defect"
    assert result["summary"]["Delta_init"] == pytest.approx(0.0, abs=1e-12)
    assert result["summary"]["rhs_beta_legacy_alias_of"] == "rhs_beta_pathwise_rms"
    assert "rhs_beta_theorem_components" in result["summary"]


def test_theorem_components_and_pathwise_rms_differ_for_nonproportional_terms():
    rows = [
        {
            "T": 1.0,
            "Ttilde": 1.0,
            "alpha": 1.0,
            "D1_model1_abs_l2h": 0.0,
            "Delta_init_abs_l2h": 1.0,
            "delta_scale_pathwise_abs_l2h": 0.0,
            "rhs_beta_pathwise_abs_l2h": 1.0,
            "rhs_beta0_pathwise_abs_l2h": 1.0,
            "beta_mode": "fixed",
            "beta_value": 0.0,
            "beta_empirical": 0.0,
            "c_beta_T": 1.0,
        },
        {
            "T": 1.0,
            "Ttilde": 1.0,
            "alpha": 1.0,
            "D1_model1_abs_l2h": 0.0,
            "Delta_init_abs_l2h": 0.0,
            "delta_scale_pathwise_abs_l2h": 1.0,
            "rhs_beta_pathwise_abs_l2h": 1.0,
            "rhs_beta0_pathwise_abs_l2h": 1.0,
            "beta_mode": "fixed",
            "beta_value": 0.0,
            "beta_empirical": 0.0,
            "c_beta_T": 1.0,
        },
    ]
    from pol.model123_1d.error_decomposition import _summary_for_dataset_rows

    summary = _summary_for_dataset_rows(rows)
    assert summary["rhs_beta_theorem_components"] != pytest.approx(summary["rhs_beta_pathwise_rms"])


def test_error_decomposition_config_contains_solver_and_input_transform_fields():
    cfg = ErrorDecompositionConfig()
    assert cfg.burgers_scheme == "split_step"
    assert cfg.burgers_dealias is False
    assert cfg.input_scale == 1.0
    assert cfg.input_shift == 0.0


def test_dataset_defect_helper_input_transform_sets_delta_init():
    cfg = ErrorDecompositionConfig(
        num_samples=2,
        nx=16,
        batch_size=2,
        target_nu=0.05,
        T=0.04,
        Ttilde_values=[0.04],
        dt=0.02,
        fine_dt=0.002,
        reservoir="burgers",
        res_burgers_nu=0.05,
        res_burgers_b=1.0,
        input_scale=2.0,
        input_shift=0.1,
        beta_mode="zero",
        dtype="float64",
        device="cpu",
    )
    x = torch.linspace(0.0, 1.0, 16, dtype=torch.float64)
    u0 = torch.stack([torch.sin(2.0 * math.pi * x), torch.cos(2.0 * math.pi * x)], dim=0)
    result = compute_time_scaled_defect_for_dataset(u0=u0, target_T=torch.zeros_like(u0), cfg=cfg)
    assert result["summary"]["Delta_init"] > 0.0
    assert all(row["Delta_init_abs_l2h"] > 0.0 for row in result["rows"])


def test_dataset_defect_helper_matching_burgers_coefficients_near_zero():
    cfg = ErrorDecompositionConfig(
        num_samples=2,
        nx=32,
        seed=3,
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
    x = torch.linspace(0.0, 1.0, 32, dtype=torch.float64)
    u0 = torch.stack([torch.sin(2.0 * math.pi * x), torch.cos(2.0 * math.pi * x)], dim=0)
    target_T = torch.zeros_like(u0)
    result = compute_time_scaled_defect_for_dataset(u0=u0, target_T=target_T, cfg=cfg, Ttilde=0.2)
    assert result["summary"]["alpha"] == 2.0
    assert result["summary"]["delta_scale_rms_abs_l2h"] < 1e-8
    assert "D1_model1_abs_l2h" in result["rows"][0]


def test_analytic_safe_poincare_shift_uses_domain_length_in_beta():
    x = torch.linspace(0.0, 1.0, 32, dtype=torch.float64)[:-1]
    states = torch.stack(
        [
            torch.sin(2.0 * math.pi * x),
            torch.cos(2.0 * math.pi * x),
        ],
        dim=0,
    ).unsqueeze(1)
    for domain_length, expected_shift in [
        (1.0, -0.05 * (2.0 * math.pi) ** 2),
        (2.0, -0.05 * math.pi**2),
    ]:
        cfg_safe = ErrorDecompositionConfig(target_nu=0.05, beta_mode="analytic_safe", domain_length=domain_length)
        cfg_poincare = ErrorDecompositionConfig(target_nu=0.05, beta_mode="analytic_safe_poincare", domain_length=domain_length)
        beta_safe, _ = compute_beta(
            calibration_target_states=states,
            calibration_surrogate_states=states,
            cfg=cfg_safe,
        )
        beta_poincare, details = compute_beta(
            calibration_target_states=states,
            calibration_surrogate_states=states,
            cfg=cfg_poincare,
        )
        assert details["poincare_shift"] == pytest.approx(expected_shift)
        assert details["poincare_lambda1"] == pytest.approx((2.0 * math.pi / domain_length) ** 2)
        assert details["domain_length"] == pytest.approx(domain_length)
        assert beta_poincare - beta_safe == pytest.approx(expected_shift)
        assert details["chosen_beta"] == pytest.approx(beta_poincare)


def test_analytic_safe_poincare_requires_matching_samplewise_means():
    target = torch.zeros(2, 1, 16, dtype=torch.float64)
    surrogate = torch.ones(2, 1, 16, dtype=torch.float64)
    cfg = ErrorDecompositionConfig(target_nu=0.05, beta_mode="analytic_safe_poincare", domain_length=2.0)
    with pytest.raises(ValueError, match="samplewise means"):
        compute_beta(calibration_target_states=target, calibration_surrogate_states=surrogate, cfg=cfg)


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
