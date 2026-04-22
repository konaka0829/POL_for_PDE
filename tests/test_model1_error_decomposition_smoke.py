import math
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pol.model123_1d.error_decomposition import (
    ErrorDecompositionConfig,
    aggregate_metric_rows,
    run_error_decomposition,
)


def test_exact_same_burgers_sanity_check():
    cfg = ErrorDecompositionConfig(
        num_samples=8,
        nx=64,
        seed=0,
        batch_size=4,
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
    rows = result["rows"]
    summary = aggregate_metric_rows(rows)
    assert len(rows) == cfg.num_samples
    assert summary[0]["D1"] < 1e-10
    assert summary[0]["Delta_init"] < 1e-10
    assert summary[0]["Delta_time"] < 1e-10
    assert summary[0]["Delta_dyn"] < 1e-10
    assert summary[0]["matched_time_error"] < 1e-10


def test_time_mismatch_inequality_sanity_check():
    cfg = ErrorDecompositionConfig(
        num_samples=8,
        nx=64,
        seed=0,
        batch_size=4,
        target_nu=0.05,
        T=0.1,
        Ttilde_values=[0.05],
        dt=0.01,
        fine_dt=0.002,
        reservoir="burgers",
        res_burgers_nu=0.05,
        res_burgers_b=1.0,
        beta_mode="fixed",
        beta_fixed=0.3,
        dtype="float64",
        device="cpu",
    )
    result = run_error_decomposition(cfg)
    rows = result["rows"]
    assert rows
    for row in rows:
        assert row["matched_time_error_abs_l2h"] < 1e-10
        assert row["Delta_dyn_abs_l2h"] < 1e-10
        assert abs(row["D1_abs_l2h"] - row["Delta_time_abs_l2h"]) < 1e-10
        assert row["D1_abs_l2h"] <= row["matched_plus_time_abs_l2h"] + 1e-12
    summary = result["summary_rows"][0]
    assert abs(summary["rhs_beta"] - summary["Delta_time"]) < 1e-10
    assert abs(summary["matched_rhs_beta"]) < 1e-10
    assert abs(summary["triangle_matched_plus_time"] - (summary["matched_time_error"] + summary["Delta_time"])) < 1e-12


def test_summary_rhs_is_theorem_consistent():
    cfg = ErrorDecompositionConfig(
        num_samples=8,
        nx=64,
        seed=0,
        batch_size=4,
        target_nu=0.05,
        T=0.1,
        Ttilde_values=[0.05, 0.1],
        dt=0.01,
        fine_dt=0.002,
        reservoir="reaction_diffusion",
        beta_mode="fixed",
        beta_fixed=0.25,
        dtype="float64",
        device="cpu",
    )
    result = run_error_decomposition(cfg)
    for row in result["summary_rows"]:
        expected_rhs = math.exp(row["beta_value"] * cfg.T) * row["Delta_init"] + row["c_beta_T"] * row["Delta_dyn"] + row["Delta_time"]
        expected_matched = math.exp(row["beta_value"] * cfg.T) * row["Delta_init"] + row["c_beta_T"] * row["Delta_dyn"]
        assert abs(row["rhs_beta"] - expected_rhs) < 1e-12
        assert abs(row["matched_rhs_beta"] - expected_matched) < 1e-12
        assert abs(row["triangle_matched_plus_time"] - (row["matched_time_error"] + row["Delta_time"])) < 1e-12


def test_fixed_and_analytic_safe_beta_modes():
    fixed_cfg = ErrorDecompositionConfig(
        num_samples=4,
        nx=32,
        seed=0,
        batch_size=2,
        target_nu=0.05,
        T=0.1,
        Ttilde_values=[0.1],
        dt=0.01,
        fine_dt=0.002,
        reservoir="reaction_diffusion",
        beta_mode="fixed",
        beta_fixed=0.7,
        dtype="float64",
        device="cpu",
    )
    fixed_result = run_error_decomposition(fixed_cfg)
    assert abs(fixed_result["beta_value"] - 0.7) < 1e-12

    analytic_cfg = ErrorDecompositionConfig(
        num_samples=4,
        nx=32,
        seed=1,
        batch_size=2,
        target_nu=0.05,
        T=0.1,
        Ttilde_values=[0.1],
        dt=0.01,
        fine_dt=0.002,
        reservoir="reaction_diffusion",
        beta_mode="analytic_safe",
        dtype="float64",
        device="cpu",
    )
    analytic_result = run_error_decomposition(analytic_cfg)
    assert math.isfinite(analytic_result["beta_value"])
    assert "M_K_hat" in analytic_result["beta_details"]


def test_end_to_end_cli_smoke(tmp_path):
    out_dir = tmp_path / "error_decomp"
    cmd = [
        sys.executable,
        "model1_error_decomposition_1d.py",
        "--num-samples",
        "8",
        "--nx",
        "64",
        "--batch-size",
        "4",
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
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    assert (out_dir / "per_sample_metrics.csv").exists()
    assert (out_dir / "per_sample_metrics.json").exists()
    assert (out_dir / "summary_metrics.csv").exists()
    assert (out_dir / "summary_metrics.json").exists()
    assert (out_dir / "matched_time_scatter_beta0_baseline.png").exists()
    assert (out_dir / "matched_time_scatter_selected_beta.png").exists()
    assert (out_dir / "time_mismatch_scatter.png").exists()
    assert (out_dir / "time_mismatch_envelope.png").exists()
    assert (out_dir / "combined_scatter_beta0_baseline.png").exists()
    assert (out_dir / "combined_scatter_selected_beta.png").exists()
    assert (out_dir / "aggregate_bound_vs_ttilde.png").exists()
