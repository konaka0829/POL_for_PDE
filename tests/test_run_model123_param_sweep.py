from argparse import Namespace

import pytest

from scripts.run_model123_param_sweep import (
    PARAMETERS,
    SweepSpec,
    build_job_env,
    build_parser,
    build_run_command,
    build_range_values,
    canonical_parameter_name,
    clip_for_log,
    dedupe_fieldnames,
    load_run_metrics,
    parse_models,
    parse_sweep_assignment,
    require_time_grid_aligned_value,
    save_all_model_param_comparison_plots,
    validate_args,
)
from model123_burgers_1d import pearson_corr_or_none, spearman_corr_or_none


def test_parse_sweep_assignment_supports_required_parameters():
    for name in [
        "Ttilde",
        "alpha",
        "res_burgers_nu",
        "res_burgers_b",
        "rd_nu",
        "rd_alpha",
        "rd_beta",
        "ks_b",
        "ks_eta",
        "ks_kappa",
        "dt",
        "K",
        "J",
    ]:
        assert parse_sweep_assignment(f"{name}=1").parameter.name == canonical_parameter_name(name)


def test_alpha_aliases_canonicalize():
    assert parse_sweep_assignment("alpha=0.5,1.0").parameter.name == "alpha"
    assert canonical_parameter_name("time_alpha") == "alpha"
    assert canonical_parameter_name("alpha_scale") == "alpha"


def test_build_range_values():
    values = build_range_values(0.5, 0.7, 0.1)
    assert values == [0.5, 0.6, 0.7]


def test_parse_models_rejects_invalid_name():
    with pytest.raises(ValueError, match="Unsupported model"):
        parse_models("model1,invalid")


def test_clip_for_log_uses_eps_for_zero():
    eps = 1e-12
    assert clip_for_log(0.0, eps) == eps
    assert clip_for_log(1e-6, eps) == 1e-6


def test_ks_kappa_sweep_allows_zero_but_rejects_negative():
    spec = parse_sweep_assignment("ks_kappa=0,1e-6")
    assert spec.values == (0.0, 1e-6)
    with pytest.raises(ValueError, match="nonnegative"):
        parse_sweep_assignment("ks_kappa=-1e-6")


def test_build_job_env_limits_blas_threads():
    env = build_job_env({})
    assert env["OMP_NUM_THREADS"] == "1"
    assert env["MKL_NUM_THREADS"] == "1"
    assert env["OPENBLAS_NUM_THREADS"] == "1"
    assert env["NUMEXPR_NUM_THREADS"] == "1"
    assert env["TORCH_NUM_THREADS"] == "1"


def test_parser_defaults_match_lightweight_model123_dataset():
    args = build_parser().parse_args(["--sweep", "Ttilde=1.0"])
    assert args.data_file == "data/burgers_model123.mat"
    assert args.dt == 1e-2
    assert args.burgers_fine_dt == 1e-4


def test_validate_args_rejects_nonpositive_max_workers_before_missing_data_file():
    args = Namespace(
        data_file="does_not_exist.mat",
        train_split=1000.0 / 1200.0,
        ntrain=1000,
        ntest=200,
        batch_size=32,
        sub=1,
        T=1.0,
        Ttilde=1.0,
        dt=1e-2,
        burgers_fine_dt=1e-4,
        max_workers=0,
        best_k=10,
        models="model1",
        sweep=["Ttilde=1.0"],
        sweep_range=[],
        reservoir="burgers",
    )
    with pytest.raises(ValueError, match="max-workers"):
        validate_args(args)


def test_load_run_metrics_reads_absolute_and_relative_metrics(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_config.json").write_text(
        """
        {
          "train_absL2h": 0.1,
          "test_absL2h": 0.2,
          "train_relL2": 0.3,
          "test_relL2": 0.4,
          "alpha": 1.0,
          "resolved_obs": "full",
          "resolved_J": 32
        }
        """,
        encoding="utf-8",
    )
    metrics = load_run_metrics(run_dir)
    assert metrics["test_absL2h"] == 0.2
    assert metrics["test_relL2"] == 0.4


def test_build_run_command_alpha_override_sets_ttilde(tmp_path):
    args = build_parser().parse_args(["--sweep", "alpha=0.5", "--T", "2.0"])
    cmd = build_run_command(args, "model1", {"alpha": 0.5}, tmp_path / "run")
    idx = cmd.index("--Ttilde")
    assert cmd[idx + 1] == "1.0"
    assert "--alpha" not in cmd


def test_validate_args_rejects_simultaneous_alpha_and_ttilde_sweeps():
    args = build_parser().parse_args(
        ["--sweep", "alpha=0.5", "--sweep", "Ttilde=1.0", "--dry-run"]
    )
    with pytest.raises(ValueError, match="Cannot sweep both alpha and Ttilde"):
        validate_args(args)


def test_alpha_derived_ttilde_alignment_validation(tmp_path):
    require_time_grid_aligned_value(1.0, 0.25, "Ttilde", alpha=0.5, T=2.0)
    args = build_parser().parse_args(["--sweep", "alpha=0.3", "--T", "1.0", "--dt", "0.2", "--dry-run"])
    with pytest.raises(ValueError, match="alpha.*T.*Ttilde.*dt|Ttilde=.*alpha=.*T=.*dt"):
        build_run_command(args, "model1", {"alpha": 0.3}, tmp_path / "run")


def test_summary_fieldnames_are_deduplicated_for_alpha_and_ttilde_sweeps():
    alpha_fields = dedupe_fieldnames(["model", "alpha", "T", "Ttilde", "alpha", "run_dir"])
    ttilde_fields = dedupe_fieldnames(["model", "Ttilde", "T", "Ttilde", "alpha", "run_dir"])
    assert alpha_fields.count("alpha") == 1
    assert ttilde_fields.count("Ttilde") == 1


def test_load_run_metrics_reads_optional_defect_metrics(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_config.json").write_text(
        """
        {
          "args": {"T": 2.0, "Ttilde": 1.0},
          "train_absL2h": 0.1,
          "test_absL2h": 0.2,
          "train_relL2": 0.3,
          "test_relL2": 0.4,
          "resolved_obs": "full",
          "resolved_J": 32
        }
        """,
        encoding="utf-8",
    )
    (run_dir / "time_scaled_defect_metrics.json").write_text(
        """
        {
          "delta_scale_rms_abs_l2h": 1.2,
          "delta_scale_mean_abs_l2h": 1.0,
          "delta_scale_std_abs_l2h": 0.2,
          "corr_error_delta_scale_pearson": 0.5,
          "corr_error_delta_scale_spearman": 0.25,
          "applies_directly_to_model1_bound": true,
          "defect_metric": "pathwise_integrated_time_scaled_generator_defect"
        }
        """,
        encoding="utf-8",
    )
    metrics = load_run_metrics(run_dir)
    assert metrics["T"] == 2.0
    assert metrics["Ttilde"] == 1.0
    assert metrics["alpha"] == 0.5
    assert metrics["delta_scale_rms_abs_l2h"] == 1.2
    assert metrics["corr_error_delta_scale_pearson"] == 0.5
    assert metrics["time_scaled_defect_metric"] == "pathwise_integrated_time_scaled_generator_defect"


def test_correlation_helpers():
    assert pearson_corr_or_none([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert pearson_corr_or_none([1, 1, 1], [2, 3, 4]) is None
    assert spearman_corr_or_none([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert spearman_corr_or_none([1, 1, 1], [10, 20, 30]) is None
    assert pearson_corr_or_none([1, float("nan"), 3], [2, 99, 6]) == pytest.approx(1.0)
    assert spearman_corr_or_none([1, float("inf"), 3], [10, 99, 30]) == pytest.approx(1.0)


def test_save_all_model_param_comparison_plots_writes_profile_and_slices(tmp_path):
    rows_by_model = {}
    for model, model_offset in [("model1", 0.0), ("model2", 0.1), ("model3", 0.2)]:
        rows = []
        for alpha in [0.5, 1.0]:
            for nu in [0.01, 0.02]:
                rows.append(
                    {
                        "model": model,
                        "alpha": alpha,
                        "res_burgers_nu": nu,
                        "status": "ok",
                        "test_absL2h": abs(alpha - 1.0) + abs(nu - 0.02) + model_offset + 0.01,
                    }
                )
        rows_by_model[model] = rows

    save_all_model_param_comparison_plots(
        model_rows=rows_by_model,
        sweep_specs=[
            SweepSpec(parameter=PARAMETERS["alpha"], values=(0.5, 1.0)),
            SweepSpec(parameter=PARAMETERS["res_burgers_nu"], values=(0.01, 0.02)),
        ],
        out_root=tmp_path,
        eps=1e-16,
    )

    for parameter in ["alpha", "res_burgers_nu"]:
        for mode in [
            "profile_optimized",
            "slice_model_best_fixed",
            "slice_common_best_fixed",
        ]:
            for ext in ["png", "pdf", "svg"]:
                assert (tmp_path / f"{parameter}_vs_error_{mode}_all_models.{ext}").exists()

    settings = (tmp_path / "all_models_param_vs_error_plot_settings.json").read_text(encoding="utf-8")
    assert "slice_model_best_fixed" in settings
    assert "slice_common_best_fixed" in settings
