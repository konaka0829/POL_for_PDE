from argparse import Namespace

import pytest

from scripts.run_model123_param_sweep import (
    PARAMETERS,
    SweepSpec,
    build_run_dir,
    build_job_env,
    build_parser,
    build_run_command,
    build_range_values,
    canonical_parameter_name,
    clip_for_log,
    dedupe_fieldnames,
    load_run_metrics,
    main as sweep_main,
    parse_models,
    parse_sweep_assignment,
    require_time_grid_aligned_value,
    save_all_model_param_comparison_plots,
    selection_metric_label_for_row,
    expected_run_values,
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
    assert args.Ttilde == 0.0


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


def test_build_run_command_unspecified_ttilde_defaults_to_T(tmp_path):
    args = build_parser().parse_args(["--sweep", "alpha=1.0", "--T", "0.1", "--dt", "0.01"])
    cmd = build_run_command(args, "model1", {}, tmp_path / "run")
    idx = cmd.index("--Ttilde")
    assert cmd[idx + 1] == "0.1"


def test_build_run_command_alpha_sweep_sets_ttilde_from_T(tmp_path):
    args = build_parser().parse_args(["--sweep", "alpha=2.0", "--T", "0.1", "--dt", "0.01"])
    cmd = build_run_command(args, "model1", {"alpha": 2.0}, tmp_path / "run")
    idx = cmd.index("--Ttilde")
    assert cmd[idx + 1] == "0.2"


def test_heat_and_advection_parameters_are_not_duplicated(tmp_path):
    heat_args = build_parser().parse_args(["--sweep", "heat_nu=0.01", "--reservoir", "heat", "--heat-nu", "0.01"])
    heat_cmd = build_run_command(heat_args, "model2", {}, tmp_path / "heat")
    assert heat_cmd.count("--heat-nu") == 1
    assert heat_cmd[heat_cmd.index("--heat-nu") + 1] == "0.01"

    adv_args = build_parser().parse_args(
        ["--sweep", "advection_c=1.5", "--reservoir", "advection", "--advection-c", "1.5"]
    )
    adv_cmd = build_run_command(adv_args, "model2", {}, tmp_path / "adv")
    assert adv_cmd.count("--advection-c") == 1
    assert adv_cmd[adv_cmd.index("--advection-c") + 1] == "1.5"


def _prepare_args_for_expected(args):
    if args.data_seed is None:
        args.data_seed = args.seed
    if args.split_seed is None:
        args.split_seed = args.seed
    if args.ridge_zeta is None and args.ridge_lambda is None:
        args.ridge_zeta = 1e-4
        args.ridge_lambda = 1e-4
    elif args.ridge_zeta is None:
        args.ridge_zeta = float(args.ridge_lambda)
    elif args.ridge_lambda is None:
        args.ridge_lambda = float(args.ridge_zeta)
    for name in [
        "expected_ic_type",
        "expected_solver",
        "expected_time_integrator",
        "expected_burgers_scheme",
        "expected_dealias",
        "expected_equation",
        "expected_domain_length",
    ]:
        if not hasattr(args, name):
            setattr(args, name, None)
    return args


def _write_matching_run_config(run_dir, args, overrides):
    expected = expected_run_values(args, "model2", overrides)
    payload_args = dict(expected)
    payload_args.update(
        {
            "data_mode": "single_split",
            "ridge_lambda": expected["ridge_zeta"],
            "standardize_features": expected["standardize_features"],
            "burgers_dealias": expected["burgers_dealias"],
        }
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(
        """
        {
          "args": %s,
          "split": {"ntrain": %d, "nval": %d, "ntest": %d, "data_seed": %d, "split_seed": %d},
          "train_absL2h": 0.1,
          "val_absL2h": 0.05,
          "test_absL2h": 0.2,
          "train_relL2": 0.3,
          "test_relL2": 0.4,
          "alpha": %s,
          "resolved_obs": "full",
          "resolved_J": %d,
          "ridge_zeta": %s,
          "ridge_convention": "%s",
          "data_file": "%s",
          "data_sha256": "%s",
          "config_name": null,
          "config_hash": null
        }
        """
        % (
            __import__("json").dumps(payload_args),
            expected["ntrain"],
            expected["nval"],
            expected["ntest"],
            expected["data_seed"],
            expected["split_seed"],
            expected["alpha"],
            expected["J"],
            expected["ridge_zeta"],
            expected["ridge_convention"],
            expected["data_file"],
            expected["data_sha256"],
        ),
        encoding="utf-8",
    )


def test_check_existing_does_not_overwrite_normal_summary_files(tmp_path):
    data_file = tmp_path / "data.pt"
    data_file.write_bytes(b"not a real dataset; audit only")
    out_root = tmp_path / "sweep"
    argv = [
        "--data-file",
        str(data_file),
        "--out-root",
        str(out_root),
        "--models",
        "model2",
        "--reservoir",
        "static",
        "--sweep",
        "alpha=1.0",
        "--check-existing",
    ]
    args = _prepare_args_for_expected(build_parser().parse_args(argv))
    run_dir = build_run_dir(out_root / "model2", {"alpha": 1.0}, ["alpha"])
    _write_matching_run_config(run_dir, args, {"alpha": 1.0})

    model_dir = out_root / "model2"
    model_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = model_dir / "summary.csv"
    summary_json = model_dir / "summary.json"
    best_csv = model_dir / "best_runs.csv"
    best_json = model_dir / "best_runs.json"
    for path in [summary_csv, summary_json, best_csv, best_json]:
        path.write_text(f"original {path.name}", encoding="utf-8")
    before = {path: path.read_text(encoding="utf-8") for path in [summary_csv, summary_json, best_csv, best_json]}

    assert sweep_main(argv) == 0
    after = {path: path.read_text(encoding="utf-8") for path in before}
    assert after == before
    assert (model_dir / "existing_audit.csv").exists()
    assert (model_dir / "existing_audit.json").exists()
    assert (model_dir / "existing_audit_summary.csv").exists()
    assert (model_dir / "existing_audit_summary.json").exists()


def test_check_existing_config_mismatch_does_not_overwrite_summary(tmp_path):
    data_file = tmp_path / "data.pt"
    data_file.write_bytes(b"not a real dataset; audit only")
    out_root = tmp_path / "sweep"
    base_argv = [
        "--data-file",
        str(data_file),
        "--out-root",
        str(out_root),
        "--models",
        "model2",
        "--reservoir",
        "static",
        "--sweep",
        "alpha=1.0",
        "--ridge-zeta",
        "1e-4",
    ]
    args = _prepare_args_for_expected(build_parser().parse_args([*base_argv, "--check-existing"]))
    run_dir = build_run_dir(out_root / "model2", {"alpha": 1.0}, ["alpha"])
    _write_matching_run_config(run_dir, args, {"alpha": 1.0})
    summary = out_root / "model2" / "summary.csv"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text("normal summary", encoding="utf-8")

    assert sweep_main([*base_argv, "--ridge-zeta", "1e-5", "--check-existing"]) == 2
    assert summary.read_text(encoding="utf-8") == "normal summary"
    audit = (out_root / "model2" / "existing_audit.json").read_text(encoding="utf-8")
    assert "config_mismatch" in audit


def test_selection_metric_label_is_set_for_new_and_reuse_rows():
    assert selection_metric_label_for_row({"val_absL2h": 0.1, "test_absL2h": 0.2}) == "val_absL2h"
    assert selection_metric_label_for_row({"val_absL2h": None, "test_absL2h": 0.2}) == "test_absL2h_legacy_fallback"


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
