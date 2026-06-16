import importlib
import warnings

import pytest
from pathlib import Path


def test_alpha_param_defect_study_import_safe():
    module = importlib.import_module("scripts.run_model123_alpha_param_defect_study")
    assert hasattr(module, "main")


def test_alpha_param_defect_study_preserves_int_parameter_values():
    module = importlib.import_module("scripts.run_model123_alpha_param_defect_study")
    args = module.build_parser().parse_args(
        ["--parameter", "K", "--parameter-values", "2,3", "--alpha-values", "1.0", "--dry-run"]
    )
    _, parameter, parameter_values, alpha_values = module.validate_args(args)
    assert parameter.name == "K"
    assert parameter_values == [2, 3]
    cmd = module.command_for_model(args, "model1", parameter.name, parameter_values[0], alpha_values[0], Path("/tmp/run"), False)
    assert cmd[cmd.index("--K") + 1] == "2"


def test_alpha_param_defect_study_rejects_misaligned_alpha_before_jobs():
    module = importlib.import_module("scripts.run_model123_alpha_param_defect_study")
    args = module.build_parser().parse_args(
        [
            "--parameter",
            "res_burgers_nu",
            "--parameter-values",
            "0.04",
            "--alpha-values",
            "0.3",
            "--T",
            "1.0",
            "--dt",
            "0.2",
            "--dry-run",
        ]
    )
    with pytest.raises(ValueError, match="Ttilde=.*alpha=.*T=.*dt"):
        module.validate_args(args)


def test_alpha_param_defect_study_zero_values_plot_without_log_warning(tmp_path):
    module = importlib.import_module("scripts.run_model123_alpha_param_defect_study")
    rows = [
        {"status": "ok", "model": "model1", "alpha": 1.0, "parameter_value": 0.05, "test_absL2h": 0.0},
        {"status": "ok", "model": "model2", "alpha": 1.0, "parameter_value": 0.05, "test_absL2h": 0.0},
    ]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module.plot_lines(
            rows,
            x_key="alpha",
            fixed_key="parameter_value",
            fixed_value=0.05,
            title="zero defect",
            xlabel="alpha",
            out_path=tmp_path / "zero_plot",
        )
    assert not any("cannot be log-scaled" in str(item.message) for item in caught)


def test_alpha_param_defect_study_positive_values_use_log_scale(tmp_path):
    module = importlib.import_module("scripts.run_model123_alpha_param_defect_study")
    rows = [
        {"status": "ok", "model": "model1", "alpha": 1.0, "parameter_value": 0.05, "test_absL2h": 1e-3},
        {"status": "ok", "model": "model1", "alpha": 2.0, "parameter_value": 0.05, "test_absL2h": 1e-2},
    ]
    module.plot_lines(
        rows,
        x_key="alpha",
        fixed_key="parameter_value",
        fixed_value=0.05,
        title="positive defect",
        xlabel="alpha",
        out_path=tmp_path / "positive_plot",
    )
    assert (tmp_path / "positive_plot.png").exists()
