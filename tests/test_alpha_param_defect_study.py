import importlib

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
