import importlib

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
