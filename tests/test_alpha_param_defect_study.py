import importlib
import json
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


def test_alpha_param_check_existing_writes_only_audit_outputs(tmp_path, monkeypatch):
    module = importlib.import_module("scripts.run_model123_alpha_param_defect_study")
    out_root = tmp_path / "alpha"
    out_root.mkdir()
    protected = [
        out_root / "summary.csv",
        out_root / "summary.json",
        out_root / "per_sample_metrics.csv",
        out_root / "per_sample_metrics.json",
        out_root / "plots" / "existing.png",
    ]
    for path in protected:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"original {path.name}", encoding="utf-8")
    before = {path: path.read_text(encoding="utf-8") for path in protected}

    def fake_audit_existing_run(*, args, model, overrides, run_dir):
        return {
            "status": "ok",
            "reason": "",
            "has_run_config": True,
            "has_defect_metrics": True,
            "run_dir": str(run_dir),
        }

    def fail_run_command(*args, **kwargs):
        raise AssertionError("--check-existing must not launch jobs")

    monkeypatch.setattr(module, "audit_existing_run", fake_audit_existing_run)
    monkeypatch.setattr(module, "run_command", fail_run_command)

    rc = module.main(
        [
            "--data-file",
            str(tmp_path / "dummy.pt"),
            "--out-root",
            str(out_root),
            "--models",
            "model1",
            "--reservoir",
            "burgers",
            "--parameter",
            "res_burgers_nu",
            "--parameter-values",
            "0.01",
            "--alpha-values",
            "1.0",
            "--check-existing",
        ]
    )
    assert rc == 0
    assert {path: path.read_text(encoding="utf-8") for path in protected} == before
    assert (out_root / "existing_audit.csv").exists()
    assert (out_root / "existing_audit.json").exists()
    assert (out_root / "existing_audit_summary.csv").exists()
    assert (out_root / "existing_audit_summary.json").exists()
    assert not (out_root / "runs").exists()
    summary = json.loads((out_root / "existing_audit_summary.json").read_text(encoding="utf-8"))
    assert summary["counts"] == {"ok": 1}


def test_alpha_param_check_existing_mismatch_preserves_normal_summary(tmp_path, monkeypatch):
    module = importlib.import_module("scripts.run_model123_alpha_param_defect_study")
    out_root = tmp_path / "alpha"
    summary_csv = out_root / "summary.csv"
    summary_json = out_root / "summary.json"
    per_sample_csv = out_root / "per_sample_metrics.csv"
    per_sample_json = out_root / "per_sample_metrics.json"
    for path in [summary_csv, summary_json, per_sample_csv, per_sample_json]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"normal {path.name}", encoding="utf-8")
    before = {path: path.read_text(encoding="utf-8") for path in [summary_csv, summary_json, per_sample_csv, per_sample_json]}

    monkeypatch.setattr(
        module,
        "audit_existing_run",
        lambda **kwargs: {
            "status": "config_mismatch",
            "reason": "mismatch: ridge_zeta",
            "has_run_config": True,
            "has_defect_metrics": True,
        },
    )
    monkeypatch.setattr(module, "run_command", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("job launched")))

    rc = module.main(
        [
            "--data-file",
            str(tmp_path / "dummy.pt"),
            "--out-root",
            str(out_root),
            "--models",
            "model1",
            "--reservoir",
            "burgers",
            "--parameter",
            "res_burgers_nu",
            "--parameter-values",
            "0.02",
            "--alpha-values",
            "1.0",
            "--ridge-zeta",
            "1e-5",
            "--check-existing",
        ]
    )
    assert rc == 2
    assert {path: path.read_text(encoding="utf-8") for path in before} == before
    audit = json.loads((out_root / "existing_audit.json").read_text(encoding="utf-8"))
    assert audit[0]["status"] == "config_mismatch"
    audit_summary = json.loads((out_root / "existing_audit_summary.json").read_text(encoding="utf-8"))
    assert audit_summary["counts"] == {"config_mismatch": 1}
