from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from pol.cli import main
import pol.paper1.runner as runner_module
from pol.paper1.run_spec import load_run_spec
from pol.paper1.runner import build_plan, execute_run, plan_to_dict
from pol.plots.types import PlotTaskSpec
from pol.runtime.recipe import RecipeResult, RecipeUsageError


ROOT = Path(__file__).resolve().parents[1]


def _spec(kind: str):
    return load_run_spec(ROOT / f"configs/runs/paper1_{kind}_smoke.json", repo_root=ROOT)


@pytest.mark.parametrize(
    ("kind", "names"), [("e0", ["e0"]), ("e1", ["e0", "e1"]), ("e2", ["e0", "master_dataset", "e2"])]
)
def test_plans_describe_direct_recipes(kind: str, names: list[str]) -> None:
    plan = build_plan(_spec(kind), repo_root=ROOT)
    assert [step.name for step in plan] == names
    assert all(step.logical_invocation[0] == "direct_recipe" for step in plan)
    assert all(step.recipe_callable.startswith("pol.paper1.recipes.") for step in plan)


@pytest.mark.parametrize("kind", ["e1", "e2"])
def test_unified_compute_always_skips_inline_plots(kind: str) -> None:
    spec = _spec(kind)
    assert build_plan(spec, repo_root=ROOT)[-1].parameters["skip_plots"] is True
    enabled = spec.__class__(**{**spec.__dict__, "skip_plots": True})
    assert build_plan(enabled, repo_root=ROOT)[-1].parameters["skip_plots"] is True


def test_plan_has_no_filesystem_or_popen(tmp_path: Path, monkeypatch, capsys) -> None:
    raw = json.loads((ROOT / "configs/runs/paper1_e0_smoke.json").read_text())
    raw["run"]["output_root"] = str(tmp_path / "out")
    source = tmp_path / "spec.json"
    source.write_text(json.dumps(raw))
    variables = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    monkeypatch.setenv(variables[0], "7")
    monkeypatch.setenv(variables[1], "11")
    monkeypatch.delenv(variables[2], raising=False)
    monkeypatch.delenv(variables[3], raising=False)
    before = {name: os.environ.get(name) for name in variables}
    monkeypatch.setattr(runner_module.subprocess, "Popen", lambda *a, **k: pytest.fail("Popen called"))
    assert main(["run", str(source), "--plan"]) == 0
    assert not (tmp_path / "out").exists()
    assert {name: os.environ.get(name) for name in variables} == before
    plan = json.loads(capsys.readouterr().out)
    assert plan["schema_version"] == "paper1-run-plan-v2"
    assert plan["steps"][0]["execution_mode"] == "direct_recipe"
    assert plan["steps"][0]["recipe_callable"].endswith(
        ".run_foundation_validation"
    )
    assert "command" not in plan["steps"][0]
    assert set(plan["steps"][0]) == {
        "name",
        "execution_mode",
        "recipe_id",
        "recipe_callable",
        "config_path",
        "output_dir",
        "parameters",
    }


def test_plan_does_not_require_legacy_wrapper_scripts(
    tmp_path: Path, monkeypatch
) -> None:
    spec = _temp_spec(tmp_path, "e2")
    original_is_file = Path.is_file

    def wrappers_are_missing(path: Path) -> bool:
        if "scripts/paper1" in path.as_posix():
            return False
        return original_is_file(path)

    monkeypatch.setattr(Path, "is_file", wrappers_are_missing)
    assert [step.recipe_id for step in build_plan(spec, repo_root=ROOT)] == [
        "e0",
        "master_dataset",
        "e2",
    ]


@pytest.mark.parametrize("failure", [False, True])
def test_cli_restores_thread_environment_after_execution(
    tmp_path: Path, monkeypatch, failure: bool
) -> None:
    raw = json.loads((ROOT / "configs/runs/paper1_e0_smoke.json").read_text())
    raw["run"]["output_root"] = str(tmp_path / "runs")
    raw["run"]["name"] = "failure" if failure else "success"
    source = tmp_path / f"{raw['run']['name']}.json"
    source.write_text(json.dumps(raw), encoding="utf-8")
    variables = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    monkeypatch.setenv(variables[0], "7")
    monkeypatch.delenv(variables[1], raising=False)
    monkeypatch.setenv(variables[2], "9")
    monkeypatch.delenv(variables[3], raising=False)
    before = {name: os.environ.get(name) for name in variables}
    monkeypatch.setattr(runner_module, "_git", lambda *args: "test")
    if failure:
        monkeypatch.setattr(
            runner_module,
            "_execute_recipe",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("simulated failure")
            ),
        )
        expected = 1
    else:
        monkeypatch.setattr(
            runner_module,
            "_execute_recipe",
            lambda step, *args, **kwargs: _result(step),
        )
        monkeypatch.setattr(runner_module, "_verify_step", lambda step: None)
        expected = 0
    assert main(["run", str(source)]) == expected
    assert {name: os.environ.get(name) for name in variables} == before


def _result(step, exit_code: int = 0) -> RecipeResult:
    return RecipeResult(
        "pass" if exit_code == 0 else "fail",
        exit_code,
        step.output_dir,
        {"status": "pass" if exit_code == 0 else "fail"},
    )


def _temp_spec(tmp_path: Path, kind: str = "e0"):
    raw = json.loads(
        (ROOT / f"configs/runs/paper1_{kind}_smoke.json").read_text()
    )
    raw["run"] = {"name": "unit", "output_root": str(tmp_path / "runs")}
    source = tmp_path / "spec.json"
    source.write_text(json.dumps(raw))
    return load_run_spec(source, repo_root=ROOT)


def _write_owned_manifest(run_dir: Path, run_name: str) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "paper1-run-manifest-v2",
                "run_name": run_name,
                "run_dir": str(run_dir),
            }
        )
    )


def test_existing_guard_force_and_manifest(tmp_path: Path, monkeypatch) -> None:
    spec = _temp_spec(tmp_path)
    _write_owned_manifest(spec.run_dir, spec.name)
    marker = spec.run_dir / "marker"
    marker.write_text("old")
    sibling = spec.output_root / "sibling"
    sibling.mkdir()
    sibling_marker = sibling / "keep"
    sibling_marker.write_text("preserve")
    with pytest.raises(FileExistsError):
        execute_run(spec, repo_root=ROOT, force=False)
    monkeypatch.setattr(runner_module, "_git", lambda *args: "test")
    monkeypatch.setattr(
        runner_module, "_execute_recipe",
        lambda step, *args, **kwargs: _result(step))
    monkeypatch.setattr("pol.paper1.runner._verify_step", lambda step: None)
    assert execute_run(spec, repo_root=ROOT, force=True) == 0
    assert not marker.exists()
    assert sibling_marker.read_text() == "preserve"
    assert spec.output_root.exists()
    manifest = json.loads((spec.run_dir / "run_manifest.json").read_text())
    assert manifest["status"] == "pass"
    assert manifest["steps"][0]["returncode"] == 0
    assert manifest["steps"][0]["execution_mode"] == "direct_recipe"
    assert manifest["steps"][0]["command"][0] == "direct_recipe"
    assert manifest["steps"][0]["recipe_callable"].endswith(
        ".run_foundation_validation"
    )


def test_child_failure_stops_and_records(tmp_path: Path, monkeypatch) -> None:
    spec = _temp_spec(tmp_path, "e2")
    calls = []

    def failed(step, *args, **kwargs):
        calls.append(step.name)
        return _result(step, 7)

    monkeypatch.setattr(runner_module, "_git", lambda *args: "test")
    monkeypatch.setattr(runner_module, "_execute_recipe", failed)
    assert execute_run(spec, repo_root=ROOT, force=False) == 1
    manifest = json.loads((spec.run_dir / "run_manifest.json").read_text())
    assert manifest["status"] == "fail"
    assert len(calls) == 1
    assert manifest["steps"][0]["returncode"] == 7
    assert manifest["steps"][0]["status"] == "fail"
    assert [step["status"] for step in manifest["steps"][1:]] == [
        "pending",
        "pending",
    ]


def test_exit_zero_missing_summary_fails(tmp_path: Path, monkeypatch) -> None:
    spec = _temp_spec(tmp_path)
    monkeypatch.setattr(runner_module, "_git", lambda *args: "test")
    monkeypatch.setattr(
        runner_module, "_execute_recipe",
        lambda step, *args, **kwargs: _result(step))
    assert execute_run(spec, repo_root=ROOT, force=False) == 1
    manifest = json.loads((spec.run_dir / "run_manifest.json").read_text())
    assert "missing saved output" in manifest["failure"]


def test_unsafe_run_directory_is_rejected(tmp_path: Path) -> None:
    spec = _temp_spec(tmp_path)
    unsafe = replace(spec, output_root=ROOT.parent, name=ROOT.name)
    with pytest.raises(ValueError, match="unsafe output root"):
        execute_run(unsafe, repo_root=ROOT, force=True)


def test_force_rejects_arbitrary_directory_without_deleting_marker(
    tmp_path: Path,
) -> None:
    spec = _temp_spec(tmp_path)
    spec.run_dir.mkdir(parents=True)
    marker = spec.run_dir / "important"
    marker.write_text("keep")
    with pytest.raises(ValueError, match="not runner-owned"):
        execute_run(spec, repo_root=ROOT, force=True)
    assert marker.read_text() == "keep"


def test_force_rejects_symlink_without_deleting_target(tmp_path: Path) -> None:
    spec = _temp_spec(tmp_path)
    spec.output_root.mkdir(parents=True)
    victim = spec.output_root / "victim"
    victim.mkdir()
    important = victim / "important.txt"
    important.write_text("keep")
    try:
        spec.run_dir.symlink_to(victim, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    with pytest.raises(ValueError, match="must not be a symlink"):
        execute_run(spec, repo_root=ROOT, force=True)
    assert important.read_text() == "keep"


def test_plan_rejects_unsafe_output_root_without_mutation(tmp_path: Path) -> None:
    spec = _temp_spec(tmp_path)
    unsafe = replace(spec, output_root=ROOT)
    before = set(ROOT.iterdir())
    with pytest.raises(ValueError, match="unsafe output root"):
        plan_to_dict(unsafe, repo_root=ROOT)
    assert set(ROOT.iterdir()) == before


def test_plan_rejects_run_directory_that_contains_repository(
    tmp_path: Path,
) -> None:
    spec = _temp_spec(tmp_path)
    output_root = tmp_path / "ancestor-output"
    run_dir = output_root / "contains-repository"
    fake_repo = run_dir / "nested" / "repository"
    fake_repo.mkdir(parents=True)
    unsafe = replace(
        spec,
        output_root=output_root,
        name=run_dir.name,
    )
    marker = fake_repo / "keep"
    marker.write_text("preserve")
    with pytest.raises(ValueError, match="contains protected path"):
        plan_to_dict(unsafe, repo_root=fake_repo)
    assert marker.read_text() == "preserve"


@pytest.mark.parametrize("protected_name", ["source", "experiment", "e0"])
def test_plan_rejects_run_directory_that_contains_input_path(
    tmp_path: Path, protected_name: str
) -> None:
    spec = _temp_spec(tmp_path, "e1")
    output_root = tmp_path / "protected-output"
    run_dir = output_root / "unit"
    protected = run_dir / f"{protected_name}.json"
    protected.parent.mkdir(parents=True)
    protected.write_text("{}")
    replacements = {
        "output_root": output_root,
        "name": run_dir.name,
        {
            "source": "source_path",
            "experiment": "experiment_config",
            "e0": "e0_config",
        }[protected_name]: protected,
    }
    unsafe = replace(spec, **replacements)
    with pytest.raises(ValueError, match="contains protected path"):
        plan_to_dict(unsafe, repo_root=ROOT)
    assert protected.read_text() == "{}"


def test_recipe_thread_scope_matches_step_policy(
    tmp_path: Path, monkeypatch
) -> None:
    spec = replace(_temp_spec(tmp_path, "e2"), torch_threads=3)
    observed = []

    def capture(*args, invocation, **kwargs):
        output_dir = args[-1]
        observed.append(
            (
                output_dir.name,
                invocation.torch_threads,
                torch.get_num_threads(),
                tuple(
                    os.environ[name]
                    for name in (
                        "OMP_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS",
                    )
                ),
            )
        )
        return RecipeResult("pass", 0, output_dir, {"status": "pass"})

    monkeypatch.setattr(runner_module, "_git", lambda *args: "test")
    monkeypatch.setattr(
        "pol.paper1.recipes.foundation_validation.run_foundation_validation",
        capture,
    )
    monkeypatch.setattr(
        "pol.paper1.recipes.master_dataset.run_master_dataset_generation",
        capture,
    )
    monkeypatch.setattr(
        "pol.paper1.recipes.surrogate_parameter_time.run_surrogate_parameter_time",
        capture,
    )
    monkeypatch.setattr(runner_module, "_verify_step", lambda step: None)
    monkeypatch.setattr(runner_module, "_run_plots", lambda *args, **kwargs: [])
    assert execute_run(spec, repo_root=ROOT, force=False) == 0
    assert observed == [
        ("e0", 1, 1, ("1",) * 4),
        ("master_dataset", 1, 1, ("1",) * 4),
        ("e2", 3, 3, ("3",) * 4),
    ]


def test_plots_only_never_calls_compute_and_records_reuse(
    tmp_path: Path, monkeypatch
) -> None:
    spec = _temp_spec(tmp_path, "e1")
    run_dir = spec.run_dir
    run_dir.mkdir(parents=True)
    manifest = {
        "schema_version": "paper1-run-manifest-v2",
        "status": "pass",
        "run_name": spec.name,
        "run_dir": str(run_dir),
        "science_fingerprint": runner_module.science_fingerprint(spec),
        "compute_status": "pass",
        "steps": [],
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(
        runner_module,
        "_execute_recipe",
        lambda *args, **kwargs: pytest.fail("compute recipe called"),
    )
    monkeypatch.setattr(
        runner_module, "_verify_compute_for_plots", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        runner_module, "_validate_current_request_binding",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        runner_module,
        "_run_plots",
        lambda *args, **kwargs: [
            {
                "recipe_id": "paper1.e1.standard.v1",
                "status": "pass",
                "executed_or_reused": "reused",
                "plot_fingerprint": "abc",
            }
        ],
    )
    assert execute_run(
        spec, repo_root=ROOT, force=False, plots_only=True
    ) == 0
    saved = json.loads((run_dir / "run_manifest.json").read_text())
    assert saved["compute_status"] == "pass"
    assert saved["plot_status"] == "pass"
    assert saved["plot_tasks"][0]["executed_or_reused"] == "reused"
    request = json.loads((run_dir / "resolved_plot_spec.json").read_text())
    assert request["request_mode"] == "plots_only"
    assert request["compute_fingerprint"] == runner_module.science_fingerprint(spec)
    assert request["requested_tasks"][0]["settings"] == dict(
        spec.plot_tasks[0].settings
    )
    assert saved["last_plot_request"] == request


def test_compute_binding_ignores_plot_settings_and_source_bytes(
    tmp_path: Path
) -> None:
    spec = _temp_spec(tmp_path, "e1")
    steps = build_plan(spec, repo_root=ROOT)
    spec.run_dir.mkdir(parents=True)
    runner_module.write_strict_json(
        spec.run_dir / "resolved_run_spec.json",
        runner_module._resolved_run_record(
            spec, steps, run_dir=spec.run_dir
        ),
    )
    changed = replace(
        spec,
        plot_tasks=(
            PlotTaskSpec(
                spec.plot_tasks[0].recipe_id,
                {**dict(spec.plot_tasks[0].settings), "dpi": 120},
            ),
        ),
    )
    runner_module._validate_current_request_binding(
        changed,
        build_plan(changed, repo_root=ROOT),
        run_dir=spec.run_dir,
    )


def test_plots_only_rejects_disabled_or_empty_plot_request(
    tmp_path: Path,
) -> None:
    spec = replace(
        _temp_spec(tmp_path, "e1"),
        plots_enabled=False,
        plot_tasks=(),
    )
    with pytest.raises(ValueError, match="enabled plot task"):
        execute_run(spec, repo_root=ROOT, force=False, plots_only=True)


def test_malformed_plot_request_records_rejection_without_compute_mutation(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs" / "unit"
    run_dir.mkdir(parents=True)
    manifest = {
        "schema_version": "paper1-run-manifest-v2",
        "status": "pass",
        "run_name": "unit",
        "run_dir": str(run_dir),
        "science_fingerprint": "saved-compute",
        "compute_status": "pass",
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest))
    raw = {
        "schema_version": "paper1-run-v2",
        "run": {"name": "unit", "output_root": str(tmp_path / "runs")},
        "experiment": {
            "kind": "e1",
            "config": "configs/paper1_e1_smoke.json",
        },
        "prerequisites": {
            "e0_config": "configs/paper1_e0_for_e1_smoke.json"
        },
        "execution": {"torch_threads": 1},
        "plots": {
            "enabled": True,
            "required": True,
            "recipes": [
                {
                    "id": "paper1.e1.standard.v1",
                    "settings": {"formats": ["bad"], "dpi": 80},
                }
            ],
        },
    }
    source = tmp_path / "bad-plot.json"
    source.write_text(json.dumps(raw))
    assert main(["run", str(source), "--plots-only"]) == 2
    assert json.loads(
        (run_dir / "run_manifest.json").read_text()
    )["compute_status"] == "pass"
    request = json.loads((run_dir / "resolved_plot_spec.json").read_text())
    assert request["request_status"] == "rejected"


@pytest.mark.parametrize(("required", "expected"), [(True, 1), (False, 0)])
def test_plot_failure_does_not_change_compute_status(
    tmp_path: Path, monkeypatch, required: bool, expected: int
) -> None:
    spec = replace(_temp_spec(tmp_path, "e1"), plots_required=required)
    run_dir = spec.run_dir
    run_dir.mkdir(parents=True)
    manifest = {
        "schema_version": "paper1-run-manifest-v2",
        "status": "pass",
        "run_name": spec.name,
        "run_dir": str(run_dir),
        "science_fingerprint": runner_module.science_fingerprint(spec),
        "compute_status": "pass",
        "steps": [],
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(
        runner_module, "_verify_compute_for_plots", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        runner_module, "_validate_current_request_binding",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        runner_module,
        "_run_plots",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("plot failed")),
    )
    assert execute_run(
        spec, repo_root=ROOT, force=False, plots_only=True
    ) == expected
    saved = json.loads((run_dir / "run_manifest.json").read_text())
    assert saved["compute_status"] == "pass"
    assert saved["plot_status"] == "fail"
    request = json.loads((run_dir / "resolved_plot_spec.json").read_text())
    assert request["status"] == "fail"
    assert request["failure"] == "RuntimeError: plot failed"


def test_plots_only_rejects_compute_tamper_before_plotting(
    tmp_path: Path, monkeypatch
) -> None:
    spec = _temp_spec(tmp_path, "e1")
    run_dir = spec.run_dir
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "paper1-run-manifest-v2",
                "status": "pass",
                "run_name": spec.name,
                "run_dir": str(run_dir),
                "science_fingerprint": runner_module.science_fingerprint(spec),
                "compute_status": "pass",
                "steps": [],
            }
        )
    )
    monkeypatch.setattr(
        runner_module,
        "_verify_compute_for_plots",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("compute artifact tampered")
        ),
    )
    monkeypatch.setattr(
        runner_module, "_validate_current_request_binding",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        runner_module,
        "_run_plots",
        lambda *args, **kwargs: pytest.fail("plot task called"),
    )
    assert execute_run(spec, repo_root=ROOT, force=False, plots_only=True) == 1
    saved = json.loads((run_dir / "run_manifest.json").read_text())
    assert saved["status"] == "fail"
    assert saved["compute_status"] == "invalid"
    assert saved["plot_status"] == "not_run"
    assert "compute artifact tampered" in saved["failure"]


def test_keyboard_interrupt_records_and_returns_130(tmp_path: Path, monkeypatch) -> None:
    spec = _temp_spec(tmp_path)

    monkeypatch.setattr(runner_module, "_git", lambda *args: "test")
    monkeypatch.setattr(
        runner_module, "_execute_recipe",
        lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()))
    assert execute_run(spec, repo_root=ROOT, force=False) == 130
    manifest = json.loads((spec.run_dir / "run_manifest.json").read_text())
    assert manifest["status"] == "interrupted"
    assert manifest["steps"][0]["status"] == "interrupted"
    assert manifest["steps"][0]["returncode"] is None
    assert manifest["steps"][0]["duration_seconds"] is not None
    assert "KeyboardInterrupt" in (spec.run_dir / "logs/01_e0.log").read_text()


def test_recipe_exception_is_logged_and_recorded(
    tmp_path: Path, monkeypatch
) -> None:
    spec = _temp_spec(tmp_path)

    monkeypatch.setattr(runner_module, "_git", lambda *args: "test")
    monkeypatch.setattr(
        runner_module, "_execute_recipe",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            OSError("simulated recipe failure")))
    assert execute_run(spec, repo_root=ROOT, force=False) == 1
    manifest = json.loads((spec.run_dir / "run_manifest.json").read_text())
    assert manifest["status"] == "fail"
    assert manifest["steps"][0]["status"] == "fail"
    assert manifest["steps"][0]["returncode"] is None
    assert manifest["steps"][0]["duration_seconds"] is not None
    assert "simulated recipe failure" in manifest["failure"]
    assert "Traceback" in (spec.run_dir / "logs/01_e0.log").read_text()


def test_recipe_usage_error_records_exit_code_two(
    tmp_path: Path, monkeypatch
) -> None:
    spec = _temp_spec(tmp_path)
    monkeypatch.setattr(runner_module, "_git", lambda *args: "test")
    monkeypatch.setattr(
        runner_module,
        "_execute_recipe",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RecipeUsageError("invalid recipe input")),
    )
    assert execute_run(spec, repo_root=ROOT, force=False) == 1
    manifest = json.loads((spec.run_dir / "run_manifest.json").read_text())
    assert manifest["steps"][0]["returncode"] == 2
    assert "invalid recipe input" in manifest["failure"]


def test_direct_runner_does_not_use_popen(tmp_path: Path, monkeypatch) -> None:
    spec = _temp_spec(tmp_path)
    monkeypatch.setattr(
        runner_module.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("Popen must not be used"),
    )
    monkeypatch.setattr(runner_module, "_git", lambda *args: "test")
    monkeypatch.setattr(
        runner_module, "_execute_recipe",
        lambda step, *args, **kwargs: _result(step))
    monkeypatch.setattr(runner_module, "_verify_step", lambda step: None)
    assert execute_run(spec, repo_root=ROOT, force=False) == 0


def test_unknown_recipe_id_is_rejected(tmp_path: Path) -> None:
    spec = _temp_spec(tmp_path)
    step = replace(build_plan(spec, repo_root=ROOT)[0], recipe_id="unknown")
    with pytest.raises(ValueError, match="unknown recipe_id"):
        runner_module._execute_recipe(
            step,
            spec,
            repo_root=ROOT,
            run_dir=spec.run_dir,
        )


@pytest.mark.parametrize(
    "result",
    (
        RecipeResult("fail", 0, Path("output"), {}),
        RecipeResult("pass", 1, Path("output"), {}),
    ),
)
def test_recipe_result_status_must_match_exit_code(result: RecipeResult) -> None:
    step = runner_module.PlannedStep(
        "e0",
        "pol.paper1.recipes.foundation_validation.run_foundation_validation",
        "e0",
        Path("output"),
        Path("e0.log"),
        Path("config.json"),
        {},
    )
    with pytest.raises(ValueError, match="status/exit_code mismatch"):
        runner_module._validate_recipe_result(step, result)


def test_recipe_result_output_dir_must_match_plan() -> None:
    step = runner_module.PlannedStep(
        "e0",
        "pol.paper1.recipes.foundation_validation.run_foundation_validation",
        "e0",
        Path("expected"),
        Path("e0.log"),
        Path("config.json"),
        {},
    )
    result = RecipeResult("pass", 0, Path("other"), {})
    with pytest.raises(ValueError, match="output_dir mismatch"):
        runner_module._validate_recipe_result(step, result)


@pytest.mark.parametrize(
    "arguments", [["--help"], ["run", "--help"]]
)
def test_cli_help(arguments: list[str]) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pol", *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout
