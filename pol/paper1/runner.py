"""Direct recipe orchestration for Paper 1 experiments."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any, Literal, Mapping

from pol.runtime.io import file_sha256, write_strict_json
from pol.runtime.recipe import (
    RecipeInvocation,
    RecipeResult,
    RecipeUsageError,
    numerical_thread_scope,
)
from pol.runtime.path_safety import resolve_safe_run_directory
from pol.paper1.config import canonical_config_json, load_config_json

from .run_spec import Paper1RunSpec, run_spec_to_resolved_dict


@dataclass(frozen=True)
class PlannedStep:
    """One direct recipe invocation and optional legacy reproduction command."""

    recipe_id: Literal["e0", "master_dataset", "e1", "e2"]
    recipe_callable: str
    name: str
    output_dir: Path
    log_path: Path
    config_path: Path
    parameters: Mapping[str, object]
    legacy_equivalent_command: tuple[str, ...] | None = None

    @property
    def logical_invocation(self) -> tuple[str, ...]:
        """Return the actual logical invocation recorded in recipe artifacts."""
        return ("direct_recipe", self.recipe_callable)


def _canonical_config_sha256(path: Path) -> str:
    canonical = canonical_config_json(load_config_json(path))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def science_fingerprint(spec: Paper1RunSpec) -> str:
    """Hash only scientific inputs and compute-affecting execution settings."""
    payload = {
        "kind": spec.kind,
        "experiment_config_sha256": _canonical_config_sha256(
            spec.experiment_config
        ),
        "e0_config_sha256": (
            _canonical_config_sha256(spec.e0_config) if spec.e0_config else None
        ),
        "recipe_protocols": {
            "e0": ("paper1-e0-v2",),
            "e1": ("paper1-e0-v2", "paper1-e1-v2"),
            "e2": (
                "paper1-e0-v2",
                "paper1-master-dataset-v1",
                "paper1-e2-v3",
            ),
        }[spec.kind],
        "torch_threads": spec.torch_threads,
        "batch_size": spec.batch_size,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_run_directory(
    spec: Paper1RunSpec, *, repo_root: Path
) -> tuple[Path, Path]:
    """Validate and return the resolved output root and lexical run directory."""
    protected_paths = [spec.source_path, spec.experiment_config]
    if spec.e0_config is not None:
        protected_paths.append(spec.e0_config)
    return resolve_safe_run_directory(
        name=spec.name,
        output_root=spec.output_root,
        repo_root=repo_root,
        protected_paths=protected_paths,
    )


def build_plan(spec: Paper1RunSpec, *, repo_root: Path) -> list[PlannedStep]:
    """Build exact child commands without mutating the filesystem."""
    root = repo_root.resolve()
    _, run_dir = resolve_run_directory(spec, repo_root=root)
    e0_config = spec.experiment_config if spec.kind == "e0" else spec.e0_config
    assert e0_config is not None
    steps = [
        PlannedStep(
            "e0",
            "pol.paper1.recipes.foundation_validation.run_foundation_validation",
            "e0",
            run_dir / "e0",
            run_dir / "logs/01_e0.log",
            e0_config,
            {
                "config_path": str(e0_config),
                "output_dir": str(run_dir / "e0"),
                "overwrite": True,
                "torch_threads": 1,
            },
            (
                sys.executable,
                str(root / "scripts/paper1/run_e0.py"),
                "--config",
                str(e0_config),
                "--output-dir",
                str(run_dir / "e0"),
                "--overwrite",
            ),
        )
    ]
    if spec.kind == "e2":
        accepted = run_dir / "e0/accepted_production_config.json"
        steps.append(
            PlannedStep(
                "master_dataset",
                "pol.paper1.recipes.master_dataset.run_master_dataset_generation",
                "master_dataset",
                run_dir / "master_dataset",
                run_dir / "logs/02_master_dataset.log",
                accepted,
                {
                    "config_path": str(accepted),
                    "master_initial_conditions": str(
                        run_dir / "e0/master_initial_conditions.pt"
                    ),
                    "output_dir": str(run_dir / "master_dataset"),
                    "overwrite": True,
                    "generate_target": True,
                    "torch_threads": 1,
                },
                (
                    sys.executable,
                    str(root / "scripts/paper1/generate_master_dataset.py"),
                    "--config",
                    str(accepted),
                    "--master-initial-conditions",
                    str(run_dir / "e0/master_initial_conditions.pt"),
                    "--output-dir",
                    str(run_dir / "master_dataset"),
                    "--overwrite",
                ),
            )
        )
    if spec.kind in {"e1", "e2"}:
        number = "02" if spec.kind == "e1" else "03"
        script = root / f"scripts/paper1/run_{spec.kind}.py"
        output = run_dir / spec.kind
        command = [
            sys.executable,
            str(script),
            "--config",
            str(spec.experiment_config),
            "--e0-dir",
            str(run_dir / "e0"),
        ]
        if spec.kind == "e2":
            command.extend(["--dataset-dir", str(run_dir / "master_dataset")])
        command.extend(
            [
                "--output-dir",
                str(output),
                "--overwrite",
                "--torch-threads",
                str(spec.torch_threads),
            ]
        )
        if spec.kind == "e2":
            command.extend(["--batch-size", str(spec.batch_size)])
        command.append("--skip-plots")
        steps.append(
            PlannedStep(
                spec.kind,
                (
                    "pol.paper1.recipes.heat_calibration.run_heat_calibration"
                    if spec.kind == "e1"
                    else (
                        "pol.paper1.recipes.surrogate_parameter_time."
                        "run_surrogate_parameter_time"
                    )
                ),
                spec.kind,
                output,
                run_dir / f"logs/{number}_{spec.kind}.log",
                spec.experiment_config,
                {
                    "config_path": str(spec.experiment_config),
                    "e0_dir": str(run_dir / "e0"),
                    **(
                        {"dataset_dir": str(run_dir / "master_dataset")}
                        if spec.kind == "e2"
                        else {}
                    ),
                    "output_dir": str(output),
                    "overwrite": True,
                    **({"resume": False} if spec.kind == "e2" else {}),
                    "skip_plots": True,
                    "torch_threads": spec.torch_threads,
                    **(
                        {"batch_size": spec.batch_size}
                        if spec.kind == "e2"
                        else {}
                    ),
                },
                tuple(command),
            )
        )
    return steps


def plan_to_dict(spec: Paper1RunSpec, *, repo_root: Path) -> dict[str, object]:
    """Return the machine-readable, side-effect-free execution plan."""
    _, run_dir = resolve_run_directory(spec, repo_root=repo_root)
    return {
        "schema_version": "paper1-run-plan-v1",
        "run_name": spec.name,
        "experiment_kind": spec.kind,
        "run_dir": str(run_dir),
        "science_fingerprint": science_fingerprint(spec),
        "plots": {
            "enabled": spec.plots_enabled,
            "required": spec.plots_required,
            "tasks": [
                {"recipe_id": task.recipe_id, "settings": dict(task.settings)}
                for task in spec.plot_tasks
            ],
        },
        "steps": [
            {
                "name": step.name,
                "execution_mode": "direct_recipe",
                "recipe_id": step.recipe_id,
                "recipe_callable": step.recipe_callable,
                "config_path": str(step.config_path),
                "output_dir": str(step.output_dir),
                "parameters": dict(step.parameters),
                "legacy_equivalent_command": (
                    list(step.legacy_equivalent_command)
                    if step.legacy_equivalent_command
                    else None
                ),
            }
            for step in build_plan(spec, repo_root=repo_root)
        ],
    }


def _git(root: Path, arguments: list[str]) -> str:
    try:
        process = subprocess.run(
            ["git", *arguments], cwd=root, capture_output=True, text=True, check=False
        )
        return process.stdout.strip() if process.returncode == 0 else "unknown"
    except OSError:
        return "unknown"


def _validate_owned_run_directory(run_dir: Path, spec: Paper1RunSpec) -> None:
    """Require an existing replacement target to be owned by this runner."""
    if run_dir.is_symlink():
        raise ValueError(f"run directory must not be a symlink: {run_dir}")
    if not run_dir.is_dir():
        raise ValueError(f"existing run path is not a directory: {run_dir}")
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.is_symlink():
        raise ValueError(f"run manifest must not be a symlink: {manifest_path}")
    if not manifest_path.is_file():
        raise ValueError(f"existing run directory is not runner-owned: {run_dir}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid runner ownership manifest: {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"runner ownership manifest must be an object: {manifest_path}")
    if manifest.get("schema_version") != "paper1-run-manifest-v1":
        raise ValueError(f"runner ownership manifest schema mismatch: {manifest_path}")
    if manifest.get("run_name") != spec.name:
        raise ValueError(f"runner ownership manifest run_name mismatch: {manifest_path}")
    if manifest.get("run_dir") != str(run_dir):
        raise ValueError(f"runner ownership manifest run_dir mismatch: {manifest_path}")


def _verify_step(step: PlannedStep) -> None:
    from .artifact_contracts import validate_step_artifacts

    if step.name != "master_dataset":
        summary_path = step.output_dir / f"{step.name}_summary.json"
        if not summary_path.is_file():
            raise ValueError(f"missing saved output: {summary_path}")
    validate_step_artifacts(step.name, step.output_dir)
    if step.name == "master_dataset":
        return
    summary_path = step.output_dir / f"{step.name}_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if step.name == "e2":
        if summary.get("test_evaluated") is not True:
            raise ValueError("E2 test_evaluated is not true")
        event_log = json.loads(
            (step.output_dir / "event_log.json").read_text(encoding="utf-8")
        )
        events = event_log.get("events", event_log)
        names = [event["event"] for event in events]
        positions = [
            names.index(name)
            for name in (
                "freeze_read_back",
                "first_test_state_solve",
                "first_test_metric",
            )
        ]
        if positions != sorted(positions) or len(set(positions)) != 3:
            raise ValueError("E2 freeze/test event order is invalid")


def _manifest(
    spec: Paper1RunSpec,
    steps: list[PlannedStep],
    root: Path,
    *,
    run_dir: Path,
) -> dict[str, Any]:
    return {
        "schema_version": "paper1-run-manifest-v1",
        "status": "running",
        "run_name": spec.name,
        "experiment_kind": spec.kind,
        "run_spec_sha256": file_sha256(spec.source_path),
        "git_commit": _git(root, ["rev-parse", "HEAD"]),
        "git_dirty_status": _git(root, ["status", "--porcelain"]),
        "python_executable": sys.executable,
        "started_at": _now(),
        "ended_at": None,
        "run_dir": str(run_dir),
        "science_fingerprint": science_fingerprint(spec),
        "compute_status": "running",
        "plot_status": "pending" if spec.plots_enabled else "disabled",
        "plot_tasks": [],
        "plots": {
            "enabled": spec.plots_enabled,
            "required": spec.plots_required,
            "tasks": [
                {"recipe_id": task.recipe_id, "settings": dict(task.settings)}
                for task in spec.plot_tasks
            ],
        },
        "steps": [
            {
                "name": step.name,
                "status": "pending",
                "execution_mode": "direct_recipe",
                "recipe_id": step.recipe_id,
                "recipe_callable": step.recipe_callable,
                "command": list(step.logical_invocation),
                "parameters": dict(step.parameters),
                "legacy_equivalent_command": (
                    list(step.legacy_equivalent_command)
                    if step.legacy_equivalent_command
                    else None
                ),
                "config_path": str(step.config_path),
                "config_sha256": (
                    file_sha256(step.config_path) if step.config_path.is_file() else None
                ),
                "output_dir": str(step.output_dir),
                "log_path": str(step.log_path),
                "started_at": None,
                "ended_at": None,
                "duration_seconds": None,
                "returncode": None,
                "failure": None,
            }
            for step in steps
        ],
        "final_result_dir": None,
        "failure": None,
    }


def _execute_recipe(
    step: PlannedStep,
    spec: Paper1RunSpec,
    *,
    repo_root: Path,
    run_dir: Path,
) -> RecipeResult:
    """Lazy-import and execute one planned recipe in the current process."""
    threads = spec.torch_threads if step.recipe_id in {"e1", "e2"} else 1
    assert threads is not None
    invocation = RecipeInvocation(
        repo_root=repo_root,
        working_directory=repo_root,
        command=step.logical_invocation,
        torch_threads=threads,
    )
    with numerical_thread_scope(threads):
        if step.recipe_id == "e0":
            from .recipes.foundation_validation import run_foundation_validation

            return run_foundation_validation(
                step.config_path,
                step.output_dir,
                overwrite=True,
                invocation=invocation,
            )
        if step.recipe_id == "master_dataset":
            from .recipes.master_dataset import run_master_dataset_generation

            return run_master_dataset_generation(
                step.config_path,
                step.output_dir,
                overwrite=True,
                generate_target=True,
                master_initial_conditions=run_dir / "e0/master_initial_conditions.pt",
                invocation=invocation,
            )
        if step.recipe_id == "e1":
            from .recipes.heat_calibration import run_heat_calibration

            return run_heat_calibration(
                step.config_path,
                run_dir / "e0",
                step.output_dir,
                overwrite=True,
                skip_plots=True,
                # Unified execution keeps compute artifacts immutable; plotting
                # is a separate artifact-only task.
                invocation=invocation,
            )
        if step.recipe_id == "e2":
            from .recipes.surrogate_parameter_time import (
                run_surrogate_parameter_time,
            )

            assert spec.batch_size is not None
            return run_surrogate_parameter_time(
                step.config_path,
                run_dir / "e0",
                run_dir / "master_dataset",
                step.output_dir,
                overwrite=True,
                resume=False,
                skip_plots=True,
                batch_size=spec.batch_size,
                invocation=invocation,
                cache_root=run_dir.parent / ".cache" / "paper1_e2_v3",
            )
        raise ValueError(f"unknown recipe_id: {step.recipe_id}")


def _validate_recipe_result(step: PlannedStep, result: RecipeResult) -> None:
    """Reject internally inconsistent or misdirected recipe results."""
    expected_status = "pass" if result.exit_code == 0 else "fail"
    if result.status != expected_status:
        raise ValueError(
            "recipe result status/exit_code mismatch: "
            f"status={result.status!r}, exit_code={result.exit_code}"
        )
    if result.output_dir != step.output_dir:
        raise ValueError(
            "recipe result output_dir mismatch: "
            f"expected {step.output_dir}, got {result.output_dir}"
        )


def _verify_compute_for_plots(spec: Paper1RunSpec, steps: list[PlannedStep]) -> None:
    for step in steps:
        _verify_step(step)
    output = steps[-1].output_dir
    if spec.kind == "e1":
        from .matrix_plugins.e1_resolution import E1ResolutionPlugin

        E1ResolutionPlugin().validate_cell(output, cell_plots=False)
    elif spec.kind == "e2":
        from .e2_qa import validate_resume_output

        if not validate_resume_output(output):
            raise ValueError("E2 compute output is not a reusable pass output")


def _run_plots(
    spec: Paper1RunSpec, *, run_dir: Path
) -> list[dict[str, Any]]:
    if not spec.plots_enabled:
        return []
    from pol.plots.runtime import execute_plot_tasks

    return execute_plot_tasks(
        experiment_kind=spec.kind,
        input_dir=run_dir / spec.kind,
        figures_dir=run_dir / "figures",
        tasks=spec.plot_tasks,
    )


def _record_plot_request(
    spec: Paper1RunSpec,
    *,
    run_dir: Path,
    manifest: dict[str, Any],
    request_mode: str,
    status: str,
    outcomes: list[dict[str, Any]],
    failure: str | None,
) -> None:
    from pol.plots.provenance import build_plot_request, write_plot_request

    request = build_plot_request(
        source_spec_path=spec.source_path,
        compute_fingerprint=science_fingerprint(spec),
        tasks=spec.plot_tasks,
        request_mode=request_mode,
        status=status,
        outcomes=outcomes,
        failure=failure,
    )
    write_plot_request(run_dir, request)
    if request_mode == "initial_run":
        manifest["initial_plot_request"] = request
    manifest["last_plot_request"] = request


def execute_run(
    spec: Paper1RunSpec,
    *,
    repo_root: Path,
    force: bool,
    plots_only: bool = False,
) -> int:
    """Execute recipes directly and maintain an atomic run manifest."""
    root = repo_root.resolve()
    output_root, run_dir = resolve_run_directory(spec, repo_root=root)
    steps = build_plan(spec, repo_root=root)
    if plots_only:
        if not spec.plots_enabled or not spec.plot_tasks:
            raise ValueError("--plots-only requires at least one enabled plot task")
        if not (run_dir.exists() or run_dir.is_symlink()):
            raise FileNotFoundError(f"plots-only requires existing run directory: {run_dir}")
        _validate_owned_run_directory(run_dir, spec)
        manifest_path = run_dir / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        try:
            expected_fingerprint = science_fingerprint(spec)
            if manifest.get("science_fingerprint") != expected_fingerprint:
                raise ValueError("science fingerprint mismatch for --plots-only")
            _verify_compute_for_plots(spec, steps)
        except Exception as exc:
            manifest["status"] = "fail"
            manifest["compute_status"] = "fail"
            manifest["plot_status"] = "not_run"
            manifest["plot_tasks"] = []
            manifest["failure"] = f"{type(exc).__name__}: {exc}"
            manifest["ended_at"] = _now()
            _record_plot_request(
                spec,
                run_dir=run_dir,
                manifest=manifest,
                request_mode="plots_only",
                status="not_run",
                outcomes=[],
                failure=manifest["failure"],
            )
            write_strict_json(manifest_path, manifest)
            return 1
        try:
            outcomes = _run_plots(spec, run_dir=run_dir)
            manifest["plot_tasks"] = outcomes
            manifest["plot_status"] = "pass" if outcomes else "disabled"
            manifest["status"] = "pass"
            manifest["failure"] = None
            manifest["ended_at"] = _now()
            _record_plot_request(
                spec,
                run_dir=run_dir,
                manifest=manifest,
                request_mode="plots_only",
                status=manifest["plot_status"],
                outcomes=outcomes,
                failure=None,
            )
            write_strict_json(manifest_path, manifest)
            return 0
        except Exception as exc:
            manifest["plot_status"] = "fail"
            manifest["plot_tasks"] = []
            manifest["failure"] = f"{type(exc).__name__}: {exc}"
            manifest["status"] = "fail" if spec.plots_required else "pass"
            manifest["ended_at"] = _now()
            _record_plot_request(
                spec,
                run_dir=run_dir,
                manifest=manifest,
                request_mode="plots_only",
                status="fail",
                outcomes=[],
                failure=manifest["failure"],
            )
            write_strict_json(manifest_path, manifest)
            return 1 if spec.plots_required else 0
    if run_dir.exists() or run_dir.is_symlink():
        if not force:
            _validate_owned_run_directory(run_dir, spec)
            manifest_path = run_dir / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected = science_fingerprint(spec)
            if manifest.get("science_fingerprint") != expected:
                raise FileExistsError(
                    f"run name exists with a different science fingerprint: "
                    f"{run_dir}; choose another name or pass --force"
                )
            if (
                manifest.get("status") != "pass"
                or manifest.get("compute_status") != "pass"
            ):
                raise FileExistsError(
                    f"existing run is not a complete pass output: {run_dir}; "
                    "pass --force"
                )
            for step in steps:
                _verify_step(step)
            if spec.plots_enabled:
                outcomes = _run_plots(spec, run_dir=run_dir)
                manifest["plot_tasks"] = outcomes
                manifest["plot_status"] = "pass"
                _record_plot_request(
                    spec,
                    run_dir=run_dir,
                    manifest=manifest,
                    request_mode="reuse",
                    status="pass",
                    outcomes=outcomes,
                    failure=None,
                )
                write_strict_json(manifest_path, manifest)
            return 0
        _validate_owned_run_directory(run_dir, spec)
        shutil.rmtree(run_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True)
    resolved = run_spec_to_resolved_dict(spec, run_dir=run_dir)
    resolved.update(
        {
            "source_run_spec_sha256": file_sha256(spec.source_path),
            "experiment_config_sha256": file_sha256(spec.experiment_config),
            "e0_config_sha256": file_sha256(spec.e0_config) if spec.e0_config else None,
            "planned_steps": [
                {
                    "name": step.name,
                    "execution_mode": "direct_recipe",
                    "recipe_id": step.recipe_id,
                    "recipe_callable": step.recipe_callable,
                    "parameters": dict(step.parameters),
                    "legacy_equivalent_command": (
                        list(step.legacy_equivalent_command)
                        if step.legacy_equivalent_command
                        else None
                    ),
                    "output_dir": str(step.output_dir),
                    "log_path": str(step.log_path),
                }
                for step in steps
            ],
        }
    )
    write_strict_json(run_dir / "resolved_run_spec.json", resolved)
    manifest = _manifest(spec, steps, root, run_dir=run_dir)
    manifest_path = run_dir / "run_manifest.json"
    write_strict_json(manifest_path, manifest)

    current_step_index: int | None = None
    current_started_monotonic: float | None = None
    try:
        for index, step in enumerate(steps):
            current_step_index = index
            record = manifest["steps"][index]
            record["status"] = "running"
            record["started_at"] = _now()
            current_started_monotonic = time.monotonic()
            if step.config_path.is_file():
                record["config_sha256"] = file_sha256(step.config_path)
            write_strict_json(manifest_path, manifest)
            with step.log_path.open("w", encoding="utf-8") as log:
                try:
                    result = _execute_recipe(
                        step, spec, repo_root=root, run_dir=run_dir)
                    payload = json.dumps(
                        result.console_payload,
                        sort_keys=True,
                        allow_nan=False,
                    )
                    log.write(payload + "\n")
                    log.flush()
                    print(f"[{step.name}] {payload}", flush=True)
                    record["returncode"] = result.exit_code
                    _validate_recipe_result(step, result)
                    if result.exit_code != 0:
                        raise RuntimeError(
                            f"step {step.name} exited with code {result.exit_code}")
                    _verify_step(step)
                except BaseException as exc:
                    if isinstance(exc, RecipeUsageError):
                        record["returncode"] = 2
                    log.write(traceback.format_exc())
                    log.flush()
                    raise
            record["status"] = "pass"
            record["ended_at"] = _now()
            record["duration_seconds"] = (
                time.monotonic() - current_started_monotonic
            )
            write_strict_json(manifest_path, manifest)
            current_step_index = None
            current_started_monotonic = None
        manifest["compute_status"] = "pass"
        try:
            outcomes = _run_plots(spec, run_dir=run_dir)
            manifest["plot_tasks"] = outcomes
            manifest["plot_status"] = "pass" if outcomes else "disabled"
            _record_plot_request(
                spec,
                run_dir=run_dir,
                manifest=manifest,
                request_mode="initial_run",
                status=manifest["plot_status"],
                outcomes=outcomes,
                failure=None,
            )
        except Exception as exc:
            manifest["plot_status"] = "fail"
            manifest["failure"] = f"{type(exc).__name__}: {exc}"
            manifest["status"] = "fail" if spec.plots_required else "pass"
            manifest["ended_at"] = _now()
            manifest["final_result_dir"] = str(run_dir / spec.kind)
            _record_plot_request(
                spec,
                run_dir=run_dir,
                manifest=manifest,
                request_mode="initial_run",
                status="fail",
                outcomes=[],
                failure=manifest["failure"],
            )
            write_strict_json(manifest_path, manifest)
            return 1 if spec.plots_required else 0
        manifest["status"] = "pass"
        manifest["ended_at"] = _now()
        manifest["final_result_dir"] = str(run_dir / spec.kind)
        write_strict_json(manifest_path, manifest)
        return 0
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        manifest["compute_status"] = "interrupted"
        manifest["failure"] = "KeyboardInterrupt"
        manifest["ended_at"] = _now()
        if current_step_index is not None:
            record = manifest["steps"][current_step_index]
            if record["status"] == "running":
                record["status"] = "interrupted"
                record["failure"] = manifest["failure"]
                record["ended_at"] = _now()
            if current_started_monotonic is not None:
                record["duration_seconds"] = (
                    time.monotonic() - current_started_monotonic
                )
        write_strict_json(manifest_path, manifest)
        return 130
    except Exception as exc:
        manifest["status"] = "fail"
        manifest["compute_status"] = "fail"
        manifest["failure"] = f"{type(exc).__name__}: {exc}"
        manifest["ended_at"] = _now()
        if current_step_index is not None:
            record = manifest["steps"][current_step_index]
            if record["status"] == "running":
                record["status"] = "fail"
                record["failure"] = manifest["failure"]
                record["ended_at"] = _now()
                if current_started_monotonic is not None:
                    record["duration_seconds"] = (
                        time.monotonic() - current_started_monotonic
                    )
        write_strict_json(manifest_path, manifest)
        return 1
