"""Thin subprocess orchestration for existing Paper 1 scripts."""
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
from typing import Any

from .datasets import load_master_dataset
from .run_spec import Paper1RunSpec, run_spec_to_resolved_dict


_CHILD_CLEANUP_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class PlannedStep:
    """One existing child script invocation."""

    name: str
    command: tuple[str, ...]
    output_dir: Path
    log_path: Path
    config_path: Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def resolve_run_directory(
    spec: Paper1RunSpec, *, repo_root: Path
) -> tuple[Path, Path]:
    """Validate and return the resolved output root and lexical run directory."""
    root = repo_root.resolve()
    output_root = spec.output_root.resolve()
    filesystem_root = Path(output_root.anchor).resolve()
    forbidden_roots = {
        filesystem_root,
        Path.home().resolve(),
        root,
        root.parent,
    }
    if output_root in forbidden_roots:
        raise ValueError(f"unsafe output root: {output_root}")

    run_dir = output_root / spec.name
    forbidden_run_dirs = {*forbidden_roots, output_root}
    if run_dir.parent != output_root or run_dir in forbidden_run_dirs:
        raise ValueError(f"unsafe run directory: {run_dir}")
    if run_dir.is_symlink():
        raise ValueError(f"run directory must not be a symlink: {run_dir}")
    resolved_target = run_dir.resolve(strict=False)
    if resolved_target != run_dir or resolved_target.parent != output_root:
        raise ValueError(f"run directory escapes output root: {run_dir}")
    return output_root, run_dir


def build_plan(spec: Paper1RunSpec, *, repo_root: Path) -> list[PlannedStep]:
    """Build exact child commands without mutating the filesystem."""
    root = repo_root.resolve()
    _, run_dir = resolve_run_directory(spec, repo_root=root)
    e0_config = spec.experiment_config if spec.kind == "e0" else spec.e0_config
    assert e0_config is not None
    steps = [
        PlannedStep(
            "e0",
            (
                sys.executable,
                str(root / "scripts/paper1/run_e0.py"),
                "--config",
                str(e0_config),
                "--output-dir",
                str(run_dir / "e0"),
                "--overwrite",
            ),
            run_dir / "e0",
            run_dir / "logs/01_e0.log",
            e0_config,
        )
    ]
    if spec.kind == "e2":
        accepted = run_dir / "e0/accepted_production_config.json"
        steps.append(
            PlannedStep(
                "master_dataset",
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
                run_dir / "master_dataset",
                run_dir / "logs/02_master_dataset.log",
                accepted,
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
        if spec.skip_plots:
            command.append("--skip-plots")
        steps.append(
            PlannedStep(
                spec.kind,
                tuple(command),
                output,
                run_dir / f"logs/{number}_{spec.kind}.log",
                spec.experiment_config,
            )
        )
    missing_scripts = [
        step.command[1] for step in steps if not Path(step.command[1]).is_file()
    ]
    if missing_scripts:
        raise ValueError(f"runner child script does not exist: {missing_scripts[0]}")
    return steps


def plan_to_dict(spec: Paper1RunSpec, *, repo_root: Path) -> dict[str, object]:
    """Return the machine-readable, side-effect-free execution plan."""
    _, run_dir = resolve_run_directory(spec, repo_root=repo_root)
    return {
        "schema_version": "paper1-run-plan-v1",
        "run_name": spec.name,
        "experiment_kind": spec.kind,
        "run_dir": str(run_dir),
        "steps": [
            {
                "name": step.name,
                "command": list(step.command),
                "output_dir": str(step.output_dir),
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


def _terminate_child(process: subprocess.Popen[str]) -> tuple[int | None, str | None]:
    """Terminate a running child and report any cleanup failure."""
    try:
        if process.poll() is None:
            process.terminate()
        try:
            return process.wait(timeout=_CHILD_CLEANUP_TIMEOUT_SECONDS), None
        except subprocess.TimeoutExpired:
            process.kill()
            return process.wait(timeout=_CHILD_CLEANUP_TIMEOUT_SECONDS), None
    except (OSError, subprocess.TimeoutExpired) as exc:
        return process.returncode, f"{type(exc).__name__}: {exc}"


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
    if step.name == "master_dataset":
        load_master_dataset(step.output_dir)
        return
    summary_path = step.output_dir / f"{step.name}_summary.json"
    if not summary_path.is_file():
        raise ValueError(f"missing saved output: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "pass":
        raise ValueError(f"{step.name} summary status is not pass")
    if step.name == "e0":
        checks = summary.get("required_checks")
        if not isinstance(checks, dict) or not checks or any(
            value != "pass" for value in checks.values()
        ):
            raise ValueError("E0 required_checks are not all pass")
        for name in (
            "accepted_production_config.json",
            "master_initial_conditions.pt",
            "master_manifest.json",
        ):
            if not (step.output_dir / name).is_file():
                raise ValueError(f"missing saved output: {step.output_dir / name}")
    else:
        if not (step.output_dir / "artifact_manifest.json").is_file():
            raise ValueError("missing artifact_manifest.json")
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


def _manifest(spec: Paper1RunSpec, steps: list[PlannedStep], root: Path) -> dict[str, Any]:
    return {
        "schema_version": "paper1-run-manifest-v1",
        "status": "running",
        "run_name": spec.name,
        "experiment_kind": spec.kind,
        "run_spec_sha256": _sha256(spec.source_path),
        "git_commit": _git(root, ["rev-parse", "HEAD"]),
        "git_dirty_status": _git(root, ["status", "--porcelain"]),
        "python_executable": sys.executable,
        "started_at": _now(),
        "ended_at": None,
        "run_dir": str(spec.run_dir.resolve()),
        "steps": [
            {
                "name": step.name,
                "status": "pending",
                "command": list(step.command),
                "config_path": str(step.config_path),
                "config_sha256": (
                    _sha256(step.config_path) if step.config_path.is_file() else None
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


def execute_run(spec: Paper1RunSpec, *, repo_root: Path, force: bool) -> int:
    """Execute the planned existing scripts and maintain an atomic manifest."""
    root = repo_root.resolve()
    output_root, run_dir = resolve_run_directory(spec, repo_root=root)
    steps = build_plan(spec, repo_root=root)
    if run_dir.exists() or run_dir.is_symlink():
        if not force:
            raise FileExistsError(f"run directory already exists: {run_dir}; pass --force")
        _validate_owned_run_directory(run_dir, spec)
        shutil.rmtree(run_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True)
    resolved = run_spec_to_resolved_dict(spec)
    resolved.update(
        {
            "source_run_spec_sha256": _sha256(spec.source_path),
            "experiment_config_sha256": _sha256(spec.experiment_config),
            "e0_config_sha256": _sha256(spec.e0_config) if spec.e0_config else None,
            "planned_steps": [
                {
                    "name": step.name,
                    "command": list(step.command),
                    "output_dir": str(step.output_dir),
                    "log_path": str(step.log_path),
                }
                for step in steps
            ],
        }
    )
    _atomic_json(run_dir / "resolved_run_spec.json", resolved)
    manifest = _manifest(spec, steps, root)
    manifest_path = run_dir / "run_manifest.json"
    _atomic_json(manifest_path, manifest)

    current_step_index: int | None = None
    current_started_monotonic: float | None = None
    process: subprocess.Popen[str] | None = None
    try:
        for index, step in enumerate(steps):
            current_step_index = index
            record = manifest["steps"][index]
            record["status"] = "running"
            record["started_at"] = _now()
            current_started_monotonic = time.monotonic()
            if step.config_path.is_file():
                record["config_sha256"] = _sha256(step.config_path)
            _atomic_json(manifest_path, manifest)
            environment = os.environ.copy()
            threads = str(spec.torch_threads if step.name in {"e1", "e2"} else 1)
            for variable in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ):
                environment[variable] = threads
            with step.log_path.open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    list(step.command),
                    cwd=root,
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(f"[{step.name}] {line}", end="", flush=True)
                returncode = process.wait()
            process = None
            record["returncode"] = returncode
            if returncode != 0:
                raise RuntimeError(f"step {step.name} exited with code {returncode}")
            _verify_step(step)
            record["status"] = "pass"
            record["ended_at"] = _now()
            record["duration_seconds"] = (
                time.monotonic() - current_started_monotonic
            )
            _atomic_json(manifest_path, manifest)
            current_step_index = None
            current_started_monotonic = None
        manifest["status"] = "pass"
        manifest["ended_at"] = _now()
        manifest["final_result_dir"] = str(run_dir / spec.kind)
        _atomic_json(manifest_path, manifest)
        return 0
    except KeyboardInterrupt:
        cleanup_failure = None
        if process is not None:
            _, cleanup_failure = _terminate_child(process)
        manifest["status"] = "interrupted"
        manifest["failure"] = "KeyboardInterrupt"
        if cleanup_failure is not None:
            manifest["failure"] += f"; child cleanup failed: {cleanup_failure}"
        manifest["ended_at"] = _now()
        if current_step_index is not None:
            record = manifest["steps"][current_step_index]
            if record["status"] == "running":
                record["status"] = "interrupted"
                record["returncode"] = process.returncode if process else None
                record["failure"] = manifest["failure"]
                record["ended_at"] = _now()
            if current_started_monotonic is not None:
                record["duration_seconds"] = (
                    time.monotonic() - current_started_monotonic
                )
        _atomic_json(manifest_path, manifest)
        return 130
    except Exception as exc:
        cleanup_failure = None
        if process is not None:
            _, cleanup_failure = _terminate_child(process)
        manifest["status"] = "fail"
        manifest["failure"] = f"{type(exc).__name__}: {exc}"
        if cleanup_failure is not None:
            manifest["failure"] += f"; child cleanup failed: {cleanup_failure}"
        manifest["ended_at"] = _now()
        if current_step_index is not None:
            record = manifest["steps"][current_step_index]
            if record["status"] == "running":
                record["status"] = "fail"
                record["returncode"] = (
                    process.returncode if process else record["returncode"]
                )
                record["failure"] = manifest["failure"]
                record["ended_at"] = _now()
                if current_started_monotonic is not None:
                    record["duration_seconds"] = (
                        time.monotonic() - current_started_monotonic
                    )
        _atomic_json(manifest_path, manifest)
        return 1
