"""Content-addressed execution for artifact-only plot recipes."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable

from pol.runtime.io import file_sha256

from .registry import get_plot_recipe
from .types import PlotContext, PlotRenderError, PlotTaskSpec


PLOT_MANIFEST_SCHEMA = "pol-plot-task-v1"
PLOT_RUNTIME_VERSION = "1"


def _safe_directory(path: Path, *, parent: Path | None = None) -> Path:
    """Validate a plot-owned lexical directory without following its final link."""
    lexical = path.absolute()
    if parent is not None and lexical.parent != parent:
        raise ValueError(f"unsafe plot directory outside its parent: {lexical}")
    if lexical.is_symlink():
        raise ValueError(f"plot directory must not be a symlink: {lexical}")
    if lexical.exists() and not lexical.is_dir():
        raise ValueError(f"plot directory is not a directory: {lexical}")
    expected = lexical.parent.resolve() / lexical.name
    if lexical.resolve(strict=False) != expected:
        raise ValueError(f"plot directory escapes its lexical parent: {lexical}")
    return lexical


def _remove_plot_directory(path: Path, *, parent: Path) -> None:
    """Remove one validated plot-owned child directory."""
    target = _safe_directory(path, parent=parent)
    if target.exists():
        shutil.rmtree(target)


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _canonical_settings(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not (-float("inf") < value < float("inf")):
            raise ValueError("plot settings must be finite")
        return value
    if isinstance(value, list):
        return [_canonical_settings(item) for item in value]
    if isinstance(value, tuple):
        return [_canonical_settings(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _canonical_settings(value[key])
            for key in sorted(value)
        }
    raise ValueError(f"unsupported plot setting type: {type(value).__name__}")


def _input_records(input_dir: Path, names: Iterable[str]) -> list[dict[str, Any]]:
    if input_dir.is_symlink() or not input_dir.is_dir():
        raise ValueError(f"unsafe or missing plot input directory: {input_dir}")
    records = []
    for name in names:
        path = input_dir / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"missing required plot input: {path}")
        records.append(
            {
                "relative_path": name,
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return records


def _fingerprint(
    recipe_id: str,
    version: str,
    settings: object,
    inputs: list[dict[str, Any]],
) -> str:
    payload = {
        "recipe_id": recipe_id,
        "recipe_version": version,
        "plot_runtime_version": PLOT_RUNTIME_VERSION,
        "settings": settings,
        "inputs": inputs,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reusable(path: Path, fingerprint: str) -> dict[str, Any] | None:
    manifest_path = path / "plot_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("schema_version") != PLOT_MANIFEST_SCHEMA
            or manifest.get("status") != "pass"
            or manifest.get("plot_fingerprint") != fingerprint
        ):
            return None
        for output in manifest["outputs"]:
            target = path / output["relative_path"]
            if (
                target.is_symlink()
                or not target.is_file()
                or target.stat().st_size != output["size_bytes"]
                or file_sha256(target) != output["sha256"]
            ):
                return None
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None
    return manifest


def execute_plot_tasks(
    *,
    experiment_kind: str,
    input_dir: Path,
    figures_dir: Path,
    tasks: tuple[PlotTaskSpec, ...],
) -> list[dict[str, Any]]:
    """Execute or reuse registered plot tasks without changing compute artifacts."""
    outcomes: list[dict[str, Any]] = []
    figures_dir = _safe_directory(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = _safe_directory(figures_dir)
    for task in tasks:
        recipe = get_plot_recipe(task.recipe_id)
        if experiment_kind not in recipe.supported_experiment_kinds:
            raise ValueError(
                f"plot recipe {task.recipe_id} does not support {experiment_kind}"
            )
        validated_settings = recipe.validate_settings(dict(task.settings))
        settings = _canonical_settings(dict(validated_settings))
        inputs = _input_records(input_dir, recipe.required_input_files)
        fingerprint = _fingerprint(
            recipe.recipe_id, recipe.version, settings, inputs
        )
        output_dir = _safe_directory(
            figures_dir / recipe.recipe_id, parent=figures_dir
        )
        reused = _reusable(output_dir, fingerprint) if output_dir.is_dir() else None
        if reused is not None:
            outcomes.append(
                {
                    "recipe_id": recipe.recipe_id,
                    "status": "pass",
                    "executed_or_reused": "reused",
                    "plot_fingerprint": fingerprint,
                    "output_dir": str(output_dir),
                }
            )
            continue
        staging = _safe_directory(
            figures_dir / f".{recipe.recipe_id}.staging", parent=figures_dir
        )
        if staging.exists():
            _remove_plot_directory(staging, parent=figures_dir)
        staging.mkdir(parents=True)
        try:
            result = recipe.render(
                PlotContext(input_dir=input_dir, output_dir=staging, settings=settings)
            )
            outputs = []
            seen_outputs: set[str] = set()
            for item in result.outputs:
                relative = str(item["relative_path"])
                relative_path = Path(relative)
                if (
                    relative_path.is_absolute()
                    or ".." in relative_path.parts
                    or relative in seen_outputs
                ):
                    raise ValueError(f"unsafe or duplicate plot output: {relative}")
                seen_outputs.add(relative)
                path = staging / relative
                if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
                    raise ValueError(f"plot renderer did not create {relative}")
                outputs.append(
                    {
                        **dict(item),
                        "relative_path": relative,
                        "size_bytes": path.stat().st_size,
                        "sha256": file_sha256(path),
                    }
                )
            if not outputs:
                raise ValueError("plot renderer produced no outputs")
            manifest = {
                "schema_version": PLOT_MANIFEST_SCHEMA,
                "status": "pass",
                "recipe_id": recipe.recipe_id,
                "recipe_version": recipe.version,
                "plot_runtime_version": PLOT_RUNTIME_VERSION,
                "plot_fingerprint": fingerprint,
                "inputs": inputs,
                "settings": settings,
                "outputs": outputs,
                "executed_or_reused": "executed",
                "failure": None,
            }
            _atomic_json(staging / "plot_manifest.json", manifest)
            if output_dir.exists():
                _remove_plot_directory(output_dir, parent=figures_dir)
            os.replace(staging, output_dir)
            outcomes.append(
                {
                    "recipe_id": recipe.recipe_id,
                    "status": "pass",
                    "executed_or_reused": "executed",
                    "plot_fingerprint": fingerprint,
                    "output_dir": str(output_dir),
                }
            )
        except Exception as exc:
            partial_outputs: list[dict[str, Any]] = []
            format_failures: list[dict[str, Any]] = []
            if isinstance(exc, PlotRenderError):
                for item in exc.failures:
                    record = dict(item)
                    relative = str(record.get("relative_path", ""))
                    relative_path = Path(relative)
                    if (
                        not relative
                        or relative_path.is_absolute()
                        or ".." in relative_path.parts
                    ):
                        raise ValueError(
                            f"unsafe failed plot output: {relative}"
                        ) from exc
                    failed_path = staging / relative
                    if failed_path.exists() and not failed_path.is_symlink():
                        failed_path.unlink()
                    format_failures.append(record)
                for item in exc.outputs:
                    record = dict(item)
                    relative = str(record["relative_path"])
                    relative_path = Path(relative)
                    path = staging / relative
                    if (
                        relative_path.is_absolute()
                        or ".." in relative_path.parts
                        or path.is_symlink()
                        or not path.is_file()
                        or path.stat().st_size <= 0
                    ):
                        raise ValueError(
                            f"invalid partial plot output: {relative}"
                        ) from exc
                    partial_outputs.append(
                        {
                            **record,
                            "relative_path": relative,
                            "size_bytes": path.stat().st_size,
                            "sha256": file_sha256(path),
                        }
                    )
            elif staging.exists():
                _remove_plot_directory(staging, parent=figures_dir)
            failure_dir = _safe_directory(
                figures_dir / f".{recipe.recipe_id}.failure",
                parent=figures_dir,
            )
            if failure_dir.exists():
                _remove_plot_directory(failure_dir, parent=figures_dir)
            if isinstance(exc, PlotRenderError):
                os.replace(staging, failure_dir)
            else:
                failure_dir.mkdir()
            _atomic_json(
                failure_dir / "plot_manifest.json",
                {
                    "schema_version": PLOT_MANIFEST_SCHEMA,
                    "status": "fail",
                    "recipe_id": recipe.recipe_id,
                    "recipe_version": recipe.version,
                    "plot_runtime_version": PLOT_RUNTIME_VERSION,
                    "plot_fingerprint": fingerprint,
                    "inputs": inputs,
                    "settings": settings,
                    "outputs": partial_outputs,
                    "format_failures": format_failures,
                    "executed_or_reused": "executed",
                    "failure": f"{type(exc).__name__}: {exc}",
                },
            )
            if output_dir.exists():
                _remove_plot_directory(output_dir, parent=figures_dir)
            os.replace(failure_dir, output_dir)
            raise
        except BaseException:
            if staging.exists():
                _remove_plot_directory(staging, parent=figures_dir)
            raise
    return outcomes
