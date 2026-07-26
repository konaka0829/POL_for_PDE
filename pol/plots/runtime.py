"""Content-addressed execution for artifact-only plot recipes."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping

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


def _safe_relative_path(raw: object) -> Path:
    relative = Path(str(raw))
    if (
        not str(raw)
        or relative.is_absolute()
        or ".." in relative.parts
        or relative == Path(".")
    ):
        raise ValueError(f"unsafe or duplicate plot output: {raw!r}")
    return relative


def _tree_files(root: Path) -> set[str]:
    """Return an exact regular-file inventory while rejecting every symlink."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"unsafe plot tree root: {root}")
    files: set[str] = set()
    pending = [root]
    while pending:
        directory = pending.pop()
        for path in directory.iterdir():
            if path.is_symlink():
                raise ValueError(f"plot tree contains symlink: {path}")
            if path.is_dir():
                pending.append(path)
            elif path.is_file():
                files.add(path.relative_to(root).as_posix())
            else:
                raise ValueError(f"plot tree contains non-regular entry: {path}")
    return files


def _output_records(
    root: Path, raw_records: Iterable[object]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_records:
        if not isinstance(raw, dict):
            raise ValueError("plot output record must be an object")
        relative = _safe_relative_path(raw.get("relative_path")).as_posix()
        if relative in seen:
            raise ValueError(f"duplicate plot output: {relative}")
        seen.add(relative)
        path = root / relative
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"plot renderer did not create {relative}")
        records.append(
            {
                **raw,
                "relative_path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return records


def _verify_exact_tree(
    root: Path, records: Iterable[Mapping[str, Any]], *, include_manifest: bool
) -> None:
    declared = {str(record["relative_path"]) for record in records}
    if include_manifest:
        declared.add("plot_manifest.json")
    actual = _tree_files(root)
    if actual != declared:
        raise ValueError(
            "plot output set mismatch; "
            f"missing={sorted(declared-actual)}, extra={sorted(actual-declared)}"
        )


def _reusable(path: Path, fingerprint: str) -> dict[str, Any] | None:
    manifest_path = path / "plot_manifest.json"
    try:
        if manifest_path.is_symlink() or not manifest_path.is_file():
            return None
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema_version") != PLOT_MANIFEST_SCHEMA
            or manifest.get("status") != "pass"
            or manifest.get("plot_fingerprint") != fingerprint
        ):
            return None
        outputs = manifest.get("outputs")
        if not isinstance(outputs, list):
            return None
        records = _output_records(path, outputs)
        _verify_exact_tree(path, records, include_manifest=True)
        for raw, output in zip(outputs, records):
            target = path / str(output["relative_path"])
            if (
                target.stat().st_size != raw.get("size_bytes")
                or file_sha256(target) != raw.get("sha256")
            ):
                return None
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None
    return manifest


def _complete_output(path: Path) -> bool:
    """Return whether an existing task directory is an intact pass output."""
    try:
        manifest_path = path / "plot_manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            return False
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        fingerprint = manifest.get("plot_fingerprint")
        return isinstance(fingerprint, str) and _reusable(
            path, fingerprint
        ) is not None
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def _publish_plot_directory(
    staging: Path, output_dir: Path, *, figures_dir: Path
) -> None:
    """Atomically replace one complete plot task directory with rollback."""
    backup = _safe_directory(
        figures_dir / f".{output_dir.name}.backup", parent=figures_dir
    )
    if backup.exists():
        _remove_plot_directory(backup, parent=figures_dir)
    moved_old = False
    if output_dir.exists():
        os.replace(output_dir, backup)
        moved_old = True
    try:
        os.replace(staging, output_dir)
    except BaseException:
        if moved_old and backup.exists() and not output_dir.exists():
            os.replace(backup, output_dir)
        raise
    if backup.exists():
        _remove_plot_directory(backup, parent=figures_dir)


def _failure_manifest(
    *,
    recipe_id: str,
    recipe_version: str,
    fingerprint: str,
    inputs: list[dict[str, Any]],
    settings: object,
    outputs: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    exc: BaseException,
) -> dict[str, Any]:
    return {
        "schema_version": PLOT_MANIFEST_SCHEMA,
        "status": "fail",
        "recipe_id": recipe_id,
        "recipe_version": recipe_version,
        "plot_runtime_version": PLOT_RUNTIME_VERSION,
        "plot_fingerprint": fingerprint,
        "inputs": inputs,
        "settings": settings,
        "outputs": outputs,
        "format_failures": failures,
        "executed_or_reused": "executed",
        "failure": f"{type(exc).__name__}: {exc}",
    }


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
        previous_complete = (
            _complete_output(output_dir) if output_dir.is_dir() else False
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
            try:
                result = recipe.render(
                    PlotContext(
                        input_dir=input_dir,
                        output_dir=staging,
                        settings=settings,
                    )
                )
                outputs = _output_records(staging, result.outputs)
            except PlotRenderError as exc:
                failures: list[dict[str, Any]] = []
                for raw in exc.failures:
                    if not isinstance(raw, dict):
                        raise ValueError(
                            "plot failure record must be an object"
                        ) from exc
                    record = dict(raw)
                    relative = _safe_relative_path(
                        record.get("relative_path")
                    ).as_posix()
                    failed_path = staging / relative
                    if failed_path.is_symlink():
                        raise ValueError(
                            f"failed plot output is a symlink: {relative}"
                        ) from exc
                    if failed_path.exists():
                        if not failed_path.is_file():
                            raise ValueError(
                                f"failed plot output is not regular: {relative}"
                            ) from exc
                        failed_path.unlink()
                    record["relative_path"] = relative
                    failures.append(record)
                outputs = _output_records(staging, exc.outputs)
                _verify_exact_tree(staging, outputs, include_manifest=False)
                manifest = _failure_manifest(
                    recipe_id=recipe.recipe_id,
                    recipe_version=recipe.version,
                    fingerprint=fingerprint,
                    inputs=inputs,
                    settings=settings,
                    outputs=outputs,
                    failures=failures,
                    exc=exc,
                )
                _atomic_json(staging / "plot_manifest.json", manifest)
                _verify_exact_tree(staging, outputs, include_manifest=True)
                if not previous_complete:
                    _publish_plot_directory(
                        staging, output_dir, figures_dir=figures_dir
                    )
                raise
            if not outputs:
                raise ValueError("plot renderer produced no outputs")
            _verify_exact_tree(staging, outputs, include_manifest=False)
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
            _verify_exact_tree(staging, outputs, include_manifest=True)
            _publish_plot_directory(staging, output_dir, figures_dir=figures_dir)
            outcomes.append(
                {
                    "recipe_id": recipe.recipe_id,
                    "status": "pass",
                    "executed_or_reused": "executed",
                    "plot_fingerprint": fingerprint,
                    "output_dir": str(output_dir),
                }
            )
        finally:
            if staging.exists():
                _remove_plot_directory(staging, parent=figures_dir)
    return outcomes
