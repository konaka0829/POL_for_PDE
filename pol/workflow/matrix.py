"""Generic process-isolated matrix planning, execution, and resume."""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import time
from typing import Any

from pol.runtime.io import file_sha256, write_csv, write_strict_json
from pol.runtime.hashing import stable_object_hash
from pol.runtime.path_safety import resolve_safe_run_directory

from .matrix_spec import MatrixRunSpec, expand_matrix
from .matrix_worker import execute_matrix_cell
from .registry import get_matrix_plugin, matrix_plugin_factory_path
from .types import MatrixCell


MATRIX_MANIFEST_SCHEMA = "paper1-matrix-manifest-v3"


def _canonical_config_sha256(path: Path, *, plugin: Any) -> str:
    canonical = plugin.canonical_config(plugin.load_base(path))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _resolved_run_dir(
    spec: MatrixRunSpec,
    repo_root: Path,
    *,
    extra_protected_paths: tuple[Path, ...] = (),
) -> tuple[Path, Path]:
    return resolve_safe_run_directory(
        name=spec.name,
        output_root=spec.output_root,
        repo_root=repo_root,
        protected_paths=(
            spec.source_path,
            spec.base_config,
            *(item.config_path for item in spec.dependencies if item.config_path),
            *extra_protected_paths,
        ),
    )


def _prepare(spec: MatrixRunSpec) -> tuple[Any, list[MatrixCell], list[dict[str, Any]], dict[str, int]]:
    plugin = get_matrix_plugin(spec.aggregation_kind)
    base = plugin.load_base(spec.base_config)
    cells, invalid, raw_counts = expand_matrix(spec, base=base, plugin=plugin)
    return plugin, cells, invalid, raw_counts


def _compute_fingerprint(
    spec: MatrixRunSpec,
    cells: list[MatrixCell],
    *,
    dependency_identities: tuple[dict[str, Any], ...],
    plugin: Any | None = None,
) -> str:
    payload = _compute_fingerprint_payload(
        spec,
        cells,
        dependency_identities=dependency_identities,
        plugin=plugin,
    )
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _compute_fingerprint_payload(
    spec: MatrixRunSpec,
    cells: list[MatrixCell],
    *,
    dependency_identities: tuple[dict[str, Any], ...],
    plugin: Any | None = None,
) -> dict[str, Any]:
    if plugin is None:
        plugin = get_matrix_plugin(spec.aggregation_kind)
    return {
        "schema_version": spec.schema_version,
        "base_config_sha256": _canonical_config_sha256(
            spec.base_config, plugin=plugin
        ),
        "dependencies": dependency_identities,
        "aggregation_kind": spec.aggregation_kind,
        "plugin_id": plugin.plugin_id,
        "experiment_kind": plugin.experiment_kind,
        "aggregation_protocol": plugin.matrix_protocol_version,
        "recipe_protocols": plugin.recipe_protocol_versions,
        "torch_threads_per_job": spec.torch_threads_per_job,
        "cells": [
            {
                "config_sha256": cell.config_sha256,
            }
            for cell in cells
        ],
    }


def _artifact_contract_fingerprint(
    compute_fingerprint: str, *, cell_plots: bool
) -> str:
    payload = {
        "compute_fingerprint": compute_fingerprint,
        "cell_plots": cell_plots,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def matrix_plan_to_dict(
    spec: MatrixRunSpec, *, repo_root: Path
) -> dict[str, object]:
    """Build the complete deterministic matrix plan without filesystem writes."""
    _, run_dir = _resolved_run_dir(spec, repo_root)
    plugin, cells, invalid, raw_counts = _prepare(spec)
    return {
        "schema_version": "paper1-matrix-plan-v1",
        "execution_mode": "process_isolated_matrix",
        "run_name": spec.name,
        "run_dir": str(run_dir),
        "aggregation_kind": spec.aggregation_kind,
        "jobs": spec.jobs,
        "torch_threads_per_job": spec.torch_threads_per_job,
        "raw_run_counts": raw_counts,
        "unique_valid_cells": len(cells),
        "invalid_runs": invalid,
        "plots": {
            "enabled": spec.plots_enabled,
            "required": spec.plots_required,
            "tasks": [
                {"recipe_id": task.recipe_id, "settings": dict(task.settings)}
                for task in spec.plot_tasks
            ],
        },
        **plugin.plan_summary(cells),
        "cells": [_cell_plan_record(cell, run_dir) for cell in cells],
    }


def _cell_plan_record(cell: MatrixCell, run_dir: Path) -> dict[str, Any]:
    return {
        "run_index": cell.run_index,
        "cell_id": cell.cell_id,
        "config_sha256": cell.config_sha256,
        "human_slug": cell.human_slug,
        "experiment_memberships": list(cell.experiment_memberships),
        "metadata": dict(cell.metadata),
        "config_path": str(run_dir / "generated_configs" / f"{cell.cell_id}.json"),
        "output_dir": str(run_dir / "cells" / cell.cell_id),
        "log_path": str(run_dir / "logs" / f"{cell.cell_id}.log"),
    }


def _validate_owned_matrix(run_dir: Path, spec: MatrixRunSpec) -> dict[str, Any]:
    manifest_path = run_dir / "matrix_manifest.json"
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise ValueError(f"unsafe existing matrix run directory: {run_dir}")
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError(f"existing directory is not matrix-runner-owned: {run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != MATRIX_MANIFEST_SCHEMA
        or manifest.get("run_name") != spec.name
        or manifest.get("run_dir") != str(run_dir)
    ):
        raise ValueError(f"matrix ownership manifest mismatch: {manifest_path}")
    return manifest


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    write_csv(path, rows, fieldnames=fields)


def _validate_aggregate(
    aggregate_dir: Path, *, plugin: Any
) -> list[dict[str, Any]]:
    """Validate the exact aggregate contract and return integrity records."""
    if aggregate_dir.is_symlink() or not aggregate_dir.is_dir():
        raise ValueError(f"unsafe or missing aggregate directory: {aggregate_dir}")
    expected = set(plugin.aggregate_artifact_names())
    actual: set[str] = set()
    for path in aggregate_dir.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"aggregate contains non-regular artifact: {path}")
        actual.add(path.name)
    if actual != expected:
        raise ValueError(
            "matrix aggregate artifact set does not match its contract; "
            f"missing={sorted(expected-actual)}, extra={sorted(actual-expected)}"
        )
    plugin.validate_aggregate(aggregate_dir)
    records = []
    for name in sorted(expected):
        path = aggregate_dir / name
        if path.stat().st_size <= 0:
            raise ValueError(f"aggregate artifact is empty: {path}")
        records.append(
            {
                "relative_path": name,
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return records


def _remove_matrix_child(path: Path, *, run_dir: Path) -> None:
    if path.parent != run_dir or path.is_symlink():
        raise ValueError(f"unsafe matrix-owned directory: {path}")
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"matrix-owned path is not a directory: {path}")
        shutil.rmtree(path)


def _collect_and_publish_aggregate(
    *,
    run_dir: Path,
    plugin: Any,
    cells: list[MatrixCell],
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """Collect into staging and atomically publish with rollback."""
    aggregate = run_dir / "aggregate"
    staging = run_dir / ".aggregate.staging"
    backup = run_dir / ".aggregate.backup"
    for temporary in (staging, backup):
        if temporary.exists() or temporary.is_symlink():
            _remove_matrix_child(temporary, run_dir=run_dir)
    staging.mkdir()
    moved_old = False
    try:
        counts = plugin.collect(staging, run_dir / "cells", cells)
        records = _validate_aggregate(staging, plugin=plugin)
        if aggregate.exists() or aggregate.is_symlink():
            if aggregate.is_symlink() or not aggregate.is_dir():
                raise ValueError(f"unsafe aggregate publish target: {aggregate}")
            os.replace(aggregate, backup)
            moved_old = True
        try:
            os.replace(staging, aggregate)
        except BaseException:
            if moved_old and backup.exists() and not aggregate.exists():
                os.replace(backup, aggregate)
            raise
        if backup.exists():
            _remove_matrix_child(backup, run_dir=run_dir)
        return counts, records
    finally:
        if staging.exists() or staging.is_symlink():
            _remove_matrix_child(staging, run_dir=run_dir)


def _shutdown_process_pool(
    pool: concurrent.futures.ProcessPoolExecutor,
    futures: list[concurrent.futures.Future[Any]],
    *,
    timeout_seconds: float = 2.0,
) -> None:
    """Cancel pending work and bound termination of spawned worker processes."""
    for future in futures:
        future.cancel()
    processes = list((getattr(pool, "_processes", None) or {}).values())
    pool.shutdown(wait=False, cancel_futures=True)
    deadline = time.monotonic() + timeout_seconds
    for process in processes:
        remaining = max(0.0, deadline - time.monotonic())
        process.join(remaining)
    for process in processes:
        if process.is_alive():
            process.terminate()
    deadline = time.monotonic() + timeout_seconds
    for process in processes:
        remaining = max(0.0, deadline - time.monotonic())
        process.join(remaining)
    for process in processes:
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(timeout_seconds)


def _reuse_result(
    plugin: Any,
    cell: MatrixCell,
    output_dir: Path,
    *,
    cell_plots: bool,
) -> dict[str, Any] | None:
    try:
        manifest_hash = plugin.validate_cell(
            output_dir,
            cell_plots=cell_plots,
            expected_config_sha256=cell.config_sha256,
            expected_config_path=(
                output_dir.parents[1]
                / "generated_configs"
                / f"{cell.cell_id}.json"
            ),
        )
    except Exception:
        return None
    return {
        "run_index": cell.run_index,
        "cell_id": cell.cell_id,
        "status": "pass",
        "executed_or_reused": "reused",
        "return_code": 0,
        "output_dir": str(output_dir),
        "artifact_manifest_sha256": manifest_hash,
        "failure_type": None,
        "failure_message": None,
    }


def _run_matrix_plots(
    spec: MatrixRunSpec, run_dir: Path, *, plugin: Any
) -> list[dict[str, Any]]:
    if not spec.plots_enabled:
        return []
    from pol.plots.runtime import execute_plot_tasks

    return execute_plot_tasks(
        experiment_kind=plugin.plot_experiment_kind,
        input_dir=run_dir / "aggregate",
        figures_dir=run_dir / "figures",
        tasks=spec.plot_tasks,
    )


def _record_matrix_plot_request(
    spec: MatrixRunSpec,
    *,
    run_dir: Path,
    manifest: dict[str, Any],
    compute_fingerprint: str,
    request_mode: str,
    status: str,
    outcomes: list[dict[str, Any]],
    failure: str | None,
) -> None:
    from pol.plots.provenance import build_plot_request, write_plot_request

    request = build_plot_request(
        source_spec_path=spec.source_path,
        compute_fingerprint=compute_fingerprint,
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


def execute_matrix_run(
    spec: MatrixRunSpec,
    *,
    repo_root: Path,
    force: bool,
    existing_dependency_paths: tuple[Path, ...] = (),
    plots_only: bool = False,
) -> int:
    """Execute a matrix with spawned workers, strict resume, and fresh aggregation."""
    root = repo_root.resolve()
    output_root, run_dir = _resolved_run_dir(
        spec,
        root,
        extra_protected_paths=(
            tuple(existing_dependency_paths)
        ),
    )
    plugin, cells, invalid, raw_counts = _prepare(spec)
    # This identity is deliberately computable without resolving/materializing
    # dependencies.  It is the ownership gate that must run before a managed
    # dependency is allowed to change the filesystem.
    planned_dependency_payload = [
        {
            "name": item.name,
            "kind": item.kind,
            "protocol_version": item.protocol_version,
            "canonical_config_hash": item.canonical_config_hash,
            "optional": item.optional,
        }
        for item in spec.dependencies
    ]
    planned_ownership_fingerprint = stable_object_hash(
        {
            "plugin_id": plugin.plugin_id,
            "matrix_protocol_version": plugin.matrix_protocol_version,
            "base_config_sha256": _canonical_config_sha256(
                spec.base_config, plugin=plugin
            ),
            "cells": [
                {"cell_id": cell.cell_id, "config_sha256": cell.config_sha256}
                for cell in cells
            ],
            "dependencies": planned_dependency_payload,
            "torch_threads_per_job": spec.torch_threads_per_job,
            "cell_plots": spec.cell_plots,
        }
    )
    old_manifest: dict[str, Any] | None = None
    if run_dir.exists() or run_dir.is_symlink():
        old_manifest = _validate_owned_matrix(run_dir, spec)
        if force and not plots_only:
            shutil.rmtree(run_dir)
            old_manifest = None
        elif not plots_only and not spec.resume:
            raise FileExistsError(
                f"matrix run directory exists and resume=false: {run_dir}"
            )
        elif (
            not force
            and old_manifest.get("planned_ownership_fingerprint")
            != planned_ownership_fingerprint
        ):
            raise ValueError(
                "matrix planned ownership fingerprint mismatch; "
                "use a new run name or --force"
            )
    if not plots_only:
        output_root.mkdir(parents=True, exist_ok=True)
        for directory in (
            run_dir,
            run_dir / "generated_configs",
            run_dir / "cells",
            run_dir / "logs",
        ):
            directory.mkdir(parents=True, exist_ok=True)
    dependencies = plugin.resolve_dependencies(
        spec.dependencies,
        run_dir=run_dir,
        base_config=spec.base_config,
        repo_root=root,
        external_paths=existing_dependency_paths,
        allow_execution=not plots_only,
    )
    compute_fingerprint = _compute_fingerprint(
        spec,
        cells,
        dependency_identities=tuple(
            dict(item) for item in dependencies.identities
        ),
        plugin=plugin,
    )
    compute_fingerprint_payload = _compute_fingerprint_payload(
        spec,
        cells,
        dependency_identities=tuple(
            dict(item) for item in dependencies.identities
        ),
        plugin=plugin,
    )
    artifact_fingerprint = _artifact_contract_fingerprint(
        compute_fingerprint, cell_plots=spec.cell_plots
    )
    if plots_only:
        if not spec.plots_enabled or not spec.plot_tasks:
            raise ValueError("--plots-only requires at least one enabled plot task")
        if not (run_dir.exists() or run_dir.is_symlink()):
            raise FileNotFoundError(
                f"plots-only requires existing matrix run directory: {run_dir}"
            )
        manifest = _validate_owned_matrix(run_dir, spec)
        try:
            if manifest.get("compute_fingerprint") != compute_fingerprint:
                raise ValueError("compute fingerprint mismatch for --plots-only")
            if (
                manifest.get("artifact_contract_fingerprint")
                != artifact_fingerprint
            ):
                raise ValueError(
                    "artifact contract fingerprint mismatch for --plots-only"
                )
            for cell in cells:
                plugin.validate_cell(
                    run_dir / "cells" / cell.cell_id,
                    cell_plots=spec.cell_plots,
                    expected_config_sha256=cell.config_sha256,
                    expected_config_path=(
                        run_dir / "generated_configs" / f"{cell.cell_id}.json"
                    ),
                )
            records = _validate_aggregate(run_dir / "aggregate", plugin=plugin)
            recorded = manifest.get("aggregate_artifacts", [])
            if records != recorded:
                raise ValueError("matrix aggregate integrity records mismatch")
            for record in recorded:
                path = run_dir / "aggregate" / record["relative_path"]
                if (
                    path.is_symlink()
                    or not path.is_file()
                    or path.stat().st_size != record["size_bytes"]
                    or file_sha256(path) != record["sha256"]
                ):
                    raise ValueError(f"matrix aggregate artifact tampered: {path}")
        except Exception as exc:
            manifest["status"] = "fail"
            manifest["compute_status"] = "fail"
            manifest["plot_status"] = "not_run"
            manifest["plot_tasks"] = []
            manifest["failure"] = f"{type(exc).__name__}: {exc}"
            _record_matrix_plot_request(
                spec,
                run_dir=run_dir,
                manifest=manifest,
                compute_fingerprint=compute_fingerprint,
                request_mode="plots_only",
                status="not_run",
                outcomes=[],
                failure=manifest["failure"],
            )
            write_strict_json(run_dir / "matrix_manifest.json", manifest)
            return 1
        try:
            outcomes = _run_matrix_plots(spec, run_dir, plugin=plugin)
            manifest["plot_tasks"] = outcomes
            manifest["plot_status"] = "pass" if outcomes else "disabled"
            manifest["status"] = "pass"
            manifest["failure"] = None
            _record_matrix_plot_request(
                spec,
                run_dir=run_dir,
                manifest=manifest,
                compute_fingerprint=compute_fingerprint,
                request_mode="plots_only",
                status=manifest["plot_status"],
                outcomes=outcomes,
                failure=None,
            )
            write_strict_json(run_dir / "matrix_manifest.json", manifest)
            return 0
        except Exception as exc:
            manifest["plot_status"] = "fail"
            manifest["plot_tasks"] = []
            manifest["failure"] = f"{type(exc).__name__}: {exc}"
            manifest["status"] = "fail" if spec.plots_required else "pass"
            _record_matrix_plot_request(
                spec,
                run_dir=run_dir,
                manifest=manifest,
                compute_fingerprint=compute_fingerprint,
                request_mode="plots_only",
                status="fail",
                outcomes=[],
                failure=manifest["failure"],
            )
            write_strict_json(run_dir / "matrix_manifest.json", manifest)
            return 1 if spec.plots_required else 0
    if old_manifest is not None and (
        old_manifest.get("artifact_contract_fingerprint")
        != artifact_fingerprint
    ):
        raise ValueError("matrix fingerprint mismatch; use a new run name or --force")

    plan = matrix_plan_to_dict(spec, repo_root=root)
    write_strict_json(run_dir / "matrix_plan.json", plan)
    resolved = {
        "schema_version": spec.schema_version,
        "source_path": str(spec.source_path),
        "source_sha256": file_sha256(spec.source_path),
        "base_config": str(spec.base_config),
        "base_config_sha256": file_sha256(spec.base_config),
        "dependencies": list(dependencies.identities),
        "run_dir": str(run_dir),
        "matrix_fingerprint": artifact_fingerprint,
        "compute_fingerprint": compute_fingerprint,
        "compute_fingerprint_payload": compute_fingerprint_payload,
        "artifact_contract_fingerprint": artifact_fingerprint,
        "dependency_identities": list(dependencies.identities),
        "planned_ownership_fingerprint": planned_ownership_fingerprint,
        "execution": {
            "jobs": spec.jobs,
            "torch_threads_per_job": spec.torch_threads_per_job,
            "resume": spec.resume,
            "cell_plots": spec.cell_plots,
        },
    }
    write_strict_json(run_dir / "resolved_matrix_spec.json", resolved)
    manifest = {
        "schema_version": MATRIX_MANIFEST_SCHEMA,
        "status": "running",
        "run_name": spec.name,
        "run_dir": str(run_dir),
        "matrix_fingerprint": artifact_fingerprint,
        "compute_fingerprint": compute_fingerprint,
        "compute_fingerprint_payload": compute_fingerprint_payload,
        "artifact_contract_fingerprint": artifact_fingerprint,
        "science_fingerprint": compute_fingerprint,
        "dependency_identities": list(dependencies.identities),
        "planned_ownership_fingerprint": planned_ownership_fingerprint,
        "compute_status": "running",
        "plot_status": "pending" if spec.plots_enabled else "disabled",
        "plot_tasks": [],
        "aggregation_kind": spec.aggregation_kind,
        "raw_run_counts": raw_counts,
        "dependencies": dict(dependencies.manifest_records),
        "cells": [_cell_plan_record(cell, run_dir) | {"status": "pending"} for cell in cells],
        "aggregate_counts": (
            old_manifest.get("aggregate_counts") if old_manifest else None
        ),
        "aggregate_artifacts": (
            old_manifest.get("aggregate_artifacts", []) if old_manifest else []
        ),
        "failure": None,
    }
    write_strict_json(run_dir / "matrix_manifest.json", manifest)
    _write_csv(
        run_dir / "matrix_invalid_runs.csv",
        [
            {
                **row,
                "overrides": json.dumps(
                    row["overrides"],
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
            }
            for row in invalid
        ],
        [
            "raw_index",
            "experiment_membership",
            "overrides",
            "failure_type",
            "failure_message",
        ],
    )
    try:
        results: dict[int, dict[str, Any]] = {}
        requests: list[dict[str, Any]] = []
        for cell in cells:
            config_path = run_dir / "generated_configs" / f"{cell.cell_id}.json"
            config_path.write_text(
                cell.canonical_config.rstrip() + "\n", encoding="utf-8"
            )
            output_dir = run_dir / "cells" / cell.cell_id
            reused = (
                _reuse_result(
                    plugin, cell, output_dir, cell_plots=spec.cell_plots
                )
                if spec.resume and output_dir.exists()
                else None
            )
            if reused is not None:
                results[cell.run_index] = reused
                continue
            requests.append(
                {
                    "plugin_id": spec.aggregation_kind,
                    "plugin_factory": matrix_plugin_factory_path(
                        spec.aggregation_kind
                    ),
                    "run_index": cell.run_index,
                    "cell_id": cell.cell_id,
                    "config_sha256": cell.config_sha256,
                    "config_path": str(config_path),
                    **dict(dependencies.request_fields),
                    "output_dir": str(output_dir),
                    "log_path": str(run_dir / "logs" / f"{cell.cell_id}.log"),
                    "repo_root": str(root),
                    "torch_threads": spec.torch_threads_per_job,
                    "cell_plots": spec.cell_plots,
                }
            )

        if requests:
            context = multiprocessing.get_context("spawn")
            pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=spec.jobs, mp_context=context
            )
            future_map = {
                pool.submit(execute_matrix_cell, request): request
                for request in requests
            }
            try:
                for future in concurrent.futures.as_completed(future_map):
                    result = future.result()
                    results[int(result["run_index"])] = result
                    manifest["cells"][int(result["run_index"])].update(result)
                    write_strict_json(run_dir / "matrix_manifest.json", manifest)
            except KeyboardInterrupt:
                _shutdown_process_pool(pool, list(future_map))
                raise
            except BaseException:
                pool.shutdown(wait=True, cancel_futures=True)
                raise
            else:
                pool.shutdown(wait=True)

        ordered = [results[cell.run_index] for cell in cells]
        for result in ordered:
            manifest["cells"][result["run_index"]].update(result)
        fields = [
            "run_index",
            "cell_id",
            "config_sha256",
            "human_slug",
            "experiment_memberships",
            "status",
            "executed_or_reused",
            "return_code",
            "output_dir",
            "artifact_manifest_sha256",
            "failure_type",
            "failure_message",
        ]
        rows = []
        for cell, result in zip(cells, ordered):
            rows.append(
                {
                    "run_index": cell.run_index,
                    "cell_id": cell.cell_id,
                    "config_sha256": cell.config_sha256,
                    "human_slug": cell.human_slug,
                    "experiment_memberships": json.dumps(
                        list(cell.experiment_memberships), separators=(",", ":")
                    ),
                    **{
                        key: result[key]
                        for key in fields
                        if key
                        not in {
                            "run_index",
                            "cell_id",
                            "config_sha256",
                            "human_slug",
                            "experiment_memberships",
                        }
                    },
                }
            )
        _write_csv(run_dir / "matrix_runs.csv", rows, fields)
        passed_cells = [
            cell
            for cell, result in zip(cells, ordered)
            if result["status"] == "pass"
        ]
        failures = [result for result in ordered if result["status"] != "pass"]
        if not failures:
            aggregate_counts, aggregate_artifacts = (
                _collect_and_publish_aggregate(
                    run_dir=run_dir,
                    plugin=plugin,
                    cells=passed_cells,
                )
            )
            manifest["aggregate_counts"] = aggregate_counts
            manifest["aggregate_artifacts"] = aggregate_artifacts
        manifest["compute_status"] = "fail" if failures else "pass"
        manifest["status"] = "fail" if failures else "pass"
        manifest["failure"] = (
            f"{len(failures)} matrix cell(s) failed" if failures else None
        )
        if failures:
            manifest["plot_status"] = "not_run"
            _record_matrix_plot_request(
                spec,
                run_dir=run_dir,
                manifest=manifest,
                compute_fingerprint=compute_fingerprint,
                request_mode="initial_run",
                status="not_run",
                outcomes=[],
                failure=manifest["failure"],
            )
        if not failures:
            try:
                outcomes = _run_matrix_plots(spec, run_dir, plugin=plugin)
                manifest["plot_tasks"] = outcomes
                manifest["plot_status"] = "pass" if outcomes else "disabled"
                _record_matrix_plot_request(
                    spec,
                    run_dir=run_dir,
                    manifest=manifest,
                    compute_fingerprint=compute_fingerprint,
                    request_mode="initial_run",
                    status=manifest["plot_status"],
                    outcomes=outcomes,
                    failure=None,
                )
            except Exception as exc:
                manifest["plot_status"] = "fail"
                manifest["failure"] = f"{type(exc).__name__}: {exc}"
                manifest["status"] = "fail" if spec.plots_required else "pass"
                _record_matrix_plot_request(
                    spec,
                    run_dir=run_dir,
                    manifest=manifest,
                    compute_fingerprint=compute_fingerprint,
                    request_mode="initial_run",
                    status="fail",
                    outcomes=[],
                    failure=manifest["failure"],
                )
        write_strict_json(run_dir / "matrix_manifest.json", manifest)
        return 1 if failures or (
            manifest["plot_status"] == "fail" and spec.plots_required
        ) else 0
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        manifest["compute_status"] = "interrupted"
        manifest["failure"] = "KeyboardInterrupt"
        write_strict_json(run_dir / "matrix_manifest.json", manifest)
        return 130
    except Exception as exc:
        manifest["status"] = "fail"
        manifest["compute_status"] = "fail"
        manifest["failure"] = f"{type(exc).__name__}: {exc}"
        write_strict_json(run_dir / "matrix_manifest.json", manifest)
        return 1
