"""Generic process-isolated matrix planning, execution, and resume."""
from __future__ import annotations

import concurrent.futures
import csv
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
from typing import Any

from pol.runtime.io import file_sha256
from pol.runtime.path_safety import resolve_safe_run_directory

from .matrix_spec import MatrixRunSpec, expand_matrix
from .matrix_worker import execute_matrix_cell
from .registry import get_matrix_plugin
from .types import MatrixCell


MATRIX_MANIFEST_SCHEMA = "paper1-matrix-manifest-v1"


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


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
            spec.e0_config,
            *extra_protected_paths,
        ),
    )


def _prepare(spec: MatrixRunSpec) -> tuple[Any, list[MatrixCell], list[dict[str, Any]], dict[str, int]]:
    plugin = get_matrix_plugin(spec.aggregation_kind)
    base = plugin.load_base(spec.base_config)
    cells, invalid, raw_counts = expand_matrix(spec, base=base, plugin=plugin)
    return plugin, cells, invalid, raw_counts


def _fingerprint(spec: MatrixRunSpec, cells: list[MatrixCell]) -> str:
    payload = {
        "schema_version": spec.schema_version,
        "base_config_sha256": file_sha256(spec.base_config),
        "e0_config_sha256": file_sha256(spec.e0_config),
        "aggregation_kind": spec.aggregation_kind,
        "invalid_run_policy": spec.invalid_run_policy,
        "cell_plots": spec.cell_plots,
        "cells": [
            {
                "config_sha256": cell.config_sha256,
                "memberships": list(cell.experiment_memberships),
            }
            for cell in cells
        ],
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _run_e0(
    spec: MatrixRunSpec,
    *,
    repo_root: Path,
    run_dir: Path,
    plugin: Any,
) -> str:
    e0_dir = run_dir / "e0"
    if e0_dir.exists():
        try:
            plugin.validate_e0(e0_dir, spec.base_config)
            return "reused"
        except Exception:
            pass
    plugin.execute_e0(
        spec.e0_config,
        e0_dir,
        base_config=spec.base_config,
        repo_root=repo_root,
    )
    return "executed"


def _reuse_result(
    plugin: Any,
    cell: MatrixCell,
    output_dir: Path,
    *,
    cell_plots: bool,
) -> dict[str, Any] | None:
    try:
        manifest_hash = plugin.validate_cell(output_dir, cell_plots=cell_plots)
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
    spec: MatrixRunSpec, run_dir: Path
) -> list[dict[str, Any]]:
    if not spec.plots_enabled:
        return []
    from pol.plots.runtime import execute_plot_tasks

    return execute_plot_tasks(
        experiment_kind="e1_matrix",
        input_dir=run_dir / "aggregate",
        figures_dir=run_dir / "figures",
        tasks=spec.plot_tasks,
    )


def execute_matrix_run(
    spec: MatrixRunSpec,
    *,
    repo_root: Path,
    force: bool,
    existing_e0_dir: Path | None = None,
    plots_only: bool = False,
) -> int:
    """Execute a matrix with spawned workers, strict resume, and fresh aggregation."""
    root = repo_root.resolve()
    output_root, run_dir = _resolved_run_dir(
        spec,
        root,
        extra_protected_paths=(
            (existing_e0_dir,) if existing_e0_dir is not None else ()
        ),
    )
    plugin, cells, invalid, raw_counts = _prepare(spec)
    fingerprint = _fingerprint(spec, cells)
    if plots_only:
        if not (run_dir.exists() or run_dir.is_symlink()):
            raise FileNotFoundError(
                f"plots-only requires existing matrix run directory: {run_dir}"
            )
        manifest = _validate_owned_matrix(run_dir, spec)
        if manifest.get("matrix_fingerprint") != fingerprint:
            raise ValueError("science fingerprint mismatch for --plots-only")
        plugin.validate_e0(run_dir / "e0", spec.base_config)
        for cell in cells:
            plugin.validate_cell(
                run_dir / "cells" / cell.cell_id,
                cell_plots=spec.cell_plots,
            )
        for record in manifest.get("aggregate_artifacts", []):
            path = run_dir / "aggregate" / record["relative_path"]
            if (
                not path.is_file()
                or path.stat().st_size != record["size_bytes"]
                or file_sha256(path) != record["sha256"]
            ):
                raise ValueError(f"matrix aggregate artifact tampered: {path}")
        if spec.plots_enabled and not manifest.get("aggregate_artifacts"):
            raise ValueError("matrix aggregate integrity records are missing")
        try:
            outcomes = _run_matrix_plots(spec, run_dir)
            manifest["plot_tasks"] = outcomes
            manifest["plot_status"] = "pass" if outcomes else "disabled"
            manifest["status"] = "pass"
            manifest["failure"] = None
            _atomic_json(run_dir / "matrix_manifest.json", manifest)
            return 0
        except Exception as exc:
            manifest["plot_status"] = "fail"
            manifest["failure"] = f"{type(exc).__name__}: {exc}"
            manifest["status"] = "fail" if spec.plots_required else "pass"
            _atomic_json(run_dir / "matrix_manifest.json", manifest)
            return 1 if spec.plots_required else 0
    old_manifest: dict[str, Any] | None = None
    if run_dir.exists() or run_dir.is_symlink():
        old_manifest = _validate_owned_matrix(run_dir, spec)
        if force:
            shutil.rmtree(run_dir)
            old_manifest = None
        elif not spec.resume:
            raise FileExistsError(
                f"matrix run directory exists and resume=false: {run_dir}"
            )
        elif old_manifest.get("matrix_fingerprint") != fingerprint:
            raise ValueError("matrix fingerprint mismatch; use a new run name or --force")
    output_root.mkdir(parents=True, exist_ok=True)
    for directory in (
        run_dir,
        run_dir / "generated_configs",
        run_dir / "cells",
        run_dir / "logs",
        run_dir / "aggregate",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    plan = matrix_plan_to_dict(spec, repo_root=root)
    _atomic_json(run_dir / "matrix_plan.json", plan)
    resolved = {
        "schema_version": spec.schema_version,
        "source_path": str(spec.source_path),
        "source_sha256": file_sha256(spec.source_path),
        "base_config": str(spec.base_config),
        "base_config_sha256": file_sha256(spec.base_config),
        "e0_config": str(spec.e0_config),
        "e0_config_sha256": file_sha256(spec.e0_config),
        "run_dir": str(run_dir),
        "matrix_fingerprint": fingerprint,
        "science_fingerprint": fingerprint,
        "compute_status": "running",
        "plot_status": "pending" if spec.plots_enabled else "disabled",
        "plot_tasks": [],
        "execution": {
            "jobs": spec.jobs,
            "torch_threads_per_job": spec.torch_threads_per_job,
            "resume": spec.resume,
            "cell_plots": spec.cell_plots,
        },
    }
    _atomic_json(run_dir / "resolved_matrix_spec.json", resolved)
    manifest = {
        "schema_version": MATRIX_MANIFEST_SCHEMA,
        "status": "running",
        "run_name": spec.name,
        "run_dir": str(run_dir),
        "matrix_fingerprint": fingerprint,
        "aggregation_kind": spec.aggregation_kind,
        "raw_run_counts": raw_counts,
        "e0": {"status": "pending", "executed_or_reused": None},
        "cells": [_cell_plan_record(cell, run_dir) | {"status": "pending"} for cell in cells],
        "failure": None,
    }
    _atomic_json(run_dir / "matrix_manifest.json", manifest)
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
        if existing_e0_dir is None:
            e0_disposition = _run_e0(
                spec, repo_root=root, run_dir=run_dir, plugin=plugin
            )
            e0_dir = run_dir / "e0"
        else:
            plugin.validate_e0(existing_e0_dir, spec.base_config)
            e0_dir = existing_e0_dir.resolve()
            e0_disposition = "external_reused"
        manifest["e0"] = {
            "status": "pass",
            "executed_or_reused": e0_disposition,
            "output_dir": str(e0_dir),
        }
        _atomic_json(run_dir / "matrix_manifest.json", manifest)

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
                    "run_index": cell.run_index,
                    "cell_id": cell.cell_id,
                    "config_path": str(config_path),
                    "e0_dir": str(e0_dir),
                    "output_dir": str(output_dir),
                    "log_path": str(run_dir / "logs" / f"{cell.cell_id}.log"),
                    "repo_root": str(root),
                    "torch_threads": spec.torch_threads_per_job,
                    "cell_plots": spec.cell_plots,
                }
            )

        if requests:
            context = multiprocessing.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=spec.jobs, mp_context=context
            ) as pool:
                future_map = {
                    pool.submit(execute_matrix_cell, request): request
                    for request in requests
                }
                try:
                    for future in concurrent.futures.as_completed(future_map):
                        result = future.result()
                        results[int(result["run_index"])] = result
                        manifest["cells"][int(result["run_index"])].update(result)
                        _atomic_json(run_dir / "matrix_manifest.json", manifest)
                except KeyboardInterrupt:
                    for future in future_map:
                        future.cancel()
                    pool.shutdown(wait=False, cancel_futures=True)
                    raise

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
        aggregate_counts = plugin.collect(
            run_dir / "aggregate", run_dir / "cells", passed_cells
        )
        aggregate_artifacts = []
        for path in sorted((run_dir / "aggregate").iterdir()):
            if path.is_file():
                aggregate_artifacts.append(
                    {
                        "relative_path": path.name,
                        "size_bytes": path.stat().st_size,
                        "sha256": file_sha256(path),
                    }
                )
        failures = [result for result in ordered if result["status"] != "pass"]
        manifest["aggregate_counts"] = aggregate_counts
        manifest["aggregate_artifacts"] = aggregate_artifacts
        manifest["compute_status"] = "fail" if failures else "pass"
        manifest["status"] = "fail" if failures else "pass"
        manifest["failure"] = (
            f"{len(failures)} matrix cell(s) failed" if failures else None
        )
        if not failures:
            try:
                outcomes = _run_matrix_plots(spec, run_dir)
                manifest["plot_tasks"] = outcomes
                manifest["plot_status"] = "pass" if outcomes else "disabled"
            except Exception as exc:
                manifest["plot_status"] = "fail"
                manifest["failure"] = f"{type(exc).__name__}: {exc}"
                manifest["status"] = "fail" if spec.plots_required else "pass"
        _atomic_json(run_dir / "matrix_manifest.json", manifest)
        return 1 if failures or (
            manifest["plot_status"] == "fail" and spec.plots_required
        ) else 0
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        manifest["compute_status"] = "interrupted"
        manifest["failure"] = "KeyboardInterrupt"
        _atomic_json(run_dir / "matrix_manifest.json", manifest)
        return 130
    except Exception as exc:
        manifest["status"] = "fail"
        manifest["compute_status"] = "fail"
        manifest["failure"] = f"{type(exc).__name__}: {exc}"
        _atomic_json(run_dir / "matrix_manifest.json", manifest)
        return 1
