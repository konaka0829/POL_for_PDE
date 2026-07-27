"""Spawn-safe worker for one isolated matrix cell."""
from __future__ import annotations

import json
from pathlib import Path
import traceback
from typing import Any

from pol.runtime.io import write_strict_json
from pol.runtime.recipe import numerical_thread_scope


def execute_matrix_cell(request: dict[str, Any]) -> dict[str, Any]:
    """Execute exactly one cell recipe and return a structured serializable result."""
    output_dir = Path(request["output_dir"])
    log_path = Path(request["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with numerical_thread_scope(request["torch_threads"]):
            from pol.workflow.registry import get_matrix_plugin

            plugin = get_matrix_plugin(request["plugin_id"])
            result = plugin.execute_cell(request)
            manifest_hash = None
            if result.exit_code == 0:
                manifest_hash = plugin.validate_cell(
                    output_dir,
                    cell_plots=request["cell_plots"],
                    expected_config_sha256=request["config_sha256"],
                    expected_config_path=Path(request["config_path"]),
                )
            record = {
                "run_index": request["run_index"],
                "cell_id": request["cell_id"],
                "status": "pass" if result.exit_code == 0 else "fail",
                "executed_or_reused": "executed",
                "return_code": result.exit_code,
                "output_dir": str(output_dir),
                "artifact_manifest_sha256": manifest_hash,
                "failure_type": (
                    None if result.exit_code == 0 else "RecipeFailure"
                ),
                "failure_message": (
                    None
                    if result.exit_code == 0
                    else str(
                        result.console_payload.get(
                            "failure_reason",
                            f"recipe exited with code {result.exit_code}",
                        )
                    )
                ),
            }
            write_strict_json(log_path, result.console_payload)
            return record
    except BaseException as exc:
        failure = {
            "run_index": request["run_index"],
            "cell_id": request["cell_id"],
            "status": "fail",
            "executed_or_reused": "executed",
            "return_code": 130 if isinstance(exc, KeyboardInterrupt) else 1,
            "output_dir": str(output_dir),
            "artifact_manifest_sha256": None,
            "failure_type": type(exc).__name__,
            "failure_message": str(exc),
        }
        log_path.write_text(
            json.dumps(failure, sort_keys=True, allow_nan=False)
            + "\n"
            + traceback.format_exc(),
            encoding="utf-8",
        )
        return failure
