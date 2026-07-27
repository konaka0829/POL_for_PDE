"""Latest plot-request provenance independent of immutable compute specs."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from pol.runtime.io import file_sha256, write_strict_json

from .types import PlotTaskSpec


PLOT_REQUEST_SCHEMA = "paper1-plot-request-v1"


def build_plot_request(
    *,
    source_spec_path: Path,
    compute_fingerprint: str,
    tasks: Iterable[PlotTaskSpec],
    request_mode: str,
    status: str,
    outcomes: Iterable[Mapping[str, Any]],
    failure: str | None,
) -> dict[str, Any]:
    """Build provenance for the latest initial or plots-only request."""
    return {
        "schema_version": PLOT_REQUEST_SCHEMA,
        "source_spec_path": str(source_spec_path),
        "source_spec_sha256": file_sha256(source_spec_path),
        "compute_fingerprint": compute_fingerprint,
        "requested_tasks": [
            {"recipe_id": task.recipe_id, "settings": dict(task.settings)}
            for task in tasks
        ],
        "request_mode": request_mode,
        "request_status": (
            "rejected" if status == "rejected" else "accepted"
        ),
        "requested_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "outcomes": [dict(outcome) for outcome in outcomes],
        "failure": failure,
    }


def write_plot_request(run_dir: Path, request: Mapping[str, Any]) -> None:
    """Atomically write the latest plot request at a run root."""
    path = run_dir / "resolved_plot_spec.json"
    write_strict_json(path, request)
