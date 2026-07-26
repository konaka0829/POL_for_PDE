"""Shared immutable matrix workflow types."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class MatrixCell:
    """One deduplicated and scientifically validated matrix cell."""

    run_index: int
    cell_id: str
    config_sha256: str
    canonical_config: str
    human_slug: str
    experiment_memberships: tuple[str, ...]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class MatrixCellResult:
    """Structured outcome returned by one isolated matrix worker."""

    run_index: int
    cell_id: str
    status: str
    executed_or_reused: str
    return_code: int
    output_dir: str
    artifact_manifest_sha256: str | None
    failure_type: str | None
    failure_message: str | None
