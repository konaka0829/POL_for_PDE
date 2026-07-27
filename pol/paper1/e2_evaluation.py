"""Frozen-plan and test-evaluation reference contracts."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FrozenPlanReference:
    path: Path
    selection_record_hash: str
    plan_content_hash: str


@dataclass(frozen=True)
class TestEvaluationResult:
    test_rows: tuple[dict[str, Any], ...]
    model3_seed_rows: tuple[dict[str, Any], ...]
    model3_aggregate_rows: tuple[dict[str, Any], ...]
