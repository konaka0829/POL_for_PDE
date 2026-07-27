"""Importable prerequisite-free plugin used to exercise spawn matrix core."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from pol.runtime.io import file_sha256, write_strict_json
from pol.runtime.recipe import RecipeResult
from pol.workflow.types import MatrixCell, ResolvedDependencies


class FakeMatrixPlugin:
    plugin_id = "test_fake_matrix_v1"
    experiment_kind = "test_fake"
    matrix_protocol_version = "test-fake-matrix-v7"
    plot_experiment_kind = "test_fake_matrix"
    recipe_protocol_versions = ("test-fake-recipe-v3",)

    def parse_dependencies(self, raw, *, repo_root):
        if raw:
            raise ValueError("fake plugin prerequisites must be empty")
        return ()

    def resolve_dependencies(
        self, specs, *, run_dir, base_config, repo_root, external_paths,
        allow_execution,
    ):
        if specs or external_paths:
            raise ValueError("fake plugin has no dependencies")
        return ResolvedDependencies((), {}, {})

    def load_base(self, path: Path) -> dict[str, Any]:
        return json.loads(path.read_text())

    def finalize_config(self, raw: dict[str, Any]):
        value = raw["science"]["value"]
        return raw, {"value": value}, f"value{value}"

    def canonical_config(self, raw: Mapping[str, Any]) -> str:
        return json.dumps(raw, sort_keys=True, separators=(",", ":"))

    def plan_summary(self, cells: list[MatrixCell]) -> dict[str, Any]:
        return {"fake_cells": len(cells)}

    def execute_cell(self, request: Mapping[str, Any]) -> RecipeResult:
        output = Path(str(request["output_dir"]))
        output.mkdir(parents=True, exist_ok=True)
        config = Path(str(request["config_path"]))
        payload = {
            "schema_version": self.recipe_protocol_versions[0],
            "config_sha256": hashlib.sha256(
                config.read_text().strip().encode("utf-8")
            ).hexdigest(),
        }
        write_strict_json(output / "result.json", payload)
        return RecipeResult("pass", 0, output, payload, output / "result.json")

    def validate_cell(self, output_dir: Path, **kwargs: Any) -> str:
        if set(path.name for path in output_dir.iterdir()) != {"result.json"}:
            raise ValueError("fake cell artifact tree mismatch")
        payload = json.loads((output_dir / "result.json").read_text())
        if payload["schema_version"] != self.recipe_protocol_versions[0]:
            raise ValueError("fake recipe protocol mismatch")
        expected = kwargs.get("expected_config_sha256")
        if expected is not None and payload["config_sha256"] != expected:
            raise ValueError("fake config hash mismatch")
        return file_sha256(output_dir / "result.json")

    def collect(self, aggregate_dir: Path, cells_dir: Path, cells):
        records = [
            json.loads((cells_dir / cell.cell_id / "result.json").read_text())
            for cell in cells
        ]
        write_strict_json(aggregate_dir / "summary.json", {"records": records})
        return {"records": len(records)}

    def validate_aggregate(self, aggregate_dir: Path) -> None:
        payload = json.loads((aggregate_dir / "summary.json").read_text())
        if not isinstance(payload.get("records"), list):
            raise ValueError("fake aggregate invalid")

    def aggregate_artifact_names(self):
        return ("summary.json",)
