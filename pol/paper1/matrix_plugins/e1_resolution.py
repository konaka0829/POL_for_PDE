"""E1 resolution-specific policy and aggregation for generic matrices."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from pol.paper1.config import (
    canonical_config_json,
    config_from_dict,
    load_config_json,
)
from pol.paper1.e1 import validate_e0_prerequisite
from pol.paper1.e1_qa import (
    expected_artifacts,
    validate_artifact_set,
    validate_plots,
    validate_saved_numeric_artifacts,
    verify_artifact_manifest,
)
from pol.runtime.io import write_strict_json
from pol.runtime.recipe import RecipeInvocation, numerical_thread_scope
from pol.workflow.types import MatrixCell


class E1ResolutionPlugin:
    """E1-specific validation, execution metadata, and aggregate collection."""

    plugin_id = "paper1_e1_resolution_v1"

    def load_base(self, path: Path) -> dict[str, Any]:
        """Load a raw E1 base configuration and validate its section."""
        raw = json.loads(path.read_text(encoding="utf-8"))
        config = config_from_dict(raw)
        if config.e1 is None:
            raise ValueError("matrix base config must contain an e1 section")
        return raw

    def finalize_config(
        self, raw: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any], str]:
        """Apply the E1 derived full-observation field and validate science."""
        spatial = raw["spatial"]
        n_tar = spatial["target_data_nx"]
        n_sur = spatial["surrogate_internal_nx"]
        observation = spatial["observation_dim"]
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (n_tar, n_sur, observation)
        ):
            raise ValueError("E1 resolution dimensions must be integers")
        raw["e1"]["require_full_observation"] = observation == n_sur
        config = config_from_dict(raw)
        if config.e1 is None:
            raise ValueError("generated matrix config lacks e1")
        metadata = {
            "n_tar": n_tar,
            "n_sur": n_sur,
            "J": observation,
            "full_observation": observation == n_sur,
        }
        slug = f"ntar{n_tar}_nsur{n_sur}_J{observation}"
        return raw, metadata, slug

    def canonical_config(self, raw: Mapping[str, Any]) -> str:
        """Return canonical scientific config JSON for cell identity."""
        return canonical_config_json(config_from_dict(dict(raw)))

    def validate_e0(self, e0_dir: Path, base_config: Path) -> None:
        """Strictly validate the shared E0 prerequisite against E1."""
        validate_e0_prerequisite(e0_dir, load_config_json(base_config))

    def execute_e0(
        self,
        e0_config: Path,
        e0_dir: Path,
        *,
        base_config: Path,
        repo_root: Path,
    ) -> None:
        """Execute and strictly validate the single shared E0 prerequisite."""
        invocation = RecipeInvocation(
            repo_root=repo_root,
            working_directory=repo_root,
            command=(
                "matrix_prerequisite",
                "pol.paper1.recipes.foundation_validation."
                "run_foundation_validation",
            ),
            torch_threads=1,
        )
        with numerical_thread_scope(1):
            from pol.paper1.recipes.foundation_validation import (
                run_foundation_validation,
            )

            result = run_foundation_validation(
                e0_config,
                e0_dir,
                overwrite=True,
                invocation=invocation,
            )
        if result.exit_code != 0:
            raise RuntimeError(
                f"E0 prerequisite failed with exit code {result.exit_code}"
            )
        self.validate_e0(e0_dir, base_config)

    def plan_summary(self, cells: list[MatrixCell]) -> dict[str, Any]:
        """Return E1-specific dimensional coverage metadata."""
        return {
            "contains_n_tar_gt_J": any(
                cell.metadata["n_tar"] > cell.metadata["J"] for cell in cells
            ),
            "contains_n_tar_lt_J": any(
                cell.metadata["n_tar"] < cell.metadata["J"] for cell in cells
            ),
            "full_observation_cells": sum(
                bool(cell.metadata["full_observation"]) for cell in cells
            ),
        }

    def validate_cell(self, output_dir: Path, *, cell_plots: bool) -> str:
        """Verify the complete E1 artifact contract and return manifest hash."""
        summary = json.loads((output_dir / "e1_summary.json").read_text())
        if summary.get("status") != "pass":
            raise ValueError("E1 cell summary is not pass")
        config = load_config_json(output_dir / "resolved_config.json")
        plot_names = validate_plots(output_dir, skip_plots=not cell_plots)
        expected = expected_artifacts(plot_names)
        validate_saved_numeric_artifacts(output_dir, config)
        verify_artifact_manifest(output_dir, expected)
        validate_artifact_set(output_dir, expected)
        return hashlib.sha256(
            (output_dir / "artifact_manifest.json").read_bytes()
        ).hexdigest()

    def collect(
        self,
        aggregate_dir: Path,
        cells_dir: Path,
        cells: list[MatrixCell],
    ) -> dict[str, int]:
        """Rebuild legacy-compatible aggregate CSVs from verified pass cells."""
        tables = {
            name: []
            for name in ("selected_results", "readout_diagnostics", "noise_summary")
        }
        # Preserve the legacy aggregate table ordering independently of generic
        # matrix run_index, whose order follows declaration order.
        for cell in sorted(
            cells,
            key=lambda item: (
                item.metadata["n_tar"],
                item.metadata["n_sur"],
                item.metadata["J"],
            ),
        ):
            output = cells_dir / cell.cell_id
            metadata = dict(cell.metadata)
            prefix = {
                "run_id": cell.human_slug,
                **metadata,
                "experiment_names": json.dumps(
                    list(cell.experiment_memberships), separators=(",", ":")
                ),
            }
            for name, rows in tables.items():
                with (output / f"{name}.csv").open(
                    newline="", encoding="utf-8"
                ) as handle:
                    rows.extend({**prefix, **row} for row in csv.DictReader(handle))
        aggregate_dir.mkdir(parents=True, exist_ok=True)
        counts: dict[str, int] = {}
        for name, rows in tables.items():
            path = aggregate_dir / f"sweep_{name}.csv"
            if not rows:
                path.write_text("", encoding="utf-8")
            else:
                fields: list[str] = []
                for row in rows:
                    fields.extend(key for key in row if key not in fields)
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
            counts[name] = len(rows)
        write_strict_json(aggregate_dir / "aggregate_summary.json", counts)
        return counts
