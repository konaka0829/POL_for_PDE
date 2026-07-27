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
from pol.runtime.io import write_strict_json
from pol.runtime.recipe import RecipeInvocation, numerical_thread_scope
from pol.workflow.types import (
    DependencySpec,
    MatrixCell,
    ResolvedDependencies,
)


class E1ResolutionPlugin:
    """E1-specific validation, execution metadata, and aggregate collection."""

    plugin_id = "paper1_e1_resolution_v1"
    experiment_kind = "e1"
    from pol.paper1.protocols import (
        E1_MATRIX_PLUGIN_PROTOCOL as matrix_protocol_version,
    )
    plot_experiment_kind = "e1_matrix"

    @property
    def recipe_protocol_versions(self) -> tuple[str, ...]:
        from pol.paper1.protocols import E1_SCHEMA_VERSION

        return (E1_SCHEMA_VERSION,)

    def parse_dependencies(
        self, raw: Mapping[str, Any], *, repo_root: Path
    ) -> tuple[DependencySpec, ...]:
        if set(raw) != {"e0_config"}:
            unknown = sorted(set(raw) - {"e0_config"})
            if unknown:
                raise ValueError(f"unknown key at $.prerequisites: {unknown[0]}")
            raise ValueError("missing required key at $.prerequisites.e0_config")
        path = Path(str(raw["e0_config"]))
        path = (path if path.is_absolute() else repo_root / path).resolve()
        if not path.is_file():
            raise ValueError(
                f"file does not exist at $.prerequisites.e0_config: {path}"
            )
        from pol.paper1.protocols import E0_SCHEMA_VERSION

        return (
            DependencySpec(
                name="foundation",
                kind="paper1_e0",
                config_path=path,
                protocol_version=E0_SCHEMA_VERSION,
                canonical_config_hash=hashlib.sha256(
                    canonical_config_json(load_config_json(path)).encode("utf-8")
                ).hexdigest(),
            ),
        )

    def resolve_dependencies(
        self,
        specs,
        *,
        run_dir: Path,
        base_config: Path,
        repo_root: Path,
        external_paths,
        allow_execution: bool,
    ) -> ResolvedDependencies:
        if len(specs) != 1:
            raise ValueError("E1 requires exactly one foundation dependency")
        spec = specs[0]
        managed = not external_paths
        # Preserve the legacy on-disk location as a plugin-owned compatibility
        # detail; generic matrix core treats this path opaquely.
        output = run_dir / "e0"
        if external_paths:
            if len(external_paths) != 1:
                raise ValueError("E1 accepts one external dependency path")
            output = Path(external_paths[0]).resolve()
            disposition = "external_reused"
        else:
            try:
                self.validate_e0(output, base_config)
                disposition = "reused"
            except Exception:
                if not allow_execution:
                    raise ValueError(
                        "plots-only requires an existing valid E0 dependency"
                    )
                if spec.config_path is None:
                    raise ValueError("managed E0 dependency requires config_path")
                self.execute_e0(
                    spec.config_path,
                    output,
                    base_config=base_config,
                    repo_root=repo_root,
                )
                disposition = "executed"
        self.validate_e0(output, base_config)
        from pol.paper1.artifact_contracts import E0ArtifactContract

        E0ArtifactContract().validate_complete(output)
        import torch

        from pol.paper1.datasets import tensor_hash
        from pol.runtime.hashing import stable_object_hash

        archive = torch.load(
            output / "master_initial_conditions.pt",
            map_location="cpu",
            weights_only=False,
        )
        values = archive.get("values") if isinstance(archive, dict) else None
        if not isinstance(values, torch.Tensor):
            raise ValueError("E0 dependency master tensor is missing")
        master_tensor_hash = tensor_hash(values)
        accepted_config_hash = hashlib.sha256(
            canonical_config_json(
                load_config_json(output / "accepted_production_config.json")
            ).encode("utf-8")
        ).hexdigest()
        artifact_identity = stable_object_hash({
            "protocol_version": spec.protocol_version,
            "accepted_config_hash": accepted_config_hash,
            "master_tensor_hash": master_tensor_hash,
        })
        identity_record = {
            "name": spec.name,
            "kind": spec.kind,
            "protocol_version": spec.protocol_version,
            "canonical_config_hash": spec.canonical_config_hash,
            "artifact_identity": artifact_identity,
            "accepted_config_hash": accepted_config_hash,
            "master_tensor_hash": master_tensor_hash,
            "binding_mode": "managed" if managed else "external",
        }
        record = {
            **identity_record,
            "status": "pass",
            "executed_or_reused": disposition,
            "output_dir": str(output),
        }
        return ResolvedDependencies(
            identities=(identity_record,),
            request_fields={"e0_dir": str(output)},
            # The legacy key remains an experiment-owned compatibility field.
            manifest_records={"e0": record},
            protected_paths=(output,) if not managed else (),
        )

    def execute_cell(self, request: Mapping[str, Any]):
        """Run the E1 recipe behind the experiment-owned plugin boundary."""
        from pol.paper1.recipes.heat_calibration import run_heat_calibration

        invocation = RecipeInvocation(
            repo_root=Path(str(request["repo_root"])),
            working_directory=Path(str(request["repo_root"])),
            command=(
                "matrix_cell",
                self.plugin_id,
                str(request["cell_id"]),
            ),
            torch_threads=int(request["torch_threads"]),
        )
        return run_heat_calibration(
            Path(str(request["config_path"])),
            Path(str(request["e0_dir"])),
            Path(str(request["output_dir"])),
            overwrite=True,
            skip_plots=not bool(request["cell_plots"]),
            invocation=invocation,
        )

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
        from pol.paper1.e1 import validate_e0_prerequisite

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

    def validate_cell(
        self,
        output_dir: Path,
        *,
        cell_plots: bool,
        expected_config_sha256: str | None = None,
        expected_config_path: Path | None = None,
    ) -> str:
        """Verify the complete E1 artifact contract and return manifest hash."""
        from pol.paper1.e1_qa import (
            expected_artifacts,
            validate_artifact_set,
            validate_plots,
            validate_saved_numeric_artifacts,
            verify_artifact_manifest,
        )

        summary = json.loads((output_dir / "e1_summary.json").read_text())
        if summary.get("status") != "pass":
            raise ValueError("E1 cell summary is not pass")
        config = load_config_json(output_dir / "resolved_config.json")
        if expected_config_sha256 is not None:
            if expected_config_path is None:
                raise ValueError("expected matrix config path is required")
            expected = load_config_json(expected_config_path)
            saved_raw = config.to_dict()
            # E0 prerequisite validation intentionally replaces only reference_nx
            # with its accepted resolution before E1 writes resolved_config.json.
            saved_raw["spatial"]["reference_nx"] = (
                expected.spatial.reference_nx
            )
            saved_hash = hashlib.sha256(
                canonical_config_json(config_from_dict(saved_raw)).encode("utf-8")
            ).hexdigest()
            if saved_hash != expected_config_sha256:
                raise ValueError(
                    "E1 cell resolved config does not match its matrix cell"
                )
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
                    sorted(cell.experiment_memberships), separators=(",", ":")
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

    def validate_aggregate(self, aggregate_dir: Path) -> None:
        """Validate aggregate row counts and legacy-compatible CSV structure."""
        summary = json.loads(
            (aggregate_dir / "aggregate_summary.json").read_text(encoding="utf-8")
        )
        expected_keys = {
            "selected_results",
            "readout_diagnostics",
            "noise_summary",
        }
        if not isinstance(summary, dict) or set(summary) != expected_keys:
            raise ValueError("aggregate summary has an invalid contract")
        for name in sorted(expected_keys):
            path = aggregate_dir / f"sweep_{name}.csv"
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                if reader.fieldnames is None:
                    raise ValueError(f"aggregate CSV lacks a header: {path}")
            if len(rows) != summary[name]:
                raise ValueError(f"aggregate row count mismatch: {path}")

    def aggregate_artifact_names(self) -> tuple[str, ...]:
        """Return the exact generic-matrix aggregate artifact contract."""
        return (
            "aggregate_summary.json",
            "sweep_noise_summary.csv",
            "sweep_readout_diagnostics.csv",
            "sweep_selected_results.csv",
        )
