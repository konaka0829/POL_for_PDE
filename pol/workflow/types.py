"""Shared immutable matrix workflow types."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from pol.runtime.recipe import RecipeResult


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


@dataclass(frozen=True)
class DependencySpec:
    """Plugin-owned prerequisite declaration understood opaquely by core."""

    name: str
    kind: str
    config_path: Path | None
    protocol_version: str
    canonical_config_hash: str | None
    optional: bool = False


@dataclass(frozen=True)
class ResolvedDependencies:
    """Validated dependency identities and worker request fields."""

    identities: tuple[Mapping[str, Any], ...]
    request_fields: Mapping[str, Any]
    manifest_records: Mapping[str, Any]
    protected_paths: tuple[Path, ...] = ()


class MatrixExperimentPlugin(Protocol):
    """Experiment-owned policy consumed by the generic matrix executor."""

    plugin_id: str
    experiment_kind: str
    matrix_protocol_version: str
    plot_experiment_kind: str

    @property
    def recipe_protocol_versions(self) -> tuple[str, ...]: ...

    def load_base(self, path: Path) -> object: ...
    def finalize_config(
        self, raw: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any], str]: ...
    def canonical_config(self, raw: Mapping[str, Any]) -> str: ...
    def plan_summary(self, cells: list[MatrixCell]) -> dict[str, Any]: ...
    def parse_dependencies(
        self, raw: Mapping[str, Any], *, repo_root: Path
    ) -> tuple[DependencySpec, ...]: ...
    def resolve_dependencies(
        self,
        specs: Sequence[DependencySpec],
        *,
        run_dir: Path,
        base_config: Path,
        repo_root: Path,
        external_paths: Sequence[Path],
        allow_execution: bool,
    ) -> ResolvedDependencies: ...
    def execute_cell(self, request: Mapping[str, Any]) -> RecipeResult:
        """Execute one finalized cell without exposing its recipe to core."""
        ...
    def validate_cell(self, output_dir: Path, **kwargs: Any) -> str: ...
    def collect(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]: ...
    def validate_aggregate(self, *args: Any, **kwargs: Any) -> None: ...
    def aggregate_artifact_names(self) -> tuple[str, ...]: ...
