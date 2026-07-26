"""Strict Phase 1 orchestration manifests for Paper 1 runs."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Literal

from .config import load_config_json


_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


@dataclass(frozen=True)
class Paper1RunSpec:
    """A resolved, validated Paper 1 orchestration manifest."""

    schema_version: str
    name: str
    output_root: Path
    kind: Literal["e0", "e1", "e2"]
    experiment_config: Path
    e0_config: Path | None
    torch_threads: int | None
    batch_size: int | None
    skip_plots: bool
    source_path: Path

    @property
    def run_dir(self) -> Path:
        """Return the run-specific output directory."""
        return self.output_root / self.name


def _object(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"expected object at {path}")
    return value


def _keys(
    value: dict[str, object], path: str, *, required: set[str], allowed: set[str]
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"unknown key at {path}: {unknown[0]}")
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"missing required key at {path}.{missing[0]}")


def _string(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"expected string at {path}")
    return value


def _positive_int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"expected positive integer at {path}")
    return value


def _resolve_repo_path(value: object, path: str, repo_root: Path) -> Path:
    raw = Path(_string(value, path))
    return (raw if raw.is_absolute() else repo_root / raw).resolve()


def load_run_spec(path: str | Path, *, repo_root: Path) -> Paper1RunSpec:
    """Load and strictly validate a ``paper1-run-v1`` JSON manifest."""
    source = Path(path).resolve()
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load run spec {source}: {exc}") from exc
    top = _object(raw, "$")
    top_keys = {"schema_version", "run", "experiment", "prerequisites", "execution"}
    _keys(top, "$", required=top_keys, allowed=top_keys)
    schema = _string(top["schema_version"], "$.schema_version")
    if schema != "paper1-run-v1":
        raise ValueError(f"unsupported value at $.schema_version: {schema}")

    run = _object(top["run"], "$.run")
    _keys(run, "$.run", required={"name", "output_root"}, allowed={"name", "output_root"})
    name = _string(run["name"], "$.run.name")
    if not _NAME_RE.fullmatch(name) or name in {".", ".."} or ".." in name:
        raise ValueError(f"unsafe directory name at $.run.name: {name!r}")
    root = repo_root.resolve()
    output_root = _resolve_repo_path(run["output_root"], "$.run.output_root", root)

    experiment = _object(top["experiment"], "$.experiment")
    _keys(
        experiment,
        "$.experiment",
        required={"kind", "config"},
        allowed={"kind", "config"},
    )
    kind_value = _string(experiment["kind"], "$.experiment.kind")
    if kind_value not in {"e0", "e1", "e2"}:
        raise ValueError(f"unsupported value at $.experiment.kind: {kind_value}")
    kind: Literal["e0", "e1", "e2"] = kind_value  # type: ignore[assignment]
    experiment_config = _resolve_repo_path(
        experiment["config"], "$.experiment.config", root
    )
    if not experiment_config.is_file():
        raise ValueError(f"config file does not exist at $.experiment.config: {experiment_config}")

    prerequisites = _object(top["prerequisites"], "$.prerequisites")
    required_prerequisites = set() if kind == "e0" else {"e0_config"}
    _keys(
        prerequisites,
        "$.prerequisites",
        required=required_prerequisites,
        allowed=required_prerequisites,
    )
    e0_config = (
        _resolve_repo_path(prerequisites["e0_config"], "$.prerequisites.e0_config", root)
        if kind != "e0"
        else None
    )
    if e0_config is not None and not e0_config.is_file():
        raise ValueError(f"config file does not exist at $.prerequisites.e0_config: {e0_config}")

    execution = _object(top["execution"], "$.execution")
    allowed_execution = (
        set()
        if kind == "e0"
        else {"torch_threads", "skip_plots"}
        if kind == "e1"
        else {"torch_threads", "batch_size", "skip_plots"}
    )
    _keys(execution, "$.execution", required=allowed_execution, allowed=allowed_execution)
    torch_threads = (
        _positive_int(execution["torch_threads"], "$.execution.torch_threads")
        if kind != "e0"
        else None
    )
    batch_size = (
        _positive_int(execution["batch_size"], "$.execution.batch_size")
        if kind == "e2"
        else None
    )
    skip_plots_value = execution.get("skip_plots", False)
    if not isinstance(skip_plots_value, bool):
        raise ValueError("expected boolean at $.execution.skip_plots")

    config = load_config_json(experiment_config)
    if getattr(config, kind) is None:
        raise ValueError(
            f"config at $.experiment.config does not contain section {kind}"
        )
    if e0_config is not None and load_config_json(e0_config).e0 is None:
        raise ValueError("config at $.prerequisites.e0_config does not contain section e0")

    return Paper1RunSpec(
        schema_version=schema,
        name=name,
        output_root=output_root,
        kind=kind,
        experiment_config=experiment_config,
        e0_config=e0_config,
        torch_threads=torch_threads,
        batch_size=batch_size,
        skip_plots=skip_plots_value,
        source_path=source,
    )


def run_spec_to_resolved_dict(spec: Paper1RunSpec) -> dict[str, object]:
    """Return the resolved public fields of a run spec."""
    return {
        "schema_version": spec.schema_version,
        "source_run_spec": str(spec.source_path),
        "run": {
            "name": spec.name,
            "output_root": str(spec.output_root),
            "run_dir": str(spec.run_dir.resolve()),
        },
        "experiment": {
            "kind": spec.kind,
            "config": str(spec.experiment_config),
        },
        "prerequisites": {
            **({"e0_config": str(spec.e0_config)} if spec.e0_config else {})
        },
        "execution": {
            **(
                {"torch_threads": spec.torch_threads, "skip_plots": spec.skip_plots}
                if spec.kind != "e0"
                else {}
            ),
            **({"batch_size": spec.batch_size} if spec.kind == "e2" else {}),
        },
    }
