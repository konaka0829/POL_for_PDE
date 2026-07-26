"""Strict Phase 1 orchestration manifests for Paper 1 runs."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Literal

from pol.plots.types import PlotTaskSpec

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
    plots_enabled: bool
    plots_required: bool
    plot_tasks: tuple[PlotTaskSpec, ...]
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


def _plot_block(
    value: object, *, kind: str
) -> tuple[bool, bool, tuple[PlotTaskSpec, ...]]:
    plots = _object(value, "$.plots")
    _keys(
        plots,
        "$.plots",
        required={"enabled", "required", "recipes"},
        allowed={"enabled", "required", "recipes"},
    )
    enabled = plots["enabled"]
    required = plots["required"]
    recipes = plots["recipes"]
    if not isinstance(enabled, bool):
        raise ValueError("expected boolean at $.plots.enabled")
    if not isinstance(required, bool):
        raise ValueError("expected boolean at $.plots.required")
    if not isinstance(recipes, list):
        raise ValueError("expected array at $.plots.recipes")
    if not enabled and recipes:
        raise ValueError("$.plots.recipes must be empty when plots are disabled")
    if required and not enabled:
        raise ValueError("$.plots.required cannot be true when plots are disabled")
    if enabled and not recipes:
        raise ValueError("$.plots.recipes must be non-empty when plots are enabled")
    supported = {
        "e1": {"paper1.e1.standard.v1"},
        "e2": {"paper1.e2.standard.v1"},
    }.get(kind, set())
    tasks: list[PlotTaskSpec] = []
    seen: set[str] = set()
    for index, item in enumerate(recipes):
        path = f"$.plots.recipes[{index}]"
        recipe = _object(item, path)
        _keys(
            recipe,
            path,
            required={"id", "formats", "dpi"},
            allowed={"id", "formats", "dpi"},
        )
        recipe_id = _string(recipe["id"], f"{path}.id")
        if recipe_id not in supported:
            raise ValueError(f"unsupported plot recipe at {path}.id: {recipe_id}")
        if recipe_id in seen:
            raise ValueError(f"duplicate plot recipe at {path}.id: {recipe_id}")
        seen.add(recipe_id)
        formats = recipe["formats"]
        if (
            not isinstance(formats, list)
            or not formats
            or any(item not in {"png", "pdf", "svg"} for item in formats)
            or len(formats) != len(set(formats))
        ):
            raise ValueError(f"expected unique png/pdf/svg array at {path}.formats")
        dpi = _positive_int(recipe["dpi"], f"{path}.dpi")
        tasks.append(
            PlotTaskSpec(
                recipe_id,
                {"formats": list(formats), "dpi": dpi},
            )
        )
    return enabled, required, tuple(tasks)


def _resolve_repo_path(value: object, path: str, repo_root: Path) -> Path:
    raw = Path(_string(value, path))
    return (raw if raw.is_absolute() else repo_root / raw).resolve()


def _validate_e2_prerequisite_config(e2_config: object, e0_config: object) -> None:
    """Reject only statically certain E2/E0 prerequisite mismatches."""
    e2 = e2_config
    e0 = e0_config
    for section_name in ("domain", "data"):
        e2_values = asdict(getattr(e2, section_name))
        e0_values = asdict(getattr(e0, section_name))
        for field_name, e2_value in e2_values.items():
            e0_value = e0_values[field_name]
            if e2_value != e0_value:
                raise ValueError(
                    "E2 prerequisite config mismatch: "
                    f"{section_name}.{field_name}={e2_value!r} "
                    f"!= e0.{section_name}.{field_name}={e0_value!r}"
                )
    for field_name in ("equation", "nu", "T", "solver", "dealias"):
        e2_value = getattr(e2.target, field_name)
        e0_value = getattr(e0.target, field_name)
        if e2_value != e0_value:
            raise ValueError(
                "E2 prerequisite config mismatch: "
                f"target.{field_name}={e2_value!r} "
                f"!= e0.target.{field_name}={e0_value!r}"
            )

    requested_time = (e2.target.dt, e2.target.fine_dt)
    candidates = {
        (candidate.dt, candidate.fine_dt)
        for candidate in e0.e0.time_candidates
    }
    if requested_time not in candidates:
        raise ValueError(
            "E2 prerequisite config mismatch: "
            f"target.(dt, fine_dt)={requested_time!r} is not present in "
            f"e0.time_candidates={list(candidates)!r}"
        )
    max_reference = max(e0.e0.reference_nx_candidates)
    if max_reference < e2.spatial.reference_nx:
        raise ValueError(
            "E2 prerequisite config mismatch: "
            f"max(e0.reference_nx_candidates)={max_reference} "
            f"< e2.spatial.reference_nx={e2.spatial.reference_nx}"
        )
    if e0.e0.q_reference_check < e2.spatial.target_output_dim:
        raise ValueError(
            "E2 prerequisite config mismatch: "
            f"e0.q_reference_check={e0.e0.q_reference_check} "
            f"< e2.spatial.target_output_dim={e2.spatial.target_output_dim}"
        )
    identity_q = e0.e0.model1_identity.target_output_dim
    if identity_q < e2.spatial.target_output_dim:
        raise ValueError(
            "E2 prerequisite config mismatch: "
            f"e0.model1_identity.target_output_dim={identity_q} "
            f"< e2.spatial.target_output_dim={e2.spatial.target_output_dim}"
        )


def load_run_spec(path: str | Path, *, repo_root: Path) -> Paper1RunSpec:
    """Load and strictly validate a ``paper1-run-v1`` JSON manifest."""
    source = Path(path).resolve()
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load run spec {source}: {exc}") from exc
    top = _object(raw, "$")
    base_keys = {"schema_version", "run", "experiment", "prerequisites", "execution"}
    if "schema_version" not in top:
        raise ValueError("missing required key at $.schema_version")
    schema = _string(top["schema_version"], "$.schema_version")
    if schema not in {"paper1-run-v1", "paper1-run-v2"}:
        raise ValueError(f"unsupported value at $.schema_version: {schema}")
    top_keys = base_keys | ({"plots"} if schema == "paper1-run-v2" else set())
    _keys(top, "$", required=top_keys, allowed=top_keys)

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
    v1_execution = (
        set()
        if kind == "e0"
        else {"torch_threads", "skip_plots"}
        if kind == "e1"
        else {"torch_threads", "batch_size", "skip_plots"}
    )
    allowed_execution = (
        v1_execution
        if schema == "paper1-run-v1"
        else v1_execution - {"skip_plots"}
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
    if schema == "paper1-run-v2":
        if kind == "e0":
            plots_enabled, plots_required, plot_tasks = _plot_block(
                top["plots"], kind=kind
            )
            if plots_enabled:
                raise ValueError("E0 does not support plots")
        else:
            plots_enabled, plots_required, plot_tasks = _plot_block(
                top["plots"], kind=kind
            )
        skip_plots_value = not plots_enabled
    else:
        plots_enabled = kind in {"e1", "e2"} and not skip_plots_value
        plots_required = plots_enabled
        plot_tasks = (
            (
                PlotTaskSpec(
                    f"paper1.{kind}.standard.v1",
                    {
                        "formats": ["png"] if kind == "e1" else ["png", "pdf"],
                        "dpi": 160 if kind == "e1" else 180,
                    },
                ),
            )
            if plots_enabled
            else ()
        )

    config = load_config_json(experiment_config)
    if getattr(config, kind) is None:
        raise ValueError(
            f"config at $.experiment.config does not contain section {kind}"
        )
    prerequisite_config = load_config_json(e0_config) if e0_config else None
    if prerequisite_config is not None and prerequisite_config.e0 is None:
        raise ValueError("config at $.prerequisites.e0_config does not contain section e0")
    if kind == "e2":
        assert prerequisite_config is not None
        _validate_e2_prerequisite_config(config, prerequisite_config)

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
        plots_enabled=plots_enabled,
        plots_required=plots_required,
        plot_tasks=plot_tasks,
        source_path=source,
    )


def run_spec_to_resolved_dict(
    spec: Paper1RunSpec, *, run_dir: Path
) -> dict[str, object]:
    """Return the resolved public fields of a run spec."""
    return {
        "schema_version": spec.schema_version,
        "source_run_spec": str(spec.source_path),
        "run": {
            "name": spec.name,
            "output_root": str(spec.output_root),
            "run_dir": str(run_dir),
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
        "plots": {
            "enabled": spec.plots_enabled,
            "required": spec.plots_required,
            "recipes": [
                {"id": task.recipe_id, **dict(task.settings)}
                for task in spec.plot_tasks
            ],
        },
    }
