"""Strict ``paper1-matrix-run-v1`` parsing and deterministic expansion."""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Literal, Mapping

from pol.plots.types import PlotTaskSpec

from .types import MatrixCell


_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_TOP_KEYS = {
    "schema_version",
    "run",
    "experiment",
    "prerequisites",
    "matrix",
    "execution",
    "aggregation",
    "plots",
}
_REQUIRED_TOP_KEYS = _TOP_KEYS - {"plots"}


@dataclass(frozen=True)
class MatrixRunSpec:
    """Resolved strict matrix orchestration specification."""

    schema_version: str
    name: str
    output_root: Path
    kind: str
    base_config: Path
    e0_config: Path
    invalid_run_policy: Literal["skip", "error"]
    experiments: tuple[Mapping[str, Any], ...]
    explicit_runs: tuple[Mapping[str, Any], ...]
    jobs: int
    torch_threads_per_job: int
    resume: bool
    cell_plots: bool
    aggregation_kind: str
    plots_enabled: bool
    plots_required: bool
    plot_tasks: tuple[PlotTaskSpec, ...]
    source_path: Path
    raw: Mapping[str, Any]

    @property
    def run_dir(self) -> Path:
        return self.output_root / self.name


def _object(value: object, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"expected object at {path}")
    return value


def _keys(
    value: Mapping[str, Any],
    path: str,
    *,
    required: set[str],
    allowed: set[str],
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"unknown key at {path}: {unknown[0]}")
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"missing required key at {path}.{missing[0]}")


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"expected non-empty string at {path}")
    return value


def _positive_int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"expected positive integer at {path}")
    return value


def _boolean(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"expected boolean at {path}")
    return value


def _plot_block(
    value: object, *, experiment_kind: str
) -> tuple[bool, bool, tuple[PlotTaskSpec, ...]]:
    plots = _object(value, "$.plots")
    _keys(
        plots,
        "$.plots",
        required={"enabled", "required", "recipes"},
        allowed={"enabled", "required", "recipes"},
    )
    enabled = _boolean(plots["enabled"], "$.plots.enabled")
    required = _boolean(plots["required"], "$.plots.required")
    raw_recipes = plots["recipes"]
    if not isinstance(raw_recipes, list):
        raise ValueError("expected array at $.plots.recipes")
    if enabled != bool(raw_recipes):
        raise ValueError("$.plots.enabled must match whether recipes are present")
    if required and not enabled:
        raise ValueError("$.plots.required cannot be true when plots are disabled")
    tasks: list[PlotTaskSpec] = []
    for index, raw in enumerate(raw_recipes):
        path = f"$.plots.recipes[{index}]"
        recipe = _object(raw, path)
        _keys(
            recipe,
            path,
            required={"id", "settings"},
            allowed={"id", "settings"},
        )
        recipe_id = _string(recipe["id"], f"{path}.id")
        settings = _object(recipe["settings"], f"{path}.settings")
        from pol.plots.registry import get_plot_recipe

        registered = get_plot_recipe(recipe_id)
        if experiment_kind not in registered.supported_experiment_kinds:
            raise ValueError(f"unsupported plot recipe at {path}.id: {recipe_id}")
        try:
            validated = registered.validate_settings(dict(settings))
        except ValueError as exc:
            raise ValueError(f"invalid settings at {path}.settings: {exc}") from exc
        tasks.append(PlotTaskSpec(recipe_id, validated))
    if len({task.recipe_id for task in tasks}) != len(tasks):
        raise ValueError("duplicate plot recipe at $.plots.recipes")
    return enabled, required, tuple(tasks)


def _repo_path(value: object, path: str, root: Path) -> Path:
    raw = Path(_string(value, path))
    resolved = (raw if raw.is_absolute() else root / raw).resolve()
    if path.endswith(("base_config", "e0_config")) and not resolved.is_file():
        raise ValueError(f"file does not exist at {path}: {resolved}")
    return resolved


def _validate_scalar(value: object, path: str) -> None:
    if value is None or isinstance(value, (str, float, bool)):
        return
    if isinstance(value, int) and not isinstance(value, bool):
        return
    raise ValueError(f"expected scalar value at {path}")


def _validate_override_map(value: object, path: str) -> dict[str, Any]:
    result = _object(value, path)
    for key, item in result.items():
        _string(key, f"{path}.<key>")
        _validate_scalar(item, f"{path}.{key}")
    return result


def _validate_experiment(value: object, path: str) -> dict[str, Any]:
    experiment = _object(value, path)
    _keys(
        experiment,
        path,
        required={"name", "fixed", "grid", "copy"},
        allowed={"name", "fixed", "grid", "copy"},
    )
    _string(experiment["name"], f"{path}.name")
    fixed = _validate_override_map(experiment["fixed"], f"{path}.fixed")
    grid = experiment["grid"]
    if not isinstance(grid, list):
        raise ValueError(f"expected array at {path}.grid")
    grid_paths: list[str] = []
    for index, raw_axis in enumerate(grid):
        axis_path = f"{path}.grid[{index}]"
        axis = _object(raw_axis, axis_path)
        _keys(
            axis,
            axis_path,
            required={"path", "values"},
            allowed={"path", "values"},
        )
        target = _string(axis["path"], f"{axis_path}.path")
        values = axis["values"]
        if not isinstance(values, list) or not values:
            raise ValueError(f"expected non-empty array at {axis_path}.values")
        for item_index, item in enumerate(values):
            _validate_scalar(item, f"{axis_path}.values[{item_index}]")
        grid_paths.append(target)
    copy_map = _object(experiment["copy"], f"{path}.copy")
    for target, source in copy_map.items():
        _string(target, f"{path}.copy.<target>")
        _string(source, f"{path}.copy.{target}")
    targets = list(fixed) + grid_paths + list(copy_map)
    duplicates = sorted({target for target in targets if targets.count(target) > 1})
    if duplicates:
        raise ValueError(f"conflicting override target at {path}: {duplicates[0]}")
    if len(grid_paths) != len(set(grid_paths)):
        raise ValueError(f"duplicate grid path at {path}")
    return experiment


def load_matrix_spec(path: str | Path, *, repo_root: Path) -> MatrixRunSpec:
    """Load and strictly validate a matrix run specification."""
    source = Path(path).resolve()
    try:
        raw_value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load matrix spec {source}: {exc}") from exc
    raw = _object(raw_value, "$")
    _keys(raw, "$", required=_REQUIRED_TOP_KEYS, allowed=_TOP_KEYS)
    if raw["schema_version"] != "paper1-matrix-run-v1":
        raise ValueError("unsupported value at $.schema_version")
    root = repo_root.resolve()
    run = _object(raw["run"], "$.run")
    _keys(run, "$.run", required={"name", "output_root"}, allowed={"name", "output_root"})
    name = _string(run["name"], "$.run.name")
    if not _NAME_RE.fullmatch(name) or ".." in name:
        raise ValueError(f"unsafe directory name at $.run.name: {name!r}")
    output_root = _repo_path(run["output_root"], "$.run.output_root", root)

    experiment = _object(raw["experiment"], "$.experiment")
    _keys(
        experiment,
        "$.experiment",
        required={"kind", "base_config"},
        allowed={"kind", "base_config"},
    )
    experiment_kind = _string(experiment["kind"], "$.experiment.kind")
    base_config = _repo_path(
        experiment["base_config"], "$.experiment.base_config", root
    )
    prerequisites = _object(raw["prerequisites"], "$.prerequisites")
    _keys(
        prerequisites,
        "$.prerequisites",
        required={"e0_config"},
        allowed={"e0_config"},
    )
    e0_config = _repo_path(
        prerequisites["e0_config"], "$.prerequisites.e0_config", root
    )

    matrix = _object(raw["matrix"], "$.matrix")
    _keys(
        matrix,
        "$.matrix",
        required={"invalid_run_policy", "experiments", "explicit_runs"},
        allowed={"invalid_run_policy", "experiments", "explicit_runs"},
    )
    policy = matrix["invalid_run_policy"]
    if policy not in {"skip", "error"}:
        raise ValueError("unsupported value at $.matrix.invalid_run_policy")
    experiments_raw = matrix["experiments"]
    explicit_raw = matrix["explicit_runs"]
    if not isinstance(experiments_raw, list):
        raise ValueError("expected array at $.matrix.experiments")
    if not isinstance(explicit_raw, list):
        raise ValueError("expected array at $.matrix.explicit_runs")
    experiments = tuple(
        _validate_experiment(item, f"$.matrix.experiments[{index}]")
        for index, item in enumerate(experiments_raw)
    )
    names = [item["name"] for item in experiments]
    if len(names) != len(set(names)):
        raise ValueError("matrix experiment names must be unique")
    explicit_runs = tuple(
        _validate_override_map(item, f"$.matrix.explicit_runs[{index}]")
        for index, item in enumerate(explicit_raw)
    )
    if not experiments and not explicit_runs:
        raise ValueError("matrix must produce at least one raw run")

    execution = _object(raw["execution"], "$.execution")
    execution_keys = {"jobs", "torch_threads_per_job", "resume", "cell_plots"}
    _keys(execution, "$.execution", required=execution_keys, allowed=execution_keys)
    jobs = _positive_int(execution["jobs"], "$.execution.jobs")
    threads = _positive_int(
        execution["torch_threads_per_job"],
        "$.execution.torch_threads_per_job",
    )
    resume = _boolean(execution["resume"], "$.execution.resume")
    cell_plots = _boolean(execution["cell_plots"], "$.execution.cell_plots")
    aggregation = _object(raw["aggregation"], "$.aggregation")
    _keys(
        aggregation,
        "$.aggregation",
        required={"kind"},
        allowed={"kind"},
    )
    aggregation_kind = _string(aggregation["kind"], "$.aggregation.kind")
    from pol.workflow.registry import get_matrix_plugin

    plugin = get_matrix_plugin(aggregation_kind)
    if plugin.experiment_kind != experiment_kind:
        raise ValueError(
            "$.experiment.kind does not match $.aggregation.kind plugin"
        )
    if "plots" in raw:
        plots_enabled, plots_required, plot_tasks = _plot_block(
            raw["plots"], experiment_kind=plugin.plot_experiment_kind
        )
    else:
        plots_enabled, plots_required, plot_tasks = False, False, ()
    return MatrixRunSpec(
        "paper1-matrix-run-v1",
        name,
        output_root,
        experiment_kind,
        base_config,
        e0_config,
        policy,
        experiments,
        explicit_runs,
        jobs,
        threads,
        resume,
        cell_plots,
        aggregation_kind,
        plots_enabled,
        plots_required,
        plot_tasks,
        source,
        raw,
    )


def _leaf(root: Mapping[str, Any], path: str) -> object:
    current: object = root
    for component in path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            raise ValueError(f"override path does not exist: {path}")
        current = current[component]
    if isinstance(current, (dict, list)):
        raise ValueError(f"override path is not a scalar leaf: {path}")
    return current


def _set_leaf(root: dict[str, Any], path: str, value: object) -> None:
    expected = _leaf(root, path)
    if isinstance(expected, bool) != isinstance(value, bool):
        raise ValueError(f"override type mismatch at {path}")
    if not isinstance(expected, bool) and type(value) is not type(expected):
        if not (
            isinstance(expected, float)
            and isinstance(value, int)
            and not isinstance(value, bool)
        ):
            raise ValueError(f"override type mismatch at {path}")
    current = root
    components = path.split(".")
    for component in components[:-1]:
        current = current[component]
    current[components[-1]] = value


def _apply_copy(root: dict[str, Any], copies: Mapping[str, str]) -> None:
    remaining = dict(copies)
    completed: set[str] = set()
    while remaining:
        progressed = False
        for target, source in list(remaining.items()):
            if source in remaining and source not in completed:
                continue
            _set_leaf(root, target, _leaf(root, source))
            completed.add(target)
            del remaining[target]
            progressed = True
        if not progressed:
            raise ValueError("copy cycle detected")


def expand_matrix(
    spec: MatrixRunSpec,
    *,
    base: Mapping[str, Any],
    plugin: Any,
) -> tuple[list[MatrixCell], list[dict[str, Any]], dict[str, int]]:
    """Expand in declaration order, validate, and deduplicate canonical cells."""
    contributions: list[tuple[str, Mapping[str, Any], Mapping[str, str]]] = []
    raw_counts: dict[str, int] = {}
    for experiment in spec.experiments:
        axes = experiment["grid"]
        products = itertools.product(*(axis["values"] for axis in axes))
        count = 0
        for product in products:
            overrides = dict(experiment["fixed"])
            overrides.update(
                {axis["path"]: value for axis, value in zip(axes, product)}
            )
            contributions.append(
                (str(experiment["name"]), overrides, experiment["copy"])
            )
            count += 1
        raw_counts[str(experiment["name"])] = count
    for index, overrides in enumerate(spec.explicit_runs):
        contributions.append(("explicit_runs", overrides, {}))
    if spec.explicit_runs:
        raw_counts["explicit_runs"] = len(spec.explicit_runs)

    unique: dict[str, MatrixCell] = {}
    invalid: list[dict[str, Any]] = []
    for raw_index, (membership, overrides, copies) in enumerate(contributions):
        candidate = copy.deepcopy(dict(base))
        try:
            for path, value in overrides.items():
                _set_leaf(candidate, path, value)
            for target, source in copies.items():
                _leaf(candidate, target)
                _leaf(candidate, source)
            _apply_copy(candidate, copies)
            finalized, metadata, slug = plugin.finalize_config(candidate)
            canonical = plugin.canonical_config(finalized)
            digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        except Exception as exc:
            invalid.append(
                {
                    "raw_index": raw_index,
                    "experiment_membership": membership,
                    "overrides": dict(overrides),
                    "failure_type": type(exc).__name__,
                    "failure_message": str(exc),
                }
            )
            continue
        if digest in unique:
            previous = unique[digest]
            memberships = list(previous.experiment_memberships)
            if membership not in memberships:
                memberships.append(membership)
            unique[digest] = MatrixCell(
                previous.run_index,
                previous.cell_id,
                previous.config_sha256,
                previous.canonical_config,
                previous.human_slug,
                tuple(memberships),
                previous.metadata,
            )
            continue
        unique[digest] = MatrixCell(
            len(unique),
            digest[:16],
            digest,
            canonical,
            slug,
            (membership,),
            metadata,
        )
    cells = list(unique.values())
    if invalid and spec.invalid_run_policy == "error":
        first = invalid[0]
        raise ValueError(
            "invalid matrix run at raw index "
            f"{first['raw_index']}: {first['failure_message']}"
        )
    if not cells:
        raise ValueError("matrix expansion produced no valid cells")
    return cells, invalid, raw_counts
