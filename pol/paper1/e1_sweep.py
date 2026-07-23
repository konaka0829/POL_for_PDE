"""Domain logic for Paper 1 E1 grid sweeps."""
from __future__ import annotations

import copy
import csv
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import config_from_dict

DIMENSION_KEYS = {"target_data_nx", "surrogate_internal_nx", "observation_dim"}
SPEC_KEYS = {"schema_version", "invalid_run_policy", "aggregate_plots", "experiments", "explicit_runs"}


@dataclass(order=True)
class SweepRun:
    """A unique E1 run and all experiments that contributed it."""

    n_tar: int
    n_sur: int
    J: int
    experiment_names: list[str] = field(default_factory=list, compare=False)

    @property
    def run_id(self) -> str:
        return f"ntar{self.n_tar}_nsur{self.n_sur}_J{self.J}"

    @property
    def full_observation(self) -> bool:
        return self.J == self.n_sur

    def metadata(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id, "n_tar": self.n_tar, "n_sur": self.n_sur,
            "J": self.J, "full_observation": self.full_observation,
            "experiment_names": self.experiment_names,
        }


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


def _dimensions(value: Any, path: str, *, lists: bool) -> dict[str, Any]:
    result = _object(value, path)
    unknown = sorted(set(result) - DIMENSION_KEYS)
    if unknown:
        raise ValueError(f"unknown sweep key: {path}.{unknown[0]}")
    for key, item in result.items():
        if lists:
            if not isinstance(item, list) or not item:
                raise ValueError(f"{path}.{key} must be a non-empty array")
            if any(not isinstance(x, int) or isinstance(x, bool) for x in item):
                raise ValueError(f"{path}.{key} values must be integers")
        elif not isinstance(item, int) or isinstance(item, bool):
            raise ValueError(f"{path}.{key} must be an integer")
    return result


def expand_sweep(spec: dict[str, Any]) -> tuple[list[SweepRun], dict[str, int]]:
    """Expand v2 experiments, deduplicate identities, and merge memberships."""
    unknown = sorted(set(spec) - SPEC_KEYS)
    if unknown:
        raise ValueError(f"unknown sweep key: {unknown[0]}")
    if spec.get("schema_version") != "paper1-e1-sweep-v2":
        raise ValueError("unsupported sweep schema; migrate to schema_version='paper1-e1-sweep-v2'")
    if spec.get("invalid_run_policy", "skip") not in {"skip", "error"}:
        raise ValueError("invalid_run_policy must be 'skip' or 'error'")
    experiments = spec.get("experiments", [])
    explicit = spec.get("explicit_runs", [])
    if not isinstance(experiments, list) or not isinstance(explicit, list):
        raise ValueError("experiments and explicit_runs must be arrays")
    unique: dict[tuple[int, int, int], SweepRun] = {}
    raw_counts: dict[str, int] = {}

    def add(values: dict[str, int], name: str) -> None:
        missing = DIMENSION_KEYS - set(values)
        if missing:
            raise ValueError(f"{name} does not define {sorted(missing)[0]}")
        key = (values["target_data_nx"], values["surrogate_internal_nx"], values["observation_dim"])
        run = unique.setdefault(key, SweepRun(*key))
        if name not in run.experiment_names:
            run.experiment_names.append(name)

    names: set[str] = set()
    for index, raw in enumerate(experiments):
        path = f"experiments[{index}]"
        experiment = _object(raw, path)
        unknown = sorted(set(experiment) - {"name", "fixed", "grid", "observation_rule"})
        if unknown:
            raise ValueError(f"unknown sweep key: {path}.{unknown[0]}")
        name = experiment.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError(f"{path}.name must be non-empty and unique")
        names.add(name)
        fixed = _dimensions(experiment.get("fixed", {}), f"{path}.fixed", lists=False)
        grid = _dimensions(experiment.get("grid", {}), f"{path}.grid", lists=True)
        overlap = set(fixed) & set(grid)
        if overlap:
            raise ValueError(f"{path}.{sorted(overlap)[0]} appears in both fixed and grid")
        rule = experiment.get("observation_rule")
        if rule not in {None, "full"}:
            raise ValueError(f"{path}.observation_rule must be 'full' when present")
        if rule == "full" and "observation_dim" in (set(fixed) | set(grid)):
            raise ValueError(f"{path}: observation_rule='full' conflicts with observation_dim")
        if rule is None and "observation_dim" not in (set(fixed) | set(grid)):
            raise ValueError(f"{path} requires observation_dim without observation_rule='full'")
        axes = sorted(grid)
        products = itertools.product(*(grid[key] for key in axes)) if axes else [()]
        count = 0
        for product in products:
            values = {**fixed, **dict(zip(axes, product))}
            if rule == "full":
                if "surrogate_internal_nx" not in values:
                    raise ValueError(f"{path} full observation requires surrogate_internal_nx")
                values["observation_dim"] = values["surrogate_internal_nx"]
            add(values, name)
            count += 1
        raw_counts[name] = count

    for index, raw in enumerate(explicit):
        path = f"explicit_runs[{index}]"
        values = _dimensions(raw, path, lists=False)
        add(values, "explicit_runs")
    if explicit:
        raw_counts["explicit_runs"] = len(explicit)
    if not unique:
        raise ValueError("the sweep specification produced no runs")
    for run in unique.values():
        run.experiment_names.sort()
    return sorted(unique.values()), raw_counts


def make_run_config(base: dict[str, Any], run: SweepRun) -> dict[str, Any]:
    """Create and validate an ordinary E1 config (the constraint source of truth)."""
    raw = copy.deepcopy(base)
    raw["spatial"]["target_data_nx"] = run.n_tar
    raw["spatial"]["surrogate_internal_nx"] = run.n_sur
    raw["spatial"]["observation_dim"] = run.J
    raw["e1"]["require_full_observation"] = run.full_observation
    config_from_dict(raw)
    return raw


def preflight(
    runs: list[SweepRun], base: dict[str, Any]
) -> tuple[list[SweepRun], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    valid, invalid, configs = [], [], {}
    q_max = max(base["e1"]["output_dims"])
    k_max = (q_max - 1) // 2
    for run in runs:
        try:
            configs[run.run_id] = make_run_config(base, run)
            valid.append(run)
        except Exception as exc:
            reason = str(exc)
            if run.n_tar <= 2 * k_max:
                reason = f"q_max={q_max} requires k_max={k_max} < target_data_nx/2, but target_data_nx={run.n_tar}"
            elif run.J <= 2 * k_max:
                reason = f"q_max={q_max} requires k_max={k_max} < observation_dim/2, but observation_dim={run.J}"
            invalid.append({**run.metadata(), "reason": reason})
    return valid, invalid, configs


def summary_passed(output_dir: Path) -> bool:
    try:
        return load_json(output_dir / "e1_summary.json").get("status") == "pass"
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect_outputs(output_root: Path, runs: list[SweepRun]) -> dict[str, int]:
    """Rebuild aggregate CSV files solely from successful run artifacts."""
    tables: dict[str, list[dict[str, Any]]] = {name: [] for name in (
        "selected_results", "readout_diagnostics", "noise_summary")}
    seen: dict[str, set[tuple[str, ...]]] = {name: set() for name in tables}
    for run in runs:
        run_dir = output_root / "runs" / run.run_id
        if not summary_passed(run_dir):
            continue
        prefix = {**run.metadata(), "experiment_names": json.dumps(run.experiment_names, separators=(",", ":"))}
        for name in tables:
            path = run_dir / f"{name}.csv"
            if not path.exists():
                raise ValueError(f"successful {run.run_id} is missing {path.name}")
            for row in read_csv(path):
                key = (run.run_id, *tuple(row.values()))
                if key in seen[name]:
                    raise ValueError(f"duplicate aggregate row in {name}: {run.run_id}")
                seen[name].add(key)
                tables[name].append({**prefix, **row})
    for name, rows in tables.items():
        write_csv(output_root / f"sweep_{name}.csv", rows)
    return {name: len(rows) for name, rows in tables.items()}
