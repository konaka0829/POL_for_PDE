"""Semantic Phase 1 regression baselines for saved Paper 1 artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from pol.runtime.io import file_sha256, write_strict_json


BASELINE_SCHEMA_VERSION = "paper1-phase1-scientific-baseline-v2"
SIGNIFICANT_DIGITS = 12

# These fields encode execution location or bindings rather than scientific results.
PROVENANCE_FIELDS = frozenset(
    {
        "attempt_history_hash",
        "bindings",
        "command",
        "cwd",
        "dataset_prerequisite_hash",
        "e0_prerequisite_hash",
        "environment",
        "frozen_plan_hash",
        "git_commit",
        "git_commit_id",
        "git_dirty_status",
        "requested_config",
        "runtime_seconds",
        "selection_record_hash",
        "started_at",
        "ended_at",
    }
)

CSV_STABLE_KEYS: Mapping[str, tuple[str, ...]] = {
    "reference_convergence.csv": ("kind", "candidate_nx"),
    "ridge_selection.csv": ("case_name", "q", "zeta"),
    "selected_results.csv": ("case_name", "q"),
    "readout_diagnostics.csv": ("case_name", "q"),
    "mode_comparison.csv": ("case_name", "q", "coefficient_index"),
    "noise_summary.csv": ("case_name", "q", "noise_level"),
    "validation_sweep.csv": (
        "family",
        "sweep_axis",
        "nu_tilde",
        "T_tilde",
        "model",
    ),
    "model3_validation_by_seed.csv": (
        "family",
        "sweep_axis",
        "nu_tilde",
        "T_tilde",
        "candidate_order",
        "seed",
    ),
    "convergence_results.csv": ("family", "n_sur"),
    "solver_metadata.csv": ("family", "sweep_axis", "nu_tilde", "T_tilde"),
    "physical_point_aliases.csv": (
        "family",
        "sweep_axis",
        "nu_tilde",
        "T_tilde",
    ),
    "test_sweep.csv": (
        "family",
        "sweep_axis",
        "nu_tilde",
        "T_tilde",
        "model",
    ),
    "model3_test_by_seed.csv": (
        "family",
        "sweep_axis",
        "nu_tilde",
        "T_tilde",
        "seed",
    ),
    "model3_test_aggregate.csv": (
        "family",
        "sweep_axis",
        "nu_tilde",
        "T_tilde",
    ),
}

E1_TABLES = (
    "ridge_selection.csv",
    "selected_results.csv",
    "readout_diagnostics.csv",
    "mode_comparison.csv",
    "noise_summary.csv",
)
E2_JSON = (
    "model_specific_optima.json",
    "shared_representatives.json",
    "coordinate_history.json",
    "convergence_summary.json",
    "selection_record.json",
)
E2_TABLES = (
    "validation_sweep.csv",
    "model3_validation_by_seed.csv",
    "convergence_results.csv",
    "solver_metadata.csv",
    "physical_point_aliases.csv",
    "test_sweep.csv",
    "model3_test_by_seed.csv",
    "model3_test_aggregate.csv",
)

_INTEGER = re.compile(r"[+-]?[0-9]+\Z")


def canonical_float(value: float) -> str:
    """Return a finite float as a 12-significant-digit decimal string."""
    if not math.isfinite(value):
        raise ValueError(f"non-finite float in scientific baseline: {value!r}")
    return format(value, f".{SIGNIFICANT_DIGITS}g")


def canonicalize_scientific(value: Any) -> Any:
    """Canonicalize scientific JSON values and remove explicit provenance."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return canonical_float(value)
    if isinstance(value, Mapping):
        return {
            str(key): canonicalize_scientific(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in PROVENANCE_FIELDS
        }
    if isinstance(value, (list, tuple)):
        return [canonicalize_scientific(item) for item in value]
    raise TypeError(f"unsupported scientific baseline value: {type(value).__name__}")


def semantic_tensor_digest(tensor: torch.Tensor) -> dict[str, object]:
    """Describe and hash a tensor after semantic decimal canonicalization."""
    value = tensor.detach().cpu().contiguous()
    if value.is_complex():
        flat: Iterable[object] = (
            component
            for item in value.reshape(-1).tolist()
            for component in (canonical_float(float(item.real)), canonical_float(float(item.imag)))
        )
        encoding = "real_imag_interleaved"
    elif value.is_floating_point():
        flat = (canonical_float(float(item)) for item in value.reshape(-1).tolist())
        encoding = "real"
    else:
        flat = (int(item) for item in value.reshape(-1).tolist())
        encoding = "integer"
    payload = json.dumps(list(flat), separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return {
        "shape": list(value.shape),
        "logical_dtype": str(value.dtype).removeprefix("torch."),
        "numel": value.numel(),
        "encoding": encoding,
        "values_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _semantic_model_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {"tensor": semantic_tensor_digest(value)}
    if isinstance(value, Mapping):
        return {
            str(key): _semantic_model_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in PROVENANCE_FIELDS
        }
    if isinstance(value, (list, tuple)):
        return [_semantic_model_value(item) for item in value]
    return canonicalize_scientific(value)


def semantic_model_digest(path: Path) -> str:
    """Hash the logical content of a saved model archive, not its zip bytes."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    canonical = _semantic_model_value(payload.get("models", payload))
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv_scalar(value: str) -> Any:
    stripped = value.strip()
    if stripped == "":
        return None
    if stripped in {"true", "false"}:
        return stripped == "true"
    if _INTEGER.fullmatch(stripped):
        return int(stripped)
    try:
        return float(stripped)
    except ValueError:
        if stripped.startswith(("[", "{")):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                pass
        return value


def canonical_csv(path: Path) -> list[dict[str, Any]]:
    """Read, canonicalize, and deterministically sort a scientific CSV table."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            canonicalize_scientific(
                {
                    key: _csv_scalar(value)
                    for key, value in row.items()
                    if key not in PROVENANCE_FIELDS
                }
            )
            for row in csv.DictReader(handle)
        ]
    stable_keys = CSV_STABLE_KEYS[path.name]
    missing = [key for key in stable_keys if rows and key not in rows[0]]
    if missing:
        raise ValueError(f"{path.name} lacks stable key: {missing[0]}")
    return sorted(
        rows,
        key=lambda row: tuple(
            json.dumps(row[key], sort_keys=True, separators=(",", ":"))
            for key in stable_keys
        ),
    )


def _verify_artifact_manifest(directory: Path) -> None:
    manifest_path = directory / "artifact_manifest.json"
    raw = _json(manifest_path)
    records = raw if isinstance(raw, list) else raw.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError(f"invalid artifact manifest: {manifest_path}")
    for record in records:
        relative = record["relative_path"]
        path = directory / relative
        size = record.get("byte_size", record.get("size_bytes"))
        if not path.is_file():
            raise ValueError(f"artifact manifest file is missing: {path}")
        if path.stat().st_size != size or file_sha256(path) != record["sha256"]:
            raise ValueError(f"artifact manifest verification failed: {path}")


def _require_pass(directory: Path, summary_name: str) -> dict[str, Any]:
    summary = _json(directory / summary_name)
    if not isinstance(summary, dict) or summary.get("status") != "pass":
        raise ValueError(f"{directory / summary_name} is not a passing summary")
    return summary


def _plot_inventory(directory: Path) -> list[dict[str, str]]:
    manifest = _json(directory / "plot_manifest.json")
    if manifest.get("status") != "pass":
        raise ValueError(f"plot manifest is not pass: {directory}")
    inventory = []
    records = manifest.get("plots", manifest.get("outputs", []))
    for record in records:
        relative = record["relative_path"]
        path = directory / relative
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"plot is missing or empty: {path}")
        inventory.append(
            {
                "logical_name": record.get("kind", Path(relative).stem),
                "format": record.get("format", path.suffix.lstrip(".").lower()),
            }
        )
    return sorted(inventory, key=lambda item: (item["logical_name"], item["format"]))


def build_e0_scientific_record(e0_dir: Path) -> dict[str, Any]:
    """Build the semantic E0 portion after validating its passing summary."""
    e0_summary = _require_pass(e0_dir, "e0_summary.json")
    master = torch.load(
        e0_dir / "master_initial_conditions.pt",
        map_location="cpu",
        weights_only=False,
    )
    master_manifest = _json(e0_dir / "master_manifest.json")
    if not isinstance(master, dict) or not isinstance(master_manifest, dict):
        raise ValueError("invalid E0 master archive or manifest")
    if master.get("metadata", {}).get("tensor_hash") != master_manifest.get(
        "tensor_hash"
    ):
        raise ValueError("E0 master archive/manifest tensor hash mismatch")
    if not torch.isfinite(master["values"]).all():
        raise ValueError("E0 master archive contains non-finite values")
    resolved_e0 = _json(e0_dir / "resolved_config.json")
    return {
        "summary": canonicalize_scientific(e0_summary),
        "accepted_science": canonicalize_scientific(
            {
                "selected_reference": e0_summary["selected_reference"],
                "dtype": resolved_e0["data"]["dtype"],
                "dealias": resolved_e0["target"]["dealias"],
                "calibration_sample_ids": resolved_e0["e0"][
                    "calibration_sample_ids"
                ],
            }
        ),
        "master_tensor": semantic_tensor_digest(master["values"]),
        "reference_convergence": canonical_csv(
            e0_dir / "reference_convergence.csv"
        ),
        "checks": {
            name: canonicalize_scientific(_json(e0_dir / name))
            for name in (
                "resampling_checks.json",
                "input_interface_checks.json",
                "model1_identity.json",
            )
        },
    }


def build_e1_scientific_record(
    e1_dir: Path, *, plot_dir: Path | None = None
) -> dict[str, Any]:
    """Build the semantic E1 portion after manifest verification."""
    e1_summary = _require_pass(e1_dir, "e1_summary.json")
    _verify_artifact_manifest(e1_dir)
    return {
        "summary": canonicalize_scientific(e1_summary),
        "tables": {name: canonical_csv(e1_dir / name) for name in E1_TABLES},
        "selected_models": semantic_model_digest(e1_dir / "selected_models.pt"),
        "plots": _plot_inventory(e1_dir if plot_dir is None else plot_dir),
    }


def build_e2_scientific_record(
    e2_dir: Path, *, plot_dir: Path | None = None
) -> dict[str, Any]:
    """Build the semantic E2 portion after contract and event verification."""
    e2_summary = _require_pass(e2_dir, "e2_summary.json")
    _verify_artifact_manifest(e2_dir)
    if e2_summary.get("test_evaluated") is not True:
        raise ValueError("E2 baseline must have test_evaluated=true")
    events = [record["event"] for record in _json(e2_dir / "event_log.json")["events"]]
    required_events = [
        "convergence_complete",
        "freeze_read_back",
        "first_test_state_solve",
        "first_test_metric",
    ]
    if events != required_events:
        raise ValueError(f"unexpected E2 event sequence: {events}")
    normalized_summary = json.loads(json.dumps(e2_summary))
    binding_check = normalized_summary.get("required_checks", {}).get(
        "test_rows_bound_to_frozen_plan"
    )
    if isinstance(binding_check, dict):
        binding_check.pop("value", None)
    return {
        "summary": canonicalize_scientific(normalized_summary),
        "json": {
            name: canonicalize_scientific(_json(e2_dir / name))
            for name in E2_JSON
        },
        "tables": {name: canonical_csv(e2_dir / name) for name in E2_TABLES},
        "selected_models": semantic_model_digest(e2_dir / "selected_models.pt"),
        "plots": _plot_inventory(e2_dir if plot_dir is None else plot_dir),
        "event_sequence": events,
    }


def build_phase1_scientific_baseline(
    e0_dir: Path,
    e1_dir: Path,
    e2_dir: Path,
    *,
    source_revision: str,
) -> dict[str, Any]:
    """Validate saved artifacts and build their semantic Phase 1 baseline."""
    if not source_revision.strip():
        raise ValueError("source_revision must not be empty")
    return {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "provenance": {
            "source_revision": source_revision,
            "canonical_float_significant_digits": SIGNIFICANT_DIGITS,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "platform": platform.platform(),
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
            "generation_policy": "validated saved artifacts only; no experiment execution",
            "excluded_fields": sorted(PROVENANCE_FIELDS),
        },
        "e0": build_e0_scientific_record(e0_dir),
        "e1": build_e1_scientific_record(e1_dir),
        "e2": build_e2_scientific_record(e2_dir),
    }


def write_phase1_scientific_baseline(
    baseline: Mapping[str, Any], output: Path, *, overwrite: bool
) -> None:
    """Write a baseline without silently replacing an existing expectation."""
    if output.exists() and not overwrite:
        raise FileExistsError(f"baseline already exists: {output}; pass --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_strict_json(output, baseline)
