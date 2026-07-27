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
from typing import Any, Mapping

import torch

from pol.runtime.io import file_sha256
from .scientific_comparison import (
    COMPARISON_POLICY_VERSION,
    DEFAULT_TOLERANCES,
    normalized_numeric_path,
)


from .protocols import (
    BASELINE_SCHEMA_VERSION,
    MATRIX_BASELINE_SCHEMA_VERSION,
)
GENERATOR_VERSION = "paper1-scientific-baseline-generator-v4"

# These fields encode execution location or bindings rather than scientific results.
PROVENANCE_FIELDS = frozenset(
    {
        "attempt_history_hash",
        "bindings",
        "binding_hash",
        "command",
        "cwd",
        "dataset_hash",
        "dataset_prerequisite_hash",
        "e0_prerequisite_hash",
        "environment",
        "feature_cache_key",
        "frozen_plan_hash",
        "git_commit",
        "git_commit_id",
        "git_dirty_status",
        "master_hash",
        "path",
        "requested_config",
        "sha256",
        "size_bytes",
        "state_cache_key",
        "tensor_hash",
        "tensor_hashes",
        "runtime_seconds",
        "selected_models_content_hash",
        "selection_record_hash",
        "state_key",
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
    "sweep_selected_results.csv": ("run_id", "case_name", "q"),
    "sweep_readout_diagnostics.csv": ("run_id", "case_name", "q"),
    "sweep_noise_summary.csv": ("run_id", "case_name", "q", "noise_level"),
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


def canonical_float(value: float) -> float:
    """Return a finite float without environment-specific decimal quantization."""
    if not math.isfinite(value):
        raise ValueError(f"non-finite float in scientific baseline: {value!r}")
    return float(value)


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
    """Describe tensor structure for cross-runtime comparison."""
    value = tensor.detach().cpu().contiguous()
    if value.is_complex():
        finite = bool(torch.isfinite(value.real).all() and torch.isfinite(value.imag).all())
    elif value.is_floating_point():
        finite = bool(torch.isfinite(value).all())
    else:
        finite = True
    if not finite:
        raise ValueError("non-finite tensor in scientific baseline")
    return {
        "shape": list(value.shape),
        "logical_dtype": str(value.dtype).removeprefix("torch."),
        "numel": value.numel(),
        "finite": True,
    }


def semantic_model_digest(path: Path) -> dict[str, Any]:
    """Describe model archive structure without cross-runtime weight digests."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    models = payload.get("models", payload)
    if not isinstance(models, Mapping):
        raise ValueError(f"model archive is not a mapping: {path}")

    def describe(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return semantic_tensor_digest(value)
        if isinstance(value, Mapping):
            described = {}
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
                child = describe(item)
                if child is not None:
                    described[str(key)] = child
            return described or None
        if isinstance(value, (list, tuple)):
            described = [describe(item) for item in value]
            return described if any(item is not None for item in described) else None
        # Hyperparameters and prediction metrics are already represented by the
        # validated summaries/tables.  The archive baseline intentionally gates
        # only model keys and tensor shape/dtype/finite structure.
        return None

    return {
        "archive_keys": sorted(str(key) for key in payload),
        "models": describe(models),
    }


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_config_sha256_for_baseline(path: Path) -> str:
    from pol.paper1.config import canonical_config_json, load_config_json

    return hashlib.sha256(
        canonical_config_json(load_config_json(path)).encode("utf-8")
    ).hexdigest()


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


def _safe_relative_path(raw: object) -> Path:
    relative = Path(str(raw))
    if (
        not str(raw)
        or relative.is_absolute()
        or ".." in relative.parts
        or relative == Path(".")
    ):
        raise ValueError(f"unsafe artifact path: {raw!r}")
    return relative


def _ensure_regular_path(root: Path, relative: Path) -> Path:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"unsafe artifact root: {root}")
    current = root
    for component in relative.parts[:-1]:
        current = current / component
        if current.is_symlink() or not current.is_dir():
            raise ValueError(f"unsafe artifact parent: {current}")
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"artifact is not a regular file: {path}")
    if path.resolve().parent != (root / relative.parent).resolve():
        raise ValueError(f"artifact escapes its root: {path}")
    return path


def _regular_file_set(root: Path) -> set[str]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"unsafe artifact root: {root}")
    files: set[str] = set()
    pending = [root]
    while pending:
        directory = pending.pop()
        for path in directory.iterdir():
            if path.is_symlink():
                raise ValueError(f"artifact tree contains symlink: {path}")
            if path.is_dir():
                pending.append(path)
            elif path.is_file():
                files.add(path.relative_to(root).as_posix())
            else:
                raise ValueError(f"artifact tree contains non-regular entry: {path}")
    return files


def _root_regular_file_set(root: Path) -> set[str]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"unsafe artifact root: {root}")
    files: set[str] = set()
    for path in root.iterdir():
        if path.is_symlink():
            raise ValueError(f"artifact root contains symlink: {path}")
        if path.is_file():
            files.add(path.name)
        elif not path.is_dir():
            raise ValueError(f"artifact root contains non-regular entry: {path}")
    return files


def _verify_artifact_manifest(
    directory: Path, *, expected: set[str]
) -> None:
    manifest_path = directory / "artifact_manifest.json"
    _ensure_regular_path(directory, Path("artifact_manifest.json"))
    raw = _json(manifest_path)
    records = raw if isinstance(raw, list) else raw.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError(f"invalid artifact manifest: {manifest_path}")
    paths: list[str] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"invalid artifact manifest record: {manifest_path}")
        relative = _safe_relative_path(record.get("relative_path"))
        name = relative.as_posix()
        paths.append(name)
        path = _ensure_regular_path(directory, relative)
        size = record.get("byte_size", record.get("size_bytes"))
        if path.stat().st_size != size or file_sha256(path) != record["sha256"]:
            raise ValueError(f"artifact manifest verification failed: {path}")
    if len(paths) != len(set(paths)):
        raise ValueError(f"duplicate artifact manifest record: {manifest_path}")
    expected_records = expected - {"artifact_manifest.json"}
    if set(paths) != expected_records:
        raise ValueError("artifact manifest record set differs from contract")
    actual = _root_regular_file_set(directory)
    if actual != expected:
        raise ValueError(
            "artifact tree differs from contract; "
            f"missing={sorted(expected-actual)}, extra={sorted(actual-expected)}"
        )


def _require_pass(directory: Path, summary_name: str) -> dict[str, Any]:
    summary = _json(directory / summary_name)
    if not isinstance(summary, dict) or summary.get("status") != "pass":
        raise ValueError(f"{directory / summary_name} is not a passing summary")
    return summary


def _plot_inventory(directory: Path) -> list[dict[str, str]]:
    _ensure_regular_path(directory, Path("plot_manifest.json"))
    manifest = _json(directory / "plot_manifest.json")
    if manifest.get("status") == "skipped":
        return []
    if manifest.get("status") != "pass":
        raise ValueError(f"plot manifest is not pass: {directory}")
    inventory = []
    records = manifest.get("plots", manifest.get("outputs", []))
    declared: list[str] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("plot manifest record must be an object")
        relative_path = _safe_relative_path(record.get("relative_path"))
        relative = relative_path.as_posix()
        declared.append(relative)
        path = _ensure_regular_path(directory, relative_path)
        if path.stat().st_size <= 0:
            raise ValueError(f"plot is missing or empty: {path}")
        inventory.append(
            {
                "logical_name": record.get("kind", Path(relative).stem),
                "format": record.get("format", path.suffix.lstrip(".").lower()),
            }
        )
    if len(declared) != len(set(declared)):
        raise ValueError("duplicate plot manifest record")
    actual = _regular_file_set(directory)
    if "outputs" in manifest:
        if actual != set(declared) | {"plot_manifest.json"}:
            raise ValueError("plot manifest and exact file tree differ")
    else:
        image_actual = {
            name
            for name in actual
            if Path(name).suffix.lower() in {".png", ".pdf", ".svg"}
        }
        if image_actual != set(declared):
            raise ValueError("inline plot manifest and image files differ")
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
    from pol.paper1.datasets import tensor_hash

    values = master.get("values")
    if not isinstance(values, torch.Tensor):
        raise ValueError("E0 master archive values tensor is missing")
    actual_hash = tensor_hash(values)
    if (
        actual_hash != master.get("metadata", {}).get("tensor_hash")
        or actual_hash != master_manifest.get("tensor_hash")
    ):
        raise ValueError("E0 master archive/manifest tensor hash mismatch")
    if not torch.isfinite(values).all():
        raise ValueError("E0 master archive contains non-finite values")
    resolved_e0 = _json(e0_dir / "resolved_config.json")
    from pol.paper1.config import canonical_config_json, load_config_json

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
        "master_tensor": semantic_tensor_digest(values),
        "accepted_config_sha256": hashlib.sha256(
            canonical_config_json(
                load_config_json(e0_dir / "accepted_production_config.json")
            ).encode("utf-8")
        ).hexdigest(),
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
    e1_dir: Path,
    *,
    plot_dir: Path | None = None,
    prerequisite_e0_dir: Path | None = None,
) -> dict[str, Any]:
    """Build the semantic E1 portion after manifest verification."""
    e1_summary = _require_pass(e1_dir, "e1_summary.json")
    from pol.paper1.config import canonical_config_json, load_config_json
    from pol.paper1.e1_qa import (
        expected_artifacts,
        read_and_scan_csv,
        scan_json_file,
        scan_models,
        validate_table_keys,
        validate_artifact_set,
        validate_plots,
        validate_saved_numeric_artifacts,
        verify_artifact_manifest,
    )

    config = load_config_json(e1_dir / "resolved_config.json")
    plot_manifest = _json(e1_dir / "plot_manifest.json")
    skip_plots = plot_manifest.get("status") == "skipped"
    plot_names = validate_plots(e1_dir, skip_plots=skip_plots)
    expected = expected_artifacts(plot_names)
    recorded_prerequisite = scan_json_file(e1_dir / "e0_prerequisite.json")
    recorded_e0 = Path(str(recorded_prerequisite.get("e0_dir", "")))
    e0_source = prerequisite_e0_dir
    if e0_source is None and recorded_e0.is_dir():
        e0_source = recorded_e0
    if e0_source is None and (e1_dir.parent / "e0").is_dir():
        e0_source = e1_dir.parent / "e0"
    if e0_source is None:
        raise ValueError("E1 prerequisite E0 directory is unavailable")
    if e0_source.resolve() == recorded_e0.resolve() and recorded_e0.is_dir():
        validate_saved_numeric_artifacts(e1_dir, config)
    else:
        # Official QA's final binding check follows the historical absolute
        # path.  A moved baseline tree instead validates the same saved tables,
        # models, JSON, and hashes against its explicitly supplied E0 copy.
        tables = {
            name: read_and_scan_csv(e1_dir / name)
            for name in (
                "ridge_selection.csv",
                "selected_results.csv",
                "readout_diagnostics.csv",
                "mode_comparison.csv",
                "noise_results.csv",
                "noise_summary.csv",
            )
        }
        validate_table_keys(tables, config)
        for path in e1_dir.glob("*.json"):
            if path.name != "artifact_manifest.json":
                scan_json_file(path)
        scan_models(e1_dir / "selected_models.pt", config)
        data_manifest = scan_json_file(e1_dir / "data_manifest.json")
        if data_manifest.get("resolved_config_hash") != file_sha256(
            e1_dir / "resolved_config.json"
        ):
            raise ValueError("data manifest resolved_config_hash mismatch")
        if data_manifest.get("e0_prerequisite_hash") != file_sha256(
            e1_dir / "e0_prerequisite.json"
        ):
            raise ValueError("data manifest e0_prerequisite_hash mismatch")
        for field, filename in (
            ("master_file_sha256", "master_initial_conditions.pt"),
            ("master_manifest_file_sha256", "master_manifest.json"),
        ):
            if data_manifest.get(field) != file_sha256(e0_source / filename):
                raise ValueError(f"data manifest {field} mismatch")
    verify_artifact_manifest(e1_dir, expected)
    validate_artifact_set(e1_dir, expected)
    _verify_artifact_manifest(e1_dir, expected=expected)
    return {
        "summary": canonicalize_scientific(e1_summary),
        "resolved_config_sha256": hashlib.sha256(
            canonical_config_json(config).encode("utf-8")
        ).hexdigest(),
        "tables": {name: canonical_csv(e1_dir / name) for name in E1_TABLES},
        "selected_models": semantic_model_digest(e1_dir / "selected_models.pt"),
        "plots": _plot_inventory(e1_dir if plot_dir is None else plot_dir),
    }


def build_e2_scientific_record(
    e2_dir: Path, *, plot_dir: Path | None = None
) -> dict[str, Any]:
    """Build the semantic E2 portion after contract and event verification."""
    e2_summary = _require_pass(e2_dir, "e2_summary.json")
    from pol.paper1.e2_qa import (
        expected_artifacts,
        validate_resume_output,
    )

    if not validate_resume_output(e2_dir):
        raise ValueError("E2 published output failed official resume validation")
    plot_manifest = _json(e2_dir / "plot_manifest.json")
    expected = expected_artifacts(
        status="pass",
        skip_plots=plot_manifest.get("status") == "skipped",
        test_evaluated=True,
    )
    _verify_artifact_manifest(e2_dir, expected=expected)
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
    from pol.paper1.config import canonical_config_json, load_config_json

    return {
        "summary": canonicalize_scientific(normalized_summary),
        "resolved_config_sha256": hashlib.sha256(
            canonical_config_json(
                load_config_json(e2_dir / "resolved_config.json")
            ).encode("utf-8")
        ).hexdigest(),
        "json": {
            name: canonicalize_scientific(_json(e2_dir / name))
            for name in E2_JSON
        },
        "tables": {name: canonical_csv(e2_dir / name) for name in E2_TABLES},
        "selected_models": semantic_model_digest(e2_dir / "selected_models.pt"),
        "plots": _plot_inventory(e2_dir if plot_dir is None else plot_dir),
        "event_sequence": events,
    }


_EXACT_NUMERIC_FIELDS = frozenset(
    {
        "T",
        "T_tilde",
        "bias_scale",
        "dt",
        "fine_dt",
        "noise_level",
        "nu",
        "nu_tilde",
        "parameter_value",
        "selected_T",
        "selected_nu",
        "selected_zeta",
        "zeta",
    }
)
_ROUND_OFF_TOKENS = (
    "imaginary",
    "off_diagonal",
    "roundoff",
    "hermitian",
    "symmetry_residual",
    "zero_mode",
)
_SELECTION_TOKENS = (
    "selection",
    "validation",
    "score",
    "objective",
    "best",
)
_FLOAT32_TOKEN = re.compile(r"(?:^|[._\-/])float32(?:$|[._\-/])")
_DTYPE_FIELDS = frozenset({"dtype", "logical_dtype", "sim_dtype", "storage_dtype"})
_STOCHASTIC_TOKENS = (
    "noise_summary",
    "model3_test_aggregate",
)
_CONDITION_SENSITIVE_TOKENS = (
    "effective_response_matrix",
    "feature_covariance_eigenvalues",
    "learned_effective_diagonal",
    "mode_comparison.csv",
    "regularized_condition_number",
)


def _numeric_category(path: str, *, dtype_context: str | None = None) -> str:
    """Classify one numeric field using its semantic path and dtype context.

    A dtype is context, not a scientific value: JSON objects and tensor
    descriptors commonly declare it on a sibling field, while E0 check names
    encode it in the table/JSON path.  Supporting both forms avoids treating
    every path containing an incidental ``32`` as float32.
    """
    field = re.split(r"[.\[]", path)[-1].rstrip("]")
    if field in _EXACT_NUMERIC_FIELDS:
        return "exact_numeric"
    lowered = path.lower()
    float32 = (
        dtype_context is not None
        and dtype_context.lower().removeprefix("torch.") == "float32"
    ) or _FLOAT32_TOKEN.search(lowered) is not None
    if any(token in lowered for token in _ROUND_OFF_TOKENS):
        return "roundoff_float32" if float32 else "roundoff_float64"
    if any(token in lowered for token in _SELECTION_TOKENS):
        return "selection_metric"
    if any(token in lowered for token in _STOCHASTIC_TOKENS):
        return "stochastic_aggregate"
    if any(token in lowered for token in _CONDITION_SENSITIVE_TOKENS):
        return "condition_sensitive"
    return "scientific_float32" if float32 else "scientific_float64"


def _numeric_paths(
    value: Any,
    path: str = "$",
    *,
    dtype_context: str | None = None,
) -> dict[str, str]:
    result: dict[str, str] = {}
    if isinstance(value, float):
        result[path] = _numeric_category(path, dtype_context=dtype_context)
    elif isinstance(value, dict):
        declared_dtype = next(
            (
                item
                for key, item in value.items()
                if str(key) in _DTYPE_FIELDS and isinstance(item, str)
            ),
            dtype_context,
        )
        for key, item in sorted(value.items()):
            result.update(
                _numeric_paths(
                    item,
                    f"{path}.{key}",
                    dtype_context=declared_dtype,
                )
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            result.update(
                _numeric_paths(
                    item,
                    f"{path}[{index}]",
                    dtype_context=dtype_context,
                )
            )
    return result


def _comparison_policy(record: Mapping[str, Any]) -> dict[str, Any]:
    compact_paths: dict[str, str] = {}
    for path, category in _numeric_paths(record).items():
        normalized = normalized_numeric_path(path)
        previous = compact_paths.get(normalized)
        if previous is not None and previous != category:
            raise ValueError(
                f"numeric policy category conflict at {normalized}: "
                f"{previous} != {category}"
            )
        compact_paths[normalized] = category
    return {
        "version": COMPARISON_POLICY_VERSION,
        "description": {
            "exact": "schema, discrete identity, dimensions, row keys, and selections",
            "ignored": sorted(PROVENANCE_FIELDS),
            "scientific_float64": "cross-runtime BLAS/FFT scientific metrics",
            "scientific_float32": "dtype-scale float32 scientific metrics",
            "roundoff_float64": "absolute dtype-scale diagnostics near theoretical zero",
            "roundoff_float32": "absolute float32 diagnostics near theoretical zero",
            "stochastic_aggregate": "stable aggregate of version-sensitive random draws",
            "condition_sensitive": "ill-conditioned raw diagnostic with exact discrete identities",
            "selection_metric": "tight metrics supporting exact selected identities",
        },
        "numeric_paths": compact_paths,
        "tolerances": {
            name: {"rtol": value.rtol, "atol": value.atol}
            for name, value in DEFAULT_TOLERANCES.items()
        },
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
    record = {
        "e0": build_e0_scientific_record(e0_dir),
        "e1": build_e1_scientific_record(
            e1_dir,
            prerequisite_e0_dir=(
                e1_dir.parent / "e0"
                if (e1_dir.parent / "e0").is_dir()
                else None
            ),
        ),
        "e2": build_e2_scientific_record(e2_dir),
    }
    return {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "comparison_policy": _comparison_policy(record),
        "provenance": {
            "source_revision": source_revision,
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
        **record,
    }


def build_e1_matrix_scientific_baseline(
    run_dir: Path, *, source_revision: str
) -> dict[str, Any]:
    """Build a portable E1 matrix baseline from one passing saved run."""
    if not source_revision.strip():
        raise ValueError("source_revision must not be empty")
    manifest = _json(run_dir / "matrix_manifest.json")
    if (
        not isinstance(manifest, dict)
        or manifest.get("status") != "pass"
        or manifest.get("compute_status") != "pass"
    ):
        raise ValueError("matrix baseline source is not a passing compute run")
    plan = _json(run_dir / "matrix_plan.json")
    from pol.paper1.matrix_plugins.e1_resolution import E1ResolutionPlugin

    aggregate_dir = run_dir / "aggregate"
    aggregate_names = set(E1ResolutionPlugin().aggregate_artifact_names())
    if _root_regular_file_set(aggregate_dir) != aggregate_names:
        raise ValueError("matrix aggregate artifact set differs from contract")
    E1ResolutionPlugin().validate_aggregate(aggregate_dir)
    cells: dict[str, Any] = {}
    for cell in manifest.get("cells", []):
        if cell.get("status") != "pass":
            raise ValueError("matrix baseline contains a non-passing cell")
        output = run_dir / "cells" / str(cell["cell_id"])
        config_path = run_dir / "generated_configs" / f"{cell['cell_id']}.json"
        if _canonical_config_sha256_for_baseline(config_path) != cell["config_sha256"]:
            raise ValueError(f"matrix cell config identity mismatch: {cell['cell_id']}")
        record = build_e1_scientific_record(
            output, prerequisite_e0_dir=run_dir / "e0"
        )
        record["summary"] = {
            key: value
            for key, value in record["summary"].items()
            if key
            not in {
                "runtime_seconds",
                "selection_record_hash",
                "frozen_plan_hash",
            }
        }
        cells[cell["human_slug"]] = record
    aggregate = {
        name: canonical_csv(aggregate_dir / name)
        for name in (
            "sweep_selected_results.csv",
            "sweep_readout_diagnostics.csv",
            "sweep_noise_summary.csv",
        )
    }
    figures = run_dir / "figures/paper1.e1.resolution_sweep.v1"
    record = {
        "plan": canonicalize_scientific(
            {
                "raw_run_counts": plan["raw_run_counts"],
                "unique_valid_cells": plan["unique_valid_cells"],
                "contains_n_tar_gt_J": plan["contains_n_tar_gt_J"],
                "contains_n_tar_lt_J": plan["contains_n_tar_lt_J"],
                "full_observation_cells": plan["full_observation_cells"],
                "cells": [
                    {
                        key: cell[key]
                        for key in (
                            "run_index",
                            "cell_id",
                            "config_sha256",
                            "human_slug",
                            "experiment_memberships",
                            "metadata",
                        )
                    }
                    for cell in plan["cells"]
                ],
            }
        ),
        "cells": cells,
        "aggregate": aggregate,
        "plots": _plot_inventory(figures),
    }
    return {
        "schema_version": MATRIX_BASELINE_SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "comparison_policy": _comparison_policy(record),
        "provenance": {
            "source_revision": source_revision,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "platform": platform.platform(),
            "generation_policy": "validated saved passing matrix artifacts only",
            "excluded_fields": sorted(PROVENANCE_FIELDS),
        },
        "record": record,
    }


def write_phase1_scientific_baseline(
    baseline: Mapping[str, Any], output: Path, *, overwrite: bool
) -> None:
    """Write a baseline without silently replacing an existing expectation."""
    if output.exists() and not overwrite:
        raise FileExistsError(f"baseline already exists: {output}; pass --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            baseline,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
