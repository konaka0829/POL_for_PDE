from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import torch

from .config import Paper1Config
from .datasets import tensor_hash
from .schemas import stable_hash_json


BASE_ARTIFACTS = {
    "ridge_selection.csv", "selected_results.csv", "readout_diagnostics.csv",
    "mode_comparison.csv", "noise_results.csv", "noise_summary.csv",
    "selected_models.pt", "e0_prerequisite.json", "data_manifest.json",
    "resolved_config.json", "environment.json", "plot_manifest.json",
    "failed_runs.json", "e1_summary.json", "artifact_manifest.json",
}
CSV_FILES = {
    "ridge_selection.csv", "selected_results.csv", "readout_diagnostics.csv",
    "mode_comparison.csv", "noise_results.csv", "noise_summary.csv",
}
JSON_COLUMNS = {
    "effective_response_matrix", "theoretical_multiplier_vector",
    "learned_effective_diagonal", "feature_covariance_eigenvalues",
}
STRING_COLUMNS = {"case_name", "regime", "component", "theory_definition"}
BOOLEAN_COLUMNS = {"selected", "identifiable", "regularized_condition_number_is_infinite"}
NULLABLE_NUMERIC_COLUMNS = {
    "regularized_condition_number", "nonzero_spectrum_pseudo_condition_number",
    "mean_identifiable_diagonal_relative_error",
    "max_identifiable_diagonal_relative_error",
    "identifiable_off_diagonal_relative_norm",
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg", ".pdf"}
REQUIRED_COLUMNS = {
    "ridge_selection.csv": {"case_name", "regime", "q", "zeta", "train_coefficient_mse", "validation_coefficient_mse", "selected"},
    "selected_results.csv": {"case_name", "regime", "q", "selected_zeta", "full_reference_field_relative_l2_mean", "output_representation_floor_mean", "field_error_to_representation_floor_ratio"},
    "readout_diagnostics.csv": {"case_name", "regime", "q", "selected_zeta", "identifiable_nonconstant_count", "nonconstant_count", "identifiable_nonconstant_fraction", "mean_identifiable_diagonal_relative_error", "max_identifiable_diagonal_relative_error", "identifiable_off_diagonal_relative_norm", "learned_operator_norm", "ideal_operator_norm"} | JSON_COLUMNS,
    "mode_comparison.csv": {"case_name", "regime", "q", "coefficient_index", "wavenumber", "component", "training_variance", "identifiable", "theoretical_multiplier", "learned_effective_diagonal", "absolute_error", "relative_error"},
    "noise_results.csv": {"case_name", "regime", "q", "noise_level", "repeat", "output_perturbation_rms", "theoretical_output_perturbation_rms", "field_relative_l2_mean"},
    "noise_summary.csv": {"case_name", "regime", "q", "noise_level", "repeats", "output_perturbation_rms_mean", "theoretical_output_perturbation_rms", "field_relative_l2_mean"},
}


def _assert_finite_tree(value: Any, location: str = "root") -> None:
    if isinstance(value, str):
        if value.strip().lower() in {"nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}:
            raise ValueError(f"string-encoded non-finite value at {location}")
        return
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"non-finite number at {location}")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_finite_tree(child, f"{location}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_finite_tree(child, f"{location}[{index}]")
        return
    raise ValueError(f"unsupported value at {location}: {type(value).__name__}")


def scan_json_file(path: Path) -> Any:
    def reject_constant(token: str) -> None:
        raise ValueError(f"non-finite JSON token {token} in {path.name}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    _assert_finite_tree(value, path.name)
    return value


def read_and_scan_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path.name}")
        missing_columns = REQUIRED_COLUMNS.get(path.name, set()) - set(reader.fieldnames)
        if missing_columns:
            raise ValueError(f"CSV missing required columns {path.name}: {sorted(missing_columns)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV has no data rows: {path.name}")
    for row_number, row in enumerate(rows, start=2):
        for column, raw in row.items():
            if raw is None:
                raise ValueError(f"missing CSV value {path.name}:{row_number}:{column}")
            if column in STRING_COLUMNS:
                if not raw:
                    raise ValueError(f"empty key/string {path.name}:{row_number}:{column}")
            elif column in BOOLEAN_COLUMNS:
                if raw not in {"True", "False", "true", "false", "0", "1"}:
                    raise ValueError(f"invalid boolean {path.name}:{row_number}:{column}")
            elif column in JSON_COLUMNS:
                parsed = json.loads(raw, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
                _assert_finite_tree(parsed, f"{path.name}:{row_number}:{column}")
            elif raw == "" and column in NULLABLE_NUMERIC_COLUMNS:
                continue
            else:
                try:
                    number = float(raw)
                except ValueError as exc:
                    raise ValueError(f"invalid numeric CSV value {path.name}:{row_number}:{column}") from exc
                if not math.isfinite(number):
                    raise ValueError(f"non-finite CSV value {path.name}:{row_number}:{column}")
    return rows


def _canonical_float(value: float | str) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite float key")
    return format(number, ".17g")


def _unique_exact(actual: Iterable[tuple[Any, ...]], expected: set[tuple[Any, ...]], label: str) -> None:
    keys = list(actual)
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate {label} key")
    if set(keys) != expected:
        missing = sorted(expected - set(keys), key=str)
        extra = sorted(set(keys) - expected, key=str)
        raise ValueError(f"{label} key mismatch; missing={missing[:1]}, extra={extra[:1]}")


def validate_table_keys(tables: dict[str, list[dict[str, str]]], config: Paper1Config) -> None:
    assert config.e1 is not None
    cases = [case.name for case in config.e1.surrogate_cases]
    qs = list(config.e1.output_dims)
    case_q = {(case, q) for case in cases for q in qs}
    _unique_exact(
        ((row["case_name"], int(row["q"])) for row in tables["selected_results.csv"]),
        case_q, "selected_results",
    )
    _unique_exact(
        ((row["case_name"], int(row["q"])) for row in tables["readout_diagnostics.csv"]),
        case_q, "readout_diagnostics",
    )
    expected_ridge = {
        (case, q, _canonical_float(zeta))
        for case in cases for q in qs for zeta in config.e1.ridge_zetas
    }
    ridge_keys = [
        (row["case_name"], int(row["q"]), _canonical_float(row["zeta"]))
        for row in tables["ridge_selection.csv"]
    ]
    _unique_exact(ridge_keys, expected_ridge, "ridge_selection")
    for case, q in case_q:
        selected = [
            row for row in tables["ridge_selection.csv"]
            if row["case_name"] == case and int(row["q"]) == q
            and row["selected"].lower() in {"true", "1"}
        ]
        if len(selected) != 1:
            raise ValueError(f"expected exactly one selected ridge row for {(case, q)}")
    expected_modes = {(case, q, index) for case in cases for q in qs for index in range(q)}
    _unique_exact(
        ((row["case_name"], int(row["q"]), int(row["coefficient_index"]))
         for row in tables["mode_comparison.csv"]),
        expected_modes, "mode_comparison",
    )
    expected_noise = {
        (case, q, _canonical_float(level), repeat)
        for case in cases for q in qs for level in config.e1.noise_levels
        for repeat in range(config.e1.noise_repeats)
    }
    _unique_exact(
        ((row["case_name"], int(row["q"]), _canonical_float(row["noise_level"]), int(row["repeat"]))
         for row in tables["noise_results.csv"]),
        expected_noise, "noise_results",
    )
    expected_summary = {
        (case, q, _canonical_float(level))
        for case in cases for q in qs for level in config.e1.noise_levels
    }
    _unique_exact(
        ((row["case_name"], int(row["q"]), _canonical_float(row["noise_level"]))
         for row in tables["noise_summary.csv"]),
        expected_summary, "noise_summary",
    )
    if any(int(row["repeats"]) != config.e1.noise_repeats for row in tables["noise_summary.csv"]):
        raise ValueError("noise_summary repeats mismatch")


def model_content_hash(payload: Any) -> str:
    def canonical(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return {"tensor_hash": tensor_hash(value), "shape": list(value.shape), "dtype": str(value.dtype)}
        if isinstance(value, dict):
            return {str(key): canonical(value[key]) for key in sorted(value)}
        if isinstance(value, (list, tuple)):
            return [canonical(item) for item in value]
        _assert_finite_tree(value)
        return value
    return stable_hash_json(canonical(payload))


def scan_models(path: Path, config: Paper1Config) -> str:
    assert config.e1 is not None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), dict):
        raise ValueError("selected_models.pt has invalid structure")
    expected = {f"{case.name}/q{q}" for case in config.e1.surrogate_cases for q in config.e1.output_dims}
    if set(payload["models"]) != expected:
        raise ValueError("selected model key set mismatch")
    dtype = config.data.torch_dtype()
    for key, model in payload["models"].items():
        q = int(key.rsplit("q", 1)[1])
        if tuple(model["W"].shape) != (q, config.spatial.observation_dim):
            raise ValueError(f"model W shape mismatch: {key}")
        if tuple(model["b"].shape) != (q,):
            raise ValueError(f"model b shape mismatch: {key}")
        for name in ("W", "b"):
            tensor = model[name]
            if tensor.dtype != dtype or not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"model tensor invalid: {key}/{name}")
        _assert_finite_tree({k: v for k, v in model.items() if not isinstance(v, torch.Tensor)}, key)
    return model_content_hash(payload)


def _safe_relative_path(raw: str) -> Path:
    pure = PurePosixPath(raw)
    if pure.is_absolute() or ".." in pure.parts or len(pure.parts) != 1:
        raise ValueError(f"unsafe artifact path: {raw}")
    return Path(raw)


def validate_plots(output_dir: Path, *, skip_plots: bool) -> set[str]:
    manifest = scan_json_file(output_dir / "plot_manifest.json")
    plots = manifest.get("plots")
    if not isinstance(plots, list):
        raise ValueError("plot manifest plots must be a list")
    image_files = {path.name for path in output_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES}
    if skip_plots:
        if manifest.get("status") != "skipped" or plots or image_files:
            raise ValueError("skip-plots manifest/files are inconsistent")
        return set()
    if manifest.get("status") != "pass":
        raise ValueError("plot manifest is not pass")
    listed: set[str] = set()
    for record in plots:
        if not isinstance(record, dict) or record.get("status") != "created":
            raise ValueError("plot manifest record status is not created")
        relative = _safe_relative_path(str(record.get("relative_path", "")))
        if relative.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"unsupported plot extension: {relative}")
        path = output_dir / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"plot missing or empty: {relative}")
        if relative.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            try:
                import matplotlib.image as mpimg
                image = mpimg.imread(path)
            except Exception as exc:
                raise ValueError(f"plot is not a readable image: {relative}") from exc
            if image.size == 0:
                raise ValueError(f"plot image has no pixels: {relative}")
        listed.add(relative.name)
    if len(listed) != len(plots) or listed != image_files:
        raise ValueError("plot manifest and image files differ")
    return listed


def expected_artifacts(plot_names: set[str]) -> set[str]:
    return BASE_ARTIFACTS | plot_names


def validate_artifact_set(output_dir: Path, expected: set[str]) -> None:
    actual = {path.name for path in output_dir.iterdir() if path.is_file()}
    if actual != expected:
        raise ValueError(f"artifact set mismatch; missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")


def artifact_records(output_dir: Path, expected: set[str]) -> list[dict[str, Any]]:
    validate_artifact_set(output_dir, expected - {"artifact_manifest.json"})
    records = []
    for name in sorted(expected - {"artifact_manifest.json"}):
        data = (output_dir / name).read_bytes()
        records.append({"relative_path": name, "byte_size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    return records


def verify_artifact_manifest(output_dir: Path, expected: set[str]) -> None:
    records = scan_json_file(output_dir / "artifact_manifest.json")
    if not isinstance(records, list):
        raise ValueError("artifact manifest must be a list")
    paths = [str(record.get("relative_path", "")) for record in records]
    for raw in paths:
        _safe_relative_path(raw)
    if len(paths) != len(set(paths)) or set(paths) != expected - {"artifact_manifest.json"}:
        raise ValueError("artifact manifest path set mismatch")
    for record in records:
        path = output_dir / str(record["relative_path"])
        data = path.read_bytes()
        if record.get("byte_size") != len(data) or record.get("sha256") != hashlib.sha256(data).hexdigest():
            raise ValueError(f"artifact manifest hash/size mismatch: {path.name}")


def validate_saved_numeric_artifacts(output_dir: Path, config: Paper1Config) -> dict[str, Any]:
    tables = {name: read_and_scan_csv(output_dir / name) for name in CSV_FILES}
    validate_table_keys(tables, config)
    for path in output_dir.glob("*.json"):
        if path.name != "artifact_manifest.json":
            scan_json_file(path)
    model_hash = scan_models(output_dir / "selected_models.pt", config)
    manifest = scan_json_file(output_dir / "data_manifest.json")
    prerequisite = scan_json_file(output_dir / "e0_prerequisite.json")
    resolved_hash = hashlib.sha256((output_dir / "resolved_config.json").read_bytes()).hexdigest()
    prerequisite_hash = hashlib.sha256((output_dir / "e0_prerequisite.json").read_bytes()).hexdigest()
    if manifest.get("resolved_config_hash") != resolved_hash:
        raise ValueError("data manifest resolved_config_hash mismatch")
    if manifest.get("e0_prerequisite_hash") != prerequisite_hash:
        raise ValueError("data manifest e0_prerequisite_hash mismatch")
    e0_dir = Path(str(prerequisite.get("e0_dir", "")))
    for field, filename in (
        ("master_file_sha256", "master_initial_conditions.pt"),
        ("master_manifest_file_sha256", "master_manifest.json"),
    ):
        path = e0_dir / filename
        if not path.is_file() or manifest.get(field) != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError(f"data manifest {field} mismatch")
    if manifest.get("master_tensor_hash") != prerequisite.get("master_tensor_hash"):
        raise ValueError("data manifest master_tensor_hash mismatch")
    recorded_model_hash = manifest.get("selected_models_content_hash")
    if recorded_model_hash is not None and recorded_model_hash != model_hash:
        raise ValueError("data manifest selected model content hash mismatch")
    return {"tables": tables, "model_content_hash": model_hash}


def scientific_acceptance_checks(
    tables: dict[str, list[dict[str, str]]], config: Paper1Config
) -> dict[str, dict[str, Any]]:
    assert config.e1 is not None
    diagnostics = tables["readout_diagnostics.csv"]
    selected = tables["selected_results.csv"]

    def threshold_check(name: str, column: str, threshold: float, *, lower: bool) -> dict[str, Any]:
        candidates = []
        for row in diagnostics:
            raw = row[column]
            if raw == "":
                candidates.append((-math.inf if lower else math.inf, row))
            else:
                candidates.append((float(raw), row))
        value, worst = (min(candidates, key=lambda item: item[0]) if lower else max(candidates, key=lambda item: item[0]))
        passed = value >= threshold if lower else value <= threshold
        return {
            "status": "pass" if passed else "fail", "value": value if math.isfinite(value) else None,
            "threshold": threshold, "worst_case_name": worst["case_name"], "worst_q": int(worst["q"]),
            "metric_definition": name,
        }

    checks = {
        "identifiable_nonconstant_fraction": threshold_check(
            "fraction of nonconstant real-Fourier coefficients above the training-variance floor",
            "identifiable_nonconstant_fraction", config.e1.min_identifiable_nonconstant_fraction,
            lower=True,
        ),
        "max_identifiable_diagonal_relative_error": threshold_check(
            "maximum relative learned-vs-theory diagonal error over identifiable nonconstant coefficients",
            "max_identifiable_diagonal_relative_error",
            config.e1.max_identifiable_diagonal_relative_error, lower=False,
        ),
        "identifiable_off_diagonal_relative_norm": threshold_check(
            "Frobenius off-diagonal leakage on identifiable nonconstant input columns divided by theory norm",
            "identifiable_off_diagonal_relative_norm",
            config.e1.max_identifiable_off_diagonal_relative_norm, lower=False,
        ),
    }
    floor_candidates = [
        (float(row["field_error_to_representation_floor_ratio"]), row) for row in selected
    ]
    floor_value, floor_worst = max(floor_candidates, key=lambda item: item[0])
    checks["field_error_to_representation_floor_ratio"] = {
        "status": "pass" if floor_value <= config.e1.max_field_error_to_representation_floor_ratio else "fail",
        "value": floor_value,
        "threshold": config.e1.max_field_error_to_representation_floor_ratio,
        "worst_case_name": floor_worst["case_name"], "worst_q": int(floor_worst["q"]),
        "metric_definition": "mean clean full-field relative error / max(mean representation floor, eps)",
    }
    by_q: dict[int, dict[str, dict[str, str]]] = {}
    for row in diagnostics:
        by_q.setdefault(int(row["q"]), {})[row["regime"]] = row
    direction_failures = []
    for q, regimes in sorted(by_q.items()):
        if set(regimes) != {"stable", "unstable"}:
            direction_failures.append((q, "missing regime"))
            continue
        stable, unstable = regimes["stable"], regimes["unstable"]
        if float(stable["ideal_operator_norm"]) > 1.0 + config.e1.operator_norm_atol:
            direction_failures.append((q, "stable theory norm > 1"))
        if float(unstable["ideal_operator_norm"]) <= 1.0 + config.e1.operator_norm_atol:
            direction_failures.append((q, "unstable theory norm <= 1"))
        if float(unstable["learned_operator_norm"]) <= float(stable["learned_operator_norm"]):
            direction_failures.append((q, "unstable learned norm <= stable learned norm"))
    unstable_rows = sorted(
        (row for row in diagnostics if row["regime"] == "unstable"), key=lambda row: int(row["q"])
    )
    for previous, current in zip(unstable_rows, unstable_rows[1:]):
        if float(current["learned_operator_norm"]) < float(previous["learned_operator_norm"]) * (1.0 - config.e1.operator_norm_monotonic_rtol):
            direction_failures.append((int(current["q"]), "unstable learned norm decreased"))
    checks["stable_unstable_operator_direction"] = {
        "status": "fail" if direction_failures else "pass",
        "value": direction_failures or "all q satisfy direction and monotonicity",
        "threshold": {"atol": config.e1.operator_norm_atol, "monotonic_rtol": config.e1.operator_norm_monotonic_rtol},
        "worst_case_name": "unstable" if direction_failures else None,
        "worst_q": direction_failures[0][0] if direction_failures else None,
        "metric_definition": "stable theory <=1; unstable theory >1; unstable learned > stable and nondecreasing in q",
    }
    return checks
