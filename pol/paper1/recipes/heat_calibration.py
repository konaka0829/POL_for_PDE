"""Import-safe artifact orchestration for Paper 1 E1."""
from __future__ import annotations

import json
import platform
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pol.runtime.provenance import git_output
from pol.runtime.io import atomic_torch_save, file_sha256, write_csv as atomic_write_csv
from pol.runtime.io import write_strict_json
from pol.runtime.recipe import (
    RecipeInvocation,
    RecipeResult,
    RecipeUsageError,
    validate_recursive_delete_target,
)

TABLE_NAMES = (
    "ridge_selection",
    "selected_results",
    "readout_diagnostics",
    "mode_comparison",
    "noise_results",
    "noise_summary",
)


def _load_science_dependencies() -> None:
    """Load Paper 1 scientific modules only when the recipe is invoked."""
    global E1_SCHEMA_VERSION, artifact_records, expected_artifacts
    global load_config_json, run_e1, save_config_json
    global scientific_acceptance_checks, validate_artifact_set
    global validate_e0_prerequisite, validate_plots
    global validate_saved_numeric_artifacts, verify_artifact_manifest

    from ..config import load_config_json, save_config_json
    from ..e1 import E1_SCHEMA_VERSION, run_e1, validate_e0_prerequisite
    from ..e1_qa import (
        artifact_records,
        expected_artifacts,
        scientific_acceptance_checks,
        validate_artifact_set,
        validate_plots,
        validate_saved_numeric_artifacts,
        verify_artifact_manifest,
    )


def write_json(path: Path, value: Any) -> None:
    write_strict_json(path, value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path.name}")
    atomic_write_csv(path, rows, fieldnames=list(rows[0]))


def check(
    status: bool, value: Any, threshold: Any, message: str
) -> dict[str, Any]:
    return {
        "status": "pass" if status else "fail",
        "value": value,
        "threshold": threshold,
        "message": message,
    }


def prepare_output(
    output_dir: Path,
    overwrite: bool,
    *,
    invocation: RecipeInvocation,
    protected_paths: tuple[Path, ...],
) -> None:
    if output_dir.is_symlink():
        raise RecipeUsageError(
            f"output path must not be a symlink: {output_dir}"
        )
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise RecipeUsageError(f"{output_dir} is nonempty; pass --overwrite")
    if overwrite and output_dir.exists():
        validate_recursive_delete_target(
            output_dir,
            repo_root=invocation.repo_root,
            protected_paths=protected_paths,
        )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def runtime_environment(
    config_path: Path,
    e0_dir: Path,
    invocation: RecipeInvocation,
) -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "torch_thread_count": torch.get_num_threads(),
        "git_commit_id": git_output(
            invocation.repo_root, ["rev-parse", "HEAD"]
        ),
        "git_dirty_status": git_output(
            invocation.repo_root, ["status", "--porcelain"]
        ),
        "command": list(invocation.command),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "requested_config": str(config_path),
        "e0_dir": str(e0_dir),
    }


def measured_checks(
    result: dict[str, Any], config: Any, prerequisite: dict[str, Any]
) -> dict[str, Any]:
    assert config.e1 is not None
    tolerance = (
        config.e1.algebraic_tolerances.float32_atol
        if config.data.dtype == "float32"
        else config.e1.algebraic_tolerances.float64_atol
    )
    regimes = {row["regime"] for row in result["selected_results"]}
    deltas = [abs(row["delta_nuT"]) for row in result["selected_results"]]
    return {
        "e0_prerequisite_passed": check(
            prerequisite.get("status") == "pass",
            prerequisite.get("status"),
            "pass",
            "schema/status/internal consistency verified; file hashes recorded",
        ),
        "both_regimes_present": check(
            regimes == {"stable", "unstable"},
            sorted(regimes),
            ["stable", "unstable"],
            "classified from measured delta_nuT",
        ),
        "no_exact_match_case": check(
            bool(deltas) and min(deltas) > tolerance,
            min(deltas) if deltas else None,
            tolerance,
            "measured delta_nuT separation",
        ),
        "finite_input_path_verified": check(
            result["checks"]["finite_input_path_verified"],
            result["data_manifest"]["finite_input_path_runtime_check"],
            None,
            "finite-only API and synthetic high-frequency check",
        ),
        "target_coefficients_agree_with_reference": check(
            result["checks"]["target_coefficients_agree_with_reference"],
            result["data_manifest"]["reference_to_target_max_coefficient_error"],
            tolerance,
            "retained target/reference Fourier coefficients",
        ),
        "heat_solver_algebraic_checks_passed": check(
            result["checks"]["heat_solver_algebraic_error"] <= tolerance,
            result["checks"]["heat_solver_algebraic_error"],
            tolerance,
            "constant/cosine/sine exact heat check",
        ),
        "real_fourier_order_verified": check(
            result["checks"]["real_fourier_coordinate_error"] <= tolerance,
            result["checks"]["real_fourier_coordinate_error"],
            tolerance,
            "measured D @ S identity",
        ),
        "ideal_readout_coordinate_check_passed": check(
            result["checks"]["ideal_readout_coordinate_error"] <= tolerance,
            result["checks"]["ideal_readout_coordinate_error"],
            tolerance,
            "measured (M D) S = M",
        ),
        "noise_zero_matches_clean": check(
            result["checks"]["noise_zero_matches_clean"],
            0.0,
            tolerance,
            "delta=0 prediction equality",
        ),
        "ridge_uses_validation_only": check(
            all(
                "test" not in key.lower()
                for row in result["ridge_selection"]
                for key in row
            ),
            config.e1.selection_metric,
            "validation-only",
            f"candidate selection; tie break={config.e1.ridge_tie_break}",
        ),
        "no_zscore_standardization": check(
            config.data.preprocessing == "l2_scaling_only"
            and "no z-score" in result["data_manifest"]["feature_preprocessing"],
            result["data_manifest"]["feature_preprocessing"],
            "l2_scaling_only; no z-score",
            "effective configuration and implemented feature metadata",
        ),
    }


def run_heat_calibration(
    config_path: Path,
    e0_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool,
    skip_plots: bool,
    invocation: RecipeInvocation,
) -> RecipeResult:
    """Run E1 artifact orchestration without argparse or printing."""
    if output_dir.is_symlink():
        raise RecipeUsageError(
            f"output path must not be a symlink: {output_dir}"
        )
    _load_science_dependencies()
    try:
        requested = load_config_json(config_path)
        if requested.e1 is None:
            raise ValueError("config must contain an e1 section")
        if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
            raise RecipeUsageError(
                f"{output_dir} is nonempty; pass --overwrite"
            )
        effective, master, prerequisite = validate_e0_prerequisite(
            e0_dir, requested
        )
    except RecipeUsageError:
        raise
    except Exception as exc:
        raise RecipeUsageError(str(exc)) from exc
    try:
        prepare_output(
            output_dir,
            overwrite,
            invocation=invocation,
            protected_paths=(config_path, e0_dir),
        )
    except RecipeUsageError:
        raise
    except Exception as exc:
        raise RecipeUsageError(str(exc)) from exc

    environment = runtime_environment(config_path, e0_dir, invocation)
    failures: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "schema_version": E1_SCHEMA_VERSION,
        "status": "fail",
        "required_checks": {},
    }
    expected: set[str] | None = None

    try:
        save_config_json(effective, output_dir / "resolved_config.json")
        write_json(output_dir / "e0_prerequisite.json", prerequisite)
        result = run_e1(effective, master)
        for name in TABLE_NAMES:
            write_csv(output_dir / f"{name}.csv", result[name])
        atomic_torch_save(
            output_dir / "selected_models.pt",
            {"schema_version": E1_SCHEMA_VERSION, "models": result["models"]},
        )
        result["data_manifest"].update(
            {
                "master_tensor_hash": prerequisite["master_tensor_hash"],
                "master_file_sha256": prerequisite["master_file_sha256"],
                "master_manifest_file_sha256": prerequisite[
                    "master_manifest_file_sha256"
                ],
                "resolved_config_hash": file_sha256(
                    output_dir / "resolved_config.json"
                ),
                "resolved_config_hash_rule": (
                    "SHA-256 of final resolved_config.json bytes"
                ),
                "e0_prerequisite_hash": file_sha256(
                    output_dir / "e0_prerequisite.json"
                ),
                "e0_prerequisite_hash_rule": (
                    "SHA-256 of e0_prerequisite.json bytes"
                ),
            }
        )
        write_json(output_dir / "data_manifest.json", result["data_manifest"])

        plot_manifest = {
            "status": "skipped" if skip_plots else "pending",
            "reason": "--skip-plots" if skip_plots else None,
            "plots": [],
        }
        if not skip_plots:
            from ..e1_plotting import create_e1_plots

            plot_manifest["plots"] = create_e1_plots(output_dir, result)
            plot_manifest["status"] = "pass"
        write_json(output_dir / "plot_manifest.json", plot_manifest)

        environment.update(
            {
                "ended_at": datetime.now(timezone.utc).isoformat(),
                "dtype": effective.data.dtype,
                "device": effective.data.device,
                "actual_solver": effective.target.solver,
            }
        )
        write_json(output_dir / "environment.json", environment)
        write_json(output_dir / "failed_runs.json", failures)

        qa = validate_saved_numeric_artifacts(output_dir, effective)
        result["data_manifest"]["selected_models_content_hash"] = qa[
            "model_content_hash"
        ]
        write_json(output_dir / "data_manifest.json", result["data_manifest"])
        qa = validate_saved_numeric_artifacts(output_dir, effective)
        if (
            qa["model_content_hash"]
            != result["data_manifest"]["selected_models_content_hash"]
        ):
            raise ValueError(
                "selected model canonical hash changed during finalization"
            )
        plot_names = validate_plots(output_dir, skip_plots=skip_plots)
        expected = expected_artifacts(plot_names)
        prefinal_expected = expected - {
            "e1_summary.json",
            "artifact_manifest.json",
        }
        validate_artifact_set(output_dir, prefinal_expected)

        checks = measured_checks(result, effective, prerequisite)
        checks.update(
            {
                "identifiability_reported": check(
                    True,
                    len(qa["tables"]["mode_comparison.csv"]),
                    "exact key/schema QA",
                    "saved mode table required columns, finite values, and keys verified",
                ),
                "all_required_artifacts_present": check(
                    True,
                    sorted(prefinal_expected),
                    sorted(prefinal_expected),
                    "prefinal expected artifact set exactly equals actual set",
                ),
                "all_required_artifacts_finite": check(
                    True,
                    "saved JSON/CSV/PT scanners passed",
                    "all finite",
                    "read-after-write recursive finite scan",
                ),
                "all_requested_q_completed": check(
                    True,
                    "exact unique (case,q) sets",
                    "config Cartesian product",
                    "selected and diagnostic key sets verified",
                ),
                "all_requested_noise_levels_completed": check(
                    True,
                    "exact unique result/summary keys",
                    "config Cartesian product",
                    "noise repeat and summary key sets verified",
                ),
                "plots_completed_or_explicitly_skipped": check(
                    True,
                    plot_manifest["status"],
                    "pass or explicit skipped",
                    "plot manifest and files verified bidirectionally",
                ),
            }
        )
        scientific = scientific_acceptance_checks(qa["tables"], effective)
        checks.update(
            {f"scientific_{name}": value for name, value in scientific.items()}
        )
        summary = {
            "schema_version": E1_SCHEMA_VERSION,
            "status": (
                "pass"
                if all(value["status"] == "pass" for value in checks.values())
                else "fail"
            ),
            "required_checks": checks,
            "scientific_acceptance": scientific,
            "profile": effective.e1.profile,
            "cases": [
                {"name": case.name, "nu": case.nu, "T": case.T}
                for case in effective.e1.surrogate_cases
            ],
            "output_dims": list(effective.e1.output_dims),
            "noise_levels": list(effective.e1.noise_levels),
            "selected_models_content_hash": qa["model_content_hash"],
        }
        if summary["status"] != "pass":
            failed = [
                name
                for name, value in checks.items()
                if value["status"] != "pass"
            ]
            raise ValueError(
                "required/scientific checks failed: " + ", ".join(failed)
            )

        summary["required_checks"]["artifact_manifest_verified"] = check(
            True,
            "final size/SHA-256 records",
            "exact final artifact set",
            "manifest is generated after final summary and read back",
        )
        write_json(output_dir / "e1_summary.json", summary)
        write_json(
            output_dir / "artifact_manifest.json",
            artifact_records(output_dir, expected),
        )
        verify_artifact_manifest(output_dir, expected)
        validate_artifact_set(output_dir, expected)
    except Exception as exc:
        failures.append(
            {
                "stage": "run_or_saved_artifact_qa",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        summary["status"] = "fail"
        summary["failure_reason"] = failures[-1]["error"]
        environment["ended_at"] = datetime.now(timezone.utc).isoformat()
        for name, value in (
            ("environment.json", environment),
            ("failed_runs.json", failures),
            ("e1_summary.json", summary),
        ):
            try:
                write_json(output_dir / name, value)
            except Exception:
                pass

    passed = summary["status"] == "pass"
    return RecipeResult(
        status="pass" if passed else "fail",
        exit_code=0 if passed else 1,
        output_dir=output_dir,
        console_payload=summary,
        summary_path=output_dir / "e1_summary.json",
    )
