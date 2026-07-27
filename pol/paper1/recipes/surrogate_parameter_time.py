"""Import-safe Paper 1 E2 surrogate-parameter/time recipe."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from pol.runtime.io import atomic_torch_save, file_sha256 as sha, write_csv as atomic_write_csv
from pol.runtime.io import write_strict_json
from pol.runtime.provenance import git_output
from pol.runtime.recipe import (
    RecipeInvocation,
    RecipeResult,
    RecipeUsageError,
    validate_recursive_delete_target,
)

TABLES = ("validation_sweep", "test_sweep", "model3_validation_by_seed", "model3_test_by_seed",
          "model3_test_aggregate", "convergence_results", "solver_metadata",
          "physical_point_aliases")


def _load_science_dependencies() -> None:
    """Load Paper 1 scientific modules only when the recipe is invoked."""
    global CSV_CONTRACT_VERSION, E0_REQUIRED_CHECKS, E2_SCHEMA_VERSION
    global assert_finite, canonical_config_json
    global dry_run_cost_summary, expected_artifacts, load_config_json
    global load_master_dataset, run_e2
    global save_config_json, spectral_resample_periodic, stable_hash, tensor_hash
    global validate_artifact_contract, validate_csv, validate_e0_prerequisite
    global validate_result_cartesian, validate_resume_output, write_manifest

    from pol.paper1.config import (
        canonical_config_json,
        load_config_json,
        save_config_json,
    )
    from pol.paper1.datasets import load_master_dataset, tensor_hash
    from pol.paper1.e1 import E0_REQUIRED_CHECKS, validate_e0_prerequisite
    from pol.paper1.e2 import (
        E2_SCHEMA_VERSION,
        dry_run_cost_summary,
        run_e2,
        stable_hash,
    )
    from pol.paper1.e2_qa import (
        CSV_CONTRACT_VERSION,
        assert_finite,
        expected_artifacts,
        validate_artifact_contract,
        validate_csv,
        validate_result_cartesian,
        validate_resume_output,
        write_manifest,
    )
    from pol.paper1.grids import spectral_resample_periodic


def write_json(path: Path, value: Any) -> None:
    write_strict_json(path, value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path.name}")
    rows = [{"schema_version": CSV_CONTRACT_VERSION, **row} for row in rows]
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    atomic_write_csv(path, rows, fieldnames=fields)


def check(status: bool, value: Any, threshold: Any, message: str) -> dict[str, Any]:
    return {"status": "pass" if status else "fail", "value": value, "threshold": threshold, "message": message}


def prerequisite(config: Any, e0_dir: Path, dataset_dir: Path) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    _, master, strict_e0 = validate_e0_prerequisite(e0_dir, config)
    e0_summary = json.loads((e0_dir / "e0_summary.json").read_text())
    if set(e0_summary.get("required_checks", {})) != E0_REQUIRED_CHECKS:
        raise ValueError("E0 required-check name set mismatch")
    accepted_path = e0_dir / "accepted_production_config.json"
    accepted = load_config_json(accepted_path)
    dataset = load_master_dataset(dataset_dir)
    if dataset.y_target_master is None:
        raise ValueError("dataset y_target_master is missing")
    if dataset.config.target.equation not in {"burgers", "viscous_burgers"}:
        raise ValueError("dataset target is not Burgers")
    for name in ("domain", "data", "target"):
        if getattr(dataset.config, name) != getattr(accepted, name):
            raise ValueError(f"dataset {name} does not match E0 accepted config")
    if dataset.config.spatial.reference_nx != accepted.spatial.reference_nx:
        raise ValueError("dataset reference_nx does not match E0 accepted config")
    for name in ("domain", "data", "target"):
        if getattr(config, name) != getattr(dataset.config, name):
            raise ValueError(f"E2 config {name} does not match dataset")
    if config.spatial.reference_nx != dataset.config.spatial.reference_nx:
        raise ValueError("E2 reference_nx does not match dataset")
    master_at_reference = spectral_resample_periodic(
        master.values_master, dataset.config.spatial.reference_nx,
        domain_length=config.domain.length)
    if tensor_hash(master_at_reference) != dataset.metadata["tensor_hashes"]["u0_master"]:
        raise ValueError("dataset u0 tensor hash does not match validated E0 master tensor")
    source_e0 = dataset.metadata.get("source_e0")
    if not isinstance(source_e0, dict):
        raise ValueError("dataset manifest lacks cryptographic source_e0 provenance")
    expected_source_files = {
        name: sha(e0_dir / name) for name in (
            "e0_summary.json", "accepted_production_config.json",
            "master_initial_conditions.pt", "master_manifest.json")}
    if source_e0.get("files") != expected_source_files:
        raise ValueError("dataset source_e0 hash chain mismatch")
    e0_report = {"schema_version": "paper1-e2-e0-prerequisite-v3", "status": "pass",
                 "e0_summary_sha256": sha(e0_dir / "e0_summary.json"),
                 "accepted_config_sha256": sha(accepted_path),
                 "master_file_sha256": sha(e0_dir / "master_initial_conditions.pt"),
                 "master_tensor_hash": strict_e0["master_tensor_hash"],
                 "master_manifest_sha256": sha(e0_dir / "master_manifest.json"),
                 "strict_validator_hash": stable_hash(strict_e0),
                 "selected_reference": json.loads(
                     (e0_dir / "e0_summary.json").read_text())["selected_reference"]}
    dataset_report = {"schema_version": "paper1-e2-dataset-prerequisite-v3", "status": "pass",
                      "dataset_hash": dataset.metadata["dataset_hash"], "split_hash": dataset.metadata["split_hash"],
                      "manifest_sha256": sha(dataset_dir / "manifest.json"),
                      "payload_sha256": sha(dataset_dir / "master_dataset.pt"),
                      "tensor_hashes": dataset.metadata["tensor_hashes"],
                      "source_e0": source_e0}
    return dataset, e0_report, dataset_report


def build_surrogate_parameter_time_cost_summary(
        config_path: Path) -> dict[str, Any]:
    """Return the E2 dry-run cost summary without creating output."""
    _load_science_dependencies()
    try:
        config = load_config_json(config_path)
        if config.e2 is None:
            raise ValueError("config has no e2 section")
    except Exception as exc:
        raise RecipeUsageError(str(exc)) from exc
    return dry_run_cost_summary(config)


def run_surrogate_parameter_time(
    config_path: Path,
    e0_dir: Path,
    dataset_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool,
    resume: bool,
    skip_plots: bool,
    batch_size: int,
    invocation: RecipeInvocation,
) -> RecipeResult:
    """Run E2 artifact orchestration and its existing scientific QA."""
    if overwrite and resume:
        raise RecipeUsageError("--overwrite and --resume are mutually exclusive")
    if batch_size <= 0:
        raise RecipeUsageError("--batch-size must be positive")
    out = output_dir
    if out.is_symlink():
        raise RecipeUsageError(f"output path must not be a symlink: {out}")
    _load_science_dependencies()
    try:
        config = load_config_json(config_path)
        if config.e2 is None:
            raise ValueError("config has no e2 section")
        # Resolve and hash every current prerequisite before considering a
        # complete-output early return.  A missing or mismatched input is never
        # hidden by --resume.
        dataset, e0_report, dataset_report = prerequisite(
            config, e0_dir, dataset_dir)
        if resume and out.exists() and (out / "e2_summary.json").exists():
            if validate_resume_output(out):
                old_config = load_config_json(out / "resolved_config.json")
                if canonical_config_json(old_config) != canonical_config_json(config):
                    raise ValueError("complete resume rejected: resolved config content mismatch")
                old_e0 = json.loads((out / "e0_prerequisite.json").read_text())
                old_dataset = json.loads((out / "dataset_prerequisite.json").read_text())
                for label, old, current in (
                        ("E0", old_e0, e0_report),
                        ("dataset", old_dataset, dataset_report)):
                    if old != current:
                        raise ValueError(
                            f"complete resume rejected: {label} content hash binding mismatch")
                plot_manifest = json.loads((out / "plot_manifest.json").read_text())
                expected_plot_status = "skipped" if skip_plots else "pass"
                if plot_manifest.get("status") != expected_plot_status:
                    raise ValueError("complete resume rejected: plot policy mismatch")
                payload = {"status": "pass", "resume": "reused_complete_output"}
                return RecipeResult(
                    "pass", 0, out, payload,
                    out / "e2_summary.json", reused_complete_output=True)
        if out.exists() and any(out.iterdir()) and not (overwrite or resume):
            raise FileExistsError(f"{out} is nonempty; pass --overwrite or --resume")
        if overwrite and out.exists():
            validate_recursive_delete_target(
                out,
                repo_root=invocation.repo_root,
                protected_paths=(config_path, e0_dir, dataset_dir),
            )
            shutil.rmtree(out)
        out.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise RecipeUsageError(str(exc)) from exc
    final_out = out
    stage = final_out / ".attempt-staging"
    if stage.is_symlink():
        raise RecipeUsageError(f"unsafe staging path: {stage} is a symlink")
    if stage.exists() and not stage.is_dir():
        raise RecipeUsageError(f"unsafe staging path: {stage} is not a directory")
    if stage.exists():
        validate_recursive_delete_target(
            stage,
            repo_root=invocation.repo_root,
            protected_paths=(config_path, e0_dir, dataset_dir),
        )
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    bindings = {
        "resolved_config_hash": hashlib.sha256(
            canonical_config_json(config).encode()).hexdigest(),
        "e0_prerequisite_hash": stable_hash(e0_report),
        "dataset_prerequisite_hash": stable_hash(dataset_report),
        "dataset_hash": dataset.metadata["dataset_hash"],
        "split_hash": dataset.metadata["split_hash"],
        "sample_ids_hash": tensor_hash(dataset.sample_ids),
        "input_tensor_hashes": dataset.metadata["tensor_hashes"],
        "protocol_version": E2_SCHEMA_VERSION,
    }
    summary: dict[str, Any] = {
        "schema_version": E2_SCHEMA_VERSION, "status": "fail",
        "required_checks": {}, "test_evaluated": False}
    try:
        save_config_json(config, stage / "resolved_config.json")
        write_json(stage / "e0_prerequisite.json", e0_report)
        write_json(stage / "dataset_prerequisite.json", dataset_report)
        result = run_e2(
            config, dataset, cache_dir=final_out / "cache",
            resume=resume, batch_size=batch_size,
            freeze_dir=stage, input_bindings=bindings)
        validate_result_cartesian(result, config)
        for name in ("validation_sweep", "model3_validation_by_seed",
                     "convergence_results", "solver_metadata",
                     "physical_point_aliases"):
            write_csv(stage / f"{name}.csv", result[name])
        if result["test_evaluated"]:
            for name in ("test_sweep", "model3_test_by_seed",
                         "model3_test_aggregate"):
                write_csv(stage / f"{name}.csv", result[name])
        write_json(stage / "selection_record.json", result["selection_record"])
        write_json(stage / "model_specific_optima.json", result["model_specific_optima"])
        write_json(stage / "shared_representatives.json", result["shared_representatives"])
        write_json(stage / "coordinate_history.json", result["coordinate_history"])
        write_json(stage / "e2_attempt_history.json", result["attempt_history"])
        write_json(stage / "convergence_summary.json", result["convergence_summary"])
        write_json(stage / "failed_runs.json", result["failed_runs"])
        write_json(stage / "event_log.json", {
            "schema_version": "paper1-e2-events-v1",
            "events": result["event_log"]})
        write_json(stage / "experiment_plan.json", dry_run_cost_summary(config))
        write_json(stage / "runtime_diagnostics.json", {
            "schema_version": "paper1-e2-runtime-v3",
            "definition": "cumulative wall-clock time inside run_e2 attempts",
            "total_runtime_seconds": result["runtime_seconds"],
            "cache": result["cache"],
            "attempts": result["attempt_history"],
        })
        if result["test_evaluated"]:
            atomic_torch_save(stage / "selected_models.pt", {
                "schema_version": E2_SCHEMA_VERSION,
                "selection_record_hash": result["selection_record_hash"],
                "frozen_plan_hash": result["frozen_plan_hash"],
                "models": result["selected_models"]})
        data_manifest = {"schema_version": "paper1-e2-data-v3", "dataset_hash": dataset.metadata["dataset_hash"],
                         "split_hash": dataset.metadata["split_hash"], "sample_ids_hash": tensor_hash(dataset.sample_ids),
                         "bindings": bindings,
                         "finite_input_path": "n_ref -> n_tar spectral low-pass -> n_sur interpolation",
                         "n_tar": config.spatial.target_data_nx, "n_sur_pilot": config.spatial.surrogate_internal_nx,
                         "actual_final_sweep_n_sur": result["pilot_n_sur"],
                         "J": config.spatial.observation_dim, "q": config.spatial.target_output_dim,
                         "cache": result["cache"]}
        write_json(stage / "data_manifest.json", data_manifest)
        environment = {"python_version": platform.python_version(), "torch_version": torch.__version__,
                       "platform": platform.platform(), "command": list(invocation.command),
                       "git_commit": git_output(invocation.repo_root, ("rev-parse", "HEAD")),
                       "ended_at": datetime.now(timezone.utc).isoformat(),
                       "runtime_seconds": result["runtime_seconds"],
                       "runtime_definition": "cumulative run_e2 wall-clock time"}
        write_json(stage / "environment.json", environment)
        plots = []
        make_plots = result["test_evaluated"] and not skip_plots
        if make_plots:
            from pol.paper1.e2_plotting import create_e2_plots

            plots = create_e2_plots(stage, result, config)
            for record in plots:
                path = stage / record["relative_path"]
                record["sha256"] = sha(path)
        plot_manifest = {
            "schema_version": "paper1-e2-plots-v3",
            "status": "pass" if make_plots else "skipped",
            "reason": None if make_plots else (
                "--skip-plots" if skip_plots
                else "pre-test scientific failure"),
            "plots": plots}
        write_json(stage / "plot_manifest.json", plot_manifest)
        actual_plot_files = {
            path.name for path in stage.iterdir()
            if path.is_file() and path.suffix.lower() in {".png", ".pdf"}}
        listed_plot_files = {record["relative_path"] for record in plots}
        if actual_plot_files != listed_plot_files:
            raise ValueError("plot manifest/file set mismatch")
        if any(
                record["size_bytes"] !=
                (stage / record["relative_path"]).stat().st_size
                or record["sha256"] != sha(
                    stage / record["relative_path"])
                for record in plots):
            raise ValueError("plot manifest size/hash mismatch")
        # Read-back structural QA.
        validate_csv(stage / "validation_sweep.csv", {"family","sweep_axis","model","validation_field_relative_l2_mean"},
                     ("family","sweep_axis","nu_tilde","T_tilde","model"))
        validate_csv(stage / "model3_validation_by_seed.csv", {"family","sweep_axis","candidate_order","seed"},
                     ("family","sweep_axis","nu_tilde","T_tilde","candidate_order","seed"))
        if result["test_evaluated"]:
            validate_csv(stage / "test_sweep.csv", {"family","sweep_axis","model","field_relative_l2_mean","selection_record_hash","frozen_plan_hash"},
                         ("family","sweep_axis","nu_tilde","T_tilde","model"))
            validate_csv(stage / "model3_test_by_seed.csv", {"family","sweep_axis","seed","field_relative_l2_mean","frozen_plan_hash"},
                         ("family","sweep_axis","nu_tilde","T_tilde","seed"))
            validate_csv(stage / "model3_test_aggregate.csv", {"family","sweep_axis","seed_count","mean","frozen_plan_hash"},
                         ("family","sweep_axis","nu_tilde","T_tilde"))
        validate_csv(stage / "convergence_results.csv", {"family","n_sur","frozen_readout","status"},
                     ("family","n_sur"))
        assert_finite(json.loads((stage / "selection_record.json").read_text()))
        conv_pass = result["convergence_summary"]["status"] == "pass"
        events = [item["event"] for item in result["event_log"]]
        event_order_ok = (
            not result["test_evaluated"] or (
                events.index("freeze_read_back")
                < events.index("first_test_state_solve")
                < events.index("first_test_metric")))
        test_hashes_ok = (
            not result["test_evaluated"] or all(
                row["selection_record_hash"] == result["selection_record_hash"]
                and row["frozen_plan_hash"] == result["frozen_plan_hash"]
                for row in result["test_sweep"]))
        checks = {
            "e0_prerequisite_passed": check(e0_report["status"] == "pass", e0_report["status"], "pass", "strict E0 validator"),
            "dataset_prerequisite_passed": check(dataset_report["status"] == "pass", dataset_report["status"], "pass", "dataset payload/hash chain"),
            "all_required_parameter_points_completed": check(not result["failed_runs"], len(result["failed_runs"]), 0, "required grid complete"),
            "model3_selection_and_evaluation_seeds_disjoint": check(not set(config.e2.model3.selection_seeds)&set(config.e2.model3.evaluation_seeds), True, True, "disjoint lists"),
            "no_zscore_standardization": check(config.data.preprocessing=="l2_scaling_only", config.data.preprocessing, "l2_scaling_only", "mean centering only"),
            "selection_frozen_before_test": check(event_order_ok, events, "freeze < test solve < metric", "measured event order"),
            "test_rows_bound_to_frozen_plan": check(test_hashes_ok, result["frozen_plan_hash"], "all rows match", "row hash cross-check"),
            "state_cache_reused_across_models": check(
                result["cache"]["states"]["solver_invocations"] == result["cache"]["states"]["misses"],
                result["cache"]["states"], "solver_invocations == unique state misses",
                "actual solver calls are counted independently from feature-cache hits"),
            "convergence_uses_no_test_ids": check(
                all(v != "test" for v in result["convergence_sample_membership"].values()),
                result["convergence_sample_membership"], "actual train/validation membership",
                "verified against shuffled dataset split"),
            "terminal_field_convergence_passed": check(conv_pass, conv_pass, True, "common-grid terminal discrepancy"),
        }
        status = "pass" if result["procedural_status"] == "pass" and all(v["status"]=="pass" for v in checks.values()) else "fail"
        attempt_history_hash = stable_hash(result["attempt_history"])
        summary = {"schema_version": E2_SCHEMA_VERSION, "status": status, "profile": config.e2.profile,
                   "required_checks": checks, "selection_record_hash": result["selection_record_hash"],
                   "frozen_plan_hash": result["frozen_plan_hash"],
                   "test_evaluated": result["test_evaluated"],
                   "failure_kind": result["failure_kind"],
                   "failure_reason": result["failure_reason"],
                   "global_n_sur_base": result["convergence_summary"]["global_n_sur_base"],
                   "final_attempt_index": len(result["attempt_history"]) - 1,
                   "actual_final_sweep_n_sur": result["pilot_n_sur"],
                   "attempt_history_hash": attempt_history_hash}
        if status == "pass":
            handoff = {"schema_version": "paper1-e2-handoff-v3", "status": "pass",
                       "families": result["shared_representatives"], "model_specific_optima": result["model_specific_optima"],
                       "final_attempt_index": len(result["attempt_history"]) - 1,
                       "actual_final_sweep_n_sur": result["pilot_n_sur"],
                       "auto_rerun_history": result["auto_rerun_history"],
                       "representative_model": config.e2.representative_model, "selection_policy": config.e2.selection_metric,
                       "family_n_sur_base": {k:v["n_sur_base"] for k,v in result["convergence_summary"]["families"].items()},
                       "global_n_sur_base": result["convergence_summary"]["global_n_sur_base"],
                       "n_tar": config.spatial.target_data_nx, "J": config.spatial.observation_dim, "q": config.spatial.target_output_dim,
                       "solver": {"burgers": asdict(config.e2.burgers), "reaction_diffusion": asdict(config.e2.reaction_diffusion)},
                       "selected_hyperparameters": result["shared_hyperparameters"],
                       "model3": asdict(config.e2.model3), "dataset_hash": dataset.metadata["dataset_hash"],
                       "split_hash": dataset.metadata["split_hash"],
                       "e0_prerequisite_hash": sha(stage/"e0_prerequisite.json"),
                       "resolved_config_hash": sha(stage/"resolved_config.json"),
                       "selection_record_hash": result["selection_record_hash"],
                       "frozen_plan_hash": result["frozen_plan_hash"],
                       "attempt_history_hash": attempt_history_hash,
                       "convergence_artifact": "convergence_summary.json", "convergence_hash": sha(stage/"convergence_summary.json"),
                       "git_commit": environment["git_commit"], "environment": "environment.json"}
            write_json(stage / "e2_handoff.json", handoff)
        write_json(stage / "e2_summary.json", summary)
    except Exception as exc:
        # Runtime failures are finalized from the staging state without
        # rewriting any file after the manifest.
        failure = {"failure_kind": "runtime_error", "stage": "run_or_qa",
                   "error": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc()}
        summary.update({"status": "fail", "test_evaluated": False,
                        "failure_kind": "runtime_error",
                        "failure_reason": failure["error"]})
        write_json(stage / "failed_runs.json", [failure])
        write_json(stage / "e2_summary.json", summary)
        if not (stage / "environment.json").exists():
            write_json(stage / "environment.json", {
                "schema_version": "paper1-e2-environment-v3",
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "runtime_definition": "runtime failure before scientific result"})
        keep = expected_artifacts(
            status="fail", skip_plots=True, test_evaluated=False,
            failure_kind="runtime_error") - {"artifact_manifest.json"}
        for path in list(stage.iterdir()):
            if path.is_file() and path.name not in keep:
                path.unlink()
    expected = expected_artifacts(
        status=summary["status"], skip_plots=skip_plots,
        test_evaluated=bool(summary["test_evaluated"]),
        failure_kind=summary.get("failure_kind"))
    validate_artifact_contract(
        stage, status=summary["status"], skip_plots=skip_plots,
        test_evaluated=bool(summary["test_evaluated"]),
        include_manifest=False, failure_kind=summary.get("failure_kind"))
    write_manifest(stage, expected - {"artifact_manifest.json"})
    validate_artifact_contract(
        stage, status=summary["status"], skip_plots=skip_plots,
        test_evaluated=bool(summary["test_evaluated"]),
        failure_kind=summary.get("failure_kind"))
    # Publish only after complete QA. Remove stale known artifacts, retain cache.
    known = expected_artifacts(status="pass", skip_plots=False, test_evaluated=True) | expected_artifacts(
        status="fail", skip_plots=True, test_evaluated=False) | expected_artifacts(
        status="fail", skip_plots=True, test_evaluated=False,
        failure_kind="runtime_error")
    for name in known:
        (final_out / name).unlink(missing_ok=True)
    for path in stage.iterdir():
        os.replace(path, final_out / path.name)
    stage.rmdir()
    exit_code = 0 if summary["status"] == "pass" else 1
    return RecipeResult(
        "pass" if exit_code == 0 else "fail",
        exit_code,
        final_out,
        summary,
        final_out / "e2_summary.json",
    )
