"""Import-safe artifact orchestration for the Paper 1 E0 gate."""
from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import platform
import time
from typing import Any

import torch

from pol.runtime.io import file_sha256, write_csv, write_e0_json
from pol.runtime.provenance import git_output
from pol.runtime.recipe import RecipeInvocation, RecipeResult, RecipeUsageError

ARTIFACTS = (
    "e0_summary.json",
    "reference_convergence.csv",
    "reference_convergence.json",
    "resampling_checks.json",
    "input_interface_checks.json",
    "model1_identity.json",
    "master_initial_conditions.pt",
    "master_manifest.json",
    "resolved_config.json",
    "environment.json",
    "accepted_production_config.json",
)


def _load_science_dependencies() -> None:
    """Load Paper 1 scientific modules only when the recipe is invoked."""
    global E0_SCHEMA_VERSION, E0SolverCache
    global build_master_grf_initial_conditions, build_required_checks
    global load_config_json, run_algebraic_checks, run_interface_checks
    global run_model1_checks, run_reference_convergence
    global save_config_json, save_master_initial_conditions

    from ..config import load_config_json, save_config_json
    from ..e0 import (
        E0_SCHEMA_VERSION,
        E0SolverCache,
        build_required_checks,
        run_algebraic_checks,
        run_interface_checks,
        run_model1_checks,
        run_reference_convergence,
        save_master_initial_conditions,
    )
    from ..initial_conditions import build_master_grf_initial_conditions


def _preflight(output_dir: Path, overwrite: bool) -> None:
    existing = [name for name in ARTIFACTS if (output_dir / name).exists()]
    if existing and not overwrite:
        raise RecipeUsageError(
            f"{output_dir} already contains E0 artifacts "
            f"({', '.join(existing)}); pass --overwrite"
        )
    if overwrite:
        for name in existing:
            (output_dir / name).unlink()


def _run_foundation_validation_staged(
    config_path: Path,
    output_dir: Path,
    *,
    overwrite: bool,
    invocation: RecipeInvocation,
) -> RecipeResult:
    """Run E0 orchestration without argparse, global cwd changes, or printing."""
    _load_science_dependencies()
    try:
        config = load_config_json(config_path)
        if config.e0 is None:
            raise ValueError("config must contain an e0 section")
    except Exception as exc:
        if isinstance(exc, RecipeUsageError):
            raise
        raise RecipeUsageError(str(exc)) from exc
    _preflight(output_dir, overwrite)

    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    config_hash = file_sha256(config_path)
    summary: dict[str, Any] = {
        "schema_version": E0_SCHEMA_VERSION,
        "status": "fail",
        "required_checks": {},
        "selected_reference": {
            "reference_nx": None,
            "solver": None,
            "requested_dt": None,
            "requested_fine_dt": None,
            "effective_inner_step": None,
            "joint_status": None,
        },
        "accepted_production_config": None,
        "num_failures": 0,
        "failure_reasons": [],
    }
    master_manifest: dict[str, Any] = {}
    cache = E0SolverCache()
    try:
        master = build_master_grf_initial_conditions(config)
        master_manifest = save_master_initial_conditions(
            master,
            output_dir / "master_initial_conditions.pt",
            output_dir / "master_manifest.json",
            config,
        )
        resampling, projector = run_algebraic_checks(config)
        resampling["fourier_projector"] = projector
        write_e0_json(output_dir / "resampling_checks.json", resampling)
        convergence = run_reference_convergence(config, master, cache=cache)
        reference_state = convergence.pop("_reference_state")
        write_e0_json(output_dir / "reference_convergence.json", convergence)
        fields = [
                "kind",
                "candidate_nx",
                "eligible_for_production",
                "status",
                "solver",
                "requested_dt",
                "requested_fine_dt",
                "outer_steps",
                "substeps_per_outer",
                "effective_inner_step",
                "relative_l2_mean",
                "relative_l2_median",
                "relative_l2_max",
                "absolute_l2_mean",
                "low_mode_relative_l2_mean",
                "master_hash",
                "sample_ids",
        ]
        csv_rows = [
            {
                        "kind": row["kind"],
                        "candidate_nx": row["candidate_nx"],
                        "eligible_for_production": row.get(
                            "eligible_for_production", ""
                        ),
                        "status": row.get("status", ""),
                        "solver": row["solver"],
                        "requested_dt": row["requested_dt"],
                        "requested_fine_dt": row["requested_fine_dt"],
                        "outer_steps": row["outer_steps"],
                        "substeps_per_outer": row["substeps_per_outer"],
                        "effective_inner_step": row["effective_inner_step"],
                        "relative_l2_mean": row["relative_l2"]["mean"],
                        "relative_l2_median": row["relative_l2"]["median"],
                        "relative_l2_max": row["relative_l2"]["max"],
                        "absolute_l2_mean": row["absolute_l2"]["mean"],
                        "low_mode_relative_l2_mean": row["low_mode_relative_l2"][
                            "mean"
                        ],
                        "master_hash": row["master_hash"],
                        "sample_ids": json.dumps(row["sample_ids"]),
            }
            for row in convergence["rows"]
        ]
        write_csv(
            output_dir / "reference_convergence.csv",
            csv_rows,
            fieldnames=fields,
        )
        chosen_time = convergence.get("selected_temporal")
        chosen_space = convergence.get("selected_spatial")
        interfaces = run_interface_checks(config, master, reference_state)
        write_e0_json(output_dir / "input_interface_checks.json", interfaces)
        model1 = run_model1_checks(config, master, cache=cache)
        write_e0_json(output_dir / "model1_identity.json", model1)
        required = build_required_checks(
            resampling, projector, convergence, interfaces, model1
        )
        summary["required_checks"] = required
        failures = [name for name, status in required.items() if status != "pass"]
        summary["failure_reasons"] = [
            f"required check failed: {name}" for name in failures
        ]
        summary["num_failures"] = len(failures)
        summary["status"] = "pass" if not failures else "fail"
        if (
            not failures
            and chosen_space
            and chosen_time
            and convergence["joint_status"] == "pass"
        ):
            summary["selected_reference"] = {
                "reference_nx": chosen_space["candidate_nx"],
                "solver": chosen_time["solver"],
                "requested_dt": chosen_time["requested_dt"],
                "requested_fine_dt": chosen_time["requested_fine_dt"],
                "effective_inner_step": chosen_time["effective_inner_step"],
                "joint_status": "pass",
            }
            accepted = replace(
                config,
                e0=None,
                spatial=replace(
                    config.spatial,
                    reference_nx=int(chosen_space["candidate_nx"]),
                ),
                target=replace(
                    config.target,
                    solver=str(chosen_time["solver"]),
                    dt=float(chosen_time["requested_dt"]),
                    fine_dt=chosen_time["requested_fine_dt"],
                ),
            )
            accepted.validate()
            save_config_json(accepted, output_dir / "accepted_production_config.json")
            summary["accepted_production_config"] = (
                "accepted_production_config.json"
            )
    except Exception as exc:
        summary["failure_reasons"].append(f"{type(exc).__name__}: {exc}")
        summary["num_failures"] = len(summary["failure_reasons"])
    finally:
        save_config_json(config, output_dir / "resolved_config.json")
        environment = {
            "full_command": list(invocation.command),
            "cwd": str(invocation.working_directory),
            "git_commit": git_output(invocation.repo_root, ["rev-parse", "HEAD"]),
            "git_dirty_status": git_output(
                invocation.repo_root, ["status", "--porcelain"]
            ),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "platform": platform.platform(),
            "device": config.data.device,
            "dtype": config.data.dtype,
            "cuda_available": torch.cuda.is_available(),
            "config_path": str(config_path),
            "config_hash": config_hash,
            "master_archive_hash": master_manifest.get("tensor_hash"),
            "solver_cache": cache.stats(),
            "runtime_seconds": time.perf_counter() - start,
        }
        write_e0_json(output_dir / "environment.json", environment)
        write_e0_json(output_dir / "e0_summary.json", summary)
    passed = summary["status"] == "pass"
    return RecipeResult(
        status="pass" if passed else "fail",
        exit_code=0 if passed else 1,
        output_dir=output_dir,
        console_payload=summary,
        summary_path=output_dir / "e0_summary.json",
    )


def run_foundation_validation(
    config_path: Path,
    output_dir: Path,
    *,
    overwrite: bool,
    invocation: RecipeInvocation,
) -> RecipeResult:
    """Run E0 through rollback-safe directory publication."""
    if output_dir.is_symlink():
        raise RecipeUsageError(f"output path must not be a symlink: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise RecipeUsageError(
            f"{output_dir} already contains E0 artifacts; pass --overwrite"
        )
    from pol.runtime.artifacts import execute_recipe_transaction
    from pol.paper1.artifact_contracts import E0ArtifactContract

    contract = E0ArtifactContract()
    return execute_recipe_transaction(
        output_dir,
        execute=lambda staging: _run_foundation_validation_staged(
            config_path,
            staging,
            overwrite=True,
            invocation=invocation,
        ),
        validate_complete=contract.validate_complete,
        validate_failure=contract.validate_failure,
    )
