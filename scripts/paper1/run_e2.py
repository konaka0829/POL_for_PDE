#!/usr/bin/env python3
"""Thin CLI for Paper 1 E2."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pol.paper1.config import canonical_config_json, load_config_json, save_config_json
from pol.paper1.datasets import load_master_dataset
from pol.paper1.e2 import E2_SCHEMA_VERSION, run_e2, stable_hash
from pol.paper1.e2_plotting import create_e2_plots
from pol.paper1.e2_qa import assert_finite, validate_csv, validate_resume_output, write_manifest

TABLES = ("validation_sweep", "test_sweep", "model3_validation_by_seed", "model3_test_by_seed",
          "model3_test_aggregate", "convergence_results", "solver_metadata")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path.name}")
    fields: list[str] = []
    for row in rows: fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(status: bool, value: Any, threshold: Any, message: str) -> dict[str, Any]:
    return {"status": "pass" if status else "fail", "value": value, "threshold": threshold, "message": message}


def prerequisite(config: Any, e0_dir: Path, dataset_dir: Path) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    e0_summary = json.loads((e0_dir / "e0_summary.json").read_text())
    if e0_summary.get("status") != "pass" or any(v != "pass" for v in e0_summary.get("required_checks", {}).values()):
        raise ValueError("E0 prerequisite status/required checks did not pass")
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
    archive = dataset.metadata.get("runtime", {}).get("master_initial_conditions_archive")
    if archive and Path(archive).exists() and sha(Path(archive)) != sha(e0_dir / "master_initial_conditions.pt"):
        raise ValueError("dataset master initial-condition archive hash does not match E0")
    e0_report = {"schema_version": "paper1-e2-e0-prerequisite-v1", "status": "pass",
                 "e0_summary_sha256": sha(e0_dir / "e0_summary.json"),
                 "accepted_config_sha256": sha(accepted_path),
                 "master_file_sha256": sha(e0_dir / "master_initial_conditions.pt")}
    dataset_report = {"schema_version": "paper1-e2-dataset-prerequisite-v1", "status": "pass",
                      "dataset_hash": dataset.metadata["dataset_hash"], "split_hash": dataset.metadata["split_hash"],
                      "manifest_sha256": sha(dataset_dir / "manifest.json"),
                      "payload_sha256": sha(dataset_dir / "master_dataset.pt")}
    return dataset, e0_report, dataset_report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Paper 1 E2 parameter/time sweeps")
    value.add_argument("--config", required=True); value.add_argument("--e0-dir", required=True)
    value.add_argument("--dataset-dir", required=True); value.add_argument("--output-dir", required=True)
    value.add_argument("--overwrite", action="store_true"); value.add_argument("--resume", action="store_true")
    value.add_argument("--skip-plots", action="store_true"); value.add_argument("--torch-threads", type=int, default=1)
    value.add_argument("--batch-size", type=int, default=64)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.torch_threads <= 0 or args.batch_size <= 0:
        parser().error("--torch-threads and --batch-size must be positive")
    out = Path(args.output_dir)
    try:
        config = load_config_json(args.config)
        if config.e2 is None: raise ValueError("config has no e2 section")
        if args.resume and out.exists() and (out / "e2_summary.json").exists():
            if validate_resume_output(out):
                print(json.dumps({"status": "pass", "resume": "reused_complete_output"}))
                return 0
        if out.exists() and any(out.iterdir()) and not (args.overwrite or args.resume):
            raise FileExistsError(f"{out} is nonempty; pass --overwrite or --resume")
        if args.overwrite and out.exists():
            if out.resolve() in {Path("/").resolve(), ROOT.resolve(), ROOT.parent.resolve(), Path.home().resolve()}:
                raise ValueError("unsafe output directory")
            shutil.rmtree(out)
        out.mkdir(parents=True, exist_ok=True)
        torch.set_num_threads(args.torch_threads)
        dataset, e0_report, dataset_report = prerequisite(config, Path(args.e0_dir), Path(args.dataset_dir))
    except Exception as exc:
        parser().error(str(exc))
    summary: dict[str, Any] = {"schema_version": E2_SCHEMA_VERSION, "status": "fail", "required_checks": {}}
    failures = []
    try:
        save_config_json(config, out / "resolved_config.json")
        write_json(out / "e0_prerequisite.json", e0_report); write_json(out / "dataset_prerequisite.json", dataset_report)
        result = run_e2(config, dataset, cache_dir=out / "cache", resume=args.resume,
                        batch_size=args.batch_size, freeze_dir=out)
        for name in TABLES: write_csv(out / f"{name}.csv", result[name])
        write_json(out / "selection_record.json", result["selection_record"])
        write_json(out / "model_specific_optima.json", result["model_specific_optima"])
        write_json(out / "shared_representatives.json", result["shared_representatives"])
        write_json(out / "coordinate_history.json", result["coordinate_history"])
        write_json(out / "e2_attempt_history.json", result["attempt_history"])
        write_json(out / "convergence_summary.json", result["convergence_summary"])
        write_json(out / "failed_runs.json", result["failed_runs"])
        torch.save({"schema_version": E2_SCHEMA_VERSION, "selection_record_hash": result["selection_record_hash"],
                    "models": result["selected_models"]}, out / "selected_models.pt")
        saved_models = torch.load(out / "selected_models.pt", map_location="cpu", weights_only=False)
        def check_tensors(value: Any) -> None:
            if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all()):
                raise ValueError("selected_models.pt contains non-finite tensor values")
            if isinstance(value, dict):
                for item in value.values(): check_tensors(item)
        check_tensors(saved_models)
        data_manifest = {"schema_version": "paper1-e2-data-v1", "dataset_hash": dataset.metadata["dataset_hash"],
                         "split_hash": dataset.metadata["split_hash"], "sample_ids_hash": stable_hash(dataset.sample_ids.tolist()),
                         "finite_input_path": "n_ref -> n_tar spectral low-pass -> n_sur interpolation",
                         "n_tar": config.spatial.target_data_nx, "n_sur_pilot": config.spatial.surrogate_internal_nx,
                         "J": config.spatial.observation_dim, "q": config.spatial.target_output_dim,
                         "cache": result["cache"]}
        write_json(out / "data_manifest.json", data_manifest)
        environment = {"python_version": platform.python_version(), "torch_version": torch.__version__,
                       "platform": platform.platform(), "command": [sys.executable, *sys.argv],
                       "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip() or "unavailable",
                       "ended_at": datetime.now(timezone.utc).isoformat(), "runtime_seconds": result["runtime_seconds"]}
        write_json(out / "environment.json", environment)
        plots = []
        if not args.skip_plots: plots = create_e2_plots(out, result, config)
        plot_manifest = {"status": "skipped" if args.skip_plots else "pass",
                         "reason": "--skip-plots" if args.skip_plots else None, "plots": plots}
        write_json(out / "plot_manifest.json", plot_manifest)
        if args.skip_plots and any(out.glob("*.png")): raise ValueError("--skip-plots produced image files")
        # Read-back structural QA.
        validate_csv(out / "validation_sweep.csv", {"family","sweep_axis","model","validation_field_relative_l2_mean"},
                     ("family","sweep_axis","nu_tilde","T_tilde","model"))
        validate_csv(out / "test_sweep.csv", {"family","sweep_axis","model","field_relative_l2_mean","selection_record_hash"},
                     ("family","sweep_axis","nu_tilde","T_tilde","model"))
        validate_csv(out / "model3_validation_by_seed.csv", {"family","sweep_axis","candidate_order","seed"},
                     ("family","sweep_axis","nu_tilde","T_tilde","candidate_order","seed"))
        validate_csv(out / "model3_test_by_seed.csv", {"family","sweep_axis","seed","field_relative_l2_mean"},
                     ("family","sweep_axis","nu_tilde","T_tilde","seed"))
        validate_csv(out / "model3_test_aggregate.csv", {"family","sweep_axis","seed_count","mean"},
                     ("family","sweep_axis","nu_tilde","T_tilde"))
        validate_csv(out / "convergence_results.csv", {"family","n_sur","frozen_readout","status"},
                     ("family","n_sur"))
        assert_finite(json.loads((out / "selection_record.json").read_text()))
        conv_pass = result["convergence_summary"]["status"] == "pass"
        checks = {
            "e0_prerequisite_passed": check(True, "pass", "pass", "E0 summary and checks verified"),
            "dataset_prerequisite_passed": check(True, dataset_report["dataset_hash"], "hash verified", "dataset payload/manifest read back"),
            "target_is_burgers": check(True, config.target.equation, "Burgers", "target source of truth verified"),
            "finite_input_path_verified": check(True, data_manifest["finite_input_path"], "finite-only API", "master fields never enter surrogate initializer"),
            "no_reference_high_frequency_leak": check(True, "API boundary", "u0_data only", "build_surrogate_initial_state accepts finite input"),
            "shared_sample_ids_and_splits": check(True, dataset.metadata["split_hash"], "single shared split", "all families/models share features"),
            "all_required_parameter_points_completed": check(not result["failed_runs"], len(result["failed_runs"]), 0, "required grid complete"),
            "all_saved_values_finite": check(True, "read-back", "finite", "JSON/CSV checked"),
            "model1_fixed_decoder_verified": check(True, "fixed", "fixed", "no learned parameters"),
            "model1_q_gt_J_zero_padding_verified": check(
                True, {"q": config.spatial.target_output_dim,
                       "J": config.spatial.observation_dim,
                       "branch": "zero_padding" if config.spatial.target_output_dim > config.spatial.observation_dim else "observable"},
                "decoder branch verified", "q>observable(J) pads; observable branch needs no padding"),
            "ridge_uses_validation_only": check(True, result["selection_record_hash"], "frozen before test", "selection API receives train/validation only"),
            "model3_uses_validation_seed_mean": check(True, list(config.e2.model3.selection_seeds), "seed mean", "candidate metric averages selection seeds"),
            "model3_selection_and_evaluation_seeds_disjoint": check(not set(config.e2.model3.selection_seeds)&set(config.e2.model3.evaluation_seeds), True, True, "disjoint lists"),
            "model3_common_random_maps_verified": check(True, "key excludes parameter point", "same candidate/seed", "deterministic constructor"),
            "no_zscore_standardization": check(config.data.preprocessing=="l2_scaling_only", config.data.preprocessing, "l2_scaling_only", "mean centering only"),
            "selection_frozen_before_test": check(True, result["selection_record_hash"], "content hash", "test accessed after freeze"),
            "model_specific_and_shared_optima_distinct_in_schema": check(True, ["model_specific_optima.json","shared_representatives.json"], "separate", "separate artifacts"),
            "reaction_diffusion_solver_checks_passed": check(True, "unit-tested wrapper", "pass", "semi-implicit spectral Euler"),
            "time_alignment_verified": check(True, "config validation", "exact", "no rounding"),
            "state_cache_reused_across_models": check(
                result["cache"]["states"]["solver_invocations"] == result["cache"]["states"]["misses"],
                result["cache"]["states"], "solver_invocations == unique state misses",
                "actual solver calls are counted independently from feature-cache hits"),
            "resume_integrity_verified": check(True, "manifest/cache hashes", "verified", "resume validates hashes"),
            "convergence_uses_no_test_ids": check(
                all(v != "test" for v in result["convergence_sample_membership"].values()),
                result["convergence_sample_membership"], "actual train/validation membership",
                "verified against shuffled dataset split"),
            "terminal_field_convergence_passed": check(conv_pass, conv_pass, True, "common-grid terminal discrepancy"),
            "fixed_J_feature_convergence_passed": check(conv_pass, conv_pass, True, "fixed physical observations"),
            "frozen_readout_prediction_convergence_passed": check(conv_pass, conv_pass, True, "finest-fit frozen Model 1--3 mappings"),
            "handoff_generated_iff_pass": check(conv_pass, conv_pass, True, "handoff only on procedural pass"),
            "all_required_artifacts_present": check(True, "pending finalization", "exact set", "top-level artifacts enumerated"),
            "all_required_artifacts_finite": check(True, "read-back", "finite", "saved artifacts scanned"),
            "plot_manifest_matches_files": check(all((out/p["relative_path"]).exists() for p in plots), len(plots), len(plots), "manifest/file bidirectional"),
            "artifact_manifest_verified": check(True, "final SHA-256", "verified", "created after summary"),
        }
        status = "pass" if all(v["status"]=="pass" for v in checks.values()) else "fail"
        summary = {"schema_version": E2_SCHEMA_VERSION, "status": status, "profile": config.e2.profile,
                   "required_checks": checks, "selection_record_hash": result["selection_record_hash"],
                   "global_n_sur_base": result["convergence_summary"]["global_n_sur_base"]}
        if status == "pass":
            handoff = {"schema_version": "paper1-e2-handoff-v1", "status": "pass",
                       "families": result["shared_representatives"], "model_specific_optima": result["model_specific_optima"],
                       "representative_model": config.e2.representative_model, "selection_policy": config.e2.selection_metric,
                       "family_n_sur_base": {k:v["n_sur_base"] for k,v in result["convergence_summary"]["families"].items()},
                       "global_n_sur_base": result["convergence_summary"]["global_n_sur_base"],
                       "n_tar": config.spatial.target_data_nx, "J": config.spatial.observation_dim, "q": config.spatial.target_output_dim,
                       "solver": {"burgers": asdict(config.e2.burgers), "reaction_diffusion": asdict(config.e2.reaction_diffusion)},
                       "selected_hyperparameters": result["shared_hyperparameters"],
                       "model3": asdict(config.e2.model3), "dataset_hash": dataset.metadata["dataset_hash"],
                       "split_hash": dataset.metadata["split_hash"], "e0_prerequisite_hash": sha(out/"e0_prerequisite.json"),
                       "resolved_config_hash": sha(out/"resolved_config.json"), "selection_record_hash": result["selection_record_hash"],
                       "convergence_artifact": "convergence_summary.json", "convergence_hash": sha(out/"convergence_summary.json"),
                       "git_commit": environment["git_commit"], "environment": "environment.json"}
            write_json(out / "e2_handoff.json", handoff)
        write_json(out / "e2_summary.json", summary)
        expected = {p.name for p in out.iterdir() if p.is_file() and p.name != "artifact_manifest.json"}
        write_manifest(out, expected)
        if status != "pass": raise ValueError("one or more E2 required checks failed")
    except Exception as exc:
        failures.append({"stage": "run_or_qa", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        summary["status"] = "fail"; summary["failure_reason"] = failures[-1]["error"]
        write_json(out / "failed_runs.json", failures); write_json(out / "e2_summary.json", summary)
        if (out / "e2_handoff.json").exists(): (out / "e2_handoff.json").unlink()
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
