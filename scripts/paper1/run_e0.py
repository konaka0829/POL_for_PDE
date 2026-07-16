#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pol.paper1.config import load_config_json, save_config_json
from pol.paper1.e0 import run_algebraic_checks, run_interface_checks, run_model1_checks, run_reference_convergence, save_master_initial_conditions
from pol.paper1.grids import spectral_resample_periodic
from pol.paper1.initial_conditions import build_master_grf_initial_conditions
from pol.paper1.solvers import solve_burgers_final_state


ARTIFACTS = ("e0_summary.json", "reference_convergence.csv", "reference_convergence.json", "resampling_checks.json", "input_interface_checks.json", "model1_identity.json", "master_initial_conditions.pt", "master_manifest.json", "resolved_config.json", "environment.json")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


import math


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _git(args: list[str]) -> str:
    try:
        p = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
        return p.stdout.strip() if p.returncode == 0 else "unknown"
    except OSError:
        return "unknown"


def _preflight(out: Path, overwrite: bool) -> None:
    existing = [name for name in ARTIFACTS if (out / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{out} already contains E0 artifacts ({', '.join(existing)}); pass --overwrite")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the current Paper 1 E0 acceptance gate")
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    out, config_path = Path(args.output_dir), Path(args.config)
    try:
        _preflight(out, args.overwrite)
        config = load_config_json(config_path)
        if config.e0 is None:
            raise ValueError("config must contain an e0 section")
    except Exception as exc:
        parser.error(str(exc))
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    command = [sys.executable, str(Path(__file__).relative_to(REPO_ROOT)), *sys.argv[1:]]
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    summary: dict[str, Any] = {"schema_version": "paper1-e0-v1", "status": "fail", "required_checks": {}, "selected_reference": {"reference_nx": None, "solver": None, "requested_dt": None, "requested_fine_dt": None, "effective_inner_step": None}, "num_failures": 0, "failure_reasons": []}
    master_manifest: dict[str, Any] = {}
    try:
        master = build_master_grf_initial_conditions(config)
        master_manifest = save_master_initial_conditions(master, out / "master_initial_conditions.pt", out / "master_manifest.json", config)
        resampling, projector = run_algebraic_checks(config)
        resampling["fourier_projector"] = projector
        _write_json(out / "resampling_checks.json", resampling)
        convergence = run_reference_convergence(config, master)
        _write_json(out / "reference_convergence.json", convergence)
        with (out / "reference_convergence.csv").open("w", newline="", encoding="utf-8") as f:
            fields = ["kind", "candidate_nx", "solver", "requested_dt", "requested_fine_dt", "outer_steps", "substeps_per_outer", "effective_inner_step", "relative_l2_mean", "relative_l2_median", "relative_l2_max", "absolute_l2_mean", "low_mode_relative_l2_mean", "master_hash", "sample_ids"]
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
            for row in convergence["rows"]:
                w.writerow({"kind": row["kind"], "candidate_nx": row["candidate_nx"], "solver": row["solver"], "requested_dt": row["requested_dt"], "requested_fine_dt": row["requested_fine_dt"], "outer_steps": row["outer_steps"], "substeps_per_outer": row["substeps_per_outer"], "effective_inner_step": row["effective_inner_step"], "relative_l2_mean": row["relative_l2"]["mean"], "relative_l2_median": row["relative_l2"]["median"], "relative_l2_max": row["relative_l2"]["max"], "absolute_l2_mean": row["absolute_l2"]["mean"], "low_mode_relative_l2_mean": row["low_mode_relative_l2"]["mean"], "master_hash": row["master_hash"], "sample_ids": json.dumps(row["sample_ids"])})
        chosen_time = convergence.get("selected_temporal")
        chosen_space = convergence.get("selected_spatial")
        finest = config.e0.reference_nx_candidates[-1]
        u0 = spectral_resample_periodic(master.values_master[:2], finest, domain_length=config.domain.length)
        ref_result = solve_burgers_final_state(u0, nu=config.target.nu, T=config.target.T, dt=config.e0.time_candidates[-1].dt, fine_dt=config.e0.time_candidates[-1].fine_dt, solver=config.target.solver, dealias=config.target.dealias, domain_length=config.domain.length)
        interfaces = run_interface_checks(config, master, ref_result.values)
        _write_json(out / "input_interface_checks.json", interfaces)
        model1 = run_model1_checks(config, master)
        _write_json(out / "model1_identity.json", model1)
        required = {"resampling": resampling["status"], "fourier_projector": projector["status"], "reference_spatial_convergence": convergence["spatial_status"], "reference_temporal_convergence": convergence["temporal_status"], "finite_data_interface": interfaces["finite_data_interface"]["status"], "no_high_frequency_leak": interfaces["no_high_frequency_leak"]["status"], "model1_full_observation_identity": model1["full_observation"]["status"], "model1_bandlimited_reduced_j": model1["bandlimited_reduced_j"]["status"]}
        summary["required_checks"] = required
        failures = [name for name, status in required.items() if status != "pass"]
        summary["failure_reasons"] = [f"required check failed: {name}" for name in failures]
        summary["num_failures"] = len(failures)
        summary["status"] = "pass" if not failures else "fail"
        if chosen_space and chosen_time:
            summary["selected_reference"] = {"reference_nx": chosen_space["candidate_nx"], "solver": chosen_time["solver"], "requested_dt": chosen_time["requested_dt"], "requested_fine_dt": chosen_time["requested_fine_dt"], "effective_inner_step": chosen_time["effective_inner_step"]}
    except Exception as exc:
        summary["failure_reasons"].append(f"{type(exc).__name__}: {exc}")
        summary["num_failures"] = len(summary["failure_reasons"])
    finally:
        save_config_json(config, out / "resolved_config.json")
        environment = {"full_command": command, "cwd": os.getcwd(), "git_commit": _git(["rev-parse", "HEAD"]), "git_dirty_status": _git(["status", "--porcelain"]), "python_version": platform.python_version(), "torch_version": torch.__version__, "platform": platform.platform(), "device": config.data.device, "dtype": config.data.dtype, "cuda_available": torch.cuda.is_available(), "config_path": str(config_path), "config_hash": config_hash, "master_archive_hash": master_manifest.get("tensor_hash"), "runtime_seconds": time.perf_counter() - start}
        _write_json(out / "environment.json", environment)
        _write_json(out / "e0_summary.json", summary)
    print(json.dumps(_json_safe(summary), sort_keys=True, allow_nan=False))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
