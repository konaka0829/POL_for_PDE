from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from PIL import Image

from pol.paper1.config import canonical_config_json, load_config_json
from pol.paper1.datasets import load_master_dataset
from pol.paper1.regression_baseline import (
    build_e0_scientific_record,
    build_e1_scientific_record,
    build_e2_scientific_record,
)
from pol.paper1.scientific_comparison import (
    assert_scientific_record_matches,
    policy_from_baseline,
)


ROOT = Path(__file__).resolve().parents[1]
PHASE1_BASELINE = json.loads(
    (ROOT / "tests/fixtures/paper1_phase1_scientific_baseline_v3.json").read_text(
        encoding="utf-8"
    )
)
_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _thread_env(threads: int = 1) -> dict[str, str]:
    env = os.environ.copy()
    for name in _THREAD_ENV_VARS:
        env[name] = str(threads)
    return env


def _run(arguments: list[str], *, env: dict[str, str]) -> None:
    subprocess.run(
        [sys.executable, *arguments],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        timeout=600,
    )


def _run_result(arguments: list[str], *, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *arguments],
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        timeout=600,
    )


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _runner(tmp_path: Path, kind: str, *, env: dict[str, str]) -> Path:
    raw = _json(ROOT / f"configs/runs/paper1_{kind}_smoke.json")
    raw["run"]["output_root"] = str(tmp_path / "runner")
    source = tmp_path / f"{kind}.json"
    source.write_text(json.dumps(raw), encoding="utf-8")
    _run(["-m", "pol", "run", str(source)], env=env)
    run_dir = tmp_path / "runner" / raw["run"]["name"]
    assert _json(run_dir / "run_manifest.json")["status"] == "pass"
    return run_dir


def _direct_e0(config: Path, output: Path, *, env: dict[str, str]) -> None:
    _run(
        [
            "scripts/paper1/run_e0.py",
            "--config",
            str(config),
            "--output-dir",
            str(output),
            "--overwrite",
        ],
        env=env,
    )


def _csv_without(path: Path, excluded: set[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {key: value for key, value in row.items() if key not in excluded}
            for row in csv.DictReader(handle)
        ]


def _drop(value, excluded: set[str]):
    if isinstance(value, dict):
        return {
            key: _drop(item, excluded)
            for key, item in value.items()
            if key not in excluded
        }
    if isinstance(value, list):
        return [_drop(item, excluded) for item in value]
    return value


def _without_inline_plot_contract(summary: dict) -> dict:
    normalized = json.loads(json.dumps(summary))
    checks = normalized.get("required_checks", {})
    for name in (
        "all_required_artifacts_present",
        "artifact_manifest_verified",
        "plots_completed_or_explicitly_skipped",
    ):
        checks.pop(name, None)
    return normalized


def _assert_png_pixels_equal(first: Path, second: Path) -> None:
    left = Image.open(first).convert("RGBA")
    right = Image.open(second).convert("RGBA")
    assert left.size == right.size
    assert left.tobytes() == right.tobytes()


@pytest.mark.slow
def test_e0_direct_and_runner_smoke_parity(tmp_path: Path) -> None:
    env = _thread_env(1)
    direct = tmp_path / "direct/e0"
    _direct_e0(ROOT / "configs/paper1_e0_smoke.json", direct, env=env)
    runner = _runner(tmp_path, "e0", env=env) / "e0"

    assert _json(direct / "e0_summary.json") == _json(runner / "e0_summary.json")
    direct_config = load_config_json(direct / "accepted_production_config.json")
    runner_config = load_config_json(runner / "accepted_production_config.json")
    assert canonical_config_json(direct_config) == canonical_config_json(runner_config)
    direct_master_manifest = _json(direct / "master_manifest.json")
    runner_master_manifest = _json(runner / "master_manifest.json")
    assert direct_master_manifest["tensor_hash"] == runner_master_manifest["tensor_hash"]
    assert_scientific_record_matches(
        build_e0_scientific_record(runner),
        PHASE1_BASELINE["e0"],
        policy_from_baseline(PHASE1_BASELINE, section="e0"),
    )


@pytest.mark.slow
def test_e1_direct_and_runner_smoke_parity(tmp_path: Path) -> None:
    env = _thread_env(1)
    direct = tmp_path / "direct"
    _direct_e0(
        ROOT / "configs/paper1_e0_for_e1_smoke.json",
        direct / "e0",
        env=env,
    )
    _run(
        [
            "scripts/paper1/run_e1.py",
            "--config",
            str(ROOT / "configs/paper1_e1_smoke.json"),
            "--e0-dir",
            str(direct / "e0"),
            "--output-dir",
            str(direct / "e1"),
            "--overwrite",
            "--torch-threads",
            "1",
        ],
        env=env,
    )
    runner = _runner(tmp_path, "e1", env=env)

    assert _without_inline_plot_contract(
        _json(direct / "e1/e1_summary.json")
    ) == _without_inline_plot_contract(_json(runner / "e1/e1_summary.json"))
    for name in (
        "ridge_selection.csv",
        "selected_results.csv",
        "readout_diagnostics.csv",
        "mode_comparison.csv",
        "noise_results.csv",
        "noise_summary.csv",
    ):
        assert (direct / "e1" / name).read_bytes() == (
            runner / "e1" / name
        ).read_bytes()
    direct_record = build_e1_scientific_record(direct / "e1")
    assert_scientific_record_matches(
        direct_record,
        PHASE1_BASELINE["e1"],
        policy_from_baseline(PHASE1_BASELINE, section="e1"),
    )
    runner_record = build_e1_scientific_record(
        runner / "e1",
        plot_dir=runner / "figures/paper1.e1.standard.v1",
    )
    assert _without_inline_plot_contract(
        runner_record["summary"]
    ) == _without_inline_plot_contract(direct_record["summary"])
    assert {
        key: value for key, value in runner_record.items() if key != "summary"
    } == {key: value for key, value in direct_record.items() if key != "summary"}
    for path in (direct / "e1").glob("*.png"):
        _assert_png_pixels_equal(
            path,
            runner / "figures/paper1.e1.standard.v1" / path.name,
        )


@pytest.mark.slow
def test_e2_direct_and_runner_smoke_parity(tmp_path: Path) -> None:
    env = _thread_env(1)
    direct = tmp_path / "direct"
    _direct_e0(ROOT / "configs/paper1_e0_smoke.json", direct / "e0", env=env)
    _run(
        [
            "scripts/paper1/generate_master_dataset.py",
            "--config",
            str(direct / "e0/accepted_production_config.json"),
            "--master-initial-conditions",
            str(direct / "e0/master_initial_conditions.pt"),
            "--output-dir",
            str(direct / "master_dataset"),
            "--overwrite",
        ],
        env=env,
    )
    _run(
        [
            "scripts/paper1/run_e2.py",
            "--config",
            str(ROOT / "configs/paper1_e2_smoke.json"),
            "--e0-dir",
            str(direct / "e0"),
            "--dataset-dir",
            str(direct / "master_dataset"),
            "--output-dir",
            str(direct / "e2"),
            "--overwrite",
            "--torch-threads",
            "1",
            "--batch-size",
            "64",
        ],
        env=env,
    )
    runner = _runner(tmp_path, "e2", env=env)

    direct_master_manifest = _json(direct / "e0/master_manifest.json")
    runner_master_manifest = _json(runner / "e0/master_manifest.json")
    assert direct_master_manifest["tensor_hash"] == runner_master_manifest["tensor_hash"]

    first = load_master_dataset(direct / "master_dataset")
    second = load_master_dataset(runner / "master_dataset")
    for key in ("dataset_hash", "split_hash", "tensor_hashes"):
        assert first.metadata[key] == second.metadata[key]
    for name in (
        "sample_ids",
        "train_indices",
        "val_indices",
        "test_indices",
        "u0_master",
        "y_target_master",
    ):
        assert torch.equal(getattr(first, name), getattr(second, name))

    direct_out, runner_out = direct / "e2", runner / "e2"
    for name in ("model_specific_optima.json", "shared_representatives.json"):
        assert _json(direct_out / name) == _json(runner_out / name)
    provenance = {
        "selection_record_hash",
        "frozen_plan_hash",
        "attempt_history_hash",
        "e0_prerequisite_hash",
        "dataset_prerequisite_hash",
    }
    for name in ("coordinate_history.json", "convergence_summary.json"):
        assert _drop(_json(direct_out / name), provenance) == _drop(
            _json(runner_out / name), provenance
        )
    assert {
        key: value
        for key, value in _json(direct_out / "selection_record.json").items()
        if key != "bindings"
    } == {
        key: value
        for key, value in _json(runner_out / "selection_record.json").items()
        if key != "bindings"
    }
    for name in (
        "validation_sweep.csv",
        "model3_validation_by_seed.csv",
        "convergence_results.csv",
        "solver_metadata.csv",
        "physical_point_aliases.csv",
    ):
        assert (direct_out / name).read_bytes() == (runner_out / name).read_bytes()
    for name in (
        "test_sweep.csv",
        "model3_test_by_seed.csv",
        "model3_test_aggregate.csv",
    ):
        assert _csv_without(direct_out / name, provenance) == _csv_without(
            runner_out / name, provenance
        )
    direct_summary = _json(direct_out / "e2_summary.json")
    runner_summary = _json(runner_out / "e2_summary.json")
    direct_summary["required_checks"]["test_rows_bound_to_frozen_plan"].pop("value")
    runner_summary["required_checks"]["test_rows_bound_to_frozen_plan"].pop("value")
    assert _drop(direct_summary, provenance) == _drop(runner_summary, provenance)
    direct_events = [
        row["event"] for row in _json(direct_out / "event_log.json")["events"]
    ]
    runner_events = [
        row["event"] for row in _json(runner_out / "event_log.json")["events"]
    ]
    assert direct_events == runner_events
    assert_scientific_record_matches(
        build_e2_scientific_record(
            runner_out,
            plot_dir=runner / "figures/paper1.e2.standard.v1",
        ),
        PHASE1_BASELINE["e2"],
        policy_from_baseline(PHASE1_BASELINE, section="e2"),
    )
    for path in direct_out.glob("*.png"):
        _assert_png_pixels_equal(
            path,
            runner / "figures/paper1.e2.standard.v1" / path.name,
        )
    for names in (direct_events, runner_events):
        assert names.index("freeze_read_back") < names.index("first_test_state_solve")
        assert names.index("first_test_state_solve") < names.index(
            "first_test_metric"
        )


@pytest.mark.slow
def test_scalar_e1_tamper_is_never_silently_reused(tmp_path: Path) -> None:
    env = _thread_env(1)
    run_dir = _runner(tmp_path, "e1", env=env)
    artifact = run_dir / "e1/selected_results.csv"
    artifact.write_bytes(artifact.read_bytes() + b"TAMPER,1\n")
    result = _run_result(["-m", "pol", "run", str(tmp_path / "e1.json")], env=env)
    assert result.returncode != 0
    assert b"TAMPER,1\n" in artifact.read_bytes()
    assert "error:" in result.stdout.lower()


@pytest.mark.slow
def test_scalar_e2_tamper_is_never_silently_reused(tmp_path: Path) -> None:
    env = _thread_env(1)
    run_dir = _runner(tmp_path, "e2", env=env)
    artifact = run_dir / "e2/test_sweep.csv"
    artifact.write_bytes(artifact.read_bytes() + b"TAMPER,1\n")
    result = _run_result(["-m", "pol", "run", str(tmp_path / "e2.json")], env=env)
    assert result.returncode != 0
    assert b"TAMPER,1\n" in artifact.read_bytes()
    assert "integrity" in result.stdout.lower() or "artifact" in result.stdout.lower()
