from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from pol.paper1.config import canonical_config_json, load_config_json
from pol.paper1.datasets import load_master_dataset


ROOT = Path(__file__).resolve().parents[1]


def _run(arguments: list[str]) -> None:
    subprocess.run(
        [sys.executable, *arguments], cwd=ROOT, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _runner(tmp_path: Path, kind: str) -> Path:
    raw = _json(ROOT / f"configs/runs/paper1_{kind}_smoke.json")
    raw["run"]["output_root"] = str(tmp_path / "runner")
    source = tmp_path / f"{kind}.json"
    source.write_text(json.dumps(raw), encoding="utf-8")
    _run(["-m", "pol", "run", str(source)])
    return tmp_path / "runner" / raw["run"]["name"]


def _direct_e0(config: Path, output: Path) -> None:
    _run([
        "scripts/paper1/run_e0.py", "--config", str(config),
        "--output-dir", str(output), "--overwrite",
    ])


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


@pytest.mark.slow
def test_direct_and_runner_smoke_scientific_parity(tmp_path: Path) -> None:
    direct = tmp_path / "direct"

    # E2 direct chain and unified chain.
    direct_e2 = direct / "e2"
    _direct_e0(ROOT / "configs/paper1_e0_smoke.json", direct_e2 / "e0")
    _run([
        "scripts/paper1/generate_master_dataset.py",
        "--config", str(direct_e2 / "e0/accepted_production_config.json"),
        "--master-initial-conditions", str(direct_e2 / "e0/master_initial_conditions.pt"),
        "--output-dir", str(direct_e2 / "master_dataset"), "--overwrite",
    ])
    _run([
        "scripts/paper1/run_e2.py",
        "--config", str(ROOT / "configs/paper1_e2_smoke.json"),
        "--e0-dir", str(direct_e2 / "e0"),
        "--dataset-dir", str(direct_e2 / "master_dataset"),
        "--output-dir", str(direct_e2 / "e2"), "--overwrite",
        "--torch-threads", "1", "--batch-size", "64",
    ])
    runner_e2 = _runner(tmp_path, "e2")

    assert _json(direct_e2 / "e0/e0_summary.json") == _json(
        runner_e2 / "e0/e0_summary.json"
    )
    direct_config = load_config_json(direct_e2 / "e0/accepted_production_config.json")
    runner_config = load_config_json(runner_e2 / "e0/accepted_production_config.json")
    assert canonical_config_json(direct_config) == canonical_config_json(runner_config)
    assert _json(direct_e2 / "e0/master_manifest.json")["tensor_hash"] == _json(
        runner_e2 / "e0/master_manifest.json"
    )["tensor_hash"]

    first = load_master_dataset(direct_e2 / "master_dataset")
    second = load_master_dataset(runner_e2 / "master_dataset")
    for key in ("dataset_hash", "split_hash", "tensor_hashes"):
        assert first.metadata[key] == second.metadata[key]
    for name in (
        "sample_ids", "train_indices", "val_indices", "test_indices",
        "u0_master", "y_target_master",
    ):
        assert torch.equal(getattr(first, name), getattr(second, name))

    direct_out, runner_out = direct_e2 / "e2", runner_e2 / "e2"
    for name in ("model_specific_optima.json", "shared_representatives.json"):
        assert _json(direct_out / name) == _json(runner_out / name)
    provenance = {
        "selection_record_hash", "frozen_plan_hash", "attempt_history_hash",
        "e0_prerequisite_hash", "dataset_prerequisite_hash",
    }
    for name in ("coordinate_history.json", "convergence_summary.json"):
        assert _drop(_json(direct_out / name), provenance) == _drop(
            _json(runner_out / name), provenance
        )
    assert {
        key: value for key, value in _json(direct_out / "selection_record.json").items()
        if key != "bindings"
    } == {
        key: value for key, value in _json(runner_out / "selection_record.json").items()
        if key != "bindings"
    }
    for name in (
        "validation_sweep.csv", "model3_validation_by_seed.csv",
        "convergence_results.csv", "solver_metadata.csv", "physical_point_aliases.csv",
    ):
        assert (direct_out / name).read_bytes() == (runner_out / name).read_bytes()
    for name in ("test_sweep.csv", "model3_test_by_seed.csv", "model3_test_aggregate.csv"):
        assert _csv_without(direct_out / name, provenance) == _csv_without(
            runner_out / name, provenance
        )
    direct_summary = _json(direct_out / "e2_summary.json")
    runner_summary = _json(runner_out / "e2_summary.json")
    direct_summary["required_checks"]["test_rows_bound_to_frozen_plan"].pop("value")
    runner_summary["required_checks"]["test_rows_bound_to_frozen_plan"].pop("value")
    assert _drop(direct_summary, provenance) == _drop(runner_summary, provenance)
    direct_events = [row["event"] for row in _json(direct_out / "event_log.json")["events"]]
    runner_events = [row["event"] for row in _json(runner_out / "event_log.json")["events"]]
    assert direct_events == runner_events
    for names in (direct_events, runner_events):
        assert names.index("freeze_read_back") < names.index("first_test_state_solve")
        assert names.index("first_test_state_solve") < names.index("first_test_metric")

    # E1 direct chain and unified chain.
    direct_e1 = direct / "e1"
    _direct_e0(ROOT / "configs/paper1_e0_for_e1_smoke.json", direct_e1 / "e0")
    _run([
        "scripts/paper1/run_e1.py",
        "--config", str(ROOT / "configs/paper1_e1_smoke.json"),
        "--e0-dir", str(direct_e1 / "e0"), "--output-dir", str(direct_e1 / "e1"),
        "--overwrite", "--torch-threads", "1",
    ])
    runner_e1 = _runner(tmp_path, "e1")
    assert _json(direct_e1 / "e1/e1_summary.json") == _json(
        runner_e1 / "e1/e1_summary.json"
    )
    for name in (
        "ridge_selection.csv", "selected_results.csv", "readout_diagnostics.csv",
        "mode_comparison.csv", "noise_results.csv", "noise_summary.csv",
    ):
        assert (direct_e1 / "e1" / name).read_bytes() == (
            runner_e1 / "e1" / name
        ).read_bytes()
