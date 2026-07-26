from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import concurrent.futures
import multiprocessing
import time
import csv

from pol.workflow.types import MatrixCell

import pytest

import pol.workflow.matrix as matrix_module
from pol.paper1.config import canonical_config_json, load_config_json
from pol.paper1.matrix_plugins.e1_resolution import E1ResolutionPlugin
from pol.workflow.matrix import execute_matrix_run
from pol.workflow.matrix_spec import load_matrix_spec
from pol.workflow.matrix_worker import execute_matrix_cell


ROOT = Path(__file__).resolve().parents[1]


def _blocking_worker() -> None:
    time.sleep(60)


def _spec(tmp_path: Path):
    raw = json.loads(
        (
            ROOT / "configs/runs/paper1_e1_resolution_sweep_smoke.json"
        ).read_text()
    )
    raw["run"] = {"name": "unit", "output_root": str(tmp_path)}
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return load_matrix_spec(path, repo_root=ROOT)


def test_failed_e0_starts_no_cell_worker(tmp_path: Path, monkeypatch) -> None:
    spec = _spec(tmp_path)
    monkeypatch.setattr(
        matrix_module,
        "_run_e0",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("simulated E0 failure")
        ),
    )
    monkeypatch.setattr(
        matrix_module.concurrent.futures,
        "ProcessPoolExecutor",
        lambda *args, **kwargs: pytest.fail("cell worker started after E0 failure"),
    )
    assert execute_matrix_run(spec, repo_root=ROOT, force=False) == 1
    manifest = json.loads((spec.run_dir / "matrix_manifest.json").read_text())
    assert manifest["status"] == "fail"
    assert "simulated E0 failure" in manifest["failure"]
    assert all(cell["status"] == "pending" for cell in manifest["cells"])


def test_matrix_worker_failure_is_structured_and_logged(tmp_path: Path) -> None:
    request = {
        "plugin_id": "paper1_e1_resolution_v1",
        "run_index": 2,
        "cell_id": "broken",
        "config_path": str(tmp_path / "missing.json"),
        "e0_dir": str(tmp_path / "e0"),
        "output_dir": str(tmp_path / "output"),
        "log_path": str(tmp_path / "broken.log"),
        "repo_root": str(ROOT),
        "torch_threads": 1,
        "cell_plots": False,
    }
    result = execute_matrix_cell(request)
    assert result["status"] == "fail"
    assert result["failure_type"]
    assert result["failure_message"]
    assert "Traceback" in (tmp_path / "broken.log").read_text()


def test_matrix_fingerprint_mismatch_is_rejected_before_execution(
    tmp_path: Path, monkeypatch
) -> None:
    spec = _spec(tmp_path)
    spec.run_dir.mkdir()
    (spec.run_dir / "matrix_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "paper1-matrix-manifest-v1",
                "run_name": spec.name,
                "run_dir": str(spec.run_dir),
                "matrix_fingerprint": "different",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        matrix_module,
        "_run_e0",
        lambda *args, **kwargs: pytest.fail("E0 started before fingerprint check"),
    )
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        execute_matrix_run(spec, repo_root=ROOT, force=False)


def test_matrix_compute_fingerprint_tracks_threads_not_schedule_or_plots(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    _, cells, _, _ = matrix_module._prepare(spec)
    binding = matrix_module._internal_e0_binding(spec)
    baseline = matrix_module._compute_fingerprint(
        spec, cells, e0_binding=binding
    )
    assert (
        matrix_module._compute_fingerprint(
            replace(spec, jobs=spec.jobs + 1), cells, e0_binding=binding
        )
        == baseline
    )
    assert (
        matrix_module._compute_fingerprint(
            replace(spec, resume=not spec.resume), cells, e0_binding=binding
        )
        == baseline
    )
    assert (
        matrix_module._compute_fingerprint(
            replace(spec, torch_threads_per_job=spec.torch_threads_per_job + 1),
            cells,
            e0_binding=binding,
        )
        != baseline
    )
    assert matrix_module._artifact_contract_fingerprint(
        baseline, cell_plots=False
    ) != matrix_module._artifact_contract_fingerprint(
        baseline, cell_plots=True
    )
    external = {
        "mode": "external",
        "accepted_config_sha256": "a" * 64,
        "master_tensor_hash": "b" * 64,
    }
    same_external = matrix_module._compute_fingerprint(
        spec, cells, e0_binding=dict(external)
    )
    assert same_external == matrix_module._compute_fingerprint(
        spec, cells, e0_binding=dict(external)
    )
    changed_external = dict(external)
    changed_external["master_tensor_hash"] = "c" * 64
    assert same_external != matrix_module._compute_fingerprint(
        spec, cells, e0_binding=changed_external
    )


def test_aggregate_publish_is_exact_and_removes_stale_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "cells").mkdir(parents=True)
    aggregate = run_dir / "aggregate"
    aggregate.mkdir()
    (aggregate / "required.csv").write_text("old\n")
    (aggregate / "stale.csv").write_text("stale\n")

    class Plugin:
        def aggregate_artifact_names(self):
            return ("required.csv",)

        def collect(self, output, cells_dir, cells):
            (output / "required.csv").write_text("new\n")
            return {"rows": 1}

        def validate_aggregate(self, output):
            assert (output / "required.csv").read_text() == "new\n"

    counts, records = matrix_module._collect_and_publish_aggregate(
        run_dir=run_dir, plugin=Plugin(), cells=[]
    )
    assert counts == {"rows": 1}
    assert [item["relative_path"] for item in records] == ["required.csv"]
    assert (aggregate / "required.csv").read_text() == "new\n"
    assert not (aggregate / "stale.csv").exists()
    assert not (run_dir / ".aggregate.staging").exists()
    assert not (run_dir / ".aggregate.backup").exists()


def test_aggregate_failure_preserves_previous_complete_output(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "cells").mkdir(parents=True)
    aggregate = run_dir / "aggregate"
    aggregate.mkdir()
    (aggregate / "required.csv").write_text("complete\n")

    class Plugin:
        def aggregate_artifact_names(self):
            return ("required.csv",)

        def collect(self, output, cells_dir, cells):
            (output / "extra.csv").write_text("unexpected\n")
            raise RuntimeError("collection failed")

        def validate_aggregate(self, output):
            raise AssertionError("must not validate")

    with pytest.raises(RuntimeError, match="collection failed"):
        matrix_module._collect_and_publish_aggregate(
            run_dir=run_dir, plugin=Plugin(), cells=[]
        )
    assert (aggregate / "required.csv").read_text() == "complete\n"
    assert not (run_dir / ".aggregate.staging").exists()
    assert not (run_dir / ".aggregate.backup").exists()


def test_legacy_aggregate_memberships_are_sorted(tmp_path: Path) -> None:
    cells_dir = tmp_path / "cells"
    output = cells_dir / "cell"
    output.mkdir(parents=True)
    tables = {
        "selected_results": ["case_name", "q"],
        "readout_diagnostics": ["case_name", "q"],
        "noise_summary": ["case_name", "q", "noise_level"],
    }
    for name, fields in tables.items():
        with (output / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow({field: 0 for field in fields})
    cell = MatrixCell(
        run_index=0,
        cell_id="cell",
        config_sha256="0" * 64,
        canonical_config="{}",
        human_slug="cell",
        experiment_memberships=("z_experiment", "a_experiment"),
        metadata={
            "n_tar": 1,
            "n_sur": 1,
            "J": 1,
            "full_observation": True,
        },
    )
    aggregate = tmp_path / "aggregate"
    aggregate.mkdir()
    E1ResolutionPlugin().collect(aggregate, cells_dir, [cell])
    with (aggregate / "sweep_selected_results.csv").open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert json.loads(row["experiment_names"]) == [
        "a_experiment",
        "z_experiment",
    ]


def test_force_rejects_non_owned_matrix_directory(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    spec.run_dir.mkdir()
    marker = spec.run_dir / "keep"
    marker.write_text("important", encoding="utf-8")
    with pytest.raises(ValueError, match="not matrix-runner-owned"):
        execute_matrix_run(spec, repo_root=ROOT, force=True)
    assert marker.read_text() == "important"


def test_keyboard_interrupt_cancels_workers_and_records_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    spec = _spec(tmp_path)
    monkeypatch.setattr(matrix_module, "_run_e0", lambda *args, **kwargs: "executed")

    class Future:
        cancelled = False

        def cancel(self):
            self.cancelled = True

    class Pool:
        instances = []

        def __init__(self, *args, **kwargs):
            self.futures = []
            self.shutdown_call = None
            self.__class__.instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, function, request):
            future = Future()
            self.futures.append(future)
            return future

        def shutdown(self, *, wait, cancel_futures):
            self.shutdown_call = (wait, cancel_futures)

    monkeypatch.setattr(
        matrix_module.concurrent.futures, "ProcessPoolExecutor", Pool
    )
    monkeypatch.setattr(
        matrix_module.concurrent.futures,
        "as_completed",
        lambda futures: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    assert execute_matrix_run(spec, repo_root=ROOT, force=False) == 130
    pool = Pool.instances[0]
    assert all(future.cancelled for future in pool.futures)
    assert pool.shutdown_call == (False, True)
    manifest = json.loads((spec.run_dir / "matrix_manifest.json").read_text())
    assert manifest["status"] == "interrupted"


def test_process_pool_shutdown_is_bounded_for_blocking_worker() -> None:
    pool = concurrent.futures.ProcessPoolExecutor(
        max_workers=1, mp_context=multiprocessing.get_context("spawn")
    )
    future = pool.submit(_blocking_worker)
    deadline = time.monotonic() + 5
    while not getattr(pool, "_processes", {}) and time.monotonic() < deadline:
        time.sleep(0.01)
    processes = list(getattr(pool, "_processes", {}).values())
    started = time.monotonic()
    matrix_module._shutdown_process_pool(
        pool, [future], timeout_seconds=0.2
    )
    assert time.monotonic() - started < 3
    assert processes and all(not process.is_alive() for process in processes)


def test_force_removes_only_owned_matrix_run_and_keeps_sibling(
    tmp_path: Path, monkeypatch
) -> None:
    spec = _spec(tmp_path)
    spec.run_dir.mkdir()
    (spec.run_dir / "matrix_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "paper1-matrix-manifest-v1",
                "run_name": spec.name,
                "run_dir": str(spec.run_dir),
            }
        ),
        encoding="utf-8",
    )
    sibling = tmp_path / "sibling"
    sibling.mkdir()
    marker = sibling / "keep"
    marker.write_text("important", encoding="utf-8")
    monkeypatch.setattr(
        matrix_module,
        "_run_e0",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("stop after safe replacement")
        ),
    )
    assert execute_matrix_run(spec, repo_root=ROOT, force=True) == 1
    assert marker.read_text() == "important"


def test_cell_validation_binds_saved_config_to_expected_matrix_cell(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "cell"
    output.mkdir()
    source = ROOT / "configs/paper1_e1_smoke.json"
    (output / "resolved_config.json").write_bytes(source.read_bytes())
    (output / "e1_summary.json").write_text(
        json.dumps({"status": "pass"}), encoding="utf-8"
    )
    (output / "artifact_manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "pol.paper1.e1_qa.validate_plots",
        lambda *args, **kwargs: set(),
    )
    monkeypatch.setattr(
        "pol.paper1.e1_qa.expected_artifacts",
        lambda *args: set(),
    )
    monkeypatch.setattr(
        "pol.paper1.e1_qa.validate_saved_numeric_artifacts",
        lambda *args: None,
    )
    monkeypatch.setattr(
        "pol.paper1.e1_qa.verify_artifact_manifest",
        lambda *args: None,
    )
    monkeypatch.setattr(
        "pol.paper1.e1_qa.validate_artifact_set",
        lambda *args: None,
    )
    expected = matrix_module.hashlib.sha256(
        canonical_config_json(load_config_json(source)).encode("utf-8")
    ).hexdigest()
    E1ResolutionPlugin().validate_cell(
        output,
        cell_plots=False,
        expected_config_sha256=expected,
        expected_config_path=source,
    )
    with pytest.raises(ValueError, match="does not match its matrix cell"):
        E1ResolutionPlugin().validate_cell(
            output,
            cell_plots=False,
            expected_config_sha256="0" * 64,
            expected_config_path=source,
        )
