from __future__ import annotations

import json
from pathlib import Path

import pytest

import pol.workflow.matrix as matrix_module
from pol.workflow.matrix import execute_matrix_run
from pol.workflow.matrix_spec import load_matrix_spec
from pol.workflow.matrix_worker import execute_matrix_cell


ROOT = Path(__file__).resolve().parents[1]


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
