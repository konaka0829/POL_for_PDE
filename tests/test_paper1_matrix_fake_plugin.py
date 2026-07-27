from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest
from pol.workflow.matrix import execute_matrix_run
from pol.workflow.matrix_spec import load_matrix_spec
from pol.workflow.registry import register_matrix_plugin


def test_prerequisite_free_importable_plugin_runs_in_spawn_and_repairs_one_cell(
    tmp_path: Path,
) -> None:
    plugin_id = "test_fake_matrix_v1"
    register_matrix_plugin(
        plugin_id, "tests.paper1_fake_matrix_plugin:FakeMatrixPlugin",
        replace=True,
    )
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"science": {"value": 0}}))
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps({
        "schema_version": "paper1-matrix-run-v1",
        "run": {"name": "fake", "output_root": str(tmp_path)},
        "experiment": {"kind": "test_fake", "base_config": str(base)},
        "prerequisites": {},
        "matrix": {
            "invalid_run_policy": "error",
            "experiments": [{
                "name": "values", "fixed": {}, "grid": [{
                    "path": "science.value", "values": [1, 2, 3],
                }], "copy": {},
            }],
            "explicit_runs": [],
        },
        "execution": {
            "jobs": 2, "torch_threads_per_job": 1,
            "resume": True, "cell_plots": False,
        },
        "aggregation": {"kind": plugin_id},
    }))
    spec = load_matrix_spec(spec_path, repo_root=Path.cwd())
    assert spec.dependencies == ()
    assert execute_matrix_run(spec, repo_root=Path.cwd(), force=False) == 0
    manifest = json.loads((tmp_path / "fake" / "matrix_manifest.json").read_text())
    assert manifest["dependency_identities"] == []
    assert manifest["compute_status"] == "pass"
    payload = manifest["compute_fingerprint_payload"]
    assert payload["recipe_protocols"] == ["test-fake-recipe-v3"]
    assert payload["aggregation_protocol"] == "test-fake-matrix-v7"
    assert manifest["cells"][0]["executed_or_reused"] == "executed"

    # Formatting and key order are provenance changes, not science changes.
    base.write_text(
        '{\n  "science": {\n    "value": 0\n  }\n}\n',
        encoding="utf-8",
    )
    formatted_spec = load_matrix_spec(spec_path, repo_root=Path.cwd())
    assert execute_matrix_run(
        formatted_spec, repo_root=Path.cwd(), force=False
    ) == 0
    formatted = json.loads(
        (tmp_path / "fake/matrix_manifest.json").read_text()
    )
    assert all(
        item["executed_or_reused"] == "reused"
        for item in formatted["cells"]
    )

    # A scientific edit is rejected before any managed output is changed.
    base.write_text(json.dumps({"science": {"value": 99}}), encoding="utf-8")
    changed_spec = load_matrix_spec(spec_path, repo_root=Path.cwd())
    run_dir = tmp_path / "fake"
    before = {
        str(path.relative_to(run_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        execute_matrix_run(changed_spec, repo_root=Path.cwd(), force=False)
    after = {
        str(path.relative_to(run_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    assert after == before
    base.write_text(json.dumps({"science": {"value": 0}}), encoding="utf-8")
    spec = load_matrix_spec(spec_path, repo_root=Path.cwd())

    cell = tmp_path / "fake" / "cells" / manifest["cells"][1]["cell_id"]
    (cell / "result.json").write_text('{"tampered":true}\n')
    assert execute_matrix_run(spec, repo_root=Path.cwd(), force=False) == 0
    repaired = json.loads((tmp_path / "fake" / "matrix_manifest.json").read_text())
    dispositions = [item["executed_or_reused"] for item in repaired["cells"]]
    assert dispositions.count("executed") == 1
    assert dispositions.count("reused") == 2
