import json
import shutil
import subprocess
import sys

import pytest

from pol.paper1.config import load_config_json
from pol.paper1.e1 import validate_e0_prerequisite


@pytest.fixture
def passing_e0(tmp_path):
    source = tmp_path / "e0"
    process = subprocess.run(
        [sys.executable, "tests/paper1_recipe_driver.py", "e0", "--config",
         "configs/paper1_e0_for_e1_smoke.json", "--output-dir", str(source),
         "--overwrite"],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    return source


def test_gate_rejects_manifest_hash(passing_e0):
    manifest_path = passing_e0 / "master_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["tensor_hash"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="hash"):
        validate_e0_prerequisite(passing_e0, load_config_json("configs/paper1_e1_smoke.json"))


def test_gate_rejects_missing_required_check(passing_e0, tmp_path):
    tampered = tmp_path / "missing-check"
    shutil.copytree(passing_e0, tampered)
    summary_path = tampered / "e0_summary.json"
    summary = json.loads(summary_path.read_text())
    del summary["required_checks"]["resampling"]
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="missing required check: resampling"):
        validate_e0_prerequisite(tampered, load_config_json("configs/paper1_e1_smoke.json"))


@pytest.mark.parametrize("artifact,mutation,match", [
    ("accepted_production_config.json", lambda doc: doc["data"].__setitem__("seed", 999), "config mismatch"),
    ("reference_convergence.json", lambda doc: doc.__setitem__("joint_status", "fail"), "joint status"),
    ("resampling_checks.json", lambda doc: doc.__setitem__("schema_version", "tampered"), "schema_version"),
    ("accepted_production_config.json", lambda doc: doc["spatial"].__setitem__("reference_nx", 64), "resolution artifacts disagree"),
    ("reference_convergence.json", lambda doc: doc["joint_row"].__setitem__("master_hash", "0"*64), "master/sample IDs"),
])
def test_gate_rejects_auxiliary_artifact_tampering(passing_e0, tmp_path, artifact, mutation, match):
    tampered=tmp_path/(artifact+match.replace(" ","-"))
    shutil.copytree(passing_e0,tampered)
    path=tampered/artifact; document=json.loads(path.read_text()); mutation(document); path.write_text(json.dumps(document))
    with pytest.raises(ValueError,match=match):
        validate_e0_prerequisite(tampered,load_config_json("configs/paper1_e1_smoke.json"))
