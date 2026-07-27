import csv
import json
import shutil
import subprocess
import sys

import pytest
import torch

from pol.paper1.config import load_config_json
from pol.paper1.e1_qa import (
    expected_artifacts,
    read_and_scan_csv,
    scan_json_file,
    scan_models,
    scientific_acceptance_checks,
    validate_artifact_set,
    validate_plots,
    validate_saved_numeric_artifacts,
    validate_table_keys,
    verify_artifact_manifest,
)


@pytest.fixture(scope="module")
def passing_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("e1-qa")
    e0, e1 = root / "e0", root / "e1"
    first = subprocess.run(
        [sys.executable, "tests/paper1_recipe_driver.py", "e0", "--config",
         "configs/paper1_e0_for_e1_smoke.json", "--output-dir", str(e0), "--overwrite"],
        capture_output=True, text=True,
    )
    assert first.returncode == 0, first.stdout + first.stderr
    second = subprocess.run(
        [sys.executable, "tests/paper1_recipe_driver.py", "e1", "--config", "configs/paper1_e1_smoke.json",
         "--e0-dir", str(e0), "--output-dir", str(e1), "--overwrite", "--skip-plots",
         "--torch-threads", "1"],
        capture_output=True, text=True,
    )
    assert second.returncode == 0, second.stdout + second.stderr
    return e1


def copied_run(passing_run, tmp_path):
    target = tmp_path / "run"
    shutil.copytree(passing_run, target)
    return target


def test_csv_json_and_pt_nonfinite_faults_fail(passing_run, tmp_path):
    csv_run = copied_run(passing_run, tmp_path / "csv")
    path = csv_run / "noise_summary.csv"
    rows = list(csv.DictReader(path.open()))
    rows[0]["output_perturbation_rms_mean"] = "inf"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with pytest.raises(ValueError, match="non-finite CSV"):
        read_and_scan_csv(path)

    json_path = tmp_path / "bad.json"
    json_path.write_text('{"value": Infinity}')
    with pytest.raises(ValueError, match="non-finite JSON"):
        scan_json_file(json_path)

    pt_run = copied_run(passing_run, tmp_path / "pt")
    pt_path = pt_run / "selected_models.pt"
    payload = torch.load(pt_path, weights_only=False)
    next(iter(payload["models"].values()))["W"][0, 0] = float("inf")
    torch.save(payload, pt_path)
    with pytest.raises(ValueError, match="model tensor invalid"):
        scan_models(pt_path, load_config_json("configs/paper1_e1_smoke.json"))


def test_missing_extra_and_manifest_tampering_fail(passing_run, tmp_path):
    run = copied_run(passing_run, tmp_path)
    expected = expected_artifacts(set())
    (run / "noise_summary.csv").unlink()
    with pytest.raises(ValueError, match="artifact set mismatch"):
        validate_artifact_set(run, expected)
    shutil.copytree(passing_run, tmp_path / "extra")
    extra = tmp_path / "extra"; (extra / "stale.txt").write_text("stale")
    with pytest.raises(ValueError, match="artifact set mismatch"):
        validate_artifact_set(extra, expected)
    manifest = json.loads((passing_run / "artifact_manifest.json").read_text())
    manifest[0]["byte_size"] += 1
    altered = copied_run(passing_run, tmp_path / "manifest")
    (altered / "artifact_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="hash/size mismatch"):
        verify_artifact_manifest(altered, expected)


def test_plot_faults_fail(tmp_path):
    out = tmp_path / "plots"; out.mkdir()
    (out / "plot_manifest.json").write_text(json.dumps({"status":"pass","plots":[{"relative_path":"never_created.png","status":"created"}]}))
    with pytest.raises(ValueError, match="missing or empty"):
        validate_plots(out, skip_plots=False)
    (out / "never_created.png").write_bytes(b"")
    with pytest.raises(ValueError, match="missing or empty"):
        validate_plots(out, skip_plots=False)
    (out / "plot_manifest.json").write_text(json.dumps({"status":"skipped","plots":[]}))
    with pytest.raises(ValueError, match="inconsistent"):
        validate_plots(out, skip_plots=True)


def test_duplicate_selected_key_fails(passing_run, tmp_path):
    run = copied_run(passing_run, tmp_path)
    path = run / "selected_results.csv"
    rows = list(csv.DictReader(path.open()))
    rows[-1]["case_name"], rows[-1]["q"] = rows[0]["case_name"], rows[0]["q"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with pytest.raises(ValueError, match="duplicate selected_results"):
        validate_saved_numeric_artifacts(run, load_config_json("configs/paper1_e1_smoke.json"))


@pytest.mark.parametrize("table,column,value,check_name", [
    ("readout_diagnostics.csv", "max_identifiable_diagonal_relative_error", "2.0", "max_identifiable_diagonal_relative_error"),
    ("readout_diagnostics.csv", "identifiable_off_diagonal_relative_norm", "2.0", "identifiable_off_diagonal_relative_norm"),
    ("selected_results.csv", "field_error_to_representation_floor_ratio", "20.0", "field_error_to_representation_floor_ratio"),
    ("readout_diagnostics.csv", "learned_operator_norm", "0.0", "stable_unstable_operator_direction"),
])
def test_scientific_faults_fail(passing_run, table, column, value, check_name):
    config=load_config_json("configs/paper1_e1_smoke.json")
    qa=validate_saved_numeric_artifacts(passing_run,config)
    rows=[dict(row) for row in qa["tables"][table]]
    if check_name=="stable_unstable_operator_direction":
        row=next(row for row in rows if row["regime"]=="unstable")
    else:
        row=rows[0]
    row[column]=value
    tables=dict(qa["tables"]); tables[table]=rows
    assert scientific_acceptance_checks(tables,config)[check_name]["status"]=="fail"


@pytest.mark.parametrize("table,match", [
    ("ridge_selection.csv", "ridge_selection"),
    ("mode_comparison.csv", "mode_comparison"),
    ("noise_results.csv", "noise_results"),
])
def test_duplicate_and_missing_cartesian_keys_fail(passing_run, table, match):
    config=load_config_json("configs/paper1_e1_smoke.json")
    tables=validate_saved_numeric_artifacts(passing_run,config)["tables"]
    altered={name:[dict(row) for row in rows] for name,rows in tables.items()}
    altered[table][-1]=dict(altered[table][0])
    with pytest.raises(ValueError,match=match):
        validate_table_keys(altered,config)


def test_noise_summary_repeat_shortfall_fails(passing_run):
    config=load_config_json("configs/paper1_e1_smoke.json")
    tables=validate_saved_numeric_artifacts(passing_run,config)["tables"]
    altered={name:[dict(row) for row in rows] for name,rows in tables.items()}
    altered["noise_summary.csv"][0]["repeats"]="1"
    with pytest.raises(ValueError,match="repeats mismatch"):
        validate_table_keys(altered,config)
