from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from pol.paper1.artifact_contracts import E0ArtifactContract
from pol.paper1.recipes.foundation_validation import run_foundation_validation
from pol.runtime.recipe import RecipeInvocation


ROOT = Path(__file__).resolve().parents[1]


def _refresh_manifest(output: Path) -> None:
    from pol.runtime.artifacts import manifest_records
    from pol.runtime.io import write_strict_json

    names = {
        path.name for path in output.iterdir()
        if path.name != "artifact_manifest.json"
    }
    write_strict_json(
        output / "artifact_manifest.json",
        {
            "schema_version": "paper1-e0-artifact-manifest-v1",
            "recipe_protocol": "paper1-e0-v3",
            "artifacts": manifest_records(output, names),
        },
    )


@pytest.fixture(scope="module")
def valid_e0(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("e0-integrity") / "e0"
    result = run_foundation_validation(
        ROOT / "configs/paper1_e0_smoke.json",
        output,
        overwrite=True,
        invocation=RecipeInvocation(
            repo_root=ROOT,
            working_directory=ROOT,
            command=("test",),
            torch_threads=1,
        ),
    )
    assert result.exit_code == 0
    E0ArtifactContract().validate_complete(output)
    return output


@pytest.mark.parametrize(
    "name",
    [
        "reference_convergence.json",
        "reference_convergence.csv",
        "e0_summary.json",
        "accepted_production_config.json",
        "resolved_config.json",
        "master_initial_conditions.pt",
        "master_manifest.json",
        "artifact_manifest.json",
    ],
)
def test_e0_rejects_byte_tamper(
    valid_e0: Path, tmp_path: Path, name: str
) -> None:
    output = tmp_path / "e0"
    shutil.copytree(valid_e0, output)
    path = output / name
    if name == "artifact_manifest.json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["artifacts"][0]["sha256"] = "0" * 64
        path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        E0ArtifactContract().validate_complete(output)


@pytest.mark.parametrize("mutation", ["missing", "extra", "symlink"])
def test_e0_rejects_tree_tamper(
    valid_e0: Path, tmp_path: Path, mutation: str
) -> None:
    output = tmp_path / "e0"
    shutil.copytree(valid_e0, output)
    if mutation == "missing":
        (output / "model1_identity.json").unlink()
    elif mutation == "extra":
        (output / "extra.json").write_text("{}\n", encoding="utf-8")
    else:
        (output / "model1_identity.json").unlink()
        (output / "model1_identity.json").symlink_to(
            valid_e0 / "model1_identity.json"
        )
    with pytest.raises(ValueError):
        E0ArtifactContract().validate_complete(output)


def test_e0_rejects_semantic_forge_with_recomputed_manifest(
    valid_e0: Path, tmp_path: Path
) -> None:
    """Byte records are necessary but are not the semantic authority."""
    from pol.runtime.artifacts import manifest_records
    from pol.runtime.io import write_strict_json

    output = tmp_path / "e0"
    shutil.copytree(valid_e0, output)
    summary_path = output / "e0_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["selected_reference"]["reference_nx"] += 1
    write_strict_json(summary_path, summary)
    names = {
        path.name
        for path in output.iterdir()
        if path.name != "artifact_manifest.json"
    }
    write_strict_json(
        output / "artifact_manifest.json",
        {
            "schema_version": "paper1-e0-artifact-manifest-v1",
            "recipe_protocol": "paper1-e0-v3",
            "artifacts": manifest_records(output, names),
        },
    )
    with pytest.raises(ValueError, match="selection mismatch"):
        E0ArtifactContract().validate_complete(output)


@pytest.mark.parametrize(
    "name,mutate",
    [
        (
            "input_interface_checks.json",
            lambda value: value["finite_data_interface"].update(status="fail"),
        ),
        (
            "resampling_checks.json",
            lambda value: next(iter(value["checks"].values())).update(
                max_abs_error=123456.0
            ),
        ),
        (
            "reference_convergence.json",
            lambda value: value["rows"][0]["relative_l2"].update(
                mean=123456.0
            ),
        ),
        (
            "model1_identity.json",
            lambda value: value["full_observation"].update(status="fail"),
        ),
    ],
)
def test_e0_rejects_nested_semantic_tamper_after_manifest_recomputed(
    valid_e0: Path, tmp_path: Path, name: str, mutate
) -> None:
    from pol.runtime.io import write_strict_json

    output = tmp_path / "e0"
    shutil.copytree(valid_e0, output)
    path = output / name
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    write_strict_json(path, value)
    _refresh_manifest(output)
    with pytest.raises(ValueError, match="scientific artifact mismatch"):
        E0ArtifactContract().validate_complete(output)


def test_e0_rejects_reference_csv_json_mismatch_after_manifest_recomputed(
    valid_e0: Path, tmp_path: Path
) -> None:
    output = tmp_path / "e0"
    shutil.copytree(valid_e0, output)
    path = output / "reference_convergence.csv"
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace(",pass,", ",fail,", 1), encoding="utf-8")
    _refresh_manifest(output)
    with pytest.raises(ValueError, match="reference_convergence.csv"):
        E0ArtifactContract().validate_complete(output)
