from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest
import torch

from pol.paper1.regression_baseline import (
    BASELINE_SCHEMA_VERSION,
    build_phase1_scientific_baseline,
    canonical_float,
    canonicalize_scientific,
    semantic_tensor_digest,
    write_phase1_scientific_baseline,
    build_e0_scientific_record,
    _plot_inventory,
    _verify_artifact_manifest,
)
from pol.paper1.datasets import tensor_hash
from pol.paper1.scientific_comparison import (
    NumericTolerance,
    ScientificComparisonPolicy,
    compare_scientific_record,
)


def test_canonical_float_preserves_finite_values_for_policy_comparison() -> None:
    assert canonical_float(1.0) == 1.0
    assert canonical_float(1.0 + 1e-15) == 1.0 + 1e-15
    assert canonical_float(-0.0) == -0.0
    with pytest.raises(ValueError, match="non-finite"):
        canonical_float(float("nan"))
    with pytest.raises(ValueError, match="non-finite"):
        canonicalize_scientific({"value": float("inf")})


def test_semantic_tensor_digest_tracks_portable_structure() -> None:
    base = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    same_semantics = base + torch.tensor([[1e-15, 0.0]], dtype=torch.float64)
    changed = base + torch.tensor([[1e-9, 0.0]], dtype=torch.float64)
    digest = semantic_tensor_digest(base)
    assert digest == semantic_tensor_digest(same_semantics)
    assert digest == semantic_tensor_digest(changed)
    assert digest != semantic_tensor_digest(base.to(torch.float32))
    assert digest != semantic_tensor_digest(base.reshape(2, 1))


def test_complex_tensor_digest_is_finite_and_ordered() -> None:
    value = torch.tensor([1 + 2j, 3 + 4j], dtype=torch.complex128)
    digest = semantic_tensor_digest(value)
    assert digest["logical_dtype"] == "complex128"
    assert digest["numel"] == 2
    with pytest.raises(ValueError, match="non-finite"):
        semantic_tensor_digest(torch.tensor([complex(float("nan"), 0)]))


def test_field_aware_comparison_absorbs_roundoff_but_detects_change() -> None:
    policy = ScientificComparisonPolicy(
        version="test",
        numeric_paths={"$.metric": "scientific"},
        tolerances={"scientific": NumericTolerance(rtol=1e-7, atol=1e-12)},
    )
    assert not compare_scientific_record(
        {"metric": 1.0 + 1e-15}, {"metric": 1.0}, policy
    )
    assert compare_scientific_record(
        {"metric": 1.0 + 1e-6}, {"metric": 1.0}, policy
    )
    assert compare_scientific_record(
        {"unexpected": 1.0},
        {"unexpected": 1.0},
        ScientificComparisonPolicy("test", {}, policy.tolerances),
    )[0].reason == "numeric path absent from policy"


def test_baseline_writer_is_deterministic_and_requires_overwrite(
    tmp_path: Path,
) -> None:
    baseline = {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "value": canonicalize_scientific({"z": 1.25, "a": True}),
    }
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_phase1_scientific_baseline(baseline, first, overwrite=False)
    write_phase1_scientific_baseline(baseline, second, overwrite=False)
    assert first.read_bytes() == second.read_bytes()
    with pytest.raises(FileExistsError, match="pass --overwrite"):
        write_phase1_scientific_baseline(baseline, first, overwrite=False)
    write_phase1_scientific_baseline(baseline, first, overwrite=True)
    assert json.loads(first.read_text()) == baseline


def test_generator_rejects_failed_and_missing_artifacts(tmp_path: Path) -> None:
    for name in ("e0", "e1", "e2"):
        directory = tmp_path / name
        directory.mkdir()
    (tmp_path / "e0/e0_summary.json").write_text(
        json.dumps({"status": "fail"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="not a passing summary"):
        build_phase1_scientific_baseline(
            tmp_path / "e0",
            tmp_path / "e1",
            tmp_path / "e2",
            source_revision="test",
        )
    (tmp_path / "e0/e0_summary.json").write_text(
        json.dumps({"status": "pass"}), encoding="utf-8"
    )
    with pytest.raises(OSError):
        build_phase1_scientific_baseline(
            tmp_path / "e0",
            tmp_path / "e1",
            tmp_path / "e2",
            source_revision="test",
        )


def _synthetic_e0(tmp_path: Path) -> Path:
    directory = tmp_path / "e0-valid"
    directory.mkdir()
    config = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/paper1_e0_smoke.json").read_text()
    )
    for name in ("resolved_config.json", "accepted_production_config.json"):
        (directory / name).write_text(json.dumps(config))
    values = torch.arange(8, dtype=torch.float64).reshape(2, 4)
    digest = tensor_hash(values)
    torch.save(
        {"values": values, "metadata": {"tensor_hash": digest}},
        directory / "master_initial_conditions.pt",
    )
    (directory / "master_manifest.json").write_text(
        json.dumps({"tensor_hash": digest})
    )
    (directory / "e0_summary.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "selected_reference": {
                    "reference_nx": 32,
                    "dt": 0.001,
                    "fine_dt": 0.0001,
                    "solver": "spectral",
                },
            }
        )
    )
    (directory / "reference_convergence.csv").write_text(
        "kind,candidate_nx,error\nreference,32,0.0\n"
    )
    for name in (
        "resampling_checks.json",
        "input_interface_checks.json",
        "model1_identity.json",
    ):
        (directory / name).write_text(json.dumps({"status": "pass"}))
    return directory


@pytest.mark.parametrize("target", ["values", "metadata", "manifest"])
def test_e0_baseline_recomputes_master_tensor_hash(
    tmp_path: Path, target: str
) -> None:
    directory = _synthetic_e0(tmp_path)
    if target == "values":
        archive = torch.load(
            directory / "master_initial_conditions.pt", weights_only=False
        )
        archive["values"][0, 0] += 1
        torch.save(archive, directory / "master_initial_conditions.pt")
    elif target == "metadata":
        archive = torch.load(
            directory / "master_initial_conditions.pt", weights_only=False
        )
        archive["metadata"]["tensor_hash"] = "0" * 64
        torch.save(archive, directory / "master_initial_conditions.pt")
    else:
        (directory / "master_manifest.json").write_text(
            json.dumps({"tensor_hash": "0" * 64})
        )
    with pytest.raises(ValueError, match="tensor hash mismatch"):
        build_e0_scientific_record(directory)


def test_e0_baseline_rejects_nonfinite_master(tmp_path: Path) -> None:
    directory = _synthetic_e0(tmp_path)
    archive = torch.load(
        directory / "master_initial_conditions.pt", weights_only=False
    )
    archive["values"][0, 0] = float("nan")
    digest = tensor_hash(archive["values"])
    archive["metadata"]["tensor_hash"] = digest
    torch.save(archive, directory / "master_initial_conditions.pt")
    (directory / "master_manifest.json").write_text(
        json.dumps({"tensor_hash": digest})
    )
    with pytest.raises(ValueError, match="non-finite"):
        build_e0_scientific_record(directory)


def _manifest(directory: Path, relative_paths: list[str]) -> None:
    records = []
    for relative in relative_paths:
        path = directory / relative
        records.append(
            {
                "relative_path": relative,
                "byte_size": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    (directory / "artifact_manifest.json").write_text(
        json.dumps({"files": records})
    )


def test_baseline_manifest_requires_exact_safe_file_set(tmp_path: Path) -> None:
    directory = tmp_path / "artifacts"
    directory.mkdir()
    (directory / "value.txt").write_text("value")
    _manifest(directory, ["value.txt"])
    expected = {"artifact_manifest.json", "value.txt"}
    _verify_artifact_manifest(directory, expected=expected)
    (directory / "extra.txt").write_text("extra")
    with pytest.raises(ValueError, match="differs from contract"):
        _verify_artifact_manifest(directory, expected=expected)


def test_baseline_manifest_rejects_duplicate_records(tmp_path: Path) -> None:
    directory = tmp_path / "artifacts"
    directory.mkdir()
    (directory / "value.txt").write_text("value")
    _manifest(directory, ["value.txt", "value.txt"])
    with pytest.raises(ValueError, match="duplicate"):
        _verify_artifact_manifest(
            directory, expected={"artifact_manifest.json", "value.txt"}
        )


@pytest.mark.parametrize("relative", ["../outside.txt", "/absolute.txt"])
def test_baseline_manifest_rejects_unsafe_paths(
    tmp_path: Path, relative: str
) -> None:
    directory = tmp_path / "artifacts"
    directory.mkdir()
    (directory / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "files": [
                    {
                        "relative_path": relative,
                        "byte_size": 0,
                        "sha256": "0" * 64,
                    }
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="unsafe"):
        _verify_artifact_manifest(
            directory, expected={"artifact_manifest.json", "value.txt"}
        )


def test_baseline_plot_inventory_rejects_symlink_parent(tmp_path: Path) -> None:
    directory = tmp_path / "plots"
    directory.mkdir()
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "figure.png").write_bytes(b"png")
    try:
        (directory / "nested").symlink_to(victim, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    (directory / "plot_manifest.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "outputs": [
                    {"relative_path": "nested/figure.png", "format": "png"}
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="unsafe"):
        _plot_inventory(directory)
