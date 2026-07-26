from __future__ import annotations

import json
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
)


def test_canonical_float_absorbs_noise_but_detects_scientific_change() -> None:
    assert canonical_float(1.0) == canonical_float(1.0 + 1e-15)
    assert canonical_float(1.0) != canonical_float(1.0 + 1e-9)
    assert canonical_float(-0.0) == "-0"
    with pytest.raises(ValueError, match="non-finite"):
        canonical_float(float("nan"))
    with pytest.raises(ValueError, match="non-finite"):
        canonicalize_scientific({"value": float("inf")})


def test_semantic_tensor_digest_tracks_shape_dtype_and_values() -> None:
    base = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    same_semantics = base + torch.tensor([[1e-15, 0.0]], dtype=torch.float64)
    changed = base + torch.tensor([[1e-9, 0.0]], dtype=torch.float64)
    digest = semantic_tensor_digest(base)
    assert digest == semantic_tensor_digest(same_semantics)
    assert digest != semantic_tensor_digest(changed)
    assert digest != semantic_tensor_digest(base.to(torch.float32))
    assert digest != semantic_tensor_digest(base.reshape(2, 1))


def test_complex_tensor_digest_is_finite_and_ordered() -> None:
    value = torch.tensor([1 + 2j, 3 + 4j], dtype=torch.complex128)
    digest = semantic_tensor_digest(value)
    assert digest["encoding"] == "real_imag_interleaved"
    assert digest["numel"] == 2
    with pytest.raises(ValueError, match="non-finite"):
        semantic_tensor_digest(torch.tensor([complex(float("nan"), 0)]))


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
