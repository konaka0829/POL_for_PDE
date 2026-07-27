import json
from pathlib import Path

import pytest
import torch

from pol.paper1.e2 import TensorCache
from pol.paper1.e2_cache import CACHE_SCHEMA_VERSION, stable_hash
from pol.paper1.e2_qa import validate_csv, validate_resume_output


def test_state_key_reused_independent_of_model_q_zeta_width(tmp_path):
    cache=TensorCache(tmp_path,resume=False)
    key={"dataset":"x","n_sur":32,"family":"burgers","nu":.1,"T":1}
    calls=[]
    first,_,digest=cache.get_or_compute("states",key,lambda:(calls.append(1) or torch.ones(2,3),{}))
    second,_,digest2=cache.get_or_compute("states",key,lambda:(calls.append(2) or torch.zeros(2,3),{}))
    assert digest==digest2 and torch.equal(first,second) and calls==[1] and cache.hits==1


def test_cache_key_tuple_round_trip_across_instances(tmp_path):
    key = {"solver": {"grid": (1, 2)}, "n_sur": 32}
    TensorCache(tmp_path, resume=False).get_or_compute(
        "states", key, lambda: (torch.ones(2, 3), {"solver": "test"}))
    calls = []
    value, _, _ = TensorCache(tmp_path, resume=True).get_or_compute(
        "states", key, lambda: (calls.append(1) or torch.zeros(2, 3), {}))
    assert torch.equal(value, torch.ones(2, 3))
    assert calls == []


def test_cache_compute_failure_releases_writer_lock(tmp_path):
    cache = TensorCache(tmp_path, resume=False)
    with pytest.raises(RuntimeError, match="injected"):
        cache.get_or_compute(
            "states", {"x": 1},
            lambda: (_ for _ in ()).throw(RuntimeError("injected")),
        )
    assert not list((tmp_path / "states").glob("*.lock"))
    value, _, _ = cache.get_or_compute(
        "states", {"x": 1}, lambda: (torch.ones(1), {})
    )
    assert torch.equal(value, torch.ones(1))


def test_cache_rejects_active_or_stale_writer_lock(tmp_path):
    cache = TensorCache(tmp_path, resume=False)
    digest = stable_hash(
        {"schema": CACHE_SCHEMA_VERSION, "kind": "states", "key": {"x": 1}}
    )
    directory = tmp_path / "states"
    directory.mkdir()
    (directory / f"{digest}.lock").write_text('{"pid":1}')
    with pytest.raises(ValueError, match="writer lock"):
        cache.get_or_compute("states", {"x": 1}, lambda: (torch.ones(1), {}))


@pytest.mark.parametrize("missing_suffix", [".pt", ".json", ".complete.json"])
def test_resume_rejects_half_cache_unit(tmp_path, missing_suffix):
    cache = TensorCache(tmp_path, resume=False)
    _, _, digest = cache.get_or_compute(
        "states", {"x": (1, 2)}, lambda: (torch.ones(2), {"solver": "test"}))
    (tmp_path / "states" / f"{digest}{missing_suffix}").unlink()
    with pytest.raises(ValueError, match="incomplete unit"):
        TensorCache(tmp_path, resume=True).get_or_compute(
            "states", {"x": (1, 2)}, lambda: (torch.zeros(2), {}))


def test_resume_rejects_extra_or_tampered_artifact(tmp_path):
    (tmp_path/"e2_summary.json").write_text('{"status":"pass"}')
    payload=tmp_path/"x.json"; payload.write_text('{"x":1}')
    import hashlib
    record={"relative_path":"e2_summary.json","size_bytes":(tmp_path/"e2_summary.json").stat().st_size,
            "sha256":hashlib.sha256((tmp_path/"e2_summary.json").read_bytes()).hexdigest()}
    record2={"relative_path":"x.json","size_bytes":payload.stat().st_size,"sha256":hashlib.sha256(payload.read_bytes()).hexdigest()}
    (tmp_path/"artifact_manifest.json").write_text(json.dumps({"files":[record,record2]}))
    with pytest.raises(ValueError, match="protocol mismatch"):
        validate_resume_output(tmp_path)
    (tmp_path/"fake.png").write_bytes(b"x")
    with pytest.raises(ValueError,match="protocol mismatch"): validate_resume_output(tmp_path)


def test_csv_qa_rejects_duplicate_and_nonfinite(tmp_path):
    path=tmp_path/"table.csv"
    path.write_text("id,value\n1,2\n1,3\n")
    with pytest.raises(ValueError,match="duplicate"): validate_csv(path,{"id","value"},("id",))
    path.write_text("id,value\n1,nan\n")
    with pytest.raises(ValueError,match="non-finite"): validate_csv(path,{"id","value"},("id",))


@pytest.mark.parametrize(
    "suffix", [".pt", ".json", ".complete.json", ".lock"]
)
def test_cache_never_follows_unit_symlinks(tmp_path, suffix):
    key = {"x": 7}
    _, _, digest = TensorCache(tmp_path, resume=False).get_or_compute(
        "states", key, lambda: (torch.ones(2), {})
    )
    unit = tmp_path / "states" / f"{digest}{suffix}"
    external = tmp_path / f"external{suffix.replace('.', '_')}"
    if suffix == ".lock":
        external.write_text("external lock", encoding="utf-8")
    else:
        external.write_bytes(unit.read_bytes())
        unit.unlink()
    unit.symlink_to(external)
    before = external.read_bytes()
    with pytest.raises(ValueError, match="unsafe"):
        TensorCache(tmp_path, resume=True).get_or_compute(
            "states", key, lambda: (torch.zeros(2), {})
        )
    assert external.read_bytes() == before


def test_nonresume_repairs_directory_substitution_without_external_access(tmp_path):
    key = {"x": 9}
    _, _, digest = TensorCache(tmp_path, resume=False).get_or_compute(
        "states", key, lambda: (torch.ones(2), {})
    )
    payload = tmp_path / "states" / f"{digest}.pt"
    payload.unlink()
    payload.mkdir()
    calls = []
    value, _, _ = TensorCache(tmp_path, resume=False).get_or_compute(
        "states", key,
        lambda: (calls.append(1) or torch.full((2,), 3.0), {}),
    )
    assert calls == [1]
    assert torch.equal(value, torch.full((2,), 3.0))
