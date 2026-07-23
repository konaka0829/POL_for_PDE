import json
from pathlib import Path

import pytest
import torch

from pol.paper1.e2 import TensorCache
from pol.paper1.e2_qa import validate_csv, validate_resume_output


def test_state_key_reused_independent_of_model_q_zeta_width(tmp_path):
    cache=TensorCache(tmp_path,resume=False)
    key={"dataset":"x","n_sur":32,"family":"burgers","nu":.1,"T":1}
    calls=[]
    first,_,digest=cache.get_or_compute("states",key,lambda:(calls.append(1) or torch.ones(2,3),{}))
    second,_,digest2=cache.get_or_compute("states",key,lambda:(calls.append(2) or torch.zeros(2,3),{}))
    assert digest==digest2 and torch.equal(first,second) and calls==[1] and cache.hits==1


def test_resume_rejects_extra_or_tampered_artifact(tmp_path):
    (tmp_path/"e2_summary.json").write_text('{"status":"pass"}')
    payload=tmp_path/"x.json"; payload.write_text('{"x":1}')
    import hashlib
    record={"relative_path":"e2_summary.json","size_bytes":(tmp_path/"e2_summary.json").stat().st_size,
            "sha256":hashlib.sha256((tmp_path/"e2_summary.json").read_bytes()).hexdigest()}
    record2={"relative_path":"x.json","size_bytes":payload.stat().st_size,"sha256":hashlib.sha256(payload.read_bytes()).hexdigest()}
    (tmp_path/"artifact_manifest.json").write_text(json.dumps({"files":[record,record2]}))
    assert validate_resume_output(tmp_path)
    (tmp_path/"fake.png").write_bytes(b"x")
    with pytest.raises(ValueError,match="artifact set mismatch"): validate_resume_output(tmp_path)


def test_csv_qa_rejects_duplicate_and_nonfinite(tmp_path):
    path=tmp_path/"table.csv"
    path.write_text("id,value\n1,2\n1,3\n")
    with pytest.raises(ValueError,match="duplicate"): validate_csv(path,{"id","value"},("id",))
    path.write_text("id,value\n1,nan\n")
    with pytest.raises(ValueError,match="non-finite"): validate_csv(path,{"id","value"},("id",))
