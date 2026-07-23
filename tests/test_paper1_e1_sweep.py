import copy, json
from pathlib import Path

import pytest

from pol.paper1.e1_sweep import expand_sweep, preflight


def load(name):
    return json.loads(Path(name).read_text())


def test_default_sweep_has_130_valid_independent_runs():
    spec=load("configs/paper1_e1_sweep_main.json")
    base=load("configs/paper1_e1_main.json")
    runs,counts=expand_sweep(spec)
    valid,invalid,_=preflight(runs,base)
    assert counts=={"full_observation_resolution":64,"target_observation_grid":64,
                    "surrogate_resolution_J65":8,"surrogate_resolution_J96":6}
    assert len(runs)==len(valid)==130 and not invalid
    assert any(r.n_tar>r.J for r in valid)
    assert any(r.n_tar<r.J for r in valid)
    assert any(len(r.experiment_names)>1 for r in valid)


def test_dedupe_and_pathful_unknown_key():
    spec={"schema_version":"paper1-e1-sweep-v2","experiments":[
        {"name":"a","fixed":{"target_data_nx":32,"surrogate_internal_nx":32,"observation_dim":32},"grid":{}},
        {"name":"b","fixed":{"target_data_nx":32,"surrogate_internal_nx":32,"observation_dim":32},"grid":{}}]}
    runs,_=expand_sweep(spec)
    assert len(runs)==1 and runs[0].experiment_names==["a","b"]
    spec["experiments"][0]["grid"]={"bogus":[1]}
    with pytest.raises(ValueError,match=r"experiments\[0\]\.grid\.bogus"): expand_sweep(spec)


def test_invalid_preflight_has_specific_reason():
    base=load("configs/paper1_e1_smoke.json")
    spec={"schema_version":"paper1-e1-sweep-v2","invalid_run_policy":"skip","experiments":[{
        "name":"bad","fixed":{"target_data_nx":32,"surrogate_internal_nx":32,"observation_dim":8},"grid":{}}]}
    runs,_=expand_sweep(spec)
    valid,invalid,_=preflight(runs,base)
    assert not valid and "observation_dim=8" in invalid[0]["reason"]
