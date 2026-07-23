import copy, json

import pytest

from pol.paper1.config import config_from_dict


def raw():
    return json.load(open("configs/paper1_e2_smoke.json"))


def test_e2_roundtrip_and_q_greater_than_j_allowed():
    cfg=config_from_dict(raw())
    assert cfg.e2.profile=="smoke"
    assert cfg.spatial.target_output_dim==17>cfg.spatial.observation_dim==16
    assert config_from_dict(cfg.to_dict())==cfg


def test_unknown_nested_key_has_path():
    value=raw(); value["e2"]["model3"]["bogus"]=1
    with pytest.raises(ValueError,match="e2.model3.bogus"): config_from_dict(value)


@pytest.mark.parametrize("mutation,match",[
    (lambda v:v["target"].update(equation="heat",solver="spectral_exact",dt=None,fine_dt=None,dealias=False),"requires a Burgers"),
    (lambda v:v["e2"]["burgers"].update(T_grid=[0.015]),"align"),
    (lambda v:v["e2"]["model3"].update(evaluation_seeds=[12]),"disjoint"),
    (lambda v:v["e2"]["convergence"].update(n_sur_candidates=[128,64]),"strictly increasing"),
])
def test_e2_invalid_config(mutation,match):
    value=raw(); mutation(value)
    with pytest.raises(ValueError,match=match): config_from_dict(value)
