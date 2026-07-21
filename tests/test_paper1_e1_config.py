import copy,json,pytest
from pol.paper1.config import config_from_dict

def raw(): return json.load(open("configs/paper1_e1_smoke.json"))
def test_e1_config_and_unknown_path():
    assert config_from_dict(raw()).e1.output_dims==(3,5,9)
    r=raw(); r["e1"]["surrogate_cases"][0]["foo"]=1
    with pytest.raises(ValueError,match=r"e1.surrogate_cases\[0\].foo"): config_from_dict(r)
@pytest.mark.parametrize("mut",[lambda r:r["e1"].update(output_dims=[3,4,9]),lambda r:r["e1"].update(output_dims=[5,3,9]),lambda r:r["e1"].update(surrogate_cases=r["e1"]["surrogate_cases"][:1])])
def test_invalid_e1(mut):
    r=raw(); mut(r)
    with pytest.raises(ValueError): config_from_dict(r)
