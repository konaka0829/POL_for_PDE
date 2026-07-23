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

@pytest.mark.parametrize("n_tar,J,n_sur", [(256,128,512),(128,256,512),(65,65,65)])
def test_target_and_observation_resolutions_are_independent(n_tar,J,n_sur):
    r=json.load(open("configs/paper1_e1_main.json"))
    r["spatial"].update(reference_nx=4096,target_data_nx=n_tar,surrogate_internal_nx=n_sur,observation_dim=J,target_output_dim=65)
    r["e1"].update(output_dims=[65],require_full_observation=J==n_sur)
    assert config_from_dict(r).spatial.observation_dim==J

@pytest.mark.parametrize("n_tar,J,n_sur,match", [
    (64,128,512,"not representable"),(128,64,512,"not representable"),
    (128,128,96,"observation_dim must be <="),
])
def test_independent_fourier_and_surrogate_limits(n_tar,J,n_sur,match):
    r=json.load(open("configs/paper1_e1_main.json"))
    r["spatial"].update(reference_nx=4096,target_data_nx=n_tar,surrogate_internal_nx=n_sur,observation_dim=J,target_output_dim=65)
    r["e1"].update(output_dims=[65],require_full_observation=False)
    with pytest.raises(ValueError,match=match): config_from_dict(r)

def test_full_observation_still_requires_equality():
    r=raw(); r["spatial"].update(surrogate_internal_nx=128,observation_dim=96)
    with pytest.raises(ValueError,match="full observation requires"): config_from_dict(r)
