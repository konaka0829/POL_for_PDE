import copy
from dataclasses import replace
import pytest

from pol.paper1.config import config_from_dict, load_config_json


def raw():
    return load_config_json("configs/paper1_e0_smoke.json").to_dict()


def test_reference_nx_and_legacy_aliases():
    value = raw()
    assert config_from_dict(value).spatial.reference_nx == 64
    value["spatial"]["target_master_nx"] = value["spatial"].pop("reference_nx")
    assert config_from_dict(value).spatial.target_master_nx == 64


def test_dual_reference_nx_same_and_conflict():
    value = raw(); value["spatial"]["target_master_nx"] = 64
    assert config_from_dict(value).spatial.reference_nx == 64
    value["spatial"]["target_master_nx"] = 32
    with pytest.raises(ValueError, match="conflicts"):
        config_from_dict(value)


@pytest.mark.parametrize("path,key", [((), "typo"), (("e0", "reference_tolerances"), "mean_relative_l22")])
def test_unknown_key_reports_path(path, key):
    value = raw(); node = value
    for part in path: node = node[part]
    node[key] = 1
    with pytest.raises(ValueError, match="unknown config key"):
        config_from_dict(value)


def test_invalid_ids_candidates_q_and_identity():
    cases = []
    value = raw(); value["e0"]["calibration_sample_ids"] = [99]; cases.append(value)
    value = raw(); value["e0"]["reference_nx_candidates"] = [64, 32]; cases.append(value)
    value = raw(); value["e0"]["q_reference_check"] = 33; cases.append(value)
    value = raw(); value["e0"]["model1_identity"]["observation_dim"] = 32; cases.append(value)
    for value in cases:
        with pytest.raises(ValueError): config_from_dict(value)


def test_time_candidate_order_duplicate_policy_and_alignment_validation():
    value = raw(); value["e0"]["time_candidates"] = list(reversed(value["e0"]["time_candidates"]))
    with pytest.raises(ValueError, match="strictly decreasing"): config_from_dict(value)
    value = raw(); value["e0"]["time_candidates"] = [{"dt": .01, "fine_dt": .006}, {"dt": .01, "fine_dt": .0051}]
    with pytest.raises(ValueError, match="duplicates"): config_from_dict(value)
    value = raw(); value["e0"]["selection_policy"] = "unknown"
    with pytest.raises(ValueError, match="unsupported"): config_from_dict(value)
    value = raw(); value["e0"]["time_candidates"][0]["dt"] = .007
    with pytest.raises(ValueError, match="aligned"): config_from_dict(value)


def test_invalid_reduced_j_validation():
    value = raw(); value["e0"]["reduced_j"]["observation_dim"] = 64
    with pytest.raises(ValueError, match="reduced_j"): config_from_dict(value)
