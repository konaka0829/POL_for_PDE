import dataclasses

import pytest

from pol.paper1.config import config_from_dict, load_config_json, save_config_json


def test_smoke_and_main_config_load():
    smoke = load_config_json("configs/paper1_smoke.json")
    main = load_config_json("configs/paper1_main.json")
    assert smoke.spatial.observation_dim == 16
    assert main.spatial.target_master_nx == 4096
    assert main.data.preprocessing == "l2_scaling_only"


def test_config_round_trip(tmp_path):
    cfg = load_config_json("configs/paper1_smoke.json")
    path = tmp_path / "roundtrip.json"
    save_config_json(cfg, path)
    got = load_config_json(path)
    assert got == cfg
    assert dataclasses.is_dataclass(got.spatial)


def test_count_mismatch_error():
    raw = load_config_json("configs/paper1_smoke.json").to_dict()
    raw["data"]["total_samples"] = 13
    with pytest.raises(ValueError, match="total_samples"):
        config_from_dict(raw)


def test_j_greater_than_nsur_error():
    raw = load_config_json("configs/paper1_smoke.json").to_dict()
    raw["spatial"]["observation_dim"] = 33
    with pytest.raises(ValueError, match="observation_dim"):
        config_from_dict(raw)


def test_even_q_error():
    raw = load_config_json("configs/paper1_smoke.json").to_dict()
    raw["spatial"]["target_output_dim"] = 16
    with pytest.raises(ValueError, match="odd"):
        config_from_dict(raw)


def test_q_over_bandwidth_error():
    raw = load_config_json("configs/paper1_smoke.json").to_dict()
    raw["spatial"]["target_data_nx"] = 16
    raw["spatial"]["target_output_dim"] = 17
    with pytest.raises(ValueError, match="representable"):
        config_from_dict(raw)


def test_target_data_nx_cannot_exceed_master_nx():
    raw = load_config_json("configs/paper1_smoke.json").to_dict()
    raw["spatial"]["target_data_nx"] = 128
    with pytest.raises(ValueError, match="target_data_nx"):
        config_from_dict(raw)
