from argparse import Namespace

import pytest
import scipy.io
import torch

from model123_burgers_1d import load_data
from pol.metadata import normalize_dataset_metadata, validate_dataset_metadata


def base_args(path, **kwargs):
    values = dict(
        data_mode="single_split",
        data_file=str(path),
        train_file=None,
        test_file=None,
        train_split=0.5,
        shuffle=False,
        seed=0,
        data_seed=0,
        split_seed=0,
        ntrain=2,
        nval=0,
        ntest=2,
        sub=1,
        T=0.1,
        dt=0.01,
        target_nu=0.05,
        data_dtype="float32",
        allow_metadata_mismatch=False,
        require_complete_metadata=False,
        expected_ic_type=None,
        expected_solver=None,
        expected_time_integrator=None,
        expected_burgers_scheme=None,
        expected_dealias=None,
        expected_equation=None,
        expected_domain_length=None,
    )
    values.update(kwargs)
    return Namespace(**values)


def write_mat(path, *, nx=8, metadata_nx=None, nu=0.05, ic_type=None, domain_length=None):
    a = torch.zeros(4, nx).numpy()
    u = torch.ones(4, nx).numpy()
    payload = {"a": a, "u": u, "T": 0.1, "dt": 0.01, "nu": nu, "nx": nx if metadata_nx is None else metadata_nx}
    if ic_type is not None:
        payload["ic_type"] = ic_type
    if domain_length is not None:
        payload["domain_length"] = domain_length
    scipy.io.savemat(path, payload)


def write_pt(path, *, nx=8, metadata_nx=None, target_nu=0.05, ic_type="grf", domain_length=None):
    payload = {
        "u0_train": torch.zeros(2, nx),
        "y_train": torch.ones(2, nx),
        "u0_val": torch.zeros(0, nx),
        "y_val": torch.ones(0, nx),
        "u0_test": torch.zeros(2, nx),
        "y_test": torch.ones(2, nx),
        "metadata": {
            "T": 0.1,
            "dt": 0.01,
            "target_nu": target_nu,
            "nx": nx if metadata_nx is None else metadata_nx,
            "ic_type": ic_type,
            **({} if domain_length is None else {"domain_length": domain_length}),
        },
    }
    torch.save(payload, path)


def test_normalize_dataset_metadata_aliases_nu_to_target_nu():
    meta = normalize_dataset_metadata({"nu": 0.05, "length": 1.0, "time_step": 0.01})
    assert meta["target_nu"] == 0.05
    assert meta["domain_length"] == 1.0
    assert meta["dt"] == 0.01


def test_subsampled_mat_validates_metadata_nx_against_raw_nx(tmp_path):
    data = tmp_path / "data.mat"
    write_mat(data, nx=8)
    args = base_args(data, sub=2)
    loaded = load_data(args)
    assert loaded[0].shape == (2, 4)
    assert loaded[6]["dataset_nx"] == 8
    assert loaded[6]["effective_nx"] == 4
    assert loaded[-1]["checks"]["nx"]["expected"] == 8
    assert loaded[-1]["checks"]["nx"]["found"] == 8
    assert loaded[-1]["checks"]["nx"]["ok"] is True


def test_subsampled_mat_still_errors_when_metadata_nx_is_wrong(tmp_path):
    data = tmp_path / "data.mat"
    write_mat(data, nx=8, metadata_nx=7)
    args = base_args(data, sub=2)
    with pytest.raises(ValueError, match="nx"):
        load_data(args)


def test_domain_length_mismatch_errors_for_pt(tmp_path):
    data = tmp_path / "data.pt"
    write_pt(data, domain_length=2.0)
    args = base_args(data, expected_domain_length=1.0)
    with pytest.raises(ValueError, match="domain_length"):
        load_data(args)


def test_mat_string_metadata_loads_without_float_cast(tmp_path):
    data = tmp_path / "data.mat"
    write_mat(data, nx=8, nu=0.05, ic_type="grf")
    args = base_args(
        data,
        expected_ic_type="grf",
        expected_solver=None,
        expected_domain_length=None,
    )
    loaded = load_data(args)
    assert loaded[0].shape == (2, 8)
    validation = loaded[-1]
    assert validation["checks"]["ic_type"]["found"] == "grf"


def test_target_nu_mismatch_errors_for_pt(tmp_path):
    data = tmp_path / "data.pt"
    write_pt(data, target_nu=0.04)
    args = base_args(data)
    with pytest.raises(ValueError, match="target_nu"):
        load_data(args)


def test_ic_type_mismatch_errors_for_pt(tmp_path):
    data = tmp_path / "data.pt"
    write_pt(data, ic_type="fourier")
    args = base_args(data, expected_ic_type="grf")
    with pytest.raises(ValueError, match="ic_type"):
        load_data(args)


def test_allow_metadata_mismatch_warns_and_continues(tmp_path):
    data = tmp_path / "data.pt"
    write_pt(data, target_nu=0.04)
    args = base_args(data, allow_metadata_mismatch=True)
    with pytest.warns(RuntimeWarning):
        loaded = load_data(args)
    assert loaded[0].shape == (2, 8)
    assert loaded[-1]["ok"] is False


def test_missing_legacy_metadata_warns_but_does_not_fail():
    result = validate_dataset_metadata(raw_metadata={}, expected={"T": 0.1}, strict=True)
    assert result["ok"] is True
    assert result["warnings"]
    assert result["has_missing"] is True


def test_require_complete_metadata_errors_on_missing():
    with pytest.raises(ValueError, match="incomplete"):
        validate_dataset_metadata(
            raw_metadata={},
            expected={"T": 0.1},
            strict=True,
            require_complete_metadata=True,
        )
