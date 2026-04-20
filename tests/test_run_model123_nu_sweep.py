import pytest

from argparse import Namespace

from scripts.run_model123_nu_sweep import (
    build_default_nu_values,
    build_job_env,
    clip_for_log,
    parse_models,
    parse_nu_values,
    validate_args,
)


def test_build_default_nu_values():
    values = build_default_nu_values()
    assert len(values) == 19
    assert values[0] == pytest.approx(1e-3)
    assert values[8] == pytest.approx(9e-3)
    assert values[9] == pytest.approx(1e-2)
    assert values[-1] == pytest.approx(1e-1)


def test_parse_models_rejects_invalid_name():
    try:
        parse_models("model1,invalid")
    except ValueError as exc:
        assert "Unsupported model" in str(exc)
    else:
        raise AssertionError("parse_models should reject invalid names")


def test_parse_nu_values_empty_uses_default_grid():
    values = parse_nu_values("")
    assert values == build_default_nu_values()


def test_clip_for_log_uses_eps_for_zero():
    eps = 1e-12
    assert clip_for_log(0.0, eps) == eps
    assert clip_for_log(1e-6, eps) == 1e-6


def test_build_job_env_limits_blas_threads():
    env = build_job_env({})
    assert env["OMP_NUM_THREADS"] == "1"
    assert env["MKL_NUM_THREADS"] == "1"
    assert env["OPENBLAS_NUM_THREADS"] == "1"
    assert env["NUMEXPR_NUM_THREADS"] == "1"
    assert env["TORCH_NUM_THREADS"] == "1"


def test_validate_args_rejects_nonpositive_max_workers():
    args = Namespace(
        data_file="data/burgers_T10_nu001.mat",
        train_split=1000.0 / 1200.0,
        ntrain=1000,
        ntest=200,
        batch_size=32,
        sub=1,
        T=1.0,
        Ttilde=1.0,
        dt=1e-4,
        burgers_fine_dt=1e-5,
        max_workers=0,
        models="model1",
        nu_values="",
    )
    with pytest.raises(ValueError, match="max-workers"):
        validate_args(args)
