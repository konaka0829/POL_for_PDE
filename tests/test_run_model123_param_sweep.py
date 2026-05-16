from argparse import Namespace

import pytest

from scripts.run_model123_param_sweep import (
    build_job_env,
    build_range_values,
    canonical_parameter_name,
    clip_for_log,
    load_run_metrics,
    parse_models,
    parse_sweep_assignment,
    validate_args,
)


def test_parse_sweep_assignment_supports_required_parameters():
    for name in [
        "Ttilde",
        "res_burgers_nu",
        "res_burgers_b",
        "rd_nu",
        "rd_alpha",
        "rd_beta",
        "ks_b",
        "ks_eta",
        "ks_kappa",
        "dt",
        "K",
        "J",
    ]:
        assert parse_sweep_assignment(f"{name}=1").parameter.name == canonical_parameter_name(name)


def test_build_range_values():
    values = build_range_values(0.5, 0.7, 0.1)
    assert values == [0.5, 0.6, 0.7]


def test_parse_models_rejects_invalid_name():
    with pytest.raises(ValueError, match="Unsupported model"):
        parse_models("model1,invalid")


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


def test_validate_args_rejects_nonpositive_max_workers_before_missing_data_file():
    args = Namespace(
        data_file="does_not_exist.mat",
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
        best_k=10,
        models="model1",
        sweep=["Ttilde=1.0"],
        sweep_range=[],
        reservoir="burgers",
    )
    with pytest.raises(ValueError, match="max-workers"):
        validate_args(args)


def test_load_run_metrics_reads_absolute_and_relative_metrics(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_config.json").write_text(
        """
        {
          "train_absL2h": 0.1,
          "test_absL2h": 0.2,
          "train_relL2": 0.3,
          "test_relL2": 0.4,
          "resolved_obs": "full",
          "resolved_J": 32
        }
        """,
        encoding="utf-8",
    )
    metrics = load_run_metrics(run_dir)
    assert metrics["test_absL2h"] == 0.2
    assert metrics["test_relL2"] == 0.4
