import json

import torch

from scripts.run_zeta_path import main as zeta_main


def _write_split_pt(path):
    nx = 8
    payload = {
        "u0_train": torch.zeros(2, nx),
        "y_train": torch.zeros(2, nx),
        "u0_val": torch.zeros(1, nx),
        "y_val": torch.zeros(1, nx),
        "u0_test": torch.zeros(1, nx),
        "y_test": torch.zeros(1, nx),
        "metadata": {"T": 0.1, "dt": 0.01, "target_nu": 0.01, "nx": nx, "domain_length": 1.0},
    }
    torch.save(payload, path)


def test_zeta_path_sim_dtype_controls_feature_dtype_and_cache_hit(tmp_path):
    data = tmp_path / "data.pt"
    out_dir = tmp_path / "out"
    cache_dir = tmp_path / "cache"
    _write_split_pt(data)
    argv = [
        "--data-file",
        str(data),
        "--output-dir",
        str(out_dir),
        "--feature-cache-dir",
        str(cache_dir),
        "--model",
        "model2",
        "--reservoir",
        "static",
        "--ntrain",
        "2",
        "--nval",
        "1",
        "--ntest",
        "1",
        "--T",
        "0.1",
        "--dt",
        "0.01",
        "--target-nu",
        "0.01",
        "--zeta-grid",
        "1e-8",
        "--data-dtype",
        "float64",
        "--sim-dtype",
        "float64",
        "--ridge-dtype",
        "float64",
        "--use-feature-cache",
    ]
    assert zeta_main(argv) == 0
    first = json.loads((out_dir / "run_config.json").read_text(encoding="utf-8"))
    assert first["dtype"]["sim_dtype"] == "float64"
    assert first["dtype"]["feature_tensor_dtype"] == "float64"
    assert first["feature_cache"]["cache_hit"] is False
    assert first["feature_cache"]["dtype"] == "float64"

    assert zeta_main(argv) == 0
    second = json.loads((out_dir / "run_config.json").read_text(encoding="utf-8"))
    assert second["feature_cache"]["cache_hit"] is True
    assert second["feature_cache"]["dtype"] == "float64"


def test_zeta_path_cache_key_includes_sim_dtype(tmp_path):
    data = tmp_path / "data.pt"
    _write_split_pt(data)
    common = [
        "--data-file",
        str(data),
        "--feature-cache-dir",
        str(tmp_path / "cache"),
        "--model",
        "model2",
        "--reservoir",
        "static",
        "--ntrain",
        "2",
        "--nval",
        "1",
        "--ntest",
        "1",
        "--T",
        "0.1",
        "--dt",
        "0.01",
        "--target-nu",
        "0.01",
        "--zeta-grid",
        "1e-8",
        "--use-feature-cache",
    ]
    zeta_main([*common, "--output-dir", str(tmp_path / "out32"), "--sim-dtype", "float32"])
    zeta_main([*common, "--output-dir", str(tmp_path / "out64"), "--sim-dtype", "float64"])
    cfg32 = json.loads((tmp_path / "out32" / "run_config.json").read_text(encoding="utf-8"))
    cfg64 = json.loads((tmp_path / "out64" / "run_config.json").read_text(encoding="utf-8"))
    assert cfg32["feature_cache"]["surrogate_hash"] != cfg64["feature_cache"]["surrogate_hash"]


def test_zeta_path_subsample_uses_raw_nx_for_validation_and_effective_nx_for_cache(tmp_path):
    data = tmp_path / "data.pt"
    _write_split_pt(data)
    out_dir = tmp_path / "out_sub2"
    assert (
        zeta_main(
            [
                "--data-file",
                str(data),
                "--output-dir",
                str(out_dir),
                "--feature-cache-dir",
                str(tmp_path / "cache"),
                "--model",
                "model2",
                "--reservoir",
                "static",
                "--ntrain",
                "2",
                "--nval",
                "1",
                "--ntest",
                "1",
                "--T",
                "0.1",
                "--dt",
                "0.01",
                "--target-nu",
                "0.01",
                "--zeta-grid",
                "1e-8",
                "--sub",
                "2",
                "--use-feature-cache",
            ]
        )
        == 0
    )
    cfg = json.loads((out_dir / "run_config.json").read_text(encoding="utf-8"))
    assert cfg["metadata_validation"]["checks"]["nx"]["expected"] == 8
    assert cfg["metadata_validation"]["checks"]["nx"]["found"] == 8
    assert cfg["split"]["dataset_nx"] == 8
    assert cfg["split"]["effective_nx"] == 4
    assert cfg["feature_cache"]["effective_nx"] == 4
    assert cfg["feature_cache"]["sub"] == 2
