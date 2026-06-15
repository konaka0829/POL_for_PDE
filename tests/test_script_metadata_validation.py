import json

import pytest
import torch

from scripts.run_headroom_burgers import main as headroom_main
from scripts.run_zeta_path import main as zeta_main


def _write_dataset(path, *, ic_type="fourier"):
    nx = 8
    torch.save(
        {
            "u0_train": torch.zeros(2, nx),
            "y_train": torch.zeros(2, nx),
            "u0_val": torch.zeros(1, nx),
            "y_val": torch.zeros(1, nx),
            "u0_test": torch.zeros(1, nx),
            "y_test": torch.zeros(1, nx),
            "metadata": {
                "T": 0.1,
                "dt": 0.01,
                "target_nu": 0.01,
                "nx": nx,
                "ic_type": ic_type,
                "domain_length": 1.0,
            },
        },
        path,
    )


def _write_config(path):
    path.write_text(
        json.dumps(
            {
                "target": {"T": 0.1, "dt": 0.01, "target_nu": 0.01},
                "domain": {"length": 1.0},
                "data": {"ntrain": 2, "nval": 1, "ntest": 1, "ic_type": "grf"},
            }
        ),
        encoding="utf-8",
    )


def test_zeta_path_detects_ic_type_mismatch_and_allow_flag(tmp_path):
    data = tmp_path / "data.pt"
    cfg = tmp_path / "config.json"
    _write_dataset(data, ic_type="fourier")
    _write_config(cfg)
    argv = [
        "--config",
        str(cfg),
        "--data-file",
        str(data),
        "--output-dir",
        str(tmp_path / "zeta"),
        "--model",
        "model2",
        "--reservoir",
        "static",
        "--zeta-grid",
        "1e-8",
    ]
    with pytest.raises(ValueError, match="ic_type"):
        zeta_main(argv)
    assert zeta_main([*argv, "--allow-metadata-mismatch"]) == 0


def test_headroom_detects_ic_type_mismatch(tmp_path):
    data = tmp_path / "data.pt"
    cfg = tmp_path / "config.json"
    _write_dataset(data, ic_type="fourier")
    _write_config(cfg)
    argv = [
        "--config",
        str(cfg),
        "--data-file",
        str(data),
        "--out-dir",
        str(tmp_path / "headroom"),
        "--zeta-grid",
        "1e-8",
    ]
    with pytest.raises(ValueError, match="ic_type"):
        headroom_main(argv)
