from argparse import Namespace
import json

import pytest
import torch

from model123_burgers_1d import compute_and_save_defect_outputs, resolve_defect_target_nu
from pol.model123_1d.error_decomposition import scaled_defect_burgers_reservoir


def _write_pt(path, *, metadata=True):
    payload = {
        "u0_train": torch.zeros(1, 16),
        "y_train": torch.zeros(1, 16),
        "u0_val": torch.zeros(0, 16),
        "y_val": torch.zeros(0, 16),
        "u0_test": torch.zeros(1, 16),
        "y_test": torch.zeros(1, 16),
    }
    if metadata:
        payload["metadata"] = {"target_nu": 0.01, "T": 0.1, "dt": 0.01, "nx": 16, "ic_type": "grf"}
    torch.save(payload, path)


def _args(path, **kwargs):
    values = {
        "defect_target_nu": None,
        "target_nu": None,
        "data_mode": "single_split",
        "data_file": str(path),
        "train_file": None,
        "test_file": None,
    }
    values.update(kwargs)
    return Namespace(**values)


def test_defect_target_nu_reads_pt_metadata(tmp_path):
    path = tmp_path / "data.pt"
    _write_pt(path)
    resolved = resolve_defect_target_nu(_args(path))
    assert resolved["target_nu"] == pytest.approx(0.01)
    assert resolved["target_nu_source"] == "dataset_metadata"


def test_defect_target_nu_explicit_override_has_priority(tmp_path):
    path = tmp_path / "data.pt"
    _write_pt(path)
    resolved = resolve_defect_target_nu(_args(path, defect_target_nu=0.02, target_nu=0.03))
    assert resolved["target_nu"] == pytest.approx(0.02)
    assert resolved["target_nu_source"] == "defect_target_nu"


def test_defect_target_nu_fallback_records_warning(tmp_path):
    path = tmp_path / "legacy.pt"
    _write_pt(path, metadata=False)
    with pytest.warns(RuntimeWarning, match="fallback"):
        resolved = resolve_defect_target_nu(_args(path))
    assert resolved["target_nu"] == pytest.approx(0.05)
    assert resolved["target_nu_source"] == "fallback"
    assert resolved["target_nu_warning"]


def test_matching_burgers_target_and_surrogate_have_zero_scaled_defect():
    x = torch.linspace(0.0, 1.0, 32, dtype=torch.float64)[:-1]
    u = torch.stack([torch.sin(2.0 * torch.pi * x), torch.cos(2.0 * torch.pi * x)])
    defect = scaled_defect_burgers_reservoir(
        u,
        alpha=1.0,
        target_nu=0.01,
        res_burgers_nu=0.01,
        res_burgers_b=1.0,
    )
    assert torch.max(torch.abs(defect)).item() < 1e-10


def test_time_scaled_defect_metrics_records_domain_metadata(tmp_path):
    data_file = tmp_path / "data.pt"
    _write_pt(data_file)
    out_dir = tmp_path / "out"
    args = Namespace(
        out_dir=str(out_dir),
        ntest=1,
        batch_size=1,
        defect_target_nu=0.01,
        target_nu=None,
        data_mode="single_split",
        data_file=str(data_file),
        train_file=None,
        test_file=None,
        expected_domain_length=2.0,
        T=0.02,
        Ttilde=0.02,
        dt=0.02,
        burgers_fine_dt=0.02,
        reservoir="burgers",
        rd_nu=1e-3,
        rd_alpha=1.0,
        rd_beta=1.0,
        res_burgers_nu=0.01,
        res_burgers_b=1.0,
        burgers_scheme="split_step",
        burgers_dealias=0,
        ks_b=1.0,
        ks_eta=1.0,
        ks_kappa=1.0,
        ks_dealias=False,
        input_scale=1.0,
        input_shift=0.0,
        defect_dtype="float64",
        device="cpu",
        defect_beta_mode="zero",
        defect_beta_fixed=0.0,
        defect_time_quadrature="trapezoid",
        model="model1",
    )
    out_dir.mkdir()
    x = torch.zeros(1, 64, dtype=torch.float64)
    metrics = compute_and_save_defect_outputs(
        args,
        64,
        x,
        x,
        torch.zeros(1, dtype=torch.float64),
        torch.zeros(1, dtype=torch.float64),
    )
    assert metrics["domain_length"] == pytest.approx(2.0)
    assert metrics["effective_nx"] == 64
    assert metrics["dx"] == pytest.approx(0.03125)
    assert metrics["l2h_convention"] == "dx=domain_length/effective_nx"
    saved = json.loads((out_dir / "time_scaled_defect_metrics.json").read_text(encoding="utf-8"))
    assert saved["domain_length"] == pytest.approx(2.0)
    row = json.loads((out_dir / "time_scaled_defect_per_sample.json").read_text(encoding="utf-8"))[0]
    assert row["effective_nx"] == 64
