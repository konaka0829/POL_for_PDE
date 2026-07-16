import torch

from pol.paper1.grids import periodic_grid
from pol.paper1.interfaces import build_surrogate_initial_state, derive_finite_resolution_data
from pol.paper1.target_representation import real_fourier_analysis
from pol.paper1.config import load_config_json
from pol.paper1.e0 import run_interface_checks
from pol.paper1.initial_conditions import build_master_grf_initial_conditions


def test_interfaces_shapes_consistency_no_mutation_and_no_leak():
    x = periodic_grid(64, 1.0)
    low = 0.2 + torch.cos(4 * torch.pi * x)
    u = torch.stack([low, low + 0.4 * torch.cos(34 * torch.pi * x)])
    y = torch.stack([low, 2 * low])
    u_copy = u.clone(); y_copy = y.clone()
    finite = derive_finite_resolution_data(u, y, target_data_nx=16, target_output_dim=7, domain_length=1.0)
    internal = build_surrogate_initial_state(finite.u0_data, surrogate_internal_nx=24, domain_length=1.0)
    assert finite.u0_data.shape == (2, 16) and internal.shape == (2, 24)
    assert finite.u0_data.dtype == u.dtype and finite.u0_data.device == u.device
    assert torch.equal(u, u_copy) and torch.equal(y, y_copy)
    assert torch.allclose(finite.u0_data[0], finite.u0_data[1], atol=1e-12)
    assert torch.allclose(internal[0], internal[1], atol=1e-12)
    assert torch.allclose(finite.target_coefficients, real_fourier_analysis(y, 7, domain_length=1.0), atol=1e-12)


def test_main_equal_ref_target_synthetic_no_leak_without_pde():
    cfg = load_config_json("configs/paper1_e0_main.json")
    master = build_master_grf_initial_conditions(cfg)
    ids = torch.tensor(cfg.e0.calibration_sample_ids)
    reference = master.values_master.index_select(0, ids)
    result = run_interface_checks(cfg, master, reference)
    leak = result["no_high_frequency_leak"]
    assert cfg.spatial.reference_nx == cfg.spatial.target_data_nx == 4096
    assert leak["status"] == "pass"
    assert leak["synthetic_reference_nx"] == 8192
    assert leak["high_mode_representable_on_synthetic_reference"]
    assert leak["high_mode_discarded_by_target_grid"]
    assert result["finite_data_interface"]["sample_ids"] == list(cfg.e0.calibration_sample_ids)
