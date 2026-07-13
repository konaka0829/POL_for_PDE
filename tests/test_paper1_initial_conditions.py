import torch

from pol.paper1.config import load_config_json
from pol.paper1.initial_conditions import build_master_grf_initial_conditions, initial_conditions_at_resolution
from pol.paper1.target_representation import real_fourier_analysis


def test_deterministic_master_grf():
    cfg = load_config_json("configs/paper1_smoke.json")
    a = build_master_grf_initial_conditions(cfg)
    b = build_master_grf_initial_conditions(cfg)
    assert torch.equal(a.sample_ids, torch.arange(cfg.data.total_samples))
    assert torch.allclose(a.values_master, b.values_master)
    assert torch.allclose(a.fourier_master, b.fourier_master)


def test_same_sample_common_low_modes_across_resolutions():
    cfg = load_config_json("configs/paper1_smoke.json")
    master = build_master_grf_initial_conditions(cfg)
    u32 = initial_conditions_at_resolution(master, 32)
    c64 = real_fourier_analysis(master.values_master, 9, domain_length=cfg.domain.length)
    c32 = real_fourier_analysis(u32, 9, domain_length=cfg.domain.length)
    assert torch.allclose(c64, c32, atol=1e-10, rtol=1e-10)


def test_no_fresh_rng_by_resolution_and_shape_dtype():
    cfg = load_config_json("configs/paper1_smoke.json")
    master = build_master_grf_initial_conditions(cfg)
    u16_a = initial_conditions_at_resolution(master, 16)
    u16_b = initial_conditions_at_resolution(master, 16)
    assert u16_a.shape == (cfg.data.total_samples, 16)
    assert u16_a.dtype == cfg.data.torch_dtype()
    assert torch.allclose(u16_a, u16_b)


def test_master_round_trip_low_band():
    cfg = load_config_json("configs/paper1_smoke.json")
    master = build_master_grf_initial_conditions(cfg)
    u32 = initial_conditions_at_resolution(master, 32)
    u64_low = initial_conditions_at_resolution(
        type(master)(
            sample_ids=master.sample_ids,
            values_master=u32,
            fourier_master=torch.fft.rfft(u32, dim=-1, norm="forward"),
            master_nx=32,
            domain_length=master.domain_length,
            seed=master.seed,
        ),
        64,
    )
    assert torch.allclose(
        real_fourier_analysis(u64_low, 17, domain_length=cfg.domain.length),
        real_fourier_analysis(master.values_master, 17, domain_length=cfg.domain.length),
        atol=1e-10,
    )
