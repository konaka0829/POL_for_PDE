import torch

from pol.model123_1d.error_decomposition import (
    burgers_generator,
    defect_burgers_reservoir,
    defect_ks_reservoir,
    defect_reaction_diffusion_reservoir,
    spectral_derivatives_1d,
)


def test_explicit_defect_formulas_match_manual_expressions():
    torch.manual_seed(0)
    z = torch.randn(3, 64, dtype=torch.float64)
    nu_star = 0.05
    ux, uxx, uxxxx = spectral_derivatives_1d(z)

    expected_burgers = (nu_star - 0.07) * uxx + (1.3 - 1.0) * z * ux
    got_burgers = defect_burgers_reservoir(z, target_nu=nu_star, res_burgers_nu=0.07, res_burgers_b=1.3)
    assert torch.allclose(got_burgers, expected_burgers, atol=1e-10, rtol=1e-10)

    expected_rd = (nu_star - 1.0e-3) * uxx - z * ux - 0.8 * z + 1.1 * z.pow(3)
    got_rd = defect_reaction_diffusion_reservoir(
        z,
        target_nu=nu_star,
        rd_nu=1.0e-3,
        rd_alpha=0.8,
        rd_beta=1.1,
    )
    assert torch.allclose(got_rd, expected_rd, atol=1e-10, rtol=1e-10)

    expected_ks = (nu_star + 0.9) * uxx + (1.4 - 1.0) * z * ux + 0.7 * uxxxx
    got_ks = defect_ks_reservoir(
        z,
        target_nu=nu_star,
        ks_b=1.4,
        ks_eta=0.9,
        ks_kappa=0.7,
    )
    assert torch.allclose(got_ks, expected_ks, atol=1e-10, rtol=1e-10)


def test_burgers_generator_matches_manual_formula():
    torch.manual_seed(1)
    z = torch.randn(2, 64, dtype=torch.float64)
    ux, uxx, _ = spectral_derivatives_1d(z)
    expected = 0.05 * uxx - z * ux
    got = burgers_generator(z, nu=0.05)
    assert torch.allclose(got, expected, atol=1e-10, rtol=1e-10)
