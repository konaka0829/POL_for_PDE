# time-scaled generator defectがコード上で正しく実装されているかを確認する
import torch

# generatorの差を出力する関数と1 次元周期関数に対してspectral method で微分を計算する関数をインポート
from pol.model123_1d.error_decomposition import (
    defect_burgers_reservoir,
    scaled_defect_burgers_reservoir,
    scaled_defect_ks_reservoir,
    scaled_defect_reaction_diffusion_reservoir,
    spectral_derivatives_1d,
)


# Burgers reservoir の scaled defect において、α=1 とした場合、古い unscaled defect alias と一致するかを確認
def test_burgers_scaled_defect_alpha_one_matches_unscaled_alias():
    torch.manual_seed(0)
    z = torch.randn(3, 64, dtype=torch.float64)
    got = scaled_defect_burgers_reservoir(
        z,
        alpha=1.0,
        target_nu=0.05,
        res_burgers_nu=0.07,
        res_burgers_b=1.3,
    )
    old = defect_burgers_reservoir(
        z,
        target_nu=0.05,
        res_burgers_nu=0.07,
        res_burgers_b=1.3,
    )
    # 2 つのテンソルの各成分が近いかを判定
    assert torch.allclose(got, old, atol=1e-10, rtol=1e-10)


# Burgers scaled defect において、time-scaled effective coefficients が target の係数と一致すると defect が 0 になるか確認
def test_burgers_scaled_defect_zero_when_effective_coefficients_match():
    torch.manual_seed(1)
    z = torch.randn(4, 64, dtype=torch.float64)
    got = scaled_defect_burgers_reservoir(
        z,
        alpha=2.0,
        target_nu=0.05,
        res_burgers_nu=0.025,
        res_burgers_b=0.5,
    )
    assert torch.linalg.norm(got).item() < 1e-10


# reaction-diffusion reservoir と KS reservoir の scaled defect が、手で書いた明示的な数式と一致するか確認する
def test_rd_and_ks_scaled_defects_match_explicit_formulas():
    torch.manual_seed(2)
    z = torch.randn(3, 64, dtype=torch.float64)
    alpha = 1.7
    target_nu = 0.05
    ux, uxx, uxxxx = spectral_derivatives_1d(z)

    expected_rd = (target_nu - alpha * 0.02) * uxx - z * ux - alpha * 0.8 * z + alpha * 1.1 * z.pow(3)
    got_rd = scaled_defect_reaction_diffusion_reservoir(
        z,
        alpha=alpha,
        target_nu=target_nu,
        rd_nu=0.02,
        rd_alpha=0.8,
        rd_beta=1.1,
    )
    assert torch.allclose(got_rd, expected_rd, atol=1e-10, rtol=1e-10)

    expected_ks = (target_nu + alpha * 0.9) * uxx + (alpha * 1.4 - 1.0) * z * ux + alpha * 0.7 * uxxxx
    got_ks = scaled_defect_ks_reservoir(
        z,
        alpha=alpha,
        target_nu=target_nu,
        ks_b=1.4,
        ks_eta=0.9,
        ks_kappa=0.7,
    )
    assert torch.allclose(got_ks, expected_ks, atol=1e-10, rtol=1e-10)
