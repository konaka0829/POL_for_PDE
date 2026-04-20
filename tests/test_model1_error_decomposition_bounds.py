from pol.model123_1d.error_decomposition import ErrorDecompositionConfig, run_error_decomposition


def test_rowwise_triangle_bound_holds():
    cfg = ErrorDecompositionConfig(
        num_samples=8,
        nx=64,
        seed=1,
        batch_size=4,
        target_nu=0.05,
        T=0.1,
        Ttilde_values=[0.05, 0.1],
        dt=0.01,
        fine_dt=0.002,
        reservoir="reaction_diffusion",
        dtype="float64",
        device="cpu",
    )
    result = run_error_decomposition(cfg)
    for row in result["rows"]:
        assert row["D1_abs_l2h"] <= row["matched_plus_time_abs_l2h"] + 1e-12


def test_empirical_rhs_is_recorded_for_both_ttilde_values():
    cfg = ErrorDecompositionConfig(
        num_samples=8,
        nx=64,
        seed=2,
        batch_size=4,
        target_nu=0.05,
        T=0.1,
        Ttilde_values=[0.05, 0.1],
        dt=0.01,
        fine_dt=0.002,
        reservoir="ks",
        dtype="float64",
        device="cpu",
    )
    result = run_error_decomposition(cfg)
    summary = result["summary_rows"]
    assert len(summary) == 2
    for row in summary:
        assert "beta_empirical" in row
        assert "rhs_beta" in row
